"""Run the research Research Agent with an offline fixture or live public providers."""

from __future__ import annotations

import argparse
import asyncio
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, cast
from uuid import uuid4

from langgraph.types import Command

from vision_research_ops.adapters.llm import FixtureStructuredLLM, build_dashscope_llm
from vision_research_ops.adapters.papers import ArxivPaperProvider
from vision_research_ops.application.research_runtime import ResearchDependencies
from vision_research_ops.application.runtime import InMemoryApprovalRecorder
from vision_research_ops.application.services.paper_models import default_sem_problem_profile
from vision_research_ops.application.services.paper_store import LocalResearchStore
from vision_research_ops.application.state import create_initial_state, workflow_state_as_jsonable
from vision_research_ops.application.workflows import build_research_graph, workflow_config
from vision_research_ops.domain import (
    Approval,
    ApprovalDecision,
    GateKind,
    JsonValue,
    PatchOperation,
    PatchOperationType,
    QuerySpec,
    ResearchBudget,
    ResearchRequest,
    WorkflowStatus,
)
from vision_research_ops.ports import StructuredLLM
from vision_research_ops.settings import Settings, load_local_env

Mode = Literal["fixture", "live"]
Decision = Literal["approve", "edit", "reject"]
FIXTURE_NOW = datetime(2026, 8, 11, 12, 0, tzinfo=UTC)
DEFAULT_FIXTURE = Path("tests/research/fixtures/arxiv_feed.xml")


def _fixture_clock() -> datetime:
    return FIXTURE_NOW


def _live_clock() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True, slots=True)
class ResearchCliOptions:
    """Validated command-line options for one manual Research Agent run."""

    mode: Mode
    fixture_xml: Path
    output_root: Path | None
    workflow_id: str | None
    decision: Decision | None
    selected_paper_ids: tuple[str, ...]


def build_parser() -> argparse.ArgumentParser:
    """Build the intentionally small manual-sample argument surface."""
    parser = argparse.ArgumentParser(
        description=(
            "Run the human-gated paper Research Agent. Fixture mode is offline; "
            "live mode uses public arXiv metadata and DashScope."
        )
    )
    parser.add_argument("--mode", choices=("fixture", "live"), default="fixture")
    parser.add_argument("--fixture-xml", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--workflow-id")
    parser.add_argument("--decision", choices=("approve", "edit", "reject"))
    parser.add_argument(
        "--selected-paper-id",
        action="append",
        default=[],
        help="Paper ID retained by an edit decision; may be repeated.",
    )
    return parser


def parse_options(argv: Sequence[str] | None = None) -> ResearchCliOptions:
    """Parse CLI arguments into a typed immutable record."""
    namespace = build_parser().parse_args(argv)
    return ResearchCliOptions(
        mode=cast(Mode, namespace.mode),
        fixture_xml=cast(Path, namespace.fixture_xml),
        output_root=cast(Path | None, namespace.output_root),
        workflow_id=cast(str | None, namespace.workflow_id),
        decision=cast(Decision | None, namespace.decision),
        selected_paper_ids=tuple(cast(list[str], namespace.selected_paper_id)),
    )


def _research_request(*, request_id: str, now: datetime) -> ResearchRequest:
    return ResearchRequest(
        schema_version="1",
        request_id=request_id,
        revision=1,
        title="Daily SEM paper research",
        research_question="Which new CV papers merit a wafer SEM classification experiment?",
        dataset_id="problem-wafer-sem-v1",
        dataset_version="profile-v1",
        query_spec=QuerySpec(
            schema_version="1",
            keywords=[
                "SEM defect classification",
                "microscopy image classification",
                "industrial anomaly classification",
            ],
            domains=["cs.CV"],
            excluded_terms=["survey"],
        ),
        candidate_limit=5,
        budget=ResearchBudget(
            schema_version="1",
            max_provider_pages=2,
            max_provider_records=20,
            max_llm_calls=5,
            max_llm_tokens=6000,
            max_cost_estimate=1.0,
            max_candidate_repositories=5,
            max_adaptation_attempts=1,
            max_workflow_walltime_seconds=180,
        ),
        requested_by="pipeline-user",
        status=WorkflowStatus.PENDING,
        created_at=now,
        updated_at=now,
    )


def _fixture_transport(path: Path) -> Callable[[str, int], bytes]:
    def read_fixture(_url: str, _timeout_seconds: int) -> bytes:
        return path.read_bytes()

    return read_fixture


def _interrupt_payload(result: dict[str, object]) -> dict[str, object] | None:
    interrupts = result.get("__interrupt__")
    if not isinstance(interrupts, list | tuple) or len(interrupts) != 1:
        return None
    value = cast(object, getattr(interrupts[0], "value", None))
    return cast(dict[str, object], value) if isinstance(value, dict) else None


def _selected_ids(
    options: ResearchCliOptions,
    *,
    input_fn: Callable[[str], str],
) -> list[str]:
    if options.selected_paper_ids:
        return list(dict.fromkeys(options.selected_paper_ids))
    raw = input_fn("输入要保留的 paper_id (多个 ID 用英文逗号分隔): ")
    return list(dict.fromkeys(value.strip() for value in raw.split(",") if value.strip()))


def _approval(
    payload: dict[str, object],
    *,
    workflow_id: str,
    decision: Decision,
    selected_ids: list[str],
    decided_at: datetime,
) -> Approval:
    approval_decision = ApprovalDecision(decision.upper())
    edits: list[PatchOperation] = []
    if approval_decision is ApprovalDecision.EDIT:
        if not selected_ids:
            raise ValueError("edit decision requires at least one --selected-paper-id")
        edits = [
            PatchOperation(
                schema_version="1",
                op=PatchOperationType.REPLACE,
                path="/selected_paper_ids",
                value=cast(list[JsonValue], selected_ids),
                reason="The pipeline reviewer selected a candidate subset.",
            )
        ]
    return Approval(
        schema_version="1",
        approval_id=f"approval-{workflow_id}-{decision}",
        gate_kind=GateKind.CANDIDATE_SELECTION,
        subject_type=cast(str, payload["subject_type"]),
        subject_id=cast(str, payload["subject_id"]),
        subject_revision=cast(int, payload["subject_revision"]),
        decision=approval_decision,
        edits=edits,
        reason="Manual pipeline review of the Research Agent candidate slate.",
        actor_id="pipeline-reviewer",
        decided_at=decided_at,
        idempotency_key=f"idempotency-{workflow_id}-{decision}",
    )


def _status_summary(
    result: dict[str, object],
    *,
    store: LocalResearchStore,
    workflow_id: str,
) -> str:
    json_state = workflow_state_as_jsonable(result)
    return json.dumps(
        {
            "workflow_id": workflow_id,
            "status": json_state.get("status"),
            "phase": json_state.get("phase"),
            "result_path": str(store.result_path(workflow_id)),
            "watermark_path": str(store.watermark_path),
        },
        ensure_ascii=False,
        indent=2,
    )


async def run(
    options: ResearchCliOptions,
    *,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
) -> int:
    """Execute one bounded Research Agent run and return a process-style status code."""
    live_settings = Settings.from_env() if options.mode == "live" else None
    started_at = datetime.now(UTC) if options.mode == "live" else FIXTURE_NOW
    workflow_id = options.workflow_id or f"research-{options.mode}-{uuid4().hex[:12]}"
    if not workflow_id.strip():
        raise ValueError("workflow_id must not be blank")
    thread_id = f"thread-{workflow_id}"
    request_id = f"request-{workflow_id}"

    if options.mode == "fixture":
        provider = ArxivPaperProvider(
            timeout_seconds=20,
            transport=_fixture_transport(options.fixture_xml),
            clock=lambda: FIXTURE_NOW,
        )
        llm: StructuredLLM = FixtureStructuredLLM()
        output_root = options.output_root or Path("var/research-fixture") / workflow_id
        clock: Callable[[], datetime] = _fixture_clock
        overlap_minutes = 60
        lookback_hours = 24
    else:
        assert live_settings is not None
        provider = ArxivPaperProvider(timeout_seconds=live_settings.arxiv_timeout_seconds)
        llm = build_dashscope_llm(live_settings)
        output_root = options.output_root or live_settings.research_output_root
        clock = _live_clock
        overlap_minutes = live_settings.research_overlap_minutes
        lookback_hours = live_settings.research_initial_lookback_hours

    request = _research_request(request_id=request_id, now=started_at)
    store = LocalResearchStore(output_root)
    dependencies = ResearchDependencies(
        request=request,
        problem_profile=default_sem_problem_profile(),
        paper_provider=provider,
        structured_llm=llm,
        store=store,
        approval_recorder=InMemoryApprovalRecorder(),
        page_size=10,
        overlap_minutes=overlap_minutes,
        initial_lookback_hours=lookback_hours,
        clock=clock,
    )
    initial_state = create_initial_state(
        {
            "schema_version": "1",
            "workflow_id": workflow_id,
            "thread_id": thread_id,
            "request_id": request_id,
            "dataset_profile_id": request.dataset_id,
        }
    )
    graph = build_research_graph()
    paused = cast(
        dict[str, object],
        await graph.ainvoke(
            initial_state,
            config=workflow_config(thread_id),
            context=dependencies,
        ),
    )
    payload = _interrupt_payload(paused)
    if payload is None:
        output_fn(_status_summary(paused, store=store, workflow_id=workflow_id))
        return 2 if paused.get("status") == WorkflowStatus.FAILED else 0

    output_fn(
        json.dumps(
            {
                "gate_kind": payload.get("gate_kind"),
                "recommended_papers": payload.get("recommended_papers", []),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    decision = options.decision
    if decision is None:
        entered = input_fn("审批决定 [approve/edit/reject]: ").strip().casefold()
        if entered not in {"approve", "edit", "reject"}:
            raise ValueError("decision must be approve, edit, or reject")
        decision = cast(Decision, entered)
    selected_ids = _selected_ids(options, input_fn=input_fn) if decision == "edit" else []
    approval = _approval(
        payload,
        workflow_id=workflow_id,
        decision=decision,
        selected_ids=selected_ids,
        decided_at=clock(),
    )
    completed = cast(
        dict[str, object],
        await graph.ainvoke(
            Command(resume=approval.model_dump(mode="json")),
            config=workflow_config(thread_id),
            context=dependencies,
        ),
    )
    output_fn(_status_summary(completed, store=store, workflow_id=workflow_id))
    return 2 if completed.get("status") == WorkflowStatus.FAILED else 0


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments and run the async graph from a normal shell."""
    options = parse_options(argv)
    load_local_env()
    return asyncio.run(run(options))


if __name__ == "__main__":
    raise SystemExit(main())
