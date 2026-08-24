"""Run the offline research-to-evaluation sample through one top-level StateGraph."""

from __future__ import annotations

import argparse
import asyncio
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from vision_research_ops.adaptation import FixtureToolCallingAdaptationPlanner
from vision_research_ops.adapters.llm import build_dashscope_adaptation_planner
from vision_research_ops.application.pipeline_runtime import (
    DecisionProvider,
    PipelineDependencies,
)
from vision_research_ops.application.services.pipeline_models import (
    LocalPipelineSummaryStore,
    PipelineDecisionConfig,
    PipelineInitialInput,
    PipelineScenario,
    PipelineStoreError,
    PipelineSummary,
    create_pipeline_decision_config,
    create_pipeline_state,
)
from vision_research_ops.application.workflows.core import workflow_config
from vision_research_ops.application.workflows.pipeline import build_pipeline_graph
from vision_research_ops.pipeline.decisions import (
    DecisionName,
    DecisionProviderError,
    InteractiveDecisionProvider,
    ScriptedDecisionProvider,
    parse_decision_script,
)
from vision_research_ops.pipeline.fixture_runtime import (
    FIXTURE_NOW,
    FixturePipelineStageDriver,
)
from vision_research_ops.settings import Settings, load_local_env

type Mode = Literal["fixture"]
type AdaptationPlannerMode = Literal["scripted", "dashscope"]


@dataclass(frozen=True, slots=True)
class PipelineCliOptions:
    """Validated command-line choices for one fixture-first pipeline run."""

    mode: Mode
    adaptation_planner_mode: AdaptationPlannerMode
    workspace: Path
    workflow_id: str
    scenario: PipelineScenario
    auto_approve_sample: bool
    decisions: tuple[DecisionName, ...]

    @property
    def scripted_fixture_decisions(self) -> bool:
        return self.auto_approve_sample or bool(self.decisions)

    @property
    def decision_config(self) -> PipelineDecisionConfig:
        """Return the canonical identity of the complete decision configuration."""
        if self.auto_approve_sample:
            return create_pipeline_decision_config(mode="auto_approve_sample")
        if self.decisions:
            return create_pipeline_decision_config(
                mode="explicit_ordered",
                decisions=self.decisions,
            )
        return create_pipeline_decision_config(mode="interactive")


def build_parser() -> argparse.ArgumentParser:
    """Build the intentionally bounded Pipeline Sample CLI."""
    parser = argparse.ArgumentParser(
        description=(
            "SEM Research Agent local pipeline sample. It composes the research-to-evaluation "
            "LangGraph workflows offline and writes only below <workspace>/var/."
        )
    )
    commands = parser.add_subparsers(dest="command", required=True)
    sample = commands.add_parser(
        "sample",
        help="run the complete offline research-to-evaluation workflow",
    )
    sample.add_argument(
        "--mode",
        choices=("fixture",),
        default="fixture",
        help="only fixture mode is supported by the complete integrated sample",
    )
    sample.add_argument(
        "--adaptation-planner",
        choices=("scripted", "dashscope"),
        default="scripted",
        help=(
            "adaptation planner only: offline scripted real tool calls (default) or live DashScope "
            "ChatOpenAI.bind_tools; all execution and Gates remain deterministic"
        ),
    )
    sample.add_argument(
        "--workspace",
        type=Path,
        required=True,
        help="run directory; generated files are confined to its var/ child",
    )
    sample.add_argument(
        "--workflow-id",
        default="sem-pipeline-sample-v1",
        help="stable local workflow ID used for deterministic exact reruns",
    )
    sample.add_argument(
        "--scenario",
        choices=("happy", "smoke-failure"),
        default="happy",
        help="happy path or deterministic adaptation failure after its one repair budget",
    )
    scripted = sample.add_mutually_exclusive_group()
    scripted.add_argument(
        "--auto-approve-sample",
        action="store_true",
        help=(
            "sample-only scripted approvals; every decision still consumes a real interrupt "
            "through Command(resume=...)"
        ),
    )
    scripted.add_argument(
        "--decisions",
        help=(
            "sample-only ordered comma list such as "
            "approve,approve,approve,approve; omitted means interactive"
        ),
    )
    return parser


def parse_options(argv: Sequence[str] | None = None) -> PipelineCliOptions:
    """Parse the sample subcommand into one immutable record."""
    namespace = build_parser().parse_args(argv)
    decisions = (
        parse_decision_script(cast(str, namespace.decisions))
        if namespace.decisions is not None
        else ()
    )
    return PipelineCliOptions(
        mode=cast(Mode, namespace.mode),
        adaptation_planner_mode=cast(AdaptationPlannerMode, namespace.adaptation_planner),
        workspace=cast(Path, namespace.workspace),
        workflow_id=cast(str, namespace.workflow_id),
        scenario=cast(PipelineScenario, namespace.scenario),
        auto_approve_sample=cast(bool, namespace.auto_approve_sample),
        decisions=decisions,
    )


def _source_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _final_event(summary: PipelineSummary, *, reused_existing: bool) -> str:
    return json.dumps(
        {
            "event": "pipeline_completed",
            "workflow_id": summary.workflow_id,
            "status": summary.status,
            "conclusion": summary.conclusion,
            "summary_ref": summary.summary_ref,
            "scripted_fixture_decisions": summary.scripted_fixture_decisions,
            "decision_config_mode": summary.decision_config.mode,
            "decision_config_fingerprint": summary.decision_config.fingerprint,
            "adaptation_planner_mode": summary.adaptation_planner_mode,
            "reused_existing": reused_existing,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _existing_summary(
    options: PipelineCliOptions,
    store: LocalPipelineSummaryStore,
) -> PipelineSummary | None:
    path = store.resolve_ref(store.summary_ref(options.workflow_id))
    if not path.exists():
        return None
    summary = store.load_summary(options.workflow_id)
    if (
        summary.mode != options.mode
        or summary.adaptation_planner_mode != options.adaptation_planner_mode
        or summary.scenario != options.scenario
        or summary.scripted_fixture_decisions != options.scripted_fixture_decisions
        or summary.decision_config != options.decision_config
    ):
        raise PipelineStoreError("PIPELINE_SUMMARY_CONFLICT")
    return summary


async def run(
    options: PipelineCliOptions,
    *,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
    source_root: Path | None = None,
) -> int:
    """Execute or verify one deterministic integrated run and return a process-style code."""
    workspace = options.workspace.resolve()
    if workspace.exists() and not workspace.is_dir():
        output_fn('{"error":"PIPELINE_WORKSPACE_INVALID"}')
        return 2
    workspace.mkdir(parents=True, exist_ok=True)
    store = LocalPipelineSummaryStore(workspace / "var")
    try:
        existing = _existing_summary(options, store)
        if existing is not None:
            output_fn(_final_event(existing, reused_existing=True))
            return 0 if existing.status == "SUCCEEDED" else 2

        decision_provider: DecisionProvider
        if options.scripted_fixture_decisions:
            decision_provider = ScriptedDecisionProvider(
                decisions=options.decisions,
                auto_approve=options.auto_approve_sample,
            )
        else:
            decision_provider = InteractiveDecisionProvider(
                input_fn=input_fn,
                output_fn=output_fn,
            )
        adaptation_planner = (
            FixtureToolCallingAdaptationPlanner()
            if options.adaptation_planner_mode == "scripted"
            else build_dashscope_adaptation_planner(Settings.from_env())
        )
        driver = FixturePipelineStageDriver(
            workspace=workspace,
            source_root=(source_root or _source_root()),
            scenario=options.scenario,
            decision_provider=decision_provider,
            decision_config=options.decision_config,
            event_sink=output_fn,
            adaptation_planner=adaptation_planner,
            adaptation_planner_mode=options.adaptation_planner_mode,
        )
        if driver.scripted_fixture_decisions != options.scripted_fixture_decisions:
            raise ValueError("decision provenance does not match the CLI options")
        thread_id = f"thread-{options.workflow_id}"
        initial = create_pipeline_state(
            PipelineInitialInput(
                workflow_id=options.workflow_id,
                thread_id=thread_id,
                scenario=options.scenario,
                scripted_fixture_decisions=options.scripted_fixture_decisions,
            )
        )
        dependencies = PipelineDependencies(
            driver=driver,
            summary_store=store,
            event_sink=output_fn,
            clock=lambda: FIXTURE_NOW,
        )
        result = await build_pipeline_graph().ainvoke(
            initial,
            config=workflow_config(thread_id),
            context=dependencies,
        )
        if not isinstance(result, dict):
            raise ValueError("top-level Pipeline graph returned an invalid result")
        summary = store.load_summary(options.workflow_id)
        output_fn(_final_event(summary, reused_existing=False))
        return 0 if summary.status == "SUCCEEDED" else 2
    except (DecisionProviderError, OSError, PipelineStoreError, TypeError, ValueError) as error:
        if isinstance(error, PipelineStoreError):
            code = error.code
        elif isinstance(error, ValueError) and str(error) == (
            "DASHSCOPE_API_KEY is required for live LLM mode"
        ):
            code = "DASHSCOPE_API_KEY_REQUIRED"
        else:
            code = "PIPELINE_CLI_FAILED"
        output_fn(
            json.dumps(
                {"error": code, "status": "FAILED"},
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        return 2


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments and run the top-level graph from a normal shell."""
    options = parse_options(argv)
    load_local_env()
    return asyncio.run(run(options))


if __name__ == "__main__":
    raise SystemExit(main())
