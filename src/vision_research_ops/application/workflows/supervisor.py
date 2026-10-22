"""Serial StateGraph supervisor with evidence-constrained worker delegation."""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable
from typing import NotRequired, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphInterrupt
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from vision_research_ops.application.services.conversation_intent import (
    normalize_arxiv_target,
    normalize_github_target,
    supported_targets,
)
from vision_research_ops.application.services.supervisor_models import (
    ModelCallBudget,
    SupervisorError,
    SupervisorPolicy,
    SupervisorProposal,
    WorkerResult,
    WorkerTask,
)


class SupervisorState(TypedDict):
    """Small task state; no prompts, source text, approvals or worker transcripts."""

    goal: str
    observations: list[dict[str, object]]
    proposal: dict[str, object]
    delegations: int
    failure_code: str | None
    completed_tasks: NotRequired[list[str]]


def build_supervisor_graph(
    *,
    policy: SupervisorPolicy,
    worker: Callable[[WorkerTask], Awaitable[WorkerResult]],
    budget: ModelCallBudget,
    context_code_urls: tuple[str, ...] = (),
) -> CompiledStateGraph[SupervisorState, None, SupervisorState, SupervisorState]:
    """Compile decide -> delegate -> observe, bounded to three serial worker calls."""

    async def decide(state: SupervisorState) -> dict[str, object]:
        results = [WorkerResult.model_validate(item) for item in state["observations"]]
        if results and results[-1].status != "success":
            return {"proposal": {"action": "finish"}}
        try:
            proposal = SupervisorProposal.model_validate(
                await policy.decide(state["goal"], results, budget)
            )
            if proposal.action != "finish" and state["delegations"] >= 3:
                raise SupervisorError("SUPERVISOR_DELEGATION_LIMIT")
            targets = supported_targets(state["goal"])
            if (
                proposal.action == "research"
                and any(item.github_url for item in targets)
                and not any(item.arxiv_id for item in targets)
                and not re.search(
                    r"论文|\bpaper\b|\bresearch\b|\barxiv\b", state["goal"], re.IGNORECASE
                )
            ):
                raise SupervisorError("SUPERVISOR_RESEARCH_NOT_REQUESTED")
            if proposal.action == "research" and proposal.target is not None:
                target = normalize_arxiv_target(proposal.target)
                if target is None or target not in {item.arxiv_id for item in targets}:
                    raise SupervisorError("SUPERVISOR_UNSUPPORTED_PAPER_TARGET")
                proposal.target = target
            if proposal.action == "code":
                explicit_code = any(item.github_url for item in targets) or re.search(
                    r"代码|仓库|\bcode\b|\brepositor(?:y|ies)\b|\bgithub\b",
                    state["goal"],
                    re.IGNORECASE,
                )
                code_declined = re.search(
                    r"不(?:分析|读取|查找|要|需要)?(?:代码|仓库)|无需(?:分析)?代码|"
                    r"\bno\s+code\b|\bwithout\s+code\b|\bpaper[- ]only\b",
                    state["goal"],
                    re.IGNORECASE,
                )
                allowed = {item.github_url for item in targets if item.github_url}
                # A fresh observation supersedes prior context, including no-code papers.
                urls = results[-1].code_urls if results else context_code_urls
                allowed.update(url for url in urls if normalize_github_target(url) == url)
                target_url = (
                    None if proposal.target is None else normalize_github_target(proposal.target)
                )
                if target_url is None or target_url not in allowed:
                    raise SupervisorError("SUPERVISOR_UNSUPPORTED_REPOSITORY_TARGET")
                if not explicit_code or code_declined:
                    raise SupervisorError("SUPERVISOR_CODE_NOT_REQUESTED")
                proposal.target = target_url
            prior_tasks = state.get("completed_tasks", [])
            task_key = f"{proposal.action}:{proposal.target or ''}"
            if proposal.action != "finish" and task_key in prior_tasks:
                raise SupervisorError("SUPERVISOR_DUPLICATE_TASK")
            return {"proposal": proposal.model_dump(mode="json")}
        except SupervisorError as error:
            return {"failure_code": str(error), "proposal": {"action": "finish"}}
        except Exception:
            return {
                "failure_code": "SUPERVISOR_PROVIDER_OR_SCHEMA_FAILED",
                "proposal": {"action": "finish"},
            }

    async def delegate(state: SupervisorState) -> dict[str, object]:
        proposal = SupervisorProposal.model_validate(state["proposal"])
        if proposal.action == "finish":
            raise ValueError("finish cannot delegate")
        task = WorkerTask(worker=proposal.action, target=proposal.target, goal=state["goal"][:2000])
        try:
            result = WorkerResult.model_validate(await worker(task))
            if result.worker != task.worker:
                raise ValueError("worker identity mismatch")
        except GraphInterrupt:
            raise
        except SupervisorError as error:
            return {"failure_code": str(error), "proposal": {"action": "finish"}}
        except Exception:
            return {"failure_code": "SUPERVISOR_WORKER_FAILED", "proposal": {"action": "finish"}}
        return {
            "observations": [*state["observations"], result.model_dump(mode="json")],
            "delegations": state["delegations"] + 1,
            "completed_tasks": [
                *state.get("completed_tasks", []),
                f"{task.worker}:{task.target or ''}",
            ],
        }

    def route(state: SupervisorState) -> str:
        return (
            END if state["failure_code"] or state["proposal"]["action"] == "finish" else "delegate"
        )

    builder = StateGraph(SupervisorState)
    builder.add_node("decide", decide)
    builder.add_node("delegate", delegate)
    builder.add_edge(START, "decide")
    builder.add_conditional_edges("decide", route, {END: END, "delegate": "delegate"})
    builder.add_conditional_edges(
        "delegate",
        lambda state: END if state["failure_code"] else "decide",
        {END: END, "decide": "decide"},
    )
    return builder.compile(checkpointer=InMemorySaver())
