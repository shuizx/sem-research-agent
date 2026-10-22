"""Fresh bounded workers wrapping existing graphs with inherited checkpoints."""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from vision_research_ops.application.repository_insight_runtime import (
    RepositoryInsightDependencies,
    create_repository_insight_state,
)
from vision_research_ops.application.research_runtime import ResearchDependencies
from vision_research_ops.application.services.conversation_intent import normalize_github_target
from vision_research_ops.application.services.supervisor_models import WorkerResult
from vision_research_ops.application.state import create_initial_state
from vision_research_ops.application.workflows import (
    build_repository_insight_graph,
    build_research_graph,
)


def research_worker(
    dependencies: ResearchDependencies, workflow_id: str
) -> Callable[[], Awaitable[WorkerResult]]:
    """Bind fresh research dependencies once, so resume retains the worker session."""
    graph = build_research_graph(inherit_checkpointer=True)
    state = create_initial_state(
        {
            "schema_version": "1",
            "workflow_id": workflow_id,
            "thread_id": f"thread-{workflow_id}",
            "request_id": dependencies.request.request_id,
            "dataset_profile_id": dependencies.request.dataset_id,
        }
    )

    async def run() -> WorkerResult:
        await graph.ainvoke(state, context=dependencies)
        result = dependencies.store.load_result(workflow_id)
        status = "failed"
        if result.status == "REJECTED":
            status = "rejected"
        elif result.status == "NO_CANDIDATES":
            status = "no_results"
        elif result.status == "COMPLETED":
            status = "success"
        selected = [item for item in result.assessments if item.selected]
        urls = list(selected[0].paper.code_urls) if len(selected) == 1 else []
        return WorkerResult.model_validate(
            {
                "worker": "research",
                "status": status,
                "code_urls": [url for url in urls if normalize_github_target(url) == url][:5],
                "artifact_ref": f"research/{workflow_id}/papers.json",
                "paper_id": selected[0].paper.paper_id if len(selected) == 1 else None,
                "summary": f"Research terminal status: {result.status}; selected: {len(selected)}.",
                "reason": None if status == "success" else result.status,
            }
        )

    return run


def code_worker(
    dependencies: RepositoryInsightDependencies, workflow_id: str, target: str
) -> Callable[[], Awaitable[WorkerResult]]:
    """Wrap the existing Code ToolNode loop; repository gate bubbles to supervisor."""
    graph = build_repository_insight_graph(inherit_checkpointer=True)
    state = create_repository_insight_state(
        {
            "schema_version": "1",
            "workflow_id": workflow_id,
            "thread_id": f"thread-{workflow_id}",
            "repository_url": target,
        }
    )

    async def run() -> WorkerResult:
        final = await graph.ainvoke(state, context=dependencies)
        status = final.get("status")
        if status == "REJECTED":
            return WorkerResult(worker="code", status="rejected", reason="HUMAN_REJECTED")
        if status != "COMPLETED":
            return WorkerResult(worker="code", status="failed", reason=final.get("failure_code"))
        result = dependencies.store.load_result(workflow_id)
        return WorkerResult(
            worker="code",
            status="success",
            artifact_ref=result.result_ref,
            summary=(
                f"Fixed repository {result.repository_url}; "
                f"commit {result.resolution.commit_sha}; fit {result.advice.adaptation_fit}."
            ),
        )

    return run
