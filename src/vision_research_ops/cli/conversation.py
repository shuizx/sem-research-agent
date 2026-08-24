"""Human-facing orchestration at the bounded conversational CLI boundary."""
# ruff: noqa: RUF001

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, cast
from uuid import uuid4

from langgraph.types import Command

from vision_research_ops.adapters.llm import FixtureStructuredLLM, build_dashscope_llm
from vision_research_ops.adapters.papers import ArxivPaperProvider
from vision_research_ops.application.repository_insight_runtime import (
    create_repository_insight_state,
)
from vision_research_ops.application.services.conversation_context import (
    ConversationContext,
    ConversationContextStoreError,
    CurrentPaperContext,
    CurrentRepositoryContext,
    LocalConversationContextStore,
)
from vision_research_ops.application.services.conversation_intent import (
    ConversationIntentName,
    ConversationRoutingError,
    StructuredIntentRouter,
    deterministic_intent,
    has_ambiguous_supported_targets,
    normalize_arxiv_target,
    normalize_github_target,
)
from vision_research_ops.application.services.paper_analysis import (
    score_assessments,
    unscored_assessment,
)
from vision_research_ops.application.services.paper_models import (
    ResearchPaperAssessment,
    default_sem_problem_profile,
)
from vision_research_ops.application.services.paper_retrieval import normalize_raw_paper
from vision_research_ops.application.services.paper_store import LocalResearchStore
from vision_research_ops.application.services.repository_insight_models import (
    RepositoryInsightResult,
)
from vision_research_ops.application.workflows import (
    build_repository_insight_graph,
    workflow_config,
)
from vision_research_ops.cli import pipeline, research
from vision_research_ops.domain import Approval, ApprovalDecision, GateKind
from vision_research_ops.ports import ExternalPaperId, OperationContext, PortError, StructuredLLM
from vision_research_ops.repository_insight import build_repository_insight_dependencies
from vision_research_ops.settings import Settings

ConversationMode = Literal["fixture", "live"]
CurrentPaperSummary = CurrentPaperContext


def _fixture_transport(path: Path) -> Callable[[str, int], bytes]:
    def read_fixture(_url: str, _timeout_seconds: int) -> bytes:
        return path.read_bytes()

    return read_fixture


def _event_object(value: str) -> dict[str, object] | None:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _gate_message(events: list[str]) -> str | None:
    fallback_kind: str | None = None
    for event in reversed(events):
        payload = _event_object(event)
        if payload is None:
            continue
        gate_kind = payload.get("gate_kind")
        if isinstance(gate_kind, str):
            fallback_kind = fallback_kind or gate_kind
            recommended = payload.get("recommended_papers")
            if isinstance(recommended, list) and recommended:
                first = recommended[0]
                if isinstance(first, dict) and isinstance(first.get("title"), str):
                    return f"人工 Gate：{gate_kind}。候选：{first['title']}。"
            evidence = payload.get("evidence")
            if isinstance(evidence, dict):
                details: list[str] = []
                for key, label in (
                    ("title", "论文"),
                    ("relevance_score", "相关度"),
                    ("repository_url", "仓库"),
                    ("confidence", "可信度"),
                    ("patch_sha256", "Patch"),
                    ("smoke_capability", "Smoke"),
                    ("spec_hash", "训练配置"),
                    ("seed", "seed"),
                ):
                    value = evidence.get(key)
                    if isinstance(value, str | int | float) and not isinstance(value, bool):
                        details.append(f"{label}={value}")
                if details:
                    return f"人工 Gate：{gate_kind}。{'；'.join(details)}。"
    return None if fallback_kind is None else f"人工 Gate：{fallback_kind}。"


class ConversationSession:
    """A serial, in-process conversation that exposes no arbitrary tools."""

    def __init__(
        self,
        *,
        workspace: Path,
        mode: ConversationMode,
        router: StructuredIntentRouter,
        input_fn: Callable[[str], str] = input,
        output_fn: Callable[[str], None] = print,
        fixture_xml: Path = research.DEFAULT_FIXTURE,
        settings: Settings | None = None,
        context_path: Path | None = None,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._workspace = workspace
        self._mode = mode
        self._router = router
        self._input_fn = input_fn
        self._output_fn = output_fn
        self._fixture_xml = fixture_xml
        self._settings = settings
        self._clock = clock
        self._debug_json = False
        self._turns = 0
        self._last_intent: ConversationIntentName | None = None
        self._context_store = (
            None if context_path is None else LocalConversationContextStore(context_path)
        )
        loaded = None if self._context_store is None else self._context_store.load()
        context = (
            ConversationContext.empty(updated_at=self._clock())
            if loaded is None
            else loaded.context
        )
        self._current_paper = context.current_paper
        self._current_repository = context.current_repository
        self._last_artifact_ref = context.last_successful_artifact_ref
        self._memory_restored = False if loaded is None else loaded.restored
        self._memory_warning = None if loaded is None else loaded.warning

    @property
    def debug_json(self) -> bool:
        """Return whether explicit raw-event output is enabled for this session."""
        return self._debug_json

    def introduction(self) -> list[str]:
        """Return bounded startup copy, including only safe context-load status."""
        mode_message = (
            "当前模式：实时 arXiv + DashScope。"
            if self._mode == "live"
            else "当前模式：本地样例；结果不是实时检索。"
        )
        lines = [
            "SEM Research Agent：受限的本地研究工作流。",
            mode_message,
            "可检索论文、分析一个 arXiv 论文、审批后读取公开 GitHub 固定源码快照并给出适配建议。",
            (
                "限制：GitHub 分析不是 git clone，不执行、patch、Smoke 或训练仓库代码，"
                "也不使用公司数据。"
            ),
            "示例：开始检索文献｜当前候选｜继续找代码｜https://arxiv.org/abs/2608.01234｜/help",
        ]
        if self._memory_warning is not None:
            lines.append(self._memory_warning)
        elif self._memory_restored:
            restored: list[str] = []
            if self._current_paper is not None:
                restored.append("当前论文")
            if self._current_repository is not None:
                restored.append("当前仓库")
            if not restored and self._last_artifact_ref is not None:
                restored.append("最近成功产物")
            lines.append(f"已恢复上次成功的会话上下文：{'、'.join(restored)}。")
        return lines

    def _emit(self, message: str) -> None:
        self._output_fn(message)

    def _ask_gate(self, events: list[str]) -> Callable[[str], str]:
        def ask(prompt: str) -> str:
            if "approve/edit/reject" not in prompt:
                return self._input_fn(prompt)
            summary = _gate_message(events)
            if summary is not None:
                self._emit(summary)
            while True:
                entered = self._input_fn("决定 [approve/edit/reject]: ").strip().casefold()
                if entered in {"approve", "edit", "reject"}:
                    return entered
                self._emit("请输入 approve、edit 或 reject。")

        return ask

    def _emit_debug(self, events: list[str]) -> None:
        if self._debug_json:
            for event in events:
                self._emit(event)

    async def handle(self, message: str) -> bool:
        """Process exactly one user turn and return whether the REPL should continue."""
        self._turns += 1
        stripped = message.strip()
        if stripped.casefold() == "/clear":
            self._clear_context()
            return True
        if stripped.casefold().startswith("/debug"):
            words = stripped.casefold().split()
            if len(words) == 2 and words[1] in {"on", "开启"}:
                self._debug_json = True
            elif len(words) == 2 and words[1] in {"off", "关闭"}:
                self._debug_json = False
            self._emit(f"调试 JSON：{'开启' if self._debug_json else '关闭'}。")
            return True
        intent = deterministic_intent(stripped)
        if intent is None:
            try:
                intent = await self._router.route(stripped)
            except ConversationRoutingError as error:
                self._emit(f"意图路由失败：{error.code}。未启动下游工作流。")
                return True
        if intent.intent is ConversationIntentName.EXIT:
            self._last_intent = intent.intent
            self._emit("已退出 SEM Research Agent 对话。")
            return False
        if intent.intent is ConversationIntentName.SHOW_HELP:
            for line in self.introduction()[1:]:
                self._emit(line)
            self._emit("命令：/help、/current、/continue、/status、/clear、/debug on|off、/exit。")
            return True
        if intent.intent is ConversationIntentName.SHOW_STATUS:
            self._emit(self._status_message())
            return True
        if intent.intent is ConversationIntentName.SHOW_CURRENT_PAPER:
            self._last_intent = intent.intent
            self._emit(self._current_paper_message())
            return True
        if intent.intent is ConversationIntentName.CONTINUE_CURRENT_PAPER:
            self._last_intent = intent.intent
            await self._continue_current_paper()
            return True
        self._last_intent = intent.intent
        if intent.intent is ConversationIntentName.OUT_OF_SCOPE:
            if has_ambiguous_supported_targets(stripped):
                self._emit(
                    "目标不明确：一次只提供一个 arXiv 论文或一个规范 GitHub 仓库地址；"
                    "本轮未启动任何下游工作流。"
                )
            else:
                self._emit(
                    "此请求超出 SEM Research Agent 范围；仅支持论文检索/arXiv 分析、"
                    "审批后的公开 GitHub 只读代码分析和本地样例流程。"
                )
            return True
        if intent.intent is ConversationIntentName.RESEARCH_LATEST:
            await self._run_research()
            return True
        if intent.intent is ConversationIntentName.ANALYZE_ARXIV_PAPER:
            await self._analyze_arxiv(intent.arxiv_id)
            return True
        if intent.intent is ConversationIntentName.ANALYZE_GITHUB_REPOSITORY:
            await self._run_repository_insight(intent.github_url)
            return True
        if intent.intent is ConversationIntentName.RUN_PIPELINE_SAMPLE:
            await self._run_pipeline()
            return True
        self._emit("意图路由失败：CONVERSATION_INTENT_UNSUPPORTED。未启动下游工作流。")
        return True

    def _current_paper_message(self) -> str:
        current = self._current_paper
        if current is None:
            paper_message = (
                "当前论文：暂无。请先检索并在候选 Gate 中选择一篇论文，或分析单篇 arXiv 论文。"
            )
        else:
            arxiv_label = current.arxiv_id or "无"
            code_urls = "、".join(current.code_urls) if current.code_urls else "无"
            paper_message = (
                f"当前论文：{current.title}（{current.paper_id}）。\n"
                f"arXiv：{arxiv_label}；建议：{current.recommendation}。\n"
                f"公开代码：{code_urls}。"
            )
        repository = self._current_repository
        if repository is None:
            repository_message = "当前仓库：暂无。"
        else:
            repository_message = (
                f"当前仓库：{repository.repository_url}。\n"
                f"固定 commit：{repository.commit_sha}；许可证：{repository.license_spdx}；"
                f"适配程度：{repository.adaptation_fit}。\n"
                f"已读文件：{'、'.join(repository.read_files)}。"
            )
        return (
            f"{paper_message}\n{repository_message}\n"
            f"最近成功产物：{self._last_artifact_ref or '无'}。"
        )

    def _status_message(self) -> str:
        current = self._current_paper
        current_label = "暂无" if current is None else f"{current.title}（{current.paper_id}）"
        repository = self._current_repository
        repository_label = "暂无" if repository is None else repository.repository_url
        return (
            "状态："
            f"turns={self._turns}，last_intent={self._last_intent or 'none'}，"
            f"当前论文={current_label}，当前仓库={repository_label}，"
            f"最近成功产物={self._last_artifact_ref or 'none'}。"
        )

    def _context_snapshot(self) -> ConversationContext:
        return ConversationContext(
            current_paper=self._current_paper,
            current_repository=self._current_repository,
            last_successful_artifact_ref=self._last_artifact_ref,
            updated_at=self._clock(),
        )

    def _persist_context(self) -> bool:
        if self._context_store is None:
            return True
        snapshot = self._context_snapshot()
        try:
            self._context_store.save(snapshot)
        except ConversationContextStoreError:
            return False
        self._memory_restored = snapshot.has_working_context
        self._memory_warning = None
        return True

    def _emit_memory_write_warning(self) -> None:
        self._emit("本次业务结果已完成，但会话记忆未保存；重启后不会恢复本次上下文。")

    def _clear_context(self) -> None:
        empty = ConversationContext.empty(updated_at=self._clock())
        if self._context_store is not None:
            try:
                self._context_store.save(empty)
            except ConversationContextStoreError:
                self._emit("会话记忆清除失败：原上下文保持不变，未删除任何业务产物。")
                return
        self._current_paper = None
        self._current_repository = None
        self._last_artifact_ref = None
        self._memory_restored = False
        self._memory_warning = None
        self._emit("会话记忆已清除；历史论文、仓库和训练等业务产物均未删除。")

    async def _continue_current_paper(self) -> None:
        current = self._current_paper
        if current is None:
            self._emit("继续找代码：暂无当前论文。请先选择或分析一篇论文。")
            return
        if len(current.code_urls) != 1:
            self._emit(
                "继续找代码：当前论文没有唯一的公开 code URL；"
                "请提供一个规范 https://github.com/<owner>/<repo> 供只读分析。"
            )
            return
        github_url = normalize_github_target(current.code_urls[0])
        if github_url is None:
            self._emit(
                "继续找代码：当前论文的唯一 code URL 不是规范 GitHub 仓库；"
                "请提供一个规范 https://github.com/<owner>/<repo> 供只读分析。"
            )
            return
        await self._run_repository_insight(github_url)

    def _set_current_paper_from_assessment(
        self, assessment: ResearchPaperAssessment, artifact_ref: str
    ) -> bool:
        """Record only a validated successful paper and clear stale repository context."""
        decision = assessment.applicability
        if decision is None:
            return False
        paper = assessment.paper
        try:
            current = CurrentPaperSummary(
                paper_id=paper.paper_id,
                title=paper.title,
                arxiv_id=paper.arxiv_id,
                code_urls=list(paper.code_urls),
                recommendation=decision.recommendation,
                artifact_ref=artifact_ref,
            )
        except ValueError:
            return False
        self._current_paper = current
        self._current_repository = None
        self._last_artifact_ref = artifact_ref
        return self._persist_context()

    def _set_current_repository_from_result(self, result: RepositoryInsightResult) -> bool:
        """Record only the small public projection of a strict completed result."""
        try:
            current = CurrentRepositoryContext(
                repository_url=result.repository_url,
                commit_sha=result.resolution.commit_sha,
                license_spdx=result.metadata.license_spdx or "UNKNOWN",
                adaptation_fit=result.advice.adaptation_fit,
                read_files=[item.path for item in result.read_files],
                result_ref=result.result_ref,
            )
        except ValueError:
            return False
        self._current_repository = current
        self._last_artifact_ref = result.result_ref
        return self._persist_context()

    def _set_last_successful_artifact(self, artifact_ref: str) -> bool:
        """Persist one successful non-paper/non-repository result reference."""
        previous = self._last_artifact_ref
        self._last_artifact_ref = artifact_ref
        try:
            return self._persist_context()
        except ValueError:
            self._last_artifact_ref = previous
            return False

    async def _run_research(self) -> None:
        workflow_id = f"conversation-research-{uuid4().hex[:12]}"
        output_root = self._workspace / "research"
        events: list[str] = []
        options = research.ResearchCliOptions(
            mode=self._mode,
            fixture_xml=self._fixture_xml,
            output_root=output_root,
            workflow_id=workflow_id,
            decision=None,
            selected_paper_ids=(),
        )
        exit_code = await research.run(
            options,
            input_fn=self._ask_gate(events),
            output_fn=events.append,
        )
        self._emit_debug(events)
        result_path = output_root / workflow_id / "papers.json"
        relative_ref = f"research/{workflow_id}/papers.json"
        if not result_path.exists():
            self._emit("论文检索失败：RESEARCH_RESULT_UNAVAILABLE。")
            return
        try:
            result = LocalResearchStore(output_root).load_result(workflow_id)
        except (OSError, TypeError, ValueError):
            self._emit("论文检索失败：RESEARCH_RESULT_INVALID。")
            return
        selected = len(result.selected_paper_ids)
        if exit_code != 0:
            self._emit(f"论文检索失败：{result.status}；产物：{relative_ref}。")
        elif result.status == "COMPLETED":
            selected_assessments = [item for item in result.assessments if item.selected]
            memory_saved = True
            if len(selected_assessments) == 1:
                memory_saved = self._set_current_paper_from_assessment(
                    selected_assessments[0], relative_ref
                )
            self._emit(
                f"论文检索完成：推荐 {len(result.recommended_paper_ids)} 篇，已选择 {selected} 篇；"
                f"产物：{relative_ref}。"
            )
            if not memory_saved:
                self._emit_memory_write_warning()
        elif result.status == "REJECTED":
            self._emit(f"论文候选已按人工决定拒绝；产物：{relative_ref}。")
        elif result.status == "NO_CANDIDATES":
            self._emit(f"论文检索完成：本次没有合格候选；产物：{relative_ref}。")
        else:
            self._emit(f"论文检索等待后续处理：{result.status}；产物：{relative_ref}。")

    def _paper_provider(self) -> ArxivPaperProvider:
        if self._mode == "fixture":
            return ArxivPaperProvider(
                transport=_fixture_transport(self._fixture_xml),
                clock=lambda: research.FIXTURE_NOW,
            )
        assert self._settings is not None
        return ArxivPaperProvider(timeout_seconds=self._settings.arxiv_timeout_seconds)

    def _paper_llm(self) -> StructuredLLM:
        if self._mode == "fixture":
            return FixtureStructuredLLM()
        assert self._settings is not None
        return build_dashscope_llm(self._settings)

    async def _analyze_arxiv(self, raw_target: str | None) -> None:
        arxiv_id = None if raw_target is None else normalize_arxiv_target(raw_target)
        if arxiv_id is None:
            self._emit("arXiv 分析失败：只接受规范 arXiv abs/pdf URL 或论文 ID。")
            return
        ctx = OperationContext(
            schema_version="1",
            correlation_id=f"conversation-arxiv-{uuid4().hex[:12]}",
            workflow_id=f"conversation-arxiv-{uuid4().hex[:12]}",
            actor_id="pipeline-user",
            sensitivity="PUBLIC",
        )
        try:
            record = await self._paper_provider().get_by_external_id(
                ExternalPaperId(schema_version="1", provider_name="arxiv", value=arxiv_id),
                ctx=ctx,
            )
            if record is None:
                self._emit(f"arXiv 分析失败：未找到 {arxiv_id}。")
                return
            paper = normalize_raw_paper(record)
            if paper.arxiv_id != arxiv_id:
                self._emit(f"arXiv 分析失败：未找到 {arxiv_id}。")
                return
            assessment = unscored_assessment(paper, request_id=f"conversation-{arxiv_id}")
            scored = await score_assessments(
                [assessment],
                problem=default_sem_problem_profile(),
                request_id=f"conversation-{arxiv_id}",
                llm=self._paper_llm(),
                max_llm_calls=1,
                ctx=ctx,
                include_ineligible=True,
            )
        except PortError as error:
            if error.failure.code == "LLM_SCHEMA_VALIDATION_FAILED":
                self._emit(
                    "论文分析失败：模型返回的结构化字段不完整，已自动纠正一次但仍未通过；"
                    "请稍后重试。论文产物和当前候选均未更新。"
                )
                if self._debug_json:
                    self._emit("调试错误码：LLM_SCHEMA_VALIDATION_FAILED。")
            else:
                self._emit(f"arXiv 分析失败：{error.failure.code}。")
            return
        except (OSError, TypeError, ValueError):
            self._emit("arXiv 分析失败：ARXIV_ANALYSIS_FAILED。")
            return
        decision = scored[0].applicability
        if decision is None:
            self._emit("arXiv 分析失败：ARXIV_ANALYSIS_SCHEMA_FAILED。")
            return
        relative_ref = f"papers/{arxiv_id}/assessment.json"
        artifact_path = self._workspace / "papers" / arxiv_id / "assessment.json"
        try:
            artifact_path.parent.mkdir(parents=True, exist_ok=True)
            artifact_path.write_text(scored[0].model_dump_json(indent=2), encoding="utf-8")
        except OSError:
            self._emit("arXiv 分析失败：ASSESSMENT_ARTIFACT_WRITE_FAILED。")
            return
        memory_saved = self._set_current_paper_from_assessment(scored[0], relative_ref)
        code_status = (
            "当前元数据中已发现公开代码链接"
            if scored[0].hard_filter.code_available
            else "当前元数据中未发现公开代码链接"
        )
        target_status = "是" if decision.applicable else "否"
        self._emit(
            f"论文：{paper.title}\n"
            f"内容概要：{decision.summary}\n"
            f"目标论文：{target_status}。\n"
            f"Recommendation：{decision.recommendation}。\n"
            f"判断理由：{decision.rationale}\n"
            f"代码状态：{code_status}。\n"
            f"风险：{decision.risks[0] if decision.risks else '需要人工审阅。'}\n"
            f"产物：{relative_ref}。"
        )
        if not memory_saved:
            self._emit_memory_write_warning()

    @staticmethod
    def _repository_gate_payload(value: Mapping[str, object]) -> dict[str, object] | None:
        interrupts = value.get("__interrupt__")
        if not isinstance(interrupts, list | tuple) or len(interrupts) != 1:
            return None
        payload = interrupts[0].value
        return payload if isinstance(payload, dict) else None

    def _repository_snapshot_approval(self, payload: Mapping[str, object]) -> Approval:
        repository_url = cast(str, payload["repository_url"])
        self._emit(
            "人工 Gate：是否下载并读取固定 commit 的只读源码快照？\n"
            f"仓库：{repository_url}\n"
            "说明：这不是 git clone；不会执行、安装、patch、Smoke 或训练仓库代码。"
        )
        while True:
            entered = self._input_fn("决定 [approve/reject]: ").strip().casefold()
            if entered in {"approve", "reject"}:
                break
            self._emit("请输入 approve 或 reject。")
        decision = ApprovalDecision.APPROVE if entered == "approve" else ApprovalDecision.REJECT
        approval_id = f"approval-repository-insight-{uuid4().hex[:12]}"
        return Approval(
            schema_version="1",
            approval_id=approval_id,
            gate_kind=GateKind.REPOSITORY_INGEST,
            subject_type=cast(str, payload["subject_type"]),
            subject_id=cast(str, payload["subject_id"]),
            subject_revision=cast(int, payload["subject_revision"]),
            decision=decision,
            edits=[],
            reason="User decision for a fixed public repository source snapshot.",
            actor_id="pipeline-user",
            decided_at=datetime.now(UTC),
            idempotency_key=f"idempotency-{approval_id}",
        )

    async def _run_repository_insight(self, raw_target: str | None) -> None:
        github_url = None if raw_target is None else normalize_github_target(raw_target)
        if github_url is None:
            self._emit(
                "GitHub 代码分析失败：只接受 https://github.com/<owner>/<repo>；"
                "未发生网络、快照、LLM、patch、Smoke 或训练调用。"
            )
            return
        workflow_id = f"conversation-repository-insight-{uuid4().hex[:12]}"
        thread_id = f"thread-{workflow_id}"
        state = create_repository_insight_state(
            {
                "schema_version": "1",
                "workflow_id": workflow_id,
                "thread_id": thread_id,
                "repository_url": github_url,
            }
        )
        try:
            dependencies = build_repository_insight_dependencies(
                workspace=self._workspace,
                mode=self._mode,
                settings=self._settings,
            )
            graph = build_repository_insight_graph()
            paused = await graph.ainvoke(
                state,
                config=workflow_config(thread_id),
                context=dependencies,
            )
            payload = self._repository_gate_payload(cast(Mapping[str, object], paused))
            if payload is None:
                failure_code = paused.get("failure_code") or "REPOSITORY_INSIGHT_GATE_MISSING"
                self._emit(f"GitHub 代码分析失败：{failure_code}。")
                return
            approval = self._repository_snapshot_approval(payload)
            final = await graph.ainvoke(
                Command(resume=approval.model_dump(mode="json")),
                config=workflow_config(thread_id),
                context=dependencies,
            )
        except (OSError, TypeError, ValueError):
            self._emit(
                "GitHub 代码分析失败：REPOSITORY_INSIGHT_START_FAILED。"
                "当前论文上下文和最近成功产物保持不变。"
            )
            return
        status = final.get("status")
        if status == "REJECTED":
            self._emit(
                "已按人工决定取消公开仓库分析；未访问 GitHub、未下载源码快照、未调用代码 LLM。"
            )
            return
        if status != "COMPLETED":
            failure_code = final.get("failure_code") or "REPOSITORY_INSIGHT_FAILED"
            self._emit(
                f"GitHub 代码分析失败：{failure_code}。"
                "未生成新的代码建议，当前论文上下文和最近成功产物保持不变。"
            )
            return
        try:
            result = dependencies.store.load_result(workflow_id)
        except (OSError, TypeError, ValueError):
            self._emit(
                "GitHub 代码分析失败：REPOSITORY_INSIGHT_RESULT_INVALID。"
                "当前论文、当前仓库和最近成功产物保持不变。"
            )
            return
        memory_saved = self._set_current_repository_from_result(result)
        entrypoints = "、".join(result.structure.entrypoint_candidates) or "未识别"
        framework = "、".join(result.structure.framework_evidence[:3]) or "证据不足"
        read_files = "、".join(item.path for item in result.read_files)
        suggestions = "\n".join(
            f"- {item.area}（{', '.join(item.target_paths)}）：{item.recommendation}"
            for item in result.advice.suggestions
        )
        risks = "；".join(result.advice.risks)
        checks = "；".join(result.advice.items_to_verify)
        limitations = "；".join(result.advice.limitations)
        self._emit(
            "公开 GitHub 代码分析完成\n"
            f"仓库：{result.repository_url}\n"
            f"固定 commit：{result.resolution.commit_sha}\n"
            f"许可证：{result.metadata.license_spdx or 'UNKNOWN'}\n"
            f"结构：入口={entrypoints}；框架证据={framework}\n"
            f"已读文件：{read_files}\n"
            f"代码概要：{result.advice.repository_summary}\n"
            f"适配程度：{result.advice.adaptation_fit}\n"
            f"建议：\n{suggestions}\n"
            f"风险：{risks}\n"
            f"待验证：{checks}\n"
            f"限制：{limitations}\n"
            "执行边界：只下载并读取固定 commit 的只读 ZIP 源码快照（不是 git clone）；"
            "未生成 patch，未运行 Smoke、训练或第三方代码，未使用公司数据。\n"
            f"产物：{result.result_ref}；报告：{result.report_ref}。"
        )
        if not memory_saved:
            self._emit_memory_write_warning()

    async def _run_pipeline(self) -> None:
        workflow_id = f"conversation-pipeline-{uuid4().hex[:12]}"
        events: list[str] = []
        options = pipeline.PipelineCliOptions(
            mode="fixture",
            adaptation_planner_mode="dashscope" if self._mode == "live" else "scripted",
            workspace=self._workspace / "pipeline",
            workflow_id=workflow_id,
            scenario="happy",
            auto_approve_sample=False,
            decisions=(),
        )
        exit_code = await pipeline.run(
            options,
            input_fn=self._ask_gate(events),
            output_fn=events.append,
        )
        self._emit_debug(events)
        summary_ref = f"pipeline/var/sample/{workflow_id}/summary.json"
        final = _event_object(events[-1]) if events else None
        status = None if final is None else final.get("status")
        if exit_code == 0 and final is not None and status == "SUCCEEDED":
            memory_saved = self._set_last_successful_artifact(summary_ref)
            self._emit(
                f"完整流程完成：{final.get('status')}，结论：{final.get('conclusion')}；产物：{summary_ref}。"
            )
            if not memory_saved:
                self._emit_memory_write_warning()
        elif final is not None:
            conclusion = final.get("conclusion")
            conclusion_text = "" if conclusion is None else f"，结论：{conclusion}"
            self._emit(f"完整流程已停止：{status}{conclusion_text}；产物：{summary_ref}。")
        else:
            self._emit(f"完整流程已停止；产物：{summary_ref}。")


__all__ = ["ConversationMode", "ConversationSession"]
