"""Bounded LangGraph tool-calling loop for the pipeline adaptation planner."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from hashlib import sha256
from time import monotonic
from typing import Annotated, Literal, Protocol, TypedDict, cast

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool, ToolException, tool
from langgraph.graph import START, StateGraph, add_messages
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, ConfigDict

from vision_research_ops.application.adaptation_runtime import AdaptationPlannerOutput
from vision_research_ops.domain import JsonObject
from vision_research_ops.ports import (
    GenerationUsage,
    LLMError,
    OperationContext,
    PortError,
    StructuredGenerationResult,
    make_failure,
)
from vision_research_ops.prompts.adaptation import TOOL_SYSTEM_PROMPT

from .adaptation_models import (
    AdaptationInputFacts,
    AdaptationPlannerTrace,
    AdaptationPlanProposal,
    PlannerToolEvent,
    PlannerToolName,
)
from .adaptation_planning import adaptation_prompt_facts, validate_proposal_against_inputs

MAX_PLANNER_TURNS = 6
MAX_TOOL_CALLS = 6
REQUIRED_PLANNER_TOOLS: tuple[PlannerToolName, ...] = (
    "inspect_repository_profile",
    "inspect_dataset_contract",
    "compare_repository_dataset",
    "validate_adaptation_plan",
)


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _hash(value: object) -> str:
    return f"sha256:{sha256(_canonical_bytes(value)).hexdigest()}"


class _NoArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, validate_default=True)


class _ValidatePlanArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, validate_default=True)
    proposal: AdaptationPlanProposal


class _ToolExecution(TypedDict):
    tool_name: PlannerToolName
    arguments: JsonObject
    output: JsonObject


class PlannerGraphState(TypedDict):
    """Private small state for the internal no-checkpoint planning subgraph."""

    messages: Annotated[list[AnyMessage], add_messages]
    planner_turns: int


class ToolCallingAgent(Protocol):
    """Small surface shared by a bound ChatOpenAI model and the scripted model."""

    async def ainvoke(self, value: object) -> AIMessage:
        """Return one assistant message, optionally carrying tool calls."""


ToolAgentFactory = Callable[[Sequence[BaseTool]], ToolCallingAgent]


def _planner_failure(
    *,
    code: str,
    message: str,
    ctx: OperationContext,
    retryable: bool = False,
) -> LLMError:
    return LLMError(
        make_failure(
            code=code,
            category="LLM_TOOL_PLANNER",
            message=message,
            retryable=retryable,
            ctx=ctx,
        )
    )


def _build_tools(
    facts: AdaptationInputFacts,
    executions: list[_ToolExecution],
) -> list[BaseTool]:
    prompt_facts = adaptation_prompt_facts(facts)
    repository_facts = cast(JsonObject, prompt_facts["repository_facts"])
    dataset_facts = cast(JsonObject, prompt_facts["dataset_facts"])
    required_contract = cast(JsonObject, prompt_facts["required_contract"])

    def record(name: PlannerToolName, arguments: JsonObject, output: JsonObject) -> JsonObject:
        executions.append({"tool_name": name, "arguments": arguments, "output": output})
        return output

    @tool(
        "inspect_repository_profile",
        args_schema=_NoArguments,
        description=(
            "Read the already validated, de-identified repository structure and license facts. "
            "This tool cannot read files, Git, URLs, or the network."
        ),
    )
    def inspect_repository_profile() -> JsonObject:
        return record("inspect_repository_profile", {}, dict(repository_facts))

    @tool(
        "inspect_dataset_contract",
        args_schema=_NoArguments,
        description=(
            "Read the already validated synthetic SEM dataset contract: modality, channels, "
            "labels, groups, and split policy. No images or paths are available."
        ),
    )
    def inspect_dataset_contract() -> JsonObject:
        return record("inspect_dataset_contract", {}, dict(dataset_facts))

    @tool(
        "compare_repository_dataset",
        args_schema=_NoArguments,
        description=(
            "Deterministically compare the supported repository template with the dataset "
            "contract and return the only adaptation areas that may be proposed."
        ),
    )
    def compare_repository_dataset() -> JsonObject:
        output: JsonObject = {
            "compatible_template": "SEM_PLAIN_PYTORCH_CONFIG_V1",
            "required_gap_areas": [
                "INPUT_CHANNELS",
                "NUM_CLASSES",
                "LABEL_MAPPING",
                "GROUP_SPLIT",
                "METRICS_OUTPUT",
            ],
            "required_contract": dict(required_contract),
            "write_or_execute_capability": False,
        }
        return record("compare_repository_dataset", {}, output)

    @tool(
        "validate_adaptation_plan",
        args_schema=_ValidatePlanArguments,
        description=(
            "Validate a complete AdaptationPlanProposal against deterministic dataset facts. "
            "Call this before finishing; it never applies a patch or executes code."
        ),
    )
    def validate_adaptation_plan(proposal: AdaptationPlanProposal) -> JsonObject:
        required_observations = {
            "inspect_repository_profile",
            "inspect_dataset_contract",
            "compare_repository_dataset",
        }
        observed = {item["tool_name"] for item in executions}
        if not required_observations.issubset(observed):
            raise ValueError("plan validation requires all read-only fact observations first")
        validate_proposal_against_inputs(proposal, facts)
        arguments = cast(JsonObject, {"proposal": proposal.model_dump(mode="json")})
        output: JsonObject = {
            "valid": True,
            "schema": "AdaptationPlanProposal",
            "policy": "SEM_PLAIN_PYTORCH_CONFIG_V1",
            "side_effects_executed": False,
        }
        return record("validate_adaptation_plan", arguments, output)

    return [
        inspect_repository_profile,
        inspect_dataset_contract,
        compare_repository_dataset,
        validate_adaptation_plan,
    ]


class LangGraphAdaptationPlanner:
    """Compose a bound tool model and real ToolNode around a strict plan tool."""

    def __init__(
        self,
        *,
        agent_factory: ToolAgentFactory,
        planner_kind: Literal["SCRIPTED_TOOL_CALLING", "DASHSCOPE_TOOL_CALLING"],
        provider_id: str,
        model_id: str,
    ) -> None:
        self._agent_factory = agent_factory
        self._planner_kind = planner_kind
        if not provider_id.strip() or not model_id.strip():
            raise ValueError("planner provider_id and model_id must be non-blank")
        self._provider_id = provider_id
        self._model_id = model_id
        self.last_graph_nodes: frozenset[str] = frozenset()
        self.last_tool_names: tuple[PlannerToolName, ...] = ()

    async def plan(
        self,
        facts: AdaptationInputFacts,
        *,
        ctx: OperationContext,
    ) -> AdaptationPlannerOutput:
        """Run the bounded observation loop and return its validated plan payload."""
        executions: list[_ToolExecution] = []
        tools = _build_tools(facts, executions)
        allowed_names = frozenset(REQUIRED_PLANNER_TOOLS)
        agent = self._agent_factory(tools)

        async def call_model(state: PlannerGraphState) -> PlannerGraphState:
            turns = state.get("planner_turns", 0)
            if turns >= MAX_PLANNER_TURNS:
                raise _planner_failure(
                    code="ADAPTATION_PLANNER_STEP_LIMIT",
                    message="The adaptation planner exceeded its bounded model-turn limit.",
                    ctx=ctx,
                )
            try:
                response = await agent.ainvoke(state["messages"])
            except PortError:
                raise
            except Exception:
                raise _planner_failure(
                    code="ADAPTATION_PLANNER_PROVIDER_FAILED",
                    message="The adaptation tool-calling model request failed.",
                    ctx=ctx,
                    retryable=True,
                ) from None
            if not isinstance(response, AIMessage) or response.invalid_tool_calls:
                raise _planner_failure(
                    code="ADAPTATION_PLANNER_MESSAGE_INVALID",
                    message="The adaptation planner returned an invalid tool message.",
                    ctx=ctx,
                )
            if len(response.tool_calls) > 1:
                raise _planner_failure(
                    code="ADAPTATION_PLANNER_PARALLEL_TOOLS_FORBIDDEN",
                    message="The bounded planner permits one tool call per turn.",
                    ctx=ctx,
                )
            prior_calls = sum(
                1 for message in state["messages"] if isinstance(message, ToolMessage)
            )
            if prior_calls + len(response.tool_calls) > MAX_TOOL_CALLS:
                raise _planner_failure(
                    code="ADAPTATION_PLANNER_TOOL_LIMIT",
                    message="The adaptation planner exceeded its bounded tool-call limit.",
                    ctx=ctx,
                )
            if any(call.get("name") not in allowed_names for call in response.tool_calls):
                raise _planner_failure(
                    code="ADAPTATION_PLANNER_TOOL_FORBIDDEN",
                    message="The adaptation planner requested a tool outside the allowlist.",
                    ctx=ctx,
                )
            return {"messages": [response], "planner_turns": turns + 1}

        def route(state: PlannerGraphState) -> Literal["tools", "__end__"]:
            last = state["messages"][-1]
            if isinstance(last, AIMessage) and last.tool_calls:
                return "tools"
            observed = {item["tool_name"] for item in executions}
            if not allowed_names.issubset(observed):
                raise _planner_failure(
                    code="ADAPTATION_PLANNER_REQUIRED_TOOLS_MISSING",
                    message="The planner stopped before all required read-only tools succeeded.",
                    ctx=ctx,
                )
            return "__end__"

        builder = StateGraph(PlannerGraphState)
        builder.add_node("planner_model", call_model)
        builder.add_node("tools", ToolNode(tools, handle_tool_errors=False))
        builder.add_edge(START, "planner_model")
        builder.add_conditional_edges("planner_model", route)
        builder.add_edge("tools", "planner_model")
        graph = builder.compile()
        self.last_graph_nodes = frozenset(graph.get_graph().nodes)
        prompt_facts = adaptation_prompt_facts(facts)
        started = monotonic()
        try:
            final_state = await graph.ainvoke(
                {
                    "messages": [
                        SystemMessage(content=TOOL_SYSTEM_PROMPT),
                        HumanMessage(content=_canonical_bytes(prompt_facts).decode("utf-8")),
                    ],
                    "planner_turns": 0,
                },
                config={"recursion_limit": 16},
            )
        except PortError:
            raise
        except (ToolException, TypeError, ValueError):
            raise _planner_failure(
                code="ADAPTATION_PLANNER_TOOL_FAILED",
                message="A bounded adaptation planning tool rejected its input or output.",
                ctx=ctx,
            ) from None

        validated_calls = [
            item for item in executions if item["tool_name"] == "validate_adaptation_plan"
        ]
        if not validated_calls:
            raise _planner_failure(
                code="ADAPTATION_PLANNER_VALIDATED_PLAN_MISSING",
                message="The planner completed without a validated plan payload.",
                ctx=ctx,
            )
        proposal_raw = validated_calls[-1]["arguments"].get("proposal")
        try:
            proposal = AdaptationPlanProposal.model_validate(proposal_raw)
            proposal_json = proposal.model_dump(mode="json")
        except (TypeError, ValueError):
            raise _planner_failure(
                code="ADAPTATION_PLANNER_VALIDATED_PLAN_INVALID",
                message="The planner validation tool did not retain a strict proposal.",
                ctx=ctx,
            ) from None

        input_tokens = 0
        output_tokens = 0
        finish_reason = "TOOL_VALIDATED"
        for message in final_state.get("messages", []):
            if not isinstance(message, AIMessage):
                continue
            usage: dict[str, object] = cast(dict[str, object], message.usage_metadata or {})
            raw_input = usage.get("input_tokens", 0)
            raw_output = usage.get("output_tokens", 0)
            if isinstance(raw_input, int) and not isinstance(raw_input, bool) and raw_input >= 0:
                input_tokens += raw_input
            if isinstance(raw_output, int) and not isinstance(raw_output, bool) and raw_output >= 0:
                output_tokens += raw_output
            raw_finish = message.response_metadata.get("finish_reason")
            if isinstance(raw_finish, str) and raw_finish.strip():
                finish_reason = raw_finish

        planner_prompt_hash = _hash(
            {
                "system": TOOL_SYSTEM_PROMPT,
                "facts": prompt_facts,
                "tools": list(REQUIRED_PLANNER_TOOLS),
            }
        )
        generation = StructuredGenerationResult[AdaptationPlanProposal](
            schema_version="1",
            value=proposal,
            provider_id=self._provider_id,
            model_id=self._model_id,
            usage=GenerationUsage(
                schema_version="1",
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
            ),
            latency_ms=max(0, int((monotonic() - started) * 1000)),
            prompt_hash=planner_prompt_hash,
            output_hash=_hash(proposal_json),
            finish_reason=finish_reason,
        )

        events = [
            PlannerToolEvent(
                call_index=index,
                tool_name=item["tool_name"],
                arguments_hash=_hash(item["arguments"]),
                output_hash=_hash(item["output"]),
            )
            for index, item in enumerate(executions, start=1)
        ]
        trace = AdaptationPlannerTrace(
            trace_id=f"planner-trace-{ctx.workflow_id}",
            workflow_id=ctx.workflow_id,
            planner_kind=self._planner_kind,
            provider_id=generation.provider_id,
            model_id=generation.model_id,
            events=events,
            prompt_hash=planner_prompt_hash,
            output_hash=generation.output_hash,
        )
        self.last_tool_names = tuple(event.tool_name for event in events)
        return AdaptationPlannerOutput(generation=generation, trace=trace)


__all__ = [
    "MAX_PLANNER_TURNS",
    "MAX_TOOL_CALLS",
    "REQUIRED_PLANNER_TOOLS",
    "LangGraphAdaptationPlanner",
    "ToolAgentFactory",
    "ToolCallingAgent",
]
