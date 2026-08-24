"""Start the default bounded conversational interface for SEM Research Agent."""
# ruff: noqa: RUF001

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from pydantic import ValidationError

from vision_research_ops.adapters.llm import build_dashscope_chat_model
from vision_research_ops.application.services.conversation_intent import (
    ConversationIntent,
    ConversationRoutingError,
    FixtureIntentRouter,
    StructuredIntentRouter,
)
from vision_research_ops.cli.conversation import (
    ConversationMode,
    ConversationSession,
)
from vision_research_ops.prompts.conversation_intent import SYSTEM_PROMPT
from vision_research_ops.settings import Settings, load_local_env

RouterMode = Literal["fixture", "live"]
LIVE_CONTEXT_PATH = Path("var/sessions/current/context.json")
DEBUG_CONTEXT_PATH = Path("var/sessions/debug/context.json")


class DashScopeIntentRouter:
    """Schema-constrained live router with no execution tools."""

    def __init__(self, settings: Settings) -> None:
        self._model = build_dashscope_chat_model(settings)

    async def route(self, message: str) -> ConversationIntent:
        try:
            structured = self._model.with_structured_output(
                ConversationIntent,
                method="function_calling",
                include_raw=True,
            )
            response = await structured.ainvoke([("system", SYSTEM_PROMPT), ("human", message)])
        except Exception:
            raise ConversationRoutingError("CONVERSATION_ROUTER_PROVIDER_FAILED") from None
        if not isinstance(response, dict) or response.get("parsing_error") is not None:
            raise ConversationRoutingError("CONVERSATION_ROUTER_SCHEMA_FAILED")
        try:
            return ConversationIntent.model_validate(response.get("parsed"))
        except (TypeError, ValidationError, ValueError):
            raise ConversationRoutingError("CONVERSATION_ROUTER_SCHEMA_FAILED") from None


@dataclass(frozen=True, slots=True)
class AgentCliOptions:
    """Validated settings for one local, serial conversation session."""

    mode: ConversationMode
    router_mode: RouterMode
    workspace: Path
    debug_json: bool
    context_path: Path | None = None


def build_parser() -> argparse.ArgumentParser:
    """Build the small default CLI surface without exposing execution controls."""
    parser = argparse.ArgumentParser(
        description="Run the bounded SEM Research Agent conversational pipeline CLI."
    )
    parser.add_argument("--mode", choices=("fixture", "live"), default="live")
    parser.add_argument(
        "--router",
        choices=("fixture", "live"),
        default="live",
        help="structured intent router; live uses DashScope and fixture provides a local sample",
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path("var/conversation"),
        help="local conversation artifacts; only existing bounded agents write below it",
    )
    parser.add_argument(
        "--debug-json",
        action="store_true",
        help="explicitly show raw internal event JSON for this local debugging session",
    )
    parser.add_argument(
        "--context-path",
        type=Path,
        default=None,
        help=("small local working-context JSON; defaults to separate stable live/debug files"),
    )
    return parser


def parse_options(argv: Sequence[str] | None = None) -> AgentCliOptions:
    """Parse command-line options into immutable, bounded choices."""
    namespace = build_parser().parse_args(argv)
    mode = cast(ConversationMode, namespace.mode)
    configured_context = cast(Path | None, namespace.context_path)
    return AgentCliOptions(
        mode=mode,
        router_mode=cast(RouterMode, namespace.router),
        workspace=cast(Path, namespace.workspace),
        debug_json=cast(bool, namespace.debug_json),
        context_path=(
            configured_context
            if configured_context is not None
            else (DEBUG_CONTEXT_PATH if mode == "fixture" else LIVE_CONTEXT_PATH)
        ),
    )


def _router(options: AgentCliOptions, settings: Settings) -> StructuredIntentRouter:
    if options.router_mode == "fixture":
        return FixtureIntentRouter()
    return DashScopeIntentRouter(settings)


async def run(
    options: AgentCliOptions,
    *,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
    settings: Settings | None = None,
) -> int:
    """Run the interactive REPL until the user exits or standard input closes."""
    resolved_settings = settings or Settings.from_env()
    session = ConversationSession(
        workspace=options.workspace,
        mode=options.mode,
        router=_router(options, resolved_settings),
        input_fn=input_fn,
        output_fn=output_fn,
        settings=resolved_settings,
        context_path=options.context_path,
    )
    for line in session.introduction():
        output_fn(line)
    if options.debug_json:
        await session.handle("/debug on")
    while True:
        try:
            message = input_fn("sem-agent> ")
        except EOFError:
            output_fn("输入结束，已退出 SEM Research Agent 对话。")
            return 0
        if not await session.handle(message):
            return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Load local settings only at the CLI boundary and start a normal event loop."""
    options = parse_options(argv)
    load_local_env()
    return asyncio.run(run(options))


if __name__ == "__main__":
    raise SystemExit(main())
