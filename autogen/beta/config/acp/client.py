# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
"""``ACPClient`` — the :class:`LLMClient` that drives a CLI agent over ACP.

One AG2 model turn maps to one ACP ``session/prompt``. The agent's own tool loop
runs inside that single call; ``session/update`` notifications stream onto the
AG2 event stream via the bridge, and the accumulated text becomes a
``ModelResponse``.

Lifecycle: the framework calls ``config.create()`` once per ``AgentRun``, so the
live ACP session is keyed by ``context.stream.id`` in a per-config registry and
reused across the run's internal model-turns. A ``weakref.finalize`` on the
stream terminates the subprocess if the run is dropped without an explicit
``config.aclose()``.
"""

import asyncio
import weakref
from asyncio.subprocess import Process
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import acp
from acp import schema

from autogen.beta.context import ConversationContext
from autogen.beta.events import BaseEvent
from autogen.beta.events.types import ModelMessage, ModelResponse
from autogen.beta.response import ResponseProto
from autogen.beta.tools.final.function_tool import FunctionToolSchema
from autogen.beta.tools.schemas import ToolSchema

from .bridge import make_bridge
from .mappers import map_usage
from .mcp_bridge import ToolMCPServer, function_schemas
from .session import ACPSession, new_prompt_text

if TYPE_CHECKING:
    from fast_depends.library.serializer import SerializerProto

    from .config import ACPConfig


def _terminate_proc(proc: Process | None) -> None:
    """Best-effort synchronous subprocess termination (finalizer safety net)."""
    try:
        if proc is not None and proc.returncode is None:
            proc.terminate()
    except ProcessLookupError:
        pass


class ACPClient:
    """ACP client implementing :class:`LLMClient`, one live session per run."""

    def __init__(self, config: "ACPConfig") -> None:
        self.config = config

    def _client_capabilities(self) -> schema.ClientCapabilities:
        return schema.ClientCapabilities(
            fs=schema.FileSystemCapabilities(read_text_file=True, write_text_file=True),
            terminal=bool(self.config.allow_terminal),
        )

    async def _session_for(
        self,
        context: ConversationContext,
        fn_schemas: list[FunctionToolSchema],
        serializer: "SerializerProto",
    ) -> ACPSession:
        key = context.stream.id
        session = self.config._sessions.get(key)
        if session is not None and session.started:
            return session

        session = ACPSession()
        session.bridge = make_bridge(self.config)

        mcp_servers: list = []
        try:
            if self.config.expose_tools and fn_schemas:
                session.mcp = ToolMCPServer(fn_schemas, serializer)
                url = await session.mcp.start(timeout=self.config.startup_timeout)
                mcp_servers = [acp.schema.HttpMcpServer(name="ag2", url=url, headers=[], type="http")]

            await session.ensure(
                session.bridge,
                self.config.command,
                cwd=self.config.cwd,
                env=self.config.env,
                protocol_version=acp.PROTOCOL_VERSION,
                client_capabilities=self._client_capabilities(),
                additional_directories=self.config.additional_directories,
                mcp_servers=mcp_servers,
            )
        except BaseException:
            # ensure() or mcp.start() failed; stop the MCP server we may have started.
            if session.mcp is not None:
                await session.mcp.stop()
                session.mcp = None
            raise

        self.config._sessions[key] = session
        # Safety net: terminate the subprocess if the stream is dropped without
        # an explicit aclose(). Keyed on the stream, not the (per-run) client.
        weakref.finalize(context.stream, _terminate_proc, session.proc)
        return session

    async def __call__(
        self,
        messages: Sequence[BaseEvent],
        context: ConversationContext,
        *,
        tools: Iterable[ToolSchema],
        response_schema: "ResponseProto | None",
        serializer: "SerializerProto",
    ) -> ModelResponse:
        fn_schemas = function_schemas(tools)
        session = await self._session_for(context, fn_schemas, serializer)
        if session.mcp is not None:
            session.mcp.current_context = context
        session.bridge.state.context = context
        session.bridge.state.begin_turn()

        text, new_count = new_prompt_text(messages, session.sent_count)

        async def _run_turn() -> schema.PromptResponse:
            return await session.conn.prompt(
                prompt=[acp.text_block(text)],
                session_id=session.session_id,
                message_id=str(uuid4()),
            )

        timed_out = False
        if self.config.turn_timeout is not None:
            # Don't cancel the prompt coroutine (that corrupts the JSON-RPC
            # connection). Signal session/cancel and let the agent return the
            # in-flight prompt with stop_reason="cancelled".
            task = asyncio.ensure_future(_run_turn())
            done, _ = await asyncio.wait({task}, timeout=self.config.turn_timeout)
            if task not in done:
                timed_out = True
                await _cancel_quietly(session)
            response = await task
        else:
            response = await _run_turn()

        session.sent_count = new_count

        finish_reason = "timeout" if timed_out else getattr(response, "stop_reason", None)

        return ModelResponse(
            message=ModelMessage(session.bridge.state.turn_text),
            usage=map_usage(_dump_optional(getattr(response, "usage", None))),
            files=session.bridge.state.turn_files,
            finish_reason=finish_reason,
            provider="acp",
            model=self.config.model,
        )


async def _cancel_quietly(session: ACPSession) -> None:
    try:
        if session.conn is not None and session.session_id is not None:
            await session.conn.cancel(session_id=session.session_id)
    except Exception:  # noqa: BLE001 — cancellation is best-effort
        pass


def _dump_optional(model: Any) -> dict | None:
    if model is None:
        return None
    return model.model_dump() if hasattr(model, "model_dump") else dict(model)
