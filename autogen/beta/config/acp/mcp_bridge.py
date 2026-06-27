# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
"""In-process MCP server exposing the AG2 agent's function tools to a CLI agent.

``ToolMCPServer`` runs a low-level ``mcp`` server over streamable HTTP (the same
plumbing as ``autogen/beta/mcp/server.py``) on an ephemeral localhost port. On an
MCP ``tools/call`` it drives the *existing* AG2 stream machinery — emitting a
singular ``ToolCallEvent`` and awaiting the matching result — so the tool runs
with full dependency injection, lands observability events on the parent stream,
and (for ``.as_tool()`` subagents) recurses into a child agent run. It holds no
executable ``Tool`` objects; only the per-turn ``current_context``.
"""

import asyncio
import contextlib
import json
from collections.abc import AsyncIterator, Iterable
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

import uvicorn
from mcp.server.lowlevel import Server
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from mcp.types import CallToolResult, ContentBlock, TextContent
from mcp.types import Tool as MCPTool
from starlette.applications import Starlette
from starlette.routing import Mount

from autogen.beta.events import (
    DataInput,
    TextInput,
    ToolCallEvent,
    ToolErrorEvent,
    ToolNotFoundEvent,
    ToolResultEvent,
)
from autogen.beta.tools.final.function_tool import FunctionToolSchema

if TYPE_CHECKING:
    from fast_depends.library.serializer import SerializerProto
    from starlette.types import Lifespan, Receive, Scope, Send

    from autogen.beta.context import ConversationContext
    from autogen.beta.tools.schemas import ToolSchema


def function_schemas(tools: "Iterable[ToolSchema]") -> list[FunctionToolSchema]:
    """Keep only AG2-executable function schemas; builtin tool schemas are dropped."""
    return [t for t in tools if isinstance(t, FunctionToolSchema)]


def map_tool_result(
    event: ToolResultEvent | ToolErrorEvent | ToolNotFoundEvent, serializer: "SerializerProto"
) -> CallToolResult:
    """Map a ``ToolResultEvent`` (or error subclass) onto an MCP ``CallToolResult``.

    Mirrors ``autogen/beta/tools/executor.py``: text parts pass through, data parts
    are serialized to text. ``ToolErrorEvent``/``ToolNotFoundEvent`` set ``isError``.
    """
    is_error = isinstance(event, (ToolErrorEvent, ToolNotFoundEvent))
    blocks: list[ContentBlock] = []
    for part in event.result.parts:
        if isinstance(part, TextInput):
            blocks.append(TextContent(type="text", text=part.content))
        elif isinstance(part, DataInput):
            blocks.append(TextContent(type="text", text=serializer.encode(part.data).decode()))
        else:
            blocks.append(TextContent(type="text", text=str(part)))
    if not blocks:
        blocks.append(TextContent(type="text", text=""))
    return CallToolResult(content=blocks, isError=is_error)


def _session_manager_lifespan(manager: StreamableHTTPSessionManager) -> "Lifespan[Any]":
    """ASGI lifespan that runs the streamable-HTTP session manager (mirrors
    ``autogen/beta/mcp/server.py``; ``manager.run()`` must be entered before serving)."""

    @asynccontextmanager
    async def lifespan(_: Starlette) -> AsyncIterator[None]:
        async with manager.run():
            yield

    return lifespan


class ToolMCPServer:
    """In-process MCP server exposing function tools to the CLI agent over HTTP."""

    def __init__(
        self,
        schemas: list[FunctionToolSchema],
        serializer: "SerializerProto",
        *,
        path: str = "/mcp",
    ) -> None:
        self._schemas = schemas
        self._serializer = serializer
        self._path = path
        self.current_context: ConversationContext | None = None
        self.url: str | None = None
        self._uvicorn: uvicorn.Server | None = None
        self._task: asyncio.Task[None] | None = None
        self.server = self._build_server()
        self.app = self._build_app()

    def list_tools(self) -> list[MCPTool]:
        return [
            MCPTool(
                name=s.function.name,
                description=s.function.description,
                inputSchema=s.function.parameters or {"type": "object", "properties": {}},
            )
            for s in self._schemas
        ]

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> CallToolResult:
        """Proxy an MCP ``tools/call`` onto the live AG2 stream and map the result.

        Emits a singular ``ToolCallEvent`` (so it never triggers the agent's
        ``ModelRequest | ToolResultsEvent`` LLM re-entry) and awaits the matching
        result, mirroring ``autogen/beta/tools/executor.py``'s ``_execute_call``.

        .. warning::
            Tools that emit a ``ClientToolCallEvent`` instead of a ``ToolResultEvent``
            are unsupported on this MCP bridge path; calling such a tool will cause
            this method to hang indefinitely.
        """
        context = self.current_context
        if context is None:
            raise RuntimeError("ToolMCPServer.call_tool invoked with no active context")

        call = ToolCallEvent(name=name, arguments=json.dumps(arguments or {}))
        async with context.stream.get(
            (ToolResultEvent.parent_id == call.id)
            | (ToolErrorEvent.parent_id == call.id)
            | (ToolNotFoundEvent.parent_id == call.id)
        ) as result_fut:
            await context.send(call)
            event = await result_fut
        return map_tool_result(event, self._serializer)

    async def start(self, *, timeout: float = 30.0) -> str:
        """Start uvicorn on the current loop on an ephemeral port; return the URL.

        Sharing the running loop is what lets ``call_tool`` touch ``context.stream``.
        """
        config = uvicorn.Config(self.app, host="127.0.0.1", port=0, log_level="warning", lifespan="on")
        self._uvicorn = uvicorn.Server(config)
        self._task = asyncio.ensure_future(self._uvicorn.serve())

        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while not self._uvicorn.started:
            if self._task.done():  # serve() failed during startup
                await self._task  # re-raise the original error
                raise RuntimeError("MCP server task exited during startup")
            if loop.time() > deadline:
                await self.stop()
                raise TimeoutError(f"MCP server did not start within {timeout}s")
            await asyncio.sleep(0.01)

        port = self._uvicorn.servers[0].sockets[0].getsockname()[1]
        self.url = f"http://127.0.0.1:{port}{self._path}"
        return self.url

    async def stop(self) -> None:
        """Gracefully stop the server; safe to call when not started or twice."""
        server, task = self._uvicorn, self._task
        self._uvicorn, self._task, self.url = None, None, None
        if server is not None:
            server.should_exit = True
        if task is not None:
            with contextlib.suppress(asyncio.CancelledError):
                await task

    def _build_server(self) -> Server:
        server: Server = Server(name="ag2-tools")
        bridge = self

        @server.list_tools()  # type: ignore[no-untyped-call, misc]
        async def _list_tools() -> list[MCPTool]:
            return bridge.list_tools()

        @server.call_tool()  # type: ignore[no-untyped-call, misc]
        async def _call_tool(name: str, arguments: dict[str, Any]) -> CallToolResult:
            return await bridge.call_tool(name, arguments or {})  # type: ignore[attr-defined]

        return server

    def _build_app(self) -> Starlette:
        self._manager = StreamableHTTPSessionManager(app=self.server, stateless=True, json_response=False)

        async def handle(scope: "Scope", receive: "Receive", send: "Send") -> None:
            await self._manager.handle_request(scope, receive, send)

        return Starlette(routes=[Mount(self._path, app=handle)], lifespan=_session_manager_lifespan(self._manager))
