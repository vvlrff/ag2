# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import contextlib
from urllib.parse import urlparse

import pytest
from fast_depends.pydantic import PydanticSerializer
from mcp.types import TextContent

from autogen.beta import Context
from autogen.beta.config.acp.mcp_bridge import (
    ToolMCPServer,
    function_schemas,
    map_tool_result,
)
from autogen.beta.events import ToolCallEvent, ToolErrorEvent, ToolNotFoundEvent, ToolResultEvent
from autogen.beta.exceptions import ToolNotFoundError
from autogen.beta.stream import MemoryStream
from autogen.beta.tools.executor import _tool_not_found
from autogen.beta.tools.final.function_tool import tool
from autogen.beta.tools.schemas import ToolSchema


def _serializer() -> PydanticSerializer:
    return PydanticSerializer(pydantic_config={"arbitrary_types_allowed": True}, use_fastdepends_errors=False)


async def _port_open(host: str, port: int) -> bool:
    try:
        _, writer = await asyncio.open_connection(host, port)
        writer.close()
        with contextlib.suppress(Exception):
            await writer.wait_closed()
        return True
    except OSError:
        return False


async def _wait_port_closed(host: str, port: int, *, timeout: float = 2.0) -> None:
    """Poll until nothing accepts on host:port, failing after ``timeout``."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if not await _port_open(host, port):
            return
        await asyncio.sleep(0.01)
    raise AssertionError(f"{host}:{port} still accepting connections after {timeout}s")


def test_function_schemas_keeps_only_function_schemas():
    fn = tool(lambda x: x, name="double").schema  # FunctionToolSchema
    builtin = ToolSchema(type="web_search")
    assert function_schemas([fn, builtin]) == [fn]


class TestMapToolResult:
    def test_text_success(self):
        call = ToolCallEvent(name="double", arguments="{}")
        event = ToolResultEvent.from_call(call, result="hi")
        result = map_tool_result(event, _serializer())
        assert result.isError is False
        assert result.content == [TextContent(type="text", text="hi")]

    def test_data_uses_serializer(self):
        call = ToolCallEvent(name="f", arguments="{}")
        event = ToolResultEvent.from_call(call, result={"a": 1})
        result = map_tool_result(event, _serializer())
        assert result.isError is False
        assert result.content == [TextContent(type="text", text='{"a":1}')]

    def test_error_sets_iserror(self):
        call = ToolCallEvent(name="boom", arguments="{}")
        event = ToolErrorEvent.from_call(call, error=ValueError("nope"))
        result = map_tool_result(event, _serializer())
        assert result.isError is True
        assert "nope" in result.content[0].text

    def test_not_found_sets_iserror(self):
        call = ToolCallEvent(name="ghost", arguments="{}")
        event = ToolNotFoundEvent.from_call(call, error=ToolNotFoundError("ghost"))
        result = map_tool_result(event, _serializer())
        assert result.isError is True


class TestListTools:
    def test_maps_function_schemas_to_mcp_tools(self):
        def double(x: int) -> int:
            """Double a number."""
            return x * 2

        schema = tool(double).schema
        server = ToolMCPServer([schema], _serializer())

        [mcp_tool] = server.list_tools()
        assert mcp_tool.name == "double"
        assert mcp_tool.description == "Double a number."
        assert mcp_tool.inputSchema == schema.function.parameters

    def test_empty_when_no_schemas(self):
        assert ToolMCPServer([], _serializer()).list_tools() == []


@pytest.mark.asyncio
class TestCallTool:
    async def test_runs_registered_tool_and_maps_result(self):
        def double(x: int) -> int:
            return x * 2

        fn = tool(double)
        ctx = Context(stream=MemoryStream())
        server = ToolMCPServer([fn.schema], _serializer())
        server.current_context = ctx

        with contextlib.ExitStack() as stack:
            fn.register(stack, ctx)
            result = await server.call_tool("double", {"x": 21})

        assert result.isError is False
        assert result.content == [TextContent(type="text", text="42")]

    async def test_maps_tool_error(self):
        def boom() -> None:
            raise ValueError("kaboom")

        fn = tool(boom)
        ctx = Context(stream=MemoryStream())
        server = ToolMCPServer([fn.schema], _serializer())
        server.current_context = ctx

        with contextlib.ExitStack() as stack:
            fn.register(stack, ctx)
            result = await server.call_tool("boom", {})

        assert result.isError is True
        assert "kaboom" in result.content[0].text

    async def test_unknown_name_maps_to_error(self):
        ctx = Context(stream=MemoryStream())
        server = ToolMCPServer([], _serializer())
        server.current_context = ctx

        # The agent's not-found subscriber emits ToolNotFoundEvent for unknown names.
        with contextlib.ExitStack() as stack:
            stack.enter_context(ctx.stream.where(ToolCallEvent).sub_scope(_tool_not_found(known_tools=set())))
            result = await server.call_tool("ghost", {})

        assert result.isError is True
        assert "ghost" in result.content[0].text

    async def test_raises_without_context(self):
        server = ToolMCPServer([], _serializer())
        server.current_context = None
        with pytest.raises(RuntimeError):
            await server.call_tool("x", {})


@pytest.mark.asyncio
class TestRunner:
    async def test_starts_on_ephemeral_port_and_stops(self):
        server = ToolMCPServer([], _serializer())

        url = await server.start()
        parsed = urlparse(url)
        assert parsed.scheme == "http"
        assert parsed.hostname == "127.0.0.1"
        assert parsed.path == "/mcp"
        assert parsed.port and parsed.port > 0
        assert await _port_open("127.0.0.1", parsed.port) is True

        await server.stop()
        await _wait_port_closed("127.0.0.1", parsed.port)

    async def test_stop_is_idempotent(self):
        server = ToolMCPServer([], _serializer())
        await server.start()
        await server.stop()
        await server.stop()  # no error
