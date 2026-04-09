# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable

import pytest

from autogen.beta.annotations import Variable
from autogen.beta.context import Context
from autogen.beta.tools import ImageGenerationTool, UserLocation, WebSearchTool
from autogen.beta.tools.builtin._resolve import resolve_variable
from autogen.beta.tools.builtin.image_generation import ImageGenerationToolSchema
from autogen.beta.tools.builtin.mcp_server import MCPServerTool, MCPServerToolSchema
from autogen.beta.tools.builtin.shell import ContainerAutoEnvironment, ShellTool, ShellToolSchema
from autogen.beta.tools.builtin.web_fetch import WebFetchTool, WebFetchToolSchema
from autogen.beta.tools.builtin.web_search import WebSearchToolSchema

# --- resolve_variable ---


def test_resolve_variable_passthrough(context: Context) -> None:
    assert resolve_variable("hello", context) == "hello"
    assert resolve_variable(42, context) == 42
    assert resolve_variable(None, context) is None


def test_resolve_variable_from_context(make_context: Callable[..., Context]) -> None:
    loc = UserLocation(country="US")
    ctx = make_context(user_location=loc)

    result = resolve_variable(Variable("user_location"), ctx)

    assert result is loc


def test_resolve_variable_default(context: Context) -> None:
    fallback = UserLocation(country="DE")

    result = resolve_variable(Variable("user_location", default=fallback), context)

    assert result is fallback


def test_resolve_variable_default_factory(context: Context) -> None:
    result = resolve_variable(Variable("counter", default_factory=dict), context)

    assert result == {}


def test_resolve_variable_context_takes_precedence_over_default(make_context: Callable[..., Context]) -> None:
    ctx = make_context(mode="fast")

    result = resolve_variable(Variable("mode", default="slow"), ctx)

    assert result == "fast"


def test_resolve_variable_missing_raises(context: Context) -> None:
    with pytest.raises(KeyError, match="user_location"):
        resolve_variable(Variable("user_location"), context)


# --- WebSearchTool ---


@pytest.mark.asyncio
async def test_web_search_tool_variable_resolved(make_context: Callable[..., Context]) -> None:
    loc = UserLocation(city="Berlin", country="DE")
    tool = WebSearchTool(user_location=Variable("loc"))
    ctx = make_context(loc=loc)

    [schema] = await tool.schemas(ctx)

    assert isinstance(schema, WebSearchToolSchema)
    assert schema.user_location is loc


@pytest.mark.asyncio
async def test_web_search_tool_variable_missing_raises(context: Context) -> None:
    tool = WebSearchTool(user_location=Variable("loc"))

    with pytest.raises(KeyError, match="loc"):
        await tool.schemas(context)


# --- WebFetchTool ---


@pytest.mark.asyncio
async def test_web_fetch_tool_variable_resolved(make_context: Callable[..., Context]) -> None:
    tool = WebFetchTool(max_uses=Variable("limit"))
    ctx = make_context(limit=10)

    [schema] = await tool.schemas(ctx)

    assert isinstance(schema, WebFetchToolSchema)
    assert schema.max_uses == 10


@pytest.mark.asyncio
async def test_web_fetch_tool_variable_missing_raises(context: Context) -> None:
    tool = WebFetchTool(max_uses=Variable("limit"))

    with pytest.raises(KeyError, match="limit"):
        await tool.schemas(context)


# --- ShellTool ---


@pytest.mark.asyncio
async def test_shell_tool_variable_resolved(make_context: Callable[..., Context]) -> None:
    env = ContainerAutoEnvironment()
    tool = ShellTool(environment=Variable("env"))
    ctx = make_context(env=env)

    [schema] = await tool.schemas(ctx)

    assert isinstance(schema, ShellToolSchema)
    assert schema.environment is env


@pytest.mark.asyncio
async def test_shell_tool_variable_missing_raises(context: Context) -> None:
    tool = ShellTool(environment=Variable("env"))

    with pytest.raises(KeyError, match="env"):
        await tool.schemas(context)


# --- ImageGenerationTool ---


@pytest.mark.asyncio
async def test_image_generation_tool_variable_resolved(make_context: Callable[..., Context]) -> None:
    tool = ImageGenerationTool(quality="high", size=Variable("image_size"))
    ctx = make_context(image_size="1536x1024")

    [schema] = await tool.schemas(ctx)

    assert isinstance(schema, ImageGenerationToolSchema)
    assert schema.quality == "high"
    assert schema.size == "1536x1024"


@pytest.mark.asyncio
async def test_image_generation_tool_variable_missing_raises(make_context: Callable[..., Context]) -> None:
    tool = ImageGenerationTool(partial_images=Variable("partial_images"))
    ctx = make_context()

    with pytest.raises(KeyError, match="partial_images"):
        await tool.schemas(ctx)


# --- MCPServerTool ---


@pytest.mark.asyncio
async def test_mcp_server_tool_variable_resolved(make_context: Callable[..., Context]) -> None:
    tool = MCPServerTool(server_url=Variable("url"), server_label="test-mcp")
    ctx = make_context(url="https://mcp.example.com/sse")

    [schema] = await tool.schemas(ctx)

    assert isinstance(schema, MCPServerToolSchema)
    assert schema.server_url == "https://mcp.example.com/sse"


@pytest.mark.asyncio
async def test_mcp_server_tool_variable_missing_raises(context: Context) -> None:
    tool = MCPServerTool(server_url=Variable("url"), server_label="test-mcp")

    with pytest.raises(KeyError, match="url"):
        await tool.schemas(context)
