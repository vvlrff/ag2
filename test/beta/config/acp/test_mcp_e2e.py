# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import contextlib
from urllib.parse import urlparse

import pytest

from autogen.beta import Agent
from autogen.beta.config.acp import ClaudeCodeConfig
from autogen.beta.events import BaseEvent, ToolCallEvent, ToolResultEvent
from autogen.beta.tools.final.function_tool import tool


async def _wait_port_closed(host: str, port: int, *, timeout: float = 2.0) -> None:
    """Poll until nothing accepts on host:port, failing after ``timeout``."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        try:
            _, writer = await asyncio.open_connection(host, port)
        except OSError:
            return
        writer.close()
        with contextlib.suppress(Exception):
            await writer.wait_closed()
        await asyncio.sleep(0.01)
    raise AssertionError(f"{host}:{port} still accepting connections after {timeout}s")


@pytest.mark.asyncio
async def test_cli_agent_calls_ag2_tool_round_trip(fake_agent_command, tmp_path):
    calls: list[tuple[int, int]] = []

    @tool
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        calls.append((a, b))
        return a + b

    cfg = ClaudeCodeConfig(command=fake_agent_command, cwd=str(tmp_path), permission_policy="auto")
    agent = Agent("acp", config=cfg, tools=[add])

    seen: list[BaseEvent] = []
    try:
        async with agent.run('call add {"a": 2, "b": 5}') as run:
            run.stream.subscribe(lambda e: seen.append(e))
            result = await run.result()
    finally:
        await cfg.aclose()

    # The AG2 @tool actually ran (observable side effect) ...
    assert calls == [(2, 5)]
    # ... its result was delivered back to the CLI agent, which echoed it ...
    assert result.body == "7"
    # ... and the call + result landed on the parent stream.
    assert any(isinstance(e, ToolCallEvent) and e.name == "add" for e in seen)
    assert any(isinstance(e, ToolResultEvent) and e.name == "add" for e in seen)


@pytest.mark.asyncio
async def test_cli_agent_calls_recursive_subagent_tool(fake_agent_command, tmp_path):
    child_ran = {"called": False}

    @tool
    def mark() -> str:
        """Record that the child agent ran."""
        child_ran["called"] = True
        return "child-done"

    # The child is itself an ACP agent driven over its own subprocess.
    child_cfg = ClaudeCodeConfig(command=fake_agent_command, cwd=str(tmp_path), permission_policy="auto")
    child = Agent("reviewer", config=child_cfg, tools=[mark])

    parent_cfg = ClaudeCodeConfig(command=fake_agent_command, cwd=str(tmp_path), permission_policy="auto")
    # as_tool() exposes the subagent task under the 'objective' parameter (not 'task').
    parent = Agent("acp", config=parent_cfg, tools=[child.as_tool(description="A reviewer subagent", name="reviewer")])

    try:
        # The parent CLI agent calls the "reviewer" subagent tool; that runs the
        # child agent, whose own CLI run calls `mark`. We assert the chain fired.
        async with parent.run('call reviewer {"objective": "call mark {}"}') as run:
            await run.result()
    finally:
        await parent_cfg.aclose()
        await child_cfg.aclose()

    assert child_ran["called"] is True


@pytest.mark.asyncio
async def test_mcp_server_stops_on_aclose(fake_agent_command, tmp_path):
    @tool
    def ping() -> str:
        """Return pong."""
        return "pong"

    cfg = ClaudeCodeConfig(command=fake_agent_command, cwd=str(tmp_path), permission_policy="auto")
    agent = Agent("acp", config=cfg, tools=[ping])

    async with agent.run("call ping {}") as run:
        await run.result()

    # A live MCP server with a bound URL exists.
    urls = [s.mcp.url for s in cfg._sessions.values() if s.mcp is not None]
    assert urls, "expected a started MCP server"
    endpoints = [urlparse(u) for u in urls]

    await cfg.aclose()

    # After aclose, the ports are no longer bound.
    for parsed in endpoints:
        assert parsed.hostname and parsed.port
        await _wait_port_closed(parsed.hostname, parsed.port)
