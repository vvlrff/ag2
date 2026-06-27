# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from autogen.beta import Agent
from autogen.beta.config.acp import ClaudeCodeConfig
from autogen.beta.events import ModelReasoning
from autogen.beta.events.tool_events import BuiltinToolCallEvent, BuiltinToolResultEvent


@pytest.mark.asyncio
async def test_ask_streams_thoughts_tools_and_returns_text(fake_agent_command, tmp_path):
    cfg = ClaudeCodeConfig(command=fake_agent_command, cwd=str(tmp_path), permission_policy="auto")
    agent = Agent("acp", config=cfg)

    seen: list = []

    try:
        async with agent.run("hello") as run:
            run.stream.subscribe(lambda e: seen.append(e))
            result = await run.result()
    finally:
        await cfg.aclose()

    assert result.body == "done"
    assert any(isinstance(e, ModelReasoning) and e.content == "planning" for e in seen)
    assert any(isinstance(e, BuiltinToolCallEvent) and e.name == "Echo" for e in seen)
    assert any(isinstance(e, BuiltinToolResultEvent) for e in seen)


@pytest.mark.asyncio
async def test_turn_timeout_surfaces_timeout(fake_agent_command, tmp_path):
    cfg = ClaudeCodeConfig(
        command=fake_agent_command,
        cwd=str(tmp_path),
        permission_policy="auto",
        turn_timeout=0.5,
    )
    agent = Agent("acp", config=cfg)

    try:
        async with agent.run("hang") as run:
            result = await run.result()
    finally:
        await cfg.aclose()
    # The turn timed out; body is whatever streamed before the timeout (empty here).
    assert result.body == ""


@pytest.mark.asyncio
async def test_aclose_terminates_subprocess(fake_agent_command, tmp_path):
    cfg = ClaudeCodeConfig(command=fake_agent_command, cwd=str(tmp_path), permission_policy="auto")
    agent = Agent("acp", config=cfg)

    async with agent.run("hello") as run:
        await run.result()

    assert cfg._sessions  # a live session was created
    procs = [s.proc for s in cfg._sessions.values()]
    await cfg.aclose()
    assert cfg._sessions == {}
    for proc in procs:
        assert proc.returncode is not None  # subprocess exited


@pytest.mark.asyncio
async def test_no_function_tools_starts_no_mcp_server(fake_agent_command, tmp_path):
    cfg = ClaudeCodeConfig(command=fake_agent_command, cwd=str(tmp_path), permission_policy="auto")
    agent = Agent("acp", config=cfg)

    try:
        async with agent.run("hello") as run:
            await run.result()
        # A session must have been created, otherwise the assertion below is vacuously true.
        assert cfg._sessions
        # No function tools were registered, so no MCP server should be live.
        assert all(s.mcp is None for s in cfg._sessions.values())
    finally:
        await cfg.aclose()
