# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
"""A minimal ACP *Agent* subprocess for tests.

Scripted behaviors keyed on the prompt text:
- ``"hang"``           : block forever (exercises the client's turn timeout).
- ``"call <tool> <json>"``: connect to the AG2-hosted MCP server and invoke
  ``<tool>`` with the JSON arguments, then stream the tool's result text back.
- anything else         : emit a thought, a message chunk, and a builtin tool call.

Run directly: ``python _fake_agent.py`` (serves ACP over stdio).
"""

import asyncio
import json

import acp
from acp import schema
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client


def _text(block: str) -> schema.TextContentBlock:
    return schema.TextContentBlock(type="text", text=block)


class FakeAgent(acp.Agent):
    def __init__(self) -> None:
        self.conn: acp.Client | None = None
        self._cancelled = asyncio.Event()
        self._mcp_url: str | None = None

    def on_connect(self, conn: acp.Client) -> None:
        self.conn = conn

    async def initialize(self, protocol_version, client_capabilities=None, client_info=None, **kw):
        return schema.InitializeResponse(protocol_version=acp.PROTOCOL_VERSION)

    async def new_session(self, cwd, additional_directories=None, mcp_servers=None, **kw):
        for srv in mcp_servers or []:
            url = getattr(srv, "url", None)
            if url:
                self._mcp_url = url
        return schema.NewSessionResponse(session_id="fake-session-1")

    async def cancel(self, session_id, **kw):
        self._cancelled.set()
        return None

    async def _call_ag2_tool(self, name: str, arguments: dict) -> str:
        async with streamable_http_client(self._mcp_url) as (read, write, _), ClientSession(read, write) as mcp:
            await mcp.initialize()
            result = await mcp.call_tool(name, arguments)
            return "".join(block.text for block in result.content if getattr(block, "type", None) == "text")

    async def prompt(self, prompt, session_id, message_id=None, **kw):
        text = "".join(getattr(b, "text", "") for b in prompt)

        async def update(u: schema.ContentChunk):
            await self.conn.session_update(session_id=session_id, update=u)

        if text.strip() == "hang":
            await self._cancelled.wait()
            self._cancelled.clear()
            return schema.PromptResponse(stop_reason="cancelled")

        if text.startswith("call "):
            _, name, raw = text.split(" ", 2)
            tool_result = await self._call_ag2_tool(name, json.loads(raw))
            await update(schema.AgentMessageChunk(session_update="agent_message_chunk", content=_text(tool_result)))
            return schema.PromptResponse(stop_reason="end_turn")

        await update(schema.AgentThoughtChunk(session_update="agent_thought_chunk", content=_text("planning")))
        await update(schema.AgentMessageChunk(session_update="agent_message_chunk", content=_text("done")))
        await update(
            schema.ToolCallStart(session_update="tool_call", tool_call_id="t1", title="Echo", status="pending")
        )
        await update(
            schema.ToolCallProgress(
                session_update="tool_call_update",
                tool_call_id="t1",
                status="completed",
                content=[schema.ContentToolCallContent(type="content", content=_text("ok"))],
            )
        )

        return schema.PromptResponse(
            stop_reason="end_turn",
            usage=schema.Usage(input_tokens=3, output_tokens=1, total_tokens=4),
        )


if __name__ == "__main__":
    asyncio.run(acp.run_agent(FakeAgent()))
