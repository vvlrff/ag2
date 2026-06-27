# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from autogen.beta.config.acp.session import ACPSession, new_prompt_text
from autogen.beta.events import ModelRequest, TextInput
from autogen.beta.events.types import ModelMessage


def _req(text: str) -> ModelRequest:
    return ModelRequest(parts=[TextInput(text)])


def test_delta_returns_only_new_requests():
    msgs = [_req("first"), ModelMessage("reply"), _req("second")]
    text, count = new_prompt_text(msgs, sent_count=0)
    assert "first" in text and "second" in text
    assert count == 2


def test_delta_skips_already_sent():
    msgs = [_req("first"), _req("second")]
    text, count = new_prompt_text(msgs, sent_count=1)
    assert text == "second"
    assert count == 2


def test_delta_empty_when_nothing_new():
    msgs = [_req("only")]
    text, count = new_prompt_text(msgs, sent_count=1)
    assert text == ""
    assert count == 1


@pytest.mark.asyncio
async def test_close_stops_mcp_server():
    session = ACPSession()
    stopped = {"called": False}

    class _FakeMCP:
        async def stop(self) -> None:
            stopped["called"] = True

    session.mcp = _FakeMCP()
    await session.close()
    assert stopped["called"] is True
    assert session.mcp is None


def test_mcp_defaults_to_none():
    assert ACPSession().mcp is None
