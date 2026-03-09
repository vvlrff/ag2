# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging

import pytest

from autogen.beta.events import HumanMessage, ModelMessage, ModelRequest, ModelResponse, ToolCall, ToolResult
from autogen.beta.middleware.base import _build_chain
from autogen.beta.middleware.builtin import (
    GuardrailTripped,
    HistoryLimiter,
    LoggingMiddleware,
    RetryMiddleware,
    RetryOnGuardrail,
    TimeoutMiddleware,
    TokenLimiter,
    ToolFilter,
    guardrail,
)

# --- Helpers ---


def _make_response(content: str = "ok") -> ModelResponse:
    return ModelResponse(message=ModelMessage(content=content))


async def _base_llm(messages, ctx, tools):
    return _make_response()


async def _base_tool(tool_call, ctx):
    return ToolResult(tool_call_id=tool_call.id, content="result")


async def _base_turn(request, ctx):
    return _make_response()


class _FakeCtx:
    """Minimal stub for Context in tests."""

    def __init__(self):
        self.prompt: list[str] = []


# --- HistoryLimiter ---


@pytest.mark.asyncio
async def test_history_limiter_truncates():
    mw = HistoryLimiter(max_events=3)
    messages = list(range(10))  # fake messages
    captured = {}

    async def capture(messages, ctx, tools):
        captured["messages"] = messages
        return _make_response()

    chain = _build_chain([mw], "on_llm_call", capture)
    await chain(messages, _FakeCtx(), [])
    assert len(captured["messages"]) == 3


@pytest.mark.asyncio
async def test_history_limiter_preserves_first_request():
    mw = HistoryLimiter(max_events=3)
    request = ModelRequest(message=HumanMessage(content="original goal"))
    filler = [_make_response(f"r{i}") for i in range(5)]
    messages = [request] + filler
    captured = {}

    async def capture(messages, ctx, tools):
        captured["messages"] = messages
        return _make_response()

    chain = _build_chain([mw], "on_llm_call", capture)
    await chain(messages, _FakeCtx(), [])
    assert len(captured["messages"]) == 3
    assert captured["messages"][0] is request


@pytest.mark.asyncio
async def test_history_limiter_max_events_one():
    """With max_events=1, should not duplicate or return all messages."""
    mw = HistoryLimiter(max_events=1)
    request = ModelRequest(message=HumanMessage(content="goal"))
    messages = [request, _make_response("r1"), _make_response("r2")]
    captured = {}

    async def capture(messages, ctx, tools):
        captured["messages"] = messages
        return _make_response()

    chain = _build_chain([mw], "on_llm_call", capture)
    await chain(messages, _FakeCtx(), [])
    assert len(captured["messages"]) == 1


@pytest.mark.asyncio
async def test_history_limiter_no_truncation_when_under():
    mw = HistoryLimiter(max_events=20)
    messages = list(range(5))

    captured = {}

    async def capture(messages, ctx, tools):
        captured["messages"] = messages
        return _make_response()

    chain = _build_chain([mw], "on_llm_call", capture)
    await chain(messages, _FakeCtx(), [])
    assert len(captured["messages"]) == 5


# --- TokenLimiter ---


@pytest.mark.asyncio
async def test_token_limiter_trims():
    mw = TokenLimiter(max_tokens=10, chars_per_token=1)
    # Each message str is ~7 chars for "msg_0" etc. but str() adds more
    messages = [f"m{i}" for i in range(100)]

    captured = {}

    async def capture(messages, ctx, tools):
        captured["messages"] = messages
        return _make_response()

    chain = _build_chain([mw], "on_llm_call", capture)
    await chain(messages, _FakeCtx(), [])
    assert len(captured["messages"]) < 100


# --- RetryMiddleware ---


@pytest.mark.asyncio
async def test_retry_middleware_retries_on_exception():
    call_count = 0

    async def failing_llm(messages, ctx, tools):
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise ValueError("transient")
        return _make_response()

    mw = RetryMiddleware(max_retries=3, retry_on=(ValueError,))
    chain = _build_chain([mw], "on_llm_call", failing_llm)
    result = await chain([], _FakeCtx(), [])
    assert result.message.content == "ok"
    assert call_count == 3


@pytest.mark.asyncio
async def test_retry_middleware_raises_after_max():
    async def always_fail(messages, ctx, tools):
        raise ValueError("fail")

    mw = RetryMiddleware(max_retries=2, retry_on=(ValueError,))
    chain = _build_chain([mw], "on_llm_call", always_fail)
    with pytest.raises(ValueError, match="fail"):
        await chain([], _FakeCtx(), [])


# --- TimeoutMiddleware ---


@pytest.mark.asyncio
async def test_timeout_middleware_raises_on_slow():
    async def slow_llm(messages, ctx, tools):
        await asyncio.sleep(10)
        return _make_response()

    mw = TimeoutMiddleware(seconds=0.01)
    chain = _build_chain([mw], "on_llm_call", slow_llm)
    with pytest.raises(asyncio.TimeoutError):
        await chain([], _FakeCtx(), [])


@pytest.mark.asyncio
async def test_timeout_middleware_passes_fast():
    mw = TimeoutMiddleware(seconds=5.0)
    chain = _build_chain([mw], "on_llm_call", _base_llm)
    result = await chain([], _FakeCtx(), [])
    assert result.message.content == "ok"


# --- ToolFilter ---


@pytest.mark.asyncio
async def test_tool_filter():
    captured = {}

    async def capture(messages, ctx, tools):
        captured["tools"] = tools
        return _make_response()

    def only_approved(tools, ctx):
        return [t for t in tools if t != "bad_tool"]

    mw = ToolFilter(filter_fn=only_approved)
    chain = _build_chain([mw], "on_llm_call", capture)
    await chain([], _FakeCtx(), ["good_tool", "bad_tool", "other_tool"])
    assert captured["tools"] == ["good_tool", "other_tool"]


# --- Guardrail ---


@pytest.mark.asyncio
async def test_guardrail_decorator_passes():
    @guardrail
    async def check_ok(response, ctx):
        pass  # no violation

    chain = _build_chain([check_ok], "on_llm_call", _base_llm)
    result = await chain([], _FakeCtx(), [])
    assert result.message.content == "ok"


@pytest.mark.asyncio
async def test_guardrail_decorator_raises():
    @guardrail
    async def check_bad(response, ctx):
        raise GuardrailTripped("bad content")

    chain = _build_chain([check_bad], "on_llm_call", _base_llm)
    with pytest.raises(GuardrailTripped, match="bad content"):
        await chain([], _FakeCtx(), [])


@pytest.mark.asyncio
async def test_guardrail_tripped_stores_response():
    resp = _make_response("bad")
    exc = GuardrailTripped("violation", response=resp)
    assert exc.response is resp


# --- RetryOnGuardrail ---


@pytest.mark.asyncio
async def test_retry_on_guardrail():
    call_count = 0
    captured_messages = []

    @guardrail
    async def strict_check(response, ctx):
        nonlocal call_count
        call_count += 1
        if call_count < 2:
            raise GuardrailTripped("try again")

    async def capture_llm(messages, ctx, tools):
        captured_messages.append(list(messages))
        return _make_response()

    mw = RetryOnGuardrail(max_retries=3)
    chain = _build_chain([mw, strict_check], "on_llm_call", capture_llm)
    ctx = _FakeCtx()
    result = await chain([], ctx, [])
    assert result.message.content == "ok"
    assert call_count == 2
    # First call: empty messages
    assert len(captured_messages[0]) == 0
    # Second call: includes feedback message
    assert len(captured_messages[1]) == 1
    assert "try again" in captured_messages[1][0].content
    # Ensure system prompt was NOT polluted
    assert len(ctx.prompt) == 0


@pytest.mark.asyncio
async def test_retry_on_guardrail_exhausted():
    @guardrail
    async def always_fail(response, ctx):
        raise GuardrailTripped("always bad")

    mw = RetryOnGuardrail(max_retries=2)
    chain = _build_chain([mw, always_fail], "on_llm_call", _base_llm)
    with pytest.raises(GuardrailTripped, match="always bad"):
        await chain([], _FakeCtx(), [])


# --- LoggingMiddleware ---


@pytest.mark.asyncio
async def test_logging_middleware_logs(caplog):
    logger = logging.getLogger("test.middleware")
    mw = LoggingMiddleware(logger=logger)

    with caplog.at_level(logging.INFO, logger="test.middleware"):
        chain = _build_chain([mw], "on_llm_call", _base_llm)
        await chain([], _FakeCtx(), [])

    assert any("LLM call" in r.message for r in caplog.records)
    assert any("LLM response" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_logging_middleware_tool(caplog):
    logger = logging.getLogger("test.middleware.tool")
    mw = LoggingMiddleware(logger=logger)
    tc = ToolCall(name="my_tool", arguments='{"x": 1}')

    with caplog.at_level(logging.INFO, logger="test.middleware.tool"):
        chain = _build_chain([mw], "on_tool_execute", _base_tool)
        await chain(tc, _FakeCtx())

    assert any("Tool execute" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_logging_middleware_turn(caplog):
    logger = logging.getLogger("test.middleware.turn")
    mw = LoggingMiddleware(logger=logger)

    with caplog.at_level(logging.INFO, logger="test.middleware.turn"):
        chain = _build_chain([mw], "on_turn", _base_turn)
        await chain(None, _FakeCtx())

    assert any("Turn started" in r.message for r in caplog.records)
    assert any("Turn finished" in r.message for r in caplog.records)
