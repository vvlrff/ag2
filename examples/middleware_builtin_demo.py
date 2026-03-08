# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Demo of all built-in middleware using TestConfig (no LLM required)."""

import asyncio
import logging

from autogen.beta import Agent, tool
from autogen.beta.events import ModelMessage, ModelResponse, ToolCall, ToolCalls
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
from autogen.beta.testing import TestConfig


# ---------------------------------------------------------------------------
# Shared tool
# ---------------------------------------------------------------------------
@tool
def add(a: int, b: int) -> str:
    """Add two numbers."""
    return str(a + b)


# ---------------------------------------------------------------------------
# 1. HistoryLimiter — only keeps last N events
# ---------------------------------------------------------------------------
async def demo_history_limiter():
    print("\n=== HistoryLimiter Demo ===")
    config = TestConfig("Response after history limiting")
    agent = Agent(
        "test",
        prompt="You are helpful.",
        config=config,
        middleware=[HistoryLimiter(max_events=5)],
    )
    conv = await agent.ask("Hello!")
    print(f"  Result: {conv.message.message.content}")
    print("  OK — HistoryLimiter passed through successfully")


# ---------------------------------------------------------------------------
# 2. TokenLimiter — trims by estimated token count
# ---------------------------------------------------------------------------
async def demo_token_limiter():
    print("\n=== TokenLimiter Demo ===")
    config = TestConfig("Response after token limiting")
    agent = Agent(
        "test",
        prompt="You are helpful.",
        config=config,
        middleware=[TokenLimiter(max_tokens=100)],
    )
    conv = await agent.ask("Hello!")
    print(f"  Result: {conv.message.message.content}")
    print("  OK — TokenLimiter passed through successfully")


# ---------------------------------------------------------------------------
# 3. RetryMiddleware — retries on transient errors
# ---------------------------------------------------------------------------
async def demo_retry_middleware():
    print("\n=== RetryMiddleware Demo ===")

    call_count = 0
    original_config = TestConfig("Success after retries")
    client = original_config.create()
    original_call = client.__call__

    async def flaky_call(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise ConnectionError(f"Transient failure #{call_count}")
        return await original_call(*args, **kwargs)

    client.__call__ = flaky_call

    # We can't easily inject the flaky client via TestConfig, so let's test
    # RetryMiddleware directly via the chain
    from autogen.beta.middleware.base import _build_chain

    attempt_count = 0

    async def flaky_base(messages, ctx, tools):
        nonlocal attempt_count
        attempt_count += 1
        if attempt_count < 3:
            raise ConnectionError(f"Transient failure #{attempt_count}")
        return ModelResponse(message=ModelMessage(content="Success after retries!"))

    mw = RetryMiddleware(max_retries=3, retry_on=(ConnectionError,))
    chain = _build_chain([mw], "on_llm_call", flaky_base)

    class FakeCtx:
        prompt = []

    result = await chain([], FakeCtx(), [])
    print(f"  Result: {result.message.content}")
    print(f"  Attempts: {attempt_count}")
    assert attempt_count == 3, f"Expected 3 attempts, got {attempt_count}"
    print("  OK — RetryMiddleware retried and succeeded")


# ---------------------------------------------------------------------------
# 4. TimeoutMiddleware — enforces time limit
# ---------------------------------------------------------------------------
async def demo_timeout_middleware():
    print("\n=== TimeoutMiddleware Demo ===")

    # Test that a fast call succeeds
    config = TestConfig("Fast response")
    agent = Agent(
        "test",
        prompt="You are helpful.",
        config=config,
        middleware=[TimeoutMiddleware(seconds=5.0)],
    )
    conv = await agent.ask("Hello!")
    print(f"  Fast call result: {conv.message.message.content}")

    # Test that a slow call times out (via direct chain)
    from autogen.beta.middleware.base import _build_chain

    async def slow_base(messages, ctx, tools):
        await asyncio.sleep(10)
        return ModelResponse(message=ModelMessage(content="Too slow"))

    mw = TimeoutMiddleware(seconds=0.05)
    chain = _build_chain([mw], "on_llm_call", slow_base)

    class FakeCtx:
        prompt = []

    try:
        await chain([], FakeCtx(), [])
        print("  ERROR — should have timed out!")
    except asyncio.TimeoutError:
        print("  Slow call correctly timed out")
    print("  OK — TimeoutMiddleware works")


# ---------------------------------------------------------------------------
# 5. ToolFilter — dynamically filter tools
# ---------------------------------------------------------------------------
async def demo_tool_filter():
    print("\n=== ToolFilter Demo ===")

    from autogen.beta.middleware.base import _build_chain

    captured_tools = []

    async def capture_base(messages, ctx, tools):
        captured_tools.extend(tools)
        return ModelResponse(message=ModelMessage(content="ok"))

    @tool
    def secret_tool(x: str) -> str:
        """A secret tool."""
        return x

    @tool
    def public_tool(x: str) -> str:
        """A public tool."""
        return x

    def only_public(tools, ctx):
        return [t for t in tools if t.name != "secret_tool"]

    mw = ToolFilter(filter_fn=only_public)
    chain = _build_chain([mw], "on_llm_call", capture_base)

    class FakeCtx:
        prompt = []

    await chain([], FakeCtx(), [secret_tool, public_tool])
    tool_names = [t.name for t in captured_tools]
    print(f"  Tools seen by LLM: {tool_names}")
    assert tool_names == ["public_tool"], f"Expected ['public_tool'], got {tool_names}"
    print("  OK — ToolFilter correctly filtered tools")


# ---------------------------------------------------------------------------
# 6. Guardrail + RetryOnGuardrail
# ---------------------------------------------------------------------------
async def demo_guardrail():
    print("\n=== Guardrail Demo ===")

    from autogen.beta.middleware.base import _build_chain

    # Test @guardrail decorator that passes
    @guardrail
    async def check_no_bad_words(response, ctx):
        if response.message and "bad" in response.message.content:
            raise GuardrailTripped("Contains bad word")

    class FakeCtx:
        prompt = []

    # Good response passes
    async def good_base(messages, ctx, tools):
        return ModelResponse(message=ModelMessage(content="This is fine"))

    chain = _build_chain([check_no_bad_words], "on_llm_call", good_base)
    result = await chain([], FakeCtx(), [])
    print(f"  Good response passed: {result.message.content}")

    # Bad response triggers guardrail
    async def bad_base(messages, ctx, tools):
        return ModelResponse(message=ModelMessage(content="This is bad content"))

    chain = _build_chain([check_no_bad_words], "on_llm_call", bad_base)
    try:
        await chain([], FakeCtx(), [])
        print("  ERROR — guardrail should have tripped!")
    except GuardrailTripped as e:
        print(f"  Guardrail correctly tripped: {e}")

    # RetryOnGuardrail: retries until guardrail passes
    call_count = 0

    async def improving_base(messages, ctx, tools):
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            return ModelResponse(message=ModelMessage(content="This is bad"))
        return ModelResponse(message=ModelMessage(content="This is good"))

    ctx = FakeCtx()
    chain = _build_chain([RetryOnGuardrail(max_retries=5), check_no_bad_words], "on_llm_call", improving_base)
    result = await chain([], ctx, [])
    print(f"  RetryOnGuardrail result: {result.message.content} (after {call_count} attempts)")
    assert call_count == 3
    assert len(ctx.prompt) == 2  # two violations appended
    print("  OK — Guardrail + RetryOnGuardrail work")


# ---------------------------------------------------------------------------
# 7. LoggingMiddleware — logs calls with timing
# ---------------------------------------------------------------------------
async def demo_logging_middleware():
    print("\n=== LoggingMiddleware Demo ===")

    # Set up a logger that captures to a list
    logger = logging.getLogger("demo.middleware")
    logger.setLevel(logging.INFO)
    captured = []

    class ListHandler(logging.Handler):
        def emit(self, record):
            captured.append(record.getMessage())

    handler = ListHandler()
    logger.addHandler(handler)

    config = TestConfig("Logged response")
    agent = Agent(
        "test",
        prompt="You are helpful.",
        config=config,
        middleware=[LoggingMiddleware(logger=logger)],
    )
    conv = await agent.ask("Hello!")
    print(f"  Result: {conv.message.message.content}")
    print(f"  Logged messages:")
    for msg in captured:
        print(f"    - {msg}")
    assert any("LLM call" in m for m in captured), "Missing LLM call log"
    assert any("LLM response" in m for m in captured), "Missing LLM response log"
    assert any("Turn started" in m for m in captured), "Missing turn started log"
    assert any("Turn finished" in m for m in captured), "Missing turn finished log"
    print("  OK — LoggingMiddleware logs correctly")

    logger.removeHandler(handler)


# ---------------------------------------------------------------------------
# 8. Stacked middleware — multiple middleware composed together
# ---------------------------------------------------------------------------
async def demo_stacked():
    print("\n=== Stacked Middleware Demo ===")

    logger = logging.getLogger("demo.stacked")
    logger.setLevel(logging.INFO)
    logged = []

    class ListHandler(logging.Handler):
        def emit(self, record):
            logged.append(record.getMessage())

    handler = ListHandler()
    logger.addHandler(handler)

    config = TestConfig("Stacked response")
    agent = Agent(
        "test",
        prompt="You are helpful.",
        config=config,
        middleware=[
            LoggingMiddleware(logger=logger),  # outermost
            TimeoutMiddleware(seconds=10.0),
            HistoryLimiter(max_events=50),
            TokenLimiter(max_tokens=5000),
        ],
    )
    conv = await agent.ask("Test stacking middleware")
    print(f"  Result: {conv.message.message.content}")
    print(f"  Log count: {len(logged)} messages")
    assert conv.message.message.content == "Stacked response"
    print("  OK — Stacked middleware works correctly")

    logger.removeHandler(handler)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def main():
    print("Built-in Middleware Demos")
    print("=" * 60)

    await demo_history_limiter()
    await demo_token_limiter()
    await demo_retry_middleware()
    await demo_timeout_middleware()
    await demo_tool_filter()
    await demo_guardrail()
    await demo_logging_middleware()
    await demo_stacked()

    print("\n" + "=" * 60)
    print("All demos passed!")


if __name__ == "__main__":
    asyncio.run(main())
