# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

from autogen.beta.middleware.base import Middleware

from .guardrail import GuardrailTripped

if TYPE_CHECKING:
    from autogen.beta.annotations import Context
    from autogen.beta.events import BaseEvent, ModelResponse
    from autogen.beta.tools import Tool


class RetryOnGuardrail(Middleware):
    """Catches ``GuardrailTripped`` and retries the LLM call with feedback.

    Wraps inner middleware (including guardrails) and on failure appends
    the violation message to the prompt so the model can self-correct.
    """

    def __init__(self, max_retries: int = 3):
        self._max_retries = max_retries

    async def on_llm_call(
        self,
        messages: list[BaseEvent],
        ctx: Context,
        tools: list[Tool],
        next: Callable[..., Awaitable[ModelResponse]],
    ) -> ModelResponse:
        current_messages = list(messages)
        from autogen.beta.events import HumanMessage

        for attempt in range(self._max_retries + 1):
            try:
                return await next(current_messages, ctx, tools)
            except GuardrailTripped as e:
                if attempt == self._max_retries:
                    raise

                # Inject feedback into the message history for the retry attempt
                # instead of polluting the global system prompt.
                feedback = HumanMessage(content=f"[Guardrail violation: {e}. Please try again.]")
                current_messages.append(feedback)
