# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

from autogen.beta.middleware.base import Middleware

if TYPE_CHECKING:
    from autogen.beta.annotations import Context
    from autogen.beta.events import BaseEvent, ModelResponse
    from autogen.beta.tools import Tool


class GuardrailTripped(Exception):  # noqa: N818
    """Raised when a guardrail check fails."""

    def __init__(self, message: str, response: ModelResponse | None = None):
        self.response = response
        super().__init__(message)


def guardrail(func: Callable[..., Awaitable[None]]) -> Middleware:
    """Create a guardrail middleware from a check function.

    The function receives ``(response, ctx)`` and should raise
    ``GuardrailTripped`` if the response violates the guardrail.

    Example::

        @guardrail
        async def no_profanity(response, ctx):
            if response.message and "badword" in response.message.content:
                raise GuardrailTripped("Profanity detected")
    """

    class _Guardrail(Middleware):
        async def on_llm_call(
            self,
            messages: list[BaseEvent],
            ctx: Context,
            tools: list[Tool],
            next: Callable[..., Awaitable[ModelResponse]],
        ) -> ModelResponse:
            response = await next(messages, ctx, tools)
            await func(response, ctx)
            return response

    mw = _Guardrail()
    mw.__name__ = func.__name__  # type: ignore[attr-defined]
    return mw
