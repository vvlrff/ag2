# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from typing import Literal

from .base import BuiltinTool

AnthropicCodeExecutionVersion = Literal["code_execution_20250825", "code_execution_20250522"]
"""Anthropic code execution tool versions.

* ``'code_execution_20250825'`` — latest; Bash commands + file operations
  (create, view, edit).  Supports any language via shell.
* ``'code_execution_20250522'`` — legacy; Python-only sandbox.
"""


@dataclass
class CodeExecutionTool(BuiltinTool):
    """Provider-native code execution tool.

    Enables the model to write and run code in a sandboxed environment on the
    provider's infrastructure.  Useful for mathematical computations, data
    analysis, and other programmatic tasks.

    Provider support:

    * **Anthropic** — Bash + file operations sandbox.
      Supported models: Claude Opus/Sonnet/Haiku 4.x and later.
      See ``anthropic_version`` for available versions.
    * **OpenAI Responses API** — Code Interpreter.  By default uses an
      ephemeral container (no setup required).  Pass ``container_id`` to
      reuse a persistent container across requests.

    Args:
        anthropic_version: Anthropic tool version to use.

            * ``'code_execution_20250825'`` *(default)* — Bash commands and
              file operations; supports any language.
            * ``'code_execution_20250522'`` — legacy Python-only sandbox.

            *Anthropic only.*
        container_id: Optional OpenAI container ID (``cntr_...``) for reusing
            a persistent container across requests.  When ``None`` (default),
            an ephemeral (``"auto"``) container is used — no setup needed.
            *OpenAI only.*

    Example::

        from autogen.beta import Agent
        from autogen.beta.config import AnthropicConfig, OpenAIResponsesConfig
        from autogen.beta.tools import CodeExecutionTool

        # Anthropic — latest Bash sandbox (default)
        agent = Agent(
            "coder",
            config=AnthropicConfig("claude-sonnet-4-6"),
            tools=[CodeExecutionTool()],
        )

        # Anthropic — legacy Python-only sandbox
        agent = Agent(
            "coder",
            config=AnthropicConfig("claude-haiku-4-5-20251001"),
            tools=[CodeExecutionTool(anthropic_version="code_execution_20250522")],
        )

        # OpenAI — ephemeral container, no setup needed
        agent = Agent(
            "coder",
            config=OpenAIResponsesConfig("gpt-4o"),
            tools=[CodeExecutionTool()],
        )

        # OpenAI — reuse a persistent container across requests
        agent = Agent(
            "coder",
            config=OpenAIResponsesConfig("gpt-4o"),
            tools=[CodeExecutionTool(container_id="cntr_abc123")],
        )
    """

    anthropic_version: AnthropicCodeExecutionVersion = field(default="code_execution_20250825")
    container_id: str | None = field(default=None)

    @property
    def kind(self) -> str:
        return "code_execution"
