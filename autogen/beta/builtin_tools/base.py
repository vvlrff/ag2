# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod


class BuiltinTool(ABC):
    """Base class for LLM provider-native builtin tools.

    Builtin tools are executed natively by the LLM provider (e.g. Anthropic web
    search, Google Gemini code execution) rather than by AG2.  They do not
    require Python functions — the provider handles execution internally.

    Unlike regular :class:`~autogen.beta.tools.Tool` instances,
    ``BuiltinTool`` subclasses only carry the configuration needed to include
    the tool in the provider API request.  Each model client maps the tool's
    :attr:`kind` to the provider-specific API format.

    Use the ready-made subclasses from :mod:`autogen.beta.tools`:

    * :class:`~autogen.beta.tools.WebSearchTool`
    * :class:`~autogen.beta.tools.CodeExecutionTool`

    .. note::
        Currently supported providers: **Anthropic** (``AnthropicConfig``) and
        **OpenAI Responses API** (``OpenAIResponsesConfig``).
        Other providers will emit a warning and ignore ``builtin_tools``.

    Example:
        ```python
        from autogen.beta import Agent
        from autogen.beta.config import AnthropicConfig, OpenAIResponsesConfig
        from autogen.beta.tools import CodeExecutionTool, WebSearchTool

        agent = Agent("a", config=AnthropicConfig("claude-haiku-4-5-20251001"),
                      builtin_tools=[WebSearchTool(anthropic_version="web_search_20250305")])

        agent = Agent("a", config=OpenAIResponsesConfig("gpt-4o-mini"),
                      builtin_tools=[WebSearchTool(search_context_size="low")])
        ```
    """

    @property
    @abstractmethod
    def kind(self) -> str:
        """Tool kind identifier used as a discriminator across providers.

        Well-known values: ``'web_search'``, ``'code_execution'``.
        Each model client checks this value to decide how to translate the tool
        to the provider's API format.  Unknown kinds are silently ignored by
        providers that do not support them.
        """
        ...

    @property
    def name(self) -> str:
        """Human-readable name for the tool (defaults to :attr:`kind`)."""
        return self.kind
