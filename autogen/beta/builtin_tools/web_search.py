# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from typing import Literal

from .base import BuiltinTool

AnthropicWebSearchVersion = Literal["web_search_20260209", "web_search_20250305"]
"""Anthropic web search tool versions.

* ``'web_search_20260209'`` — latest; enables dynamic filtering: Claude executes
  code internally to post-process search results before they reach the context
  window, improving accuracy and reducing token usage.  Requires a model that
  supports code execution (Opus 4.6, Sonnet 4.6).  **Not** ZDR-eligible.
* ``'web_search_20250305'`` — basic; no dynamic filtering.  ZDR-eligible.
"""

OpenAIWebSearchVersion = Literal["web_search_preview", "web_search_preview_2025_03_11"]
"""OpenAI Responses API web search tool versions.

* ``'web_search_preview'`` — always resolves to the latest preview release.
* ``'web_search_preview_2025_03_11'`` — pinned to the March 2025 release.
"""


@dataclass
class WebSearchTool(BuiltinTool):
    """Provider-native web search tool.

    Enables the model to search the web in real time.  The search is executed
    entirely on the provider's infrastructure — no Python-side execution is
    required.

    Parameters are forwarded to the provider API where supported; unsupported
    parameters are silently ignored.

    Provider support:

    * **Anthropic** — ``anthropic_version``, ``max_uses``, ``allowed_domains``,
      ``blocked_domains``, ``user_location``
    * **OpenAI Responses API** — ``openai_version``, ``search_context_size``,
      ``user_location``, ``allowed_domains``

    Args:
        anthropic_version: Anthropic tool version to use.

            * ``'web_search_20260209'`` *(default)* — dynamic filtering via
              code execution; more accurate, lower token usage.  Not ZDR-eligible.
            * ``'web_search_20250305'`` — basic search.  ZDR-eligible.

            *Anthropic only.*
        openai_version: OpenAI Responses API tool version to use.

            * ``'web_search_preview'`` *(default)* — latest preview release.
            * ``'web_search_preview_2025_03_11'`` — pinned March 2025 release.

            *OpenAI only.*
        max_uses: Maximum number of web searches per request.
            ``None`` means unlimited.  *Anthropic only.*
        allowed_domains: Restrict searches to these domains only.
            *Anthropic and OpenAI.*
        blocked_domains: Never include results from these domains.
            *Anthropic only.*  Cannot be combined with ``allowed_domains``
            on Anthropic.
        search_context_size: Amount of web context retrieved per search.
            One of ``'low'``, ``'medium'`` (default), ``'high'``.
            *OpenAI only.*
        user_location: Localise results based on the user's location.
            Dict with optional keys ``city``, ``country``, ``region``,
            ``timezone``.  For OpenAI ``country`` must be a 2-letter code
            (e.g. ``'US'``).  *Anthropic and OpenAI.*

    Example:
        ```python
        from autogen.beta import Agent
        from autogen.beta.config import AnthropicConfig, OpenAIResponsesConfig
        from autogen.beta.builtin_tools import WebSearchTool

        # Anthropic — latest version with dynamic filtering (default)
        agent = Agent(
            "searcher",
            config=AnthropicConfig("claude-sonnet-4-6"),
            builtin_tools=[WebSearchTool(max_uses=3)],
        )

        # Anthropic — basic version, ZDR-compatible
        agent = Agent(
            "searcher",
            config=AnthropicConfig("claude-haiku-4-5-20251001"),
            builtin_tools=[WebSearchTool(anthropic_version="web_search_20250305")],
        )

        # OpenAI — pinned version
        agent = Agent(
            "searcher",
            config=OpenAIResponsesConfig("gpt-4o-mini"),
            builtin_tools=[WebSearchTool(
                openai_version="web_search_preview_2025_03_11",
                search_context_size="low",
            )],
        )

        ```
    """

    anthropic_version: AnthropicWebSearchVersion = field(default="web_search_20260209")
    openai_version: OpenAIWebSearchVersion = field(default="web_search_preview")

    max_uses: int | None = field(default=None)
    allowed_domains: list[str] | None = field(default=None)
    blocked_domains: list[str] | None = field(default=None)
    search_context_size: Literal["low", "medium", "high"] = field(default="medium")
    user_location: dict[str, str] | None = field(default=None)

    @property
    def kind(self) -> str:
        return "web_search"
