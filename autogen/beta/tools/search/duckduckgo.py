# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable
from contextlib import ExitStack
from typing import TYPE_CHECKING, Annotated

from ddgs.ddgs import DDGS
from pydantic import Field

from autogen.beta.middleware import BaseMiddleware, ToolMiddleware
from autogen.beta.tools.final import tool
from autogen.beta.tools.final.function_tool import FunctionTool
from autogen.beta.tools.tool import Tool

if TYPE_CHECKING:
    from autogen.beta.annotations import Context


class DuckDuckGoSearchTool(Tool):
    __slots__ = ("_tool", "name")

    def __init__(
        self,
        max_results: int = 5,
        region: str = "us-en",
        safesearch: str = "moderate",
        client: DDGS | None = None,
        name: str = "duckduckgo_search",
        *,
        description: str = ("Search the web using DuckDuckGo. Returns titles, URLs, and snippets for each result."),
        middleware: Iterable[ToolMiddleware] = (),
    ) -> None:
        _client = client if client is not None else DDGS()

        def duckduckgo_search(
            query: Annotated[str, Field(description="The search query string.")],
        ) -> str:
            """Search the web using DuckDuckGo and return results."""
            results = _client.text(
                query,
                region=region,
                safesearch=safesearch,
                max_results=max_results,
            )
            if not results:
                return "No results found."
            lines = []
            for i, r in enumerate(results, 1):
                lines.append(f"{i}. {r['title']}\n   {r['href']}\n   {r['body']}")
            return "\n\n".join(lines)

        self._tool: FunctionTool = tool(
            duckduckgo_search,
            name=name,
            description=description,
            middleware=middleware,
        )
        self.name = name

    async def schemas(self, context: "Context") -> list:
        return await self._tool.schemas(context)

    def register(
        self,
        stack: ExitStack,
        context: "Context",
        *,
        middleware: Iterable[BaseMiddleware] = (),
    ) -> None:
        self._tool.register(stack, context, middleware=middleware)
