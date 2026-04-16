# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable
from contextlib import ExitStack
from typing import TYPE_CHECKING, Annotated

from ddgs.ddgs import DDGS
from pydantic import Field, TypeAdapter
from typing_extensions import TypedDict

from autogen.beta.middleware import BaseMiddleware, ToolMiddleware
from autogen.beta.tools.final import tool
from autogen.beta.tools.final.function_tool import FunctionTool
from autogen.beta.tools.tool import Tool

if TYPE_CHECKING:
    from autogen.beta.annotations import Context


class DuckDuckGoResult(TypedDict):
    """A single DuckDuckGo search result."""

    title: str
    """The title of the search result."""
    href: str
    """The URL of the search result."""
    body: str
    """The body/snippet of the search result."""


_result_adapter = TypeAdapter(list[DuckDuckGoResult])


class DuckDuckGoSearchTool(Tool):
    """Client-side web search via DuckDuckGo.

    Works with any LLM provider. Does not require an API key.

    Requires ``pip install "ag2[duckduckgo]"``.

    Examples::

        ddg = DuckDuckGoSearchTool()
        agent = Agent("researcher", config=config, tools=[ddg])

        # With configuration
        ddg = DuckDuckGoSearchTool(max_results=10, region="us-en")

        # With custom DDGS client (e.g. proxy)
        from ddgs.ddgs import DDGS

        ddg = DuckDuckGoSearchTool(client=DDGS(proxy="socks5://..."))
    """

    __slots__ = ("_tool", "name")

    def __init__(
        self,
        max_results: int = 5,
        region: str = "wt-wt",
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
        ) -> list[DuckDuckGoResult]:
            """Search the web using DuckDuckGo and return results."""
            results = _client.text(
                query,
                region=region,
                safesearch=safesearch,
                max_results=max_results,
            )
            return _result_adapter.validate_python(results)

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
