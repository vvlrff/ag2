# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json
from unittest.mock import MagicMock

import pytest

from autogen.beta import Agent, MemoryStream
from autogen.beta.context import ConversationContext
from autogen.beta.events import ToolCallEvent, ToolCallsEvent, ToolResultEvent
from autogen.beta.events.types import ModelResponse
from autogen.beta.testing import TestConfig
from autogen.beta.tools import DuckDuckGoSearchTool

SAMPLE_RESULTS = [
    {"title": "AG2 Framework", "href": "https://ag2.ai", "body": "AG2 is an agent framework."},
    {"title": "GitHub - AG2", "href": "https://github.com/ag2ai/ag2", "body": "Open source repo."},
]


def _make_tool_call(query: str) -> ToolCallEvent:
    return ToolCallEvent(
        arguments=json.dumps({"query": query}),
        name="duckduckgo_search",
    )


def _make_config(query: str, final_reply: str = "done") -> TestConfig:
    return TestConfig(
        ModelResponse(tool_calls=ToolCallsEvent([_make_tool_call(query)])),
        final_reply,
    )


@pytest.mark.asyncio
class TestSchema:
    async def test_schema_has_query_param(self, context: ConversationContext) -> None:
        ddg = DuckDuckGoSearchTool(client=MagicMock())

        [schema] = await ddg.schemas(context)

        assert schema.function.name == "duckduckgo_search"
        params = schema.function.parameters
        assert "query" in params["properties"]
        assert params["properties"]["query"]["type"] == "string"
        assert "query" in params["required"]

    async def test_custom_name_and_description(self, context: ConversationContext) -> None:
        ddg = DuckDuckGoSearchTool(client=MagicMock(), name="my_search", description="Custom search tool.")

        [schema] = await ddg.schemas(context)

        assert schema.function.name == "my_search"
        assert schema.function.description == "Custom search tool."


@pytest.mark.asyncio
class TestSearchExecution:
    async def test_search_returns_formatted_string(self) -> None:
        mock_client = MagicMock()
        mock_client.text.return_value = SAMPLE_RESULTS

        ddg = DuckDuckGoSearchTool(client=mock_client)
        agent = Agent("a", config=_make_config("AG2 framework"), tools=[ddg])

        tool_results: list[str] = []
        stream = MemoryStream()
        stream.where(ToolResultEvent).subscribe(lambda e: tool_results.append(e.result.content))

        await agent.ask("search", stream=stream)

        assert tool_results
        result = tool_results[0]
        assert "1. AG2 Framework" in result
        assert "https://ag2.ai" in result
        assert "AG2 is an agent framework." in result
        assert "2. GitHub - AG2" in result
        assert "https://github.com/ag2ai/ag2" in result
        mock_client.text.assert_called_once_with("AG2 framework", region="wt-wt", safesearch="moderate", max_results=5)

    async def test_search_empty_results(self) -> None:
        mock_client = MagicMock()
        mock_client.text.return_value = []

        ddg = DuckDuckGoSearchTool(client=mock_client)
        agent = Agent("a", config=_make_config("nonexistent query"), tools=[ddg])

        tool_results: list[str] = []
        stream = MemoryStream()
        stream.where(ToolResultEvent).subscribe(lambda e: tool_results.append(e.result.content))

        await agent.ask("search", stream=stream)

        assert tool_results
        assert tool_results[0] == "No results found."

    async def test_custom_client_used(self) -> None:
        mock_client = MagicMock()
        mock_client.text.return_value = SAMPLE_RESULTS

        ddg = DuckDuckGoSearchTool(client=mock_client, max_results=3, region="us-en", safesearch="off")
        agent = Agent("a", config=_make_config("test query"), tools=[ddg])

        await agent.ask("search")

        mock_client.text.assert_called_once_with("test query", region="us-en", safesearch="off", max_results=3)

    async def test_custom_tool_name_in_agent(self) -> None:
        mock_client = MagicMock()
        mock_client.text.return_value = SAMPLE_RESULTS

        ddg = DuckDuckGoSearchTool(client=mock_client, name="web_search")

        tool_call = ToolCallEvent(
            arguments=json.dumps({"query": "test"}),
            name="web_search",
        )
        config = TestConfig(
            ModelResponse(tool_calls=ToolCallsEvent([tool_call])),
            "done",
        )
        agent = Agent("a", config=config, tools=[ddg])

        tool_results: list[str] = []
        stream = MemoryStream()
        stream.where(ToolResultEvent).subscribe(lambda e: tool_results.append(e.result.content))

        await agent.ask("search", stream=stream)

        assert tool_results
        assert "1. AG2 Framework" in tool_results[0]
