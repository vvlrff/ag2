# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json
from collections.abc import Iterable
from typing import Any

import pytest
from openai.types.responses import (
    Response,
    ResponseCodeInterpreterToolCall,
    ResponseFunctionToolCall,
    ResponseFunctionWebSearch,
    ResponseOutputMessage,
)
from openai.types.responses.response_function_web_search import ActionSearch
from openai.types.responses.response_output_item import ImageGenerationCall
from openai.types.responses.response_output_text import ResponseOutputText

from autogen.beta import MemoryStream
from autogen.beta.config.openai import (
    OpenAIResponsesClient,
    OpenAIServerToolCallEvent,
    OpenAIServerToolResultEvent,
)
from autogen.beta.context import ConversationContext
from autogen.beta.events import BaseEvent, ModelMessage, ModelResponse, ToolCallEvent, ToolCallsEvent
from autogen.beta.events.tool_events import ToolResult
from autogen.beta.tools.builtin.code_execution import CODE_EXECUTION_TOOL_NAME
from autogen.beta.tools.builtin.image_generation import IMAGE_GENERATION_TOOL_NAME
from autogen.beta.tools.builtin.web_search import WEB_SEARCH_TOOL_NAME


async def _process(output: Iterable[Any]) -> tuple[ModelResponse, list[BaseEvent]]:
    client = OpenAIResponsesClient(api_key="test")
    response = Response.model_construct(
        id="r1",
        object="response",
        model="gpt-5",
        output=list(output),
        usage=None,
    )
    stream = MemoryStream()
    context = ConversationContext(stream=stream)
    result = await client._process_response(response, context)
    return result, list(await stream.history.get_events())


@pytest.mark.asyncio
async def test_process_response_routes_all_item_types() -> None:
    web = ResponseFunctionWebSearch(
        id="ws_1",
        action=ActionSearch(type="search", query="bitcoin"),
        status="completed",
        type="web_search_call",
    )
    code = ResponseCodeInterpreterToolCall(
        id="ci_1",
        code="print(1)",
        status="completed",
        type="code_interpreter_call",
        outputs=None,
        container_id="c_1",
    )
    image = ImageGenerationCall(
        id="ig_1",
        status="completed",
        type="image_generation_call",
        result="YWJj",  # base64 "abc"
        revised_prompt=None,
        output_format="png",
    )
    msg = ResponseOutputMessage(
        id="msg_1",
        type="message",
        role="assistant",
        status="completed",
        content=[ResponseOutputText(type="output_text", text="Done.", annotations=[])],
    )
    user_tool = ResponseFunctionToolCall(
        id="id_1",
        call_id="call_1",
        name="multiply",
        arguments='{"a": 2, "b": 3}',
        type="function_call",
    )

    response, events = await _process([web, code, image, msg, user_tool])

    assert response.message == ModelMessage("Done.")
    assert response.tool_calls == ToolCallsEvent([
        ToolCallEvent(id="call_1", name="multiply", arguments='{"a": 2, "b": 3}'),
    ])
    assert [f.data for f in response.files] == [b"abc"]
    assert events == [
        OpenAIServerToolCallEvent(
            id="ws_1", name=WEB_SEARCH_TOOL_NAME, arguments=web.action.model_dump_json(), item=web
        ),
        OpenAIServerToolResultEvent(parent_id="ws_1", name=WEB_SEARCH_TOOL_NAME, result=ToolResult()),
        OpenAIServerToolCallEvent(
            id="ci_1", name=CODE_EXECUTION_TOOL_NAME, arguments=json.dumps({"code": "print(1)"}), item=code
        ),
        OpenAIServerToolResultEvent(parent_id="ci_1", name=CODE_EXECUTION_TOOL_NAME, result=ToolResult()),
        OpenAIServerToolCallEvent(id="ig_1", name=IMAGE_GENERATION_TOOL_NAME, arguments="", item=image),
        OpenAIServerToolResultEvent(parent_id="ig_1", name=IMAGE_GENERATION_TOOL_NAME, result=ToolResult()),
        ModelMessage("Done."),
    ]
