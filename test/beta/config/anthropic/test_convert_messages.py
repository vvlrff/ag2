# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import base64
from collections.abc import Callable
from typing import Any

import pytest
from anthropic.types import (
    BashCodeExecutionToolResultBlock,
    ServerToolUseBlock,
    TextEditorCodeExecutionToolResultBlock,
    WebSearchToolResultBlock,
)
from anthropic.types.bash_code_execution_tool_result_error import BashCodeExecutionToolResultError
from anthropic.types.text_editor_code_execution_tool_result_error import TextEditorCodeExecutionToolResultError
from dirty_equals import IsPartialDict

from autogen.beta import ToolResult
from autogen.beta.config.anthropic import (
    AnthropicServerToolCallEvent,
    AnthropicServerToolResultEvent,
)
from autogen.beta.config.anthropic.mappers import convert_messages
from autogen.beta.events import (
    AudioInput,
    BinaryInput,
    BinaryType,
    DocumentInput,
    FileIdInput,
    ImageInput,
    ModelRequest,
    ModelResponse,
    TextInput,
    ToolCallEvent,
    ToolCallsEvent,
    ToolResultEvent,
    ToolResultsEvent,
    VideoInput,
)
from autogen.beta.events.types import ModelMessage
from autogen.beta.exceptions import UnsupportedInputError


def _model_response_with_tool_call(arguments: str | None) -> ModelResponse:
    """Helper to build a ModelResponse containing a single tool call."""
    return ModelResponse(
        message=None,
        tool_calls=ToolCallsEvent(
            calls=[ToolCallEvent(id="tc_1", name="list_items", arguments=arguments)],
        ),
    )


@pytest.mark.parametrize(
    ("arguments", "expected_input"),
    [
        pytest.param("", {}, id="empty_string"),
        pytest.param(None, {}, id="none"),
        pytest.param("{}", {}, id="empty_object"),
        pytest.param('{"category": "books"}', {"category": "books"}, id="valid_object"),
    ],
)
def test_tool_call_arguments_parsed_into_input(arguments: str | None, expected_input: dict) -> None:
    result = convert_messages([_model_response_with_tool_call(arguments)])

    assert result == [
        IsPartialDict({
            "role": "assistant",
            "content": [IsPartialDict({"type": "tool_use", "input": expected_input})],
        }),
    ]


def test_full_sequence_with_empty_args() -> None:
    """A request -> response-with-tool-call -> tool-result sequence should convert cleanly."""
    events = [
        ModelRequest([TextInput("What items do we have?")]),
        _model_response_with_tool_call(""),
        ToolResultsEvent(
            results=[
                ToolResultEvent(
                    parent_id="tc_1",
                    name="list_items",
                    result=ToolResult(content="apple, banana"),
                )
            ],
        ),
    ]
    result = convert_messages(events)

    assert result[0] == IsPartialDict({"role": "user"})
    assert result[1] == IsPartialDict({
        "role": "assistant",
        "content": [IsPartialDict({"input": {}})],
    })
    assert result[2] == IsPartialDict({
        "role": "user",
        "content": [IsPartialDict({"type": "tool_result"})],
    })


class TestImageUrlInput:
    IMAGE_URL = "https://example.com/image.png"

    def test_converts_to_image_url_block(self) -> None:
        result = convert_messages([ModelRequest([ImageInput(url=self.IMAGE_URL)])])

        assert result == [
            {
                "role": "user",
                "content": [{"type": "image", "source": {"type": "url", "url": self.IMAGE_URL}}],
            }
        ]


class TestImageBinaryInput:
    SAMPLE_BYTES = b"\x89PNG\r\n\x1a\nfake"

    def test_converts_to_image_base64_block(self) -> None:
        result = convert_messages([ModelRequest([ImageInput(data=self.SAMPLE_BYTES, media_type="image/png")])])

        expected_b64 = base64.b64encode(self.SAMPLE_BYTES).decode()
        assert result == [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": "image/png", "data": expected_b64},
                    }
                ],
            }
        ]

    def test_vendor_metadata_cache_control_merges(self) -> None:
        result = convert_messages([
            ModelRequest([
                BinaryInput(
                    data=self.SAMPLE_BYTES,
                    media_type="image/png",
                    vendor_metadata={"cache_control": {"type": "ephemeral"}},
                    kind=BinaryType.IMAGE,
                )
            ])
        ])

        assert result == [
            IsPartialDict({
                "role": "user",
                "content": [IsPartialDict({"type": "image", "cache_control": {"type": "ephemeral"}})],
            })
        ]

    def test_vendor_metadata_filename_filtered_out(self) -> None:
        result = convert_messages([
            ModelRequest([
                BinaryInput(
                    data=self.SAMPLE_BYTES,
                    media_type="image/png",
                    vendor_metadata={"filename": "photo.png"},
                    kind=BinaryType.IMAGE,
                )
            ])
        ])

        content = result[0]["content"][0]
        assert "filename" not in content


class TestDocumentUrlInput:
    DOC_URL = "https://example.com/doc.pdf"

    def test_converts_to_document_url_block(self) -> None:
        result = convert_messages([ModelRequest([DocumentInput(url=self.DOC_URL)])])

        assert result == [
            {
                "role": "user",
                "content": [{"type": "document", "source": {"type": "url", "url": self.DOC_URL}}],
            }
        ]


class TestDocumentBinaryInput:
    SAMPLE_BYTES = b"%PDF-1.4"

    def test_converts_to_document_base64_block(self) -> None:
        result = convert_messages([ModelRequest([DocumentInput(data=self.SAMPLE_BYTES, media_type="application/pdf")])])

        expected_b64 = base64.b64encode(self.SAMPLE_BYTES).decode()
        assert result == [
            {
                "role": "user",
                "content": [
                    {
                        "type": "document",
                        "source": {"type": "base64", "media_type": "application/pdf", "data": expected_b64},
                    }
                ],
            }
        ]

    def test_vendor_metadata_merges(self) -> None:
        result = convert_messages([
            ModelRequest([
                BinaryInput(
                    data=self.SAMPLE_BYTES,
                    media_type="application/pdf",
                    vendor_metadata={"cache_control": {"type": "ephemeral"}},
                    kind=BinaryType.DOCUMENT,
                )
            ])
        ])

        assert result == [
            IsPartialDict({
                "role": "user",
                "content": [IsPartialDict({"type": "document", "cache_control": {"type": "ephemeral"}})],
            })
        ]


class TestFileIdInput:
    FILE_ID = "file_011CNha8iCJcU1wXNR6q4V8w"

    def test_no_filename_defaults_to_document(self) -> None:
        result = convert_messages([ModelRequest([FileIdInput(file_id=self.FILE_ID)])])

        assert result == [
            {
                "role": "user",
                "content": [{"type": "document", "source": {"type": "file", "file_id": self.FILE_ID}}],
            }
        ]

    def test_image_filename_uses_image_block(self) -> None:
        result = convert_messages([ModelRequest([FileIdInput(file_id=self.FILE_ID, filename="photo.jpg")])])

        assert result == [
            {
                "role": "user",
                "content": [{"type": "image", "source": {"type": "file", "file_id": self.FILE_ID}}],
            }
        ]

    def test_pdf_filename_uses_document_block(self) -> None:
        result = convert_messages([ModelRequest([FileIdInput(file_id=self.FILE_ID, filename="report.pdf")])])

        assert result == [
            {
                "role": "user",
                "content": [{"type": "document", "source": {"type": "file", "file_id": self.FILE_ID}}],
            }
        ]


class TestMultipleInputs:
    def test_multiple_inputs_grouped_into_one_message(self) -> None:
        result = convert_messages([
            ModelRequest([
                TextInput("Describe these images."),
                ImageInput(url="https://example.com/a.png"),
                ImageInput(url="https://example.com/b.jpg"),
            ])
        ])

        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert len(result[0]["content"]) == 3
        assert result[0]["content"][0] == {"type": "text", "text": "Describe these images."}
        assert result[0]["content"][1] == IsPartialDict({"type": "image"})
        assert result[0]["content"][2] == IsPartialDict({"type": "image"})


@pytest.mark.parametrize(
    ("input_factory", "match"),
    [
        pytest.param(
            lambda: AudioInput(url="https://example.com/audio.wav"),
            "UrlInput.*audio.*anthropic",
            id="audio_url",
        ),
        pytest.param(
            lambda: VideoInput(url="https://example.com/video.mp4"),
            "UrlInput.*video.*anthropic",
            id="video_url",
        ),
        pytest.param(
            lambda: AudioInput(data=b"\x00audio", media_type="audio/wav"),
            "BinaryInput.*audio.*anthropic",
            id="audio_binary",
        ),
        pytest.param(
            lambda: VideoInput(data=b"\x00video", media_type="video/mp4"),
            "BinaryInput.*video.*anthropic",
            id="video_binary",
        ),
        pytest.param(
            lambda: BinaryInput(data=b"\x00", media_type="application/octet-stream", kind=BinaryType.BINARY),
            "BinaryInput.*binary.*anthropic",
            id="generic_binary",
        ),
    ],
)
def test_unsupported_input_raises(input_factory: Callable[[], Any], match: str) -> None:
    with pytest.raises(UnsupportedInputError, match=match):
        convert_messages([ModelRequest([input_factory()])])


def _server_tool_use_block(
    *,
    id: str = "stu_1",
    name: str = "web_search",
    input: dict | None = None,
) -> ServerToolUseBlock:
    return ServerToolUseBlock(
        id=id,
        name=name,
        input=input if input is not None else {"query": "bitcoin price"},
        type="server_tool_use",
    )


def _web_search_result_block(
    *,
    tool_use_id: str = "stu_1",
    content: list | None = None,
) -> WebSearchToolResultBlock:
    return WebSearchToolResultBlock(
        tool_use_id=tool_use_id,
        type="web_search_tool_result",
        content=content if content is not None else [],
    )


class TestAnthropicServerToolCallEvent:
    def test_emits_wrapped_sdk_block_as_assistant_content(self) -> None:
        block = _server_tool_use_block()
        result = convert_messages([
            AnthropicServerToolCallEvent(
                id=block.id,
                name="web_search",
                arguments="{}",
                block=block,
            ),
        ])

        assert result == [{"role": "assistant", "content": [block.model_dump(exclude_none=True, mode="json")]}]

    def test_appends_to_existing_assistant_message(self) -> None:
        block = _server_tool_use_block(input={"query": "test"})
        result = convert_messages([
            ModelResponse(message=ModelMessage("Let me search for that."), tool_calls=ToolCallsEvent()),
            AnthropicServerToolCallEvent(id=block.id, name="web_search", arguments="{}", block=block),
        ])

        assert result == [
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Let me search for that."},
                    block.model_dump(exclude_none=True, mode="json"),
                ],
            }
        ]


class TestAnthropicServerToolResultEvent:
    def test_emits_wrapped_sdk_block_as_assistant_content(self) -> None:
        block = _web_search_result_block()
        result = convert_messages([
            AnthropicServerToolResultEvent(
                parent_id=block.tool_use_id,
                name="web_search",
                result=ToolResult(),
                block=block,
            ),
        ])

        assert result == [{"role": "assistant", "content": [block.model_dump(exclude_none=True, mode="json")]}]

    def test_call_and_result_blocks_stack_into_one_assistant_message(self) -> None:
        call_block = _server_tool_use_block(input={"query": "test"})
        result_block = _web_search_result_block()
        result = convert_messages([
            AnthropicServerToolCallEvent(id=call_block.id, name="web_search", arguments="{}", block=call_block),
            AnthropicServerToolResultEvent(
                parent_id=result_block.tool_use_id,
                name="web_search",
                result=ToolResult(),
                block=result_block,
            ),
        ])

        assert result == [
            {
                "role": "assistant",
                "content": [
                    call_block.model_dump(exclude_none=True, mode="json"),
                    result_block.model_dump(exclude_none=True, mode="json"),
                ],
            }
        ]


def test_full_sequence_round_trip() -> None:
    """ModelRequest -> call+result -> ModelResponse -> ModelRequest."""
    call_block = _server_tool_use_block(input={"query": "bitcoin"})
    result_block = _web_search_result_block()
    events = [
        ModelRequest([TextInput("Search for bitcoin price")]),
        AnthropicServerToolCallEvent(id=call_block.id, name="web_search", arguments="{}", block=call_block),
        AnthropicServerToolResultEvent(
            parent_id=result_block.tool_use_id,
            name="web_search",
            result=ToolResult(),
            block=result_block,
        ),
        ModelResponse(message=ModelMessage("Bitcoin is $74,000."), tool_calls=ToolCallsEvent()),
        ModelRequest([TextInput("What was the exact price?")]),
    ]

    result = convert_messages(events)

    assert result == [
        {"role": "user", "content": [{"type": "text", "text": "Search for bitcoin price"}]},
        {
            "role": "assistant",
            "content": [
                call_block.model_dump(exclude_none=True, mode="json"),
                result_block.model_dump(exclude_none=True, mode="json"),
            ],
        },
        {"role": "assistant", "content": [{"type": "text", "text": "Bitcoin is $74,000."}]},
        {"role": "user", "content": [{"type": "text", "text": "What was the exact price?"}]},
    ]


@pytest.mark.parametrize(
    ("call_block", "result_block"),
    [
        pytest.param(
            ServerToolUseBlock(
                id="stu_1",
                name="bash_code_execution",
                input={"command": "echo hello"},
                type="server_tool_use",
            ),
            BashCodeExecutionToolResultBlock(
                tool_use_id="stu_1",
                type="bash_code_execution_tool_result",
                content=BashCodeExecutionToolResultError(
                    error_code="unavailable",
                    type="bash_code_execution_tool_result_error",
                ),
            ),
            id="bash",
        ),
        pytest.param(
            ServerToolUseBlock(
                id="stu_2",
                name="text_editor_code_execution",
                input={"command": "view", "path": "/a.txt"},
                type="server_tool_use",
            ),
            TextEditorCodeExecutionToolResultBlock(
                tool_use_id="stu_2",
                type="text_editor_code_execution_tool_result",
                content=TextEditorCodeExecutionToolResultError(
                    error_code="file_not_found",
                    type="text_editor_code_execution_tool_result_error",
                ),
            ),
            id="text_editor",
        ),
    ],
)
def test_code_execution_subtool_preserves_block_shape(
    call_block: ServerToolUseBlock,
    result_block: Any,
) -> None:
    """Anthropic's code_execution tool reports results via sub-tool-specific block
    types. The typed event preserves the original block shape so replay stays lossless."""
    result = convert_messages([
        AnthropicServerToolCallEvent(id=call_block.id, name="code_execution", arguments="{}", block=call_block),
        AnthropicServerToolResultEvent(
            parent_id=result_block.tool_use_id, name="code_execution", result=ToolResult(), block=result_block
        ),
    ])

    assert result == [
        {
            "role": "assistant",
            "content": [
                call_block.model_dump(exclude_none=True, mode="json"),
                result_block.model_dump(exclude_none=True, mode="json"),
            ],
        }
    ]
