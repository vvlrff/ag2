# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import base64
from collections.abc import Callable
from typing import Any

import pytest
from dirty_equals import IsPartialDict
from openai.types.responses import ResponseFunctionWebSearch, ResponseReasoningItem
from openai.types.responses.response_function_web_search import ActionSearch
from openai.types.responses.response_reasoning_item import Summary

from autogen.beta.config.openai import (
    OpenAIReasoningEvent,
    OpenAIServerToolCallEvent,
    OpenAIServerToolResultEvent,
)
from autogen.beta.config.openai.mappers import convert_messages, events_to_responses_input
from autogen.beta.events import (
    AudioInput,
    BinaryInput,
    BinaryType,
    DocumentInput,
    FileIdInput,
    ImageInput,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextInput,
    ToolCallsEvent,
)
from autogen.beta.events.tool_events import ToolResult
from autogen.beta.exceptions import UnsupportedInputError


class TestTextInput:
    def test_completions(self) -> None:
        result = convert_messages([], [ModelRequest([TextInput("hello")])])

        assert result[1] == {"role": "user", "content": "hello"}

    def test_responses(self) -> None:
        result = events_to_responses_input([ModelRequest([TextInput("hello")])])

        assert result == [{"role": "user", "content": [{"type": "input_text", "text": "hello"}]}]

    def test_completions_text_with_image_url(self) -> None:
        """Text + image in one ModelRequest must produce a single message with content array."""
        image_url = "https://example.com/image.png"
        result = convert_messages(
            [],
            [
                ModelRequest([TextInput("describe this"), ImageInput(url=image_url)]),
            ],
        )

        assert len(result) == 2  # system + one user message
        assert result[1] == {
            "role": "user",
            "content": [
                {"type": "text", "text": "describe this"},
                {"type": "image_url", "image_url": {"url": image_url}},
            ],
        }


class TestImageUrlInput:
    IMAGE_URL = "https://example.com/image.png"

    def test_completions(self) -> None:
        result = convert_messages([], [ModelRequest([ImageInput(url=self.IMAGE_URL)])])

        assert result[1] == {
            "role": "user",
            "content": [{"type": "image_url", "image_url": {"url": self.IMAGE_URL}}],
        }

    def test_responses(self) -> None:
        result = events_to_responses_input([ModelRequest([ImageInput(url=self.IMAGE_URL)])])

        assert result == [
            {
                "role": "user",
                "content": [{"type": "input_image", "image_url": self.IMAGE_URL}],
            }
        ]


class TestFileIdInput:
    FILE_ID = "file-6F2ksmvXxt4VdoqmHRw6kL"

    def test_completions_raises(self) -> None:
        with pytest.raises(UnsupportedInputError, match="FileIdInput.*openai-completions"):
            convert_messages([], [ModelRequest([FileIdInput(file_id=self.FILE_ID)])])

    def test_responses(self) -> None:
        result = events_to_responses_input([ModelRequest([FileIdInput(file_id=self.FILE_ID)])])

        assert result == [
            {
                "role": "user",
                "content": [{"type": "input_file", "file_id": self.FILE_ID}],
            }
        ]

    def test_responses_with_filename(self) -> None:
        result = events_to_responses_input([ModelRequest([FileIdInput(file_id=self.FILE_ID, filename="report.pdf")])])

        assert result == [
            {
                "role": "user",
                "content": [{"type": "input_file", "file_id": self.FILE_ID, "filename": "report.pdf"}],
            }
        ]


@pytest.mark.parametrize(
    ("mapper", "match"),
    [
        pytest.param(
            lambda req: convert_messages([], [req]),
            "UrlInput.*audio.*openai-completions",
            id="completions",
        ),
        pytest.param(
            lambda req: events_to_responses_input([req]),
            "UrlInput.*audio.*openai-responses",
            id="responses",
        ),
    ],
)
def test_audio_url_raises_on_both_apis(mapper: Callable[[ModelRequest], Any], match: str) -> None:
    with pytest.raises(UnsupportedInputError, match=match):
        mapper(ModelRequest([AudioInput(url="https://example.com/audio.wav")]))


@pytest.mark.parametrize(
    ("media_type", "expected_format"),
    [
        pytest.param("audio/wav", "wav", id="wav"),
        pytest.param("audio/mpeg", "mp3", id="mp3"),
    ],
)
def test_audio_binary_completions_format_detection(media_type: str, expected_format: str) -> None:
    sample_bytes = b"\x00\x01\x02audio"
    result = convert_messages([], [ModelRequest([AudioInput(data=sample_bytes, media_type=media_type)])])

    expected_b64 = base64.b64encode(sample_bytes).decode()
    assert result[1] == {
        "role": "user",
        "content": [{"type": "input_audio", "input_audio": {"data": expected_b64, "format": expected_format}}],
    }


class TestBinaryInput:
    SAMPLE_BYTES = b"\x89PNG\r\n\x1a\nfake"

    def test_completions(self) -> None:
        result = convert_messages([], [ModelRequest([ImageInput(data=self.SAMPLE_BYTES, media_type="image/png")])])

        expected_url = f"data:image/png;base64,{base64.b64encode(self.SAMPLE_BYTES).decode()}"
        assert result[1] == {
            "role": "user",
            "content": [{"type": "image_url", "image_url": {"url": expected_url}}],
        }

    def test_completions_with_vendor_metadata(self) -> None:
        result = convert_messages(
            [],
            [
                ModelRequest([
                    BinaryInput(
                        data=self.SAMPLE_BYTES,
                        media_type="image/png",
                        vendor_metadata={"detail": "low"},
                        kind=BinaryType.IMAGE,
                    )
                ])
            ],
        )

        assert result[1] == IsPartialDict({
            "role": "user",
            "content": [IsPartialDict({"type": "image_url", "detail": "low"})],
        })

    def test_responses(self) -> None:
        result = events_to_responses_input([
            ModelRequest([BinaryInput(data=self.SAMPLE_BYTES, media_type="image/png")])
        ])

        expected_data = f"data:image/png;base64,{base64.b64encode(self.SAMPLE_BYTES).decode()}"
        assert result == [
            {
                "role": "user",
                "content": [{"type": "input_file", "file_data": expected_data}],
            }
        ]

    def test_responses_with_vendor_metadata(self) -> None:
        result = events_to_responses_input(
            [
                ModelRequest([
                    BinaryInput(
                        data=self.SAMPLE_BYTES, media_type="image/png", vendor_metadata={"filename": "test.png"}
                    )
                ])
            ],
        )

        assert result == [
            IsPartialDict({
                "role": "user",
                "content": [IsPartialDict({"type": "input_file", "filename": "test.png"})],
            })
        ]


class TestDocumentUrlInput:
    DOC_URL = "https://example.com/document.pdf"

    def test_completions_raises(self) -> None:
        with pytest.raises(UnsupportedInputError, match="UrlInput.*document.*openai-completions"):
            convert_messages([], [ModelRequest([DocumentInput(url=self.DOC_URL)])])

    def test_responses(self) -> None:
        result = events_to_responses_input([ModelRequest([DocumentInput(url=self.DOC_URL)])])

        assert result == [
            {
                "role": "user",
                "content": [{"type": "input_file", "file_url": self.DOC_URL}],
            }
        ]


def _web_search_sdk_item(*, id: str = "ws_1", query: str = "bitcoin") -> ResponseFunctionWebSearch:
    return ResponseFunctionWebSearch(
        id=id,
        action=ActionSearch(type="search", query=query),
        status="completed",
        type="web_search_call",
    )


def _reasoning_sdk_item(*, id: str = "rs_1", text: str = "thinking") -> ResponseReasoningItem:
    return ResponseReasoningItem(
        id=id,
        type="reasoning",
        summary=[Summary(type="summary_text", text=text)],
    )


class TestOpenAIServerToolCallEvent:
    def test_emits_wrapped_sdk_item_as_input(self) -> None:
        item = _web_search_sdk_item()
        result = events_to_responses_input([
            OpenAIServerToolCallEvent(id=item.id, name="web_search", arguments="{}", item=item),
        ])

        assert result == [item.model_dump(exclude_none=True, mode="json")]


class TestOpenAIServerToolResultEvent:
    def test_is_observability_only_and_not_replayed(self) -> None:
        """Result event carries no payload — paired call event already covers the item."""
        result = events_to_responses_input([
            OpenAIServerToolResultEvent(parent_id="ws_1", name="web_search", result=ToolResult()),
        ])

        assert result == []


class TestOpenAIReasoningEvent:
    def test_emits_wrapped_sdk_item_as_input(self) -> None:
        item = _reasoning_sdk_item()
        result = events_to_responses_input([OpenAIReasoningEvent("thinking", item=item)])

        assert result == [item.model_dump(exclude_none=True, mode="json")]

    def test_two_empty_reasoning_items_with_distinct_ids_are_not_equal(self) -> None:
        a = OpenAIReasoningEvent("", item=_reasoning_sdk_item(id="rs_a", text=""))
        b = OpenAIReasoningEvent("", item=_reasoning_sdk_item(id="rs_b", text=""))

        assert a != b


def test_full_sequence_round_trip() -> None:
    reasoning = _reasoning_sdk_item(text="I'll search")
    web = _web_search_sdk_item()
    events = [
        ModelRequest([TextInput("Search for bitcoin price")]),
        OpenAIReasoningEvent("I'll search", item=reasoning),
        OpenAIServerToolCallEvent(id=web.id, name="web_search", arguments="{}", item=web),
        OpenAIServerToolResultEvent(parent_id=web.id, name="web_search", result=ToolResult()),
        ModelResponse(message=ModelMessage("Bitcoin is $74,000."), tool_calls=ToolCallsEvent()),
        ModelRequest([TextInput("What was the exact price?")]),
    ]

    result = events_to_responses_input(events)

    assert result == [
        {"role": "user", "content": [{"type": "input_text", "text": "Search for bitcoin price"}]},
        reasoning.model_dump(exclude_none=True, mode="json"),
        web.model_dump(exclude_none=True, mode="json"),
        {"role": "assistant", "content": [{"type": "output_text", "text": "Bitcoin is $74,000."}]},
        {"role": "user", "content": [{"type": "input_text", "text": "What was the exact price?"}]},
    ]
