# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import TypeAlias

from openai.types.responses import (
    ResponseCodeInterpreterToolCall,
    ResponseFunctionWebSearch,
    ResponseReasoningItem,
)
from openai.types.responses.response_output_item import ImageGenerationCall

from autogen.beta.events import BuiltinToolCallEvent, BuiltinToolResultEvent, ModelReasoning
from autogen.beta.events.base import Field

OpenAIServerToolItem: TypeAlias = ResponseFunctionWebSearch | ResponseCodeInterpreterToolCall | ImageGenerationCall


class OpenAIServerToolCallEvent(BuiltinToolCallEvent):
    item: OpenAIServerToolItem = Field(repr=False)


class OpenAIServerToolResultEvent(BuiltinToolResultEvent):
    """Observability-only companion to :class:`OpenAIServerToolCallEvent`."""


class OpenAIReasoningEvent(ModelReasoning):
    item: ResponseReasoningItem = Field(repr=False)
