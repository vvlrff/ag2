# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import TypeAlias

from anthropic.types import (
    BashCodeExecutionToolResultBlock,
    CodeExecutionToolResultBlock,
    ServerToolUseBlock,
    TextEditorCodeExecutionToolResultBlock,
    WebFetchToolResultBlock,
    WebSearchToolResultBlock,
)

from autogen.beta.events import BuiltinToolCallEvent, BuiltinToolResultEvent
from autogen.beta.events.base import Field

AnthropicServerToolResultBlockType: TypeAlias = (
    WebSearchToolResultBlock
    | WebFetchToolResultBlock
    | CodeExecutionToolResultBlock
    | BashCodeExecutionToolResultBlock
    | TextEditorCodeExecutionToolResultBlock
)


class AnthropicServerToolCallEvent(BuiltinToolCallEvent):
    block: ServerToolUseBlock = Field(repr=False)


class AnthropicServerToolResultEvent(BuiltinToolResultEvent):
    block: AnthropicServerToolResultBlockType = Field(repr=False)
