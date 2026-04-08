# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from contextlib import ExitStack
from unittest.mock import MagicMock

import pytest

from autogen.beta.context import Context
from autogen.beta.tools.builtin.skills import Skill, SkillsTool, SkillsToolSchema


def _ctx() -> Context:
    return Context(stream=MagicMock())


@pytest.mark.asyncio
async def test_skills_tool_strings() -> None:
    t = SkillsTool("pptx", "xlsx")

    [schema] = await t.schemas(_ctx())

    assert isinstance(schema, SkillsToolSchema)
    assert schema.type == "skills"
    assert len(schema.skills) == 2
    assert schema.skills[0] == Skill(id="pptx")
    assert schema.skills[1] == Skill(id="xlsx")


@pytest.mark.asyncio
async def test_skills_tool_skill_objects() -> None:
    t = SkillsTool(Skill("openai-spreadsheets"), Skill("skill_abc123", version=2))

    [schema] = await t.schemas(_ctx())

    assert schema.skills[0] == Skill(id="openai-spreadsheets", version=None)
    assert schema.skills[1] == Skill(id="skill_abc123", version=2)


@pytest.mark.asyncio
async def test_skills_tool_mixed() -> None:
    t = SkillsTool("pptx", Skill("xlsx", version="20251013"))

    [schema] = await t.schemas(_ctx())

    assert schema.skills[0] == Skill(id="pptx", version=None)
    assert schema.skills[1] == Skill(id="xlsx", version="20251013")


@pytest.mark.asyncio
async def test_skills_tool_no_args() -> None:
    t = SkillsTool()

    [schema] = await t.schemas(_ctx())

    assert schema.skills == []


@pytest.mark.asyncio
async def test_skills_tool_register_is_noop() -> None:
    t = SkillsTool("pptx")
    with ExitStack() as stack:
        t.register(stack, _ctx())  # must not raise
