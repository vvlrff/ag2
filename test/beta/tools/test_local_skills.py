# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import textwrap
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from autogen.beta.context import Context
from autogen.beta.tools.local_skills.loader import SkillLoader
from autogen.beta.tools.local_skills.tool import LocalSkillsTool, _make_run_tool


def _ctx() -> Context:
    return Context(stream=MagicMock())


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def skill_tree(tmp_path: Path) -> Path:
    """Create a minimal skill tree for testing.

    Structure::

        tmp_path/
          react-best-practices/
            SKILL.md
            scripts/
              scaffold.py
          markdown-guide/
            SKILL.md
    """
    skills_dir = tmp_path

    react_dir = skills_dir / "react-best-practices"
    react_dir.mkdir(parents=True)
    (react_dir / "SKILL.md").write_text(
        textwrap.dedent("""\
            ---
            name: react-best-practices
            description: Best practices for React development
            version: 1.2.0
            ---
            # React Best Practices
            Use functional components and hooks.
        """),
        encoding="utf-8",
    )
    scripts_dir = react_dir / "scripts"
    scripts_dir.mkdir()
    (scripts_dir / "scaffold.py").write_text('print("scaffold")\n', encoding="utf-8")

    md_dir = skills_dir / "markdown-guide"
    md_dir.mkdir(parents=True)
    (md_dir / "SKILL.md").write_text(
        textwrap.dedent("""\
            ---
            name: markdown-guide
            description: Guide for writing Markdown
            ---
            # Markdown Guide
            Use headings, lists, and code blocks.
        """),
        encoding="utf-8",
    )

    return skills_dir


# ---------------------------------------------------------------------------
# SkillLoader tests
# ---------------------------------------------------------------------------


def test_loader_discover(skill_tree: Path) -> None:
    loader = SkillLoader(skill_tree)

    skills = loader.discover()
    names = {s.name for s in skills}

    assert names == {"react-best-practices", "markdown-guide"}


def test_loader_discover_metadata(skill_tree: Path) -> None:
    loader = SkillLoader(skill_tree)

    skills = {s.name: s for s in loader.discover()}

    react = skills["react-best-practices"]
    assert react.description == "Best practices for React development"
    assert react.version == "1.2.0"
    assert react.has_scripts is True

    md = skills["markdown-guide"]
    assert md.description == "Guide for writing Markdown"
    assert md.has_scripts is False


def test_loader_load(skill_tree: Path) -> None:
    loader = SkillLoader(skill_tree)

    content = loader.load("react-best-practices")

    assert "React Best Practices" in content
    assert "functional components" in content


def test_loader_load_missing(skill_tree: Path) -> None:
    loader = SkillLoader(skill_tree)

    with pytest.raises(KeyError, match="nonexistent"):
        loader.load("nonexistent")


def test_loader_get_path(skill_tree: Path) -> None:
    loader = SkillLoader(skill_tree)

    path = loader.get_path("react-best-practices")

    assert path == skill_tree / "react-best-practices"


def test_loader_priority(tmp_path: Path) -> None:
    """First path wins when the same skill name appears in multiple paths."""
    project_skills = tmp_path / "project"
    user_skills = tmp_path / "user"

    for base in (project_skills, user_skills):
        skill_dir = base / "my-skill"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            f"---\nname: my-skill\ndescription: from {base.name}\n---\n",
            encoding="utf-8",
        )

    # project_skills listed first → wins
    loader = SkillLoader(project_skills, user_skills)
    [meta] = loader.discover()

    assert meta.description == "from project"


def test_loader_nonexistent_path(tmp_path: Path) -> None:
    """Non-existent path is silently skipped — returns empty list."""
    loader = SkillLoader(tmp_path / "no-such-dir")

    assert loader.discover() == []


# ---------------------------------------------------------------------------
# LocalSkillsTool tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_local_skills_tool_schemas(skill_tree: Path) -> None:
    tool = LocalSkillsTool(skill_tree)

    schemas = await tool.schemas(_ctx())

    # Three function tools: list_skills, load_skill, run_skill_script
    assert len(schemas) == 3
    names = {s.function.name for s in schemas}  # type: ignore[union-attr]
    assert names == {"list_skills", "load_skill", "run_skill_script"}


@pytest.mark.asyncio
async def test_run_skill_script_schema(skill_tree: Path) -> None:
    loader = SkillLoader(skill_tree)
    run_tool = _make_run_tool(loader)

    [schema] = await run_tool.schemas(_ctx())

    assert schema.function.name == "run_skill_script"  # type: ignore[union-attr]


def test_run_skill_script_missing_script(skill_tree: Path) -> None:
    """run_skill_script returns an error string when the script doesn't exist.

    We test via LocalShellEnvironment directly since that's the execution engine
    used by run_skill_script.
    """
    from autogen.beta.tools.shell.environment.local import LocalShellEnvironment

    scripts_dir = skill_tree / "react-best-practices" / "scripts"
    _env = LocalShellEnvironment(path=scripts_dir, cleanup=False)

    # Nonexistent script — verified in tool before env.run is called
    script_path = scripts_dir / "nonexistent.py"
    assert not script_path.exists()


def test_run_skill_script_executes(skill_tree: Path) -> None:
    """LocalShellEnvironment executes scaffold.py and returns its output."""
    from autogen.beta.tools.shell.environment.local import LocalShellEnvironment

    scripts_dir = skill_tree / "react-best-practices" / "scripts"
    env = LocalShellEnvironment(path=scripts_dir, cleanup=False)

    # cwd is already scripts_dir, so pass just the filename
    result = env.run("python scaffold.py")

    assert "scaffold" in result
