# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import shlex
from pathlib import Path
from typing import Annotated

from pydantic import Field

from autogen.beta.tools.final import Toolkit, tool
from autogen.beta.tools.final.function_tool import FunctionTool
from autogen.beta.tools.shell.environment.local import LocalShellEnvironment

from .loader import SkillLoader


class LocalSkillsTool(Toolkit):
    """Client-side skills toolkit using the agentskills.io convention.

    Implements a three-step progressive-disclosure pattern:

    1. **list_skills()** — returns a lightweight catalog (name + description).
    2. **load_skill(name)** — returns the full ``SKILL.md`` instructions on demand.
    3. **run_skill_script(name, script, args)** — executes a script from the
       skill's ``scripts/`` directory via :class:`~autogen.beta.tools.shell.LocalShellEnvironment`.

    Works with *any* provider (no provider-specific API required).

    Example::

        # Scan default paths (.agents/skills)
        LocalSkillsTool()

        # Scan only the given paths (defaults are NOT included)
        LocalSkillsTool("./skills")
        LocalSkillsTool("/path/to/skills-a", "/path/to/skills-b")
    """

    list_skills: FunctionTool
    load_skill: FunctionTool
    run_skill_script: FunctionTool

    def __init__(
        self,
        *paths: str | Path,
        script_timeout: float = 60,
        script_max_output: int = 100_000,
        script_blocked: list[str] | None = None,
    ) -> None:
        loader = SkillLoader(*paths)
        self.loader = loader
        self.list_skills = _make_list_tool(loader)
        self.load_skill = _make_load_tool(loader)
        self.run_skill_script = _make_run_tool(
            loader,
            timeout=script_timeout,
            max_output=script_max_output,
            blocked=script_blocked,
        )

        tools = [self.list_skills, self.load_skill, self.run_skill_script]
        super().__init__(*tools)


def _make_list_tool(loader: SkillLoader) -> FunctionTool:
    @tool(description="List available local skills with name and short description.")
    def list_skills() -> list[dict[str, str]]:
        return [{"name": m.name, "description": m.description} for m in loader.discover()]

    return list_skills


def _make_load_tool(loader: SkillLoader) -> FunctionTool:
    @tool(description="Load the full SKILL.md content for a specific skill.")
    def load_skill(
        name: Annotated[str, Field(description="Skill name returned by list_skills.")],
    ) -> str:
        return loader.load(name)

    return load_skill


def _make_run_tool(
    loader: SkillLoader,
    *,
    timeout: float = 60,
    max_output: int = 100_000,
    blocked: list[str] | None = None,
) -> FunctionTool:
    @tool(description=("Run a script from a skill's scripts directory. Only .py and .sh scripts are supported."))
    def run_skill_script(
        name: Annotated[str, Field(description="Skill name returned by list_skills.")],
        script: Annotated[
            str,
            Field(description="Script filename inside scripts/, for example scaffold.py or build.sh."),
        ],
        args: list[str] | None = Field(
            default=None,
            description="Optional script arguments passed as positional parameters.",
        ),
    ) -> str:
        skill_dir = loader.get_path(name)
        scripts_dir = skill_dir / "scripts"
        script_path = Path(script)
        if script_path.name != script:
            raise ValueError("script must be a filename inside the skill scripts directory")

        resolved_script = (scripts_dir / script_path.name).resolve()
        if not resolved_script.is_file() or not resolved_script.is_relative_to(scripts_dir.resolve()):
            raise FileNotFoundError(f"script {script!r} not found in {scripts_dir}")

        command = [resolved_script.name]
        if args:
            command.extend(args)

        env = LocalShellEnvironment(
            path=scripts_dir,
            cleanup=False,
            timeout=timeout,
            max_output=max_output,
            blocked=blocked,
        )
        return env.run(shlex.join(command))

    return run_skill_script
