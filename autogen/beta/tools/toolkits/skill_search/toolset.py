# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import atexit
import os
import shutil
from collections.abc import Iterable
from pathlib import Path

from autogen.beta.exceptions import SkillDownloadError, SkillInstallError
from autogen.beta.middleware import ToolMiddleware
from autogen.beta.tools.final import Toolkit, tool
from autogen.beta.tools.final.function_tool import FunctionTool
from autogen.beta.tools.local_skills.tool import LocalSkillsTool

from .client import SkillsClient
from .extractor import format_install_result
from .lock import SkillsLock


class SkillSearchToolset(Toolkit):
    """Toolkit for dynamically searching and installing skills from the
    `skills.sh <https://skills.sh>`_ ecosystem.

    Does **not** require Node.js. Uses HTTP + GitHub Tarball API directly.
    A ``GITHUB_TOKEN`` environment variable is read automatically to raise the
    GitHub rate limit from 60 to 5,000 requests per hour.

    Example::

        import asyncio
        from autogen.beta import Agent
        from autogen.beta.config import AnthropicConfig
        from autogen.beta.tools import SkillSearchToolset

        config = AnthropicConfig(model="claude-sonnet-4-5")
        skills = SkillSearchToolset()

        agent = Agent(
            "coder",
            "You are a helpful coding assistant. Use skills to extend your capabilities.",
            config=config,
            tools=[skills],
        )


        async def main():
            reply = await agent.ask("Find and install a skill for React best practices, then tell me the top 3 rules.")
            print(await reply.content())


        asyncio.run(main())

    Individual tools are available as attributes::

        agent = Agent("a", config=config, tools=[skills.search_skills, skills.install_skill])
    """

    search_skills: FunctionTool
    install_skill: FunctionTool
    remove_skill: FunctionTool
    list_skills: FunctionTool
    load_skill: FunctionTool
    run_skill_script: FunctionTool

    def __init__(
        self,
        *extra_paths: str | Path,
        install_dir: str | Path | None = None,
        cleanup: bool = False,
        github_token: str | None = None,
        verify: bool | str = True,
        script_timeout: float = 60,
        script_max_output: int = 100_000,
        script_blocked: list[str] | None = None,
        middleware: Iterable[ToolMiddleware] = (),
    ) -> None:
        _install_dir = Path(install_dir) if install_dir is not None else Path(".agents/skills")
        client = SkillsClient(github_token or os.environ.get("GITHUB_TOKEN"), verify=verify)
        lock = SkillsLock(_install_dir / "skills-lock.json")
        local = LocalSkillsTool(
            _install_dir,
            *extra_paths,
            script_timeout=script_timeout,
            script_max_output=script_max_output,
            script_blocked=script_blocked,
        )

        self.search_skills = _make_search_tool(client)
        self.install_skill = _make_install_tool(client, lock, local, _install_dir)
        self.remove_skill = _make_remove_tool(_install_dir, lock, local)
        self.list_skills = local.list_skills
        self.load_skill = local.load_skill
        self.run_skill_script = local.run_skill_script

        if cleanup:
            atexit.register(shutil.rmtree, str(_install_dir), True)
        atexit.register(_schedule_close, client)

        super().__init__(
            self.search_skills,
            self.install_skill,
            self.remove_skill,
            self.list_skills,
            self.load_skill,
            self.run_skill_script,
            middleware=middleware,
        )


def _make_search_tool(client: SkillsClient) -> FunctionTool:
    @tool
    async def search_skills(query: str, limit: int = 10) -> str:
        """Search for skills on skills.sh.

        Returns a formatted list of matching skills with ready-to-use install commands.

        Args:
            query: Search query (e.g. ``"react performance"``).
            limit: Maximum number of results to return (default: 10).
        """
        try:
            skills = await client.search(query, limit)
        except Exception as e:
            return f"Error searching skills.sh: {e}"

        if not skills:
            return f'No skills found for "{query}".'

        lines: list[str] = [f'Found {len(skills)} skill(s) for "{query}":\n']
        for i, s in enumerate(skills, 1):
            name = s.get("name") or s.get("skillId") or "unknown"
            installs: int = s.get("installs", 0)
            skill_id_val: str = s.get("skillId") or ""
            source: str = s.get("source") or ""
            install_id = f"{source}/{skill_id_val}" if skill_id_val and source else source or skill_id_val
            lines.append(f"{i}. {name} ({installs:,} installs)")
            lines.append(f'   \u2192 install_skill("{install_id}")')
            lines.append("")
        return "\n".join(lines)

    return search_skills


def _make_install_tool(
    client: SkillsClient,
    lock: SkillsLock,
    local: LocalSkillsTool,
    install_dir: Path,
) -> FunctionTool:
    @tool
    async def install_skill(skill_id: str) -> str:
        """Download and install a skill from skills.sh.

        Args:
            skill_id: The skill identifier from search results, e.g.:
                      ``"vercel-labs/agent-skills/react-best-practices"`` (monorepo),
                      ``"mvanhorn/last30days-skill"`` (standalone repo).
        """
        parts = skill_id.split("/")
        if len(parts) >= 3:
            source, sid = f"{parts[0]}/{parts[1]}", "/".join(parts[2:])
        elif len(parts) == 2:
            source, sid = skill_id, ""
        else:
            return f"Invalid skill_id format: {skill_id!r}. Expected 'owner/repo/skill-name' or 'owner/repo'."

        try:
            install_dir.mkdir(parents=True, exist_ok=True)
            meta, computed_hash = await client.download_skill(source, sid, install_dir)
            lock.record(meta.name, source, computed_hash)
            local.loader.invalidate()
            return format_install_result(meta, install_dir)
        except (SkillDownloadError, SkillInstallError) as e:
            return str(e)
        except Exception as e:
            return f"Error installing skill: {e}"

    return install_skill


def _make_remove_tool(install_dir: Path, lock: SkillsLock, local: LocalSkillsTool) -> FunctionTool:
    @tool
    def remove_skill(name: str) -> str:
        """Remove an installed skill by name.

        Args:
            name: Skill name as returned by list_skills().
        """
        skill_path = (install_dir / name).resolve()
        if not skill_path.is_relative_to(install_dir.resolve()) or not skill_path.exists():
            return f"Cannot remove '{name}': not in install_dir {install_dir}"
        shutil.rmtree(skill_path)
        lock.remove(name)
        local.loader.invalidate()
        return f"Removed: {name}"

    return remove_skill


def _schedule_close(client: SkillsClient) -> None:
    """Schedule async client close at process exit (best-effort)."""
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(client.close())
    except RuntimeError:
        # No running event loop (normal at process exit after asyncio.run()).
        # The AsyncClient was created in the now-closed loop — attempting
        # asyncio.run(client.close()) would spin up a *new* loop, but the
        # transport objects are still bound to the old one and will raise
        # "RuntimeError: Event loop is closed" when aclose() tries to
        # schedule callbacks.  Drop the reference instead; the OS reclaims
        # all TCP sockets at process exit.
        client._client = None
