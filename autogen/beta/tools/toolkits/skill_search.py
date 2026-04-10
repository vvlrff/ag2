# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import os
import shutil
import tarfile
import tempfile
from collections.abc import Iterable
from pathlib import Path

import httpx

from autogen.beta.middleware import ToolMiddleware
from autogen.beta.tools.final import Toolkit, tool
from autogen.beta.tools.final.function_tool import FunctionTool
from autogen.beta.tools.local_skills.tool import LocalSkillsTool

_EXCLUDE_NAMES = frozenset({".git", ".env", "__pycache__", ".DS_Store", "node_modules"})
_MAX_FILE_BYTES = 25 * 1024 * 1024  # 25 MB


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
        middleware: Iterable[ToolMiddleware] = (),
    ) -> None:
        _install_dir = Path(install_dir) if install_dir is not None else Path(".agents/skills")
        client = _SkillsClient(github_token or os.environ.get("GITHUB_TOKEN"))

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
                _install_dir.mkdir(parents=True, exist_ok=True)
                name = await client.download_skill(source, sid, _install_dir)
                return f"Installed: {name} \u2192 {_install_dir / name}/"
            except RuntimeError as e:
                return str(e)
            except Exception as e:
                return f"Error installing skill: {e}"

        @tool
        def remove_skill(name: str) -> str:
            """Remove an installed skill by name.

            Args:
                name: Skill name as returned by list_skills().
            """
            skill_path = (_install_dir / name).resolve()
            if not skill_path.is_relative_to(_install_dir.resolve()) or not skill_path.exists():
                return f"Cannot remove '{name}': not in install_dir {_install_dir}"
            shutil.rmtree(skill_path)
            return f"Removed: {name}"

        local = LocalSkillsTool(_install_dir, *extra_paths)

        self.search_skills = search_skills
        self.install_skill = install_skill
        self.remove_skill = remove_skill
        self.list_skills = local.list_skills
        self.load_skill = local.load_skill
        self.run_skill_script = local.run_skill_script

        if cleanup:
            import atexit

            atexit.register(shutil.rmtree, str(_install_dir), True)

        super().__init__(
            self.search_skills,
            self.install_skill,
            self.remove_skill,
            self.list_skills,
            self.load_skill,
            self.run_skill_script,
            middleware=middleware,
        )


# ---------------------------------------------------------------------------
# Internal HTTP client
# ---------------------------------------------------------------------------


class _SkillsClient:
    SKILLS_SH_API = "https://skills.sh/api"
    GITHUB_API = "https://api.github.com"

    def __init__(self, github_token: str | None = None) -> None:
        self._github_token = github_token

    async def search(self, query: str, limit: int = 10) -> list[dict]:
        url = f"{self.SKILLS_SH_API}/search"
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(url, params={"q": query, "limit": limit})
            response.raise_for_status()
            return response.json().get("skills", [])

    async def download_skill(self, source: str, skill_id: str, dest: Path) -> str:
        """Download a skill via GitHub Tarball API and extract it to *dest*.

        Args:
            source:   ``"owner/repo"``
            skill_id: directory name inside the repo (e.g. ``"react-best-practices"``),
                      or empty string for a standalone repo.
            dest:     parent directory where the extracted skill folder is placed.

        Returns:
            The skill name read from ``SKILL.md`` frontmatter.
        """
        headers: dict[str, str] = {
            "Accept": "application/vnd.github+json",
            "User-Agent": "ag2-skill-search/1.0",
        }
        if self._github_token:
            headers["Authorization"] = f"Bearer {self._github_token}"

        with tempfile.TemporaryDirectory() as tmp_dir:
            tar_path = Path(tmp_dir) / "skill.tar.gz"

            async with httpx.AsyncClient(follow_redirects=True, timeout=120) as client:
                async with client.stream("GET", f"{self.GITHUB_API}/repos/{source}/tarball", headers=headers) as resp:
                    if resp.status_code == 403:
                        raise RuntimeError(
                            "GitHub rate limit exceeded. Set GITHUB_TOKEN or pass github_token= to SkillSearchToolset."
                        )
                    if resp.status_code == 404:
                        raise RuntimeError(
                            f"Skill not found: {source}. Check that the repository exists and is public."
                        )
                    resp.raise_for_status()
                    with tar_path.open("wb") as fh:
                        async for chunk in resp.aiter_bytes():
                            fh.write(chunk)

            return _extract_skill(tar_path, skill_id, dest)


def _extract_skill(tar_path: Path, skill_id: str, dest: Path) -> str:
    """Extract a skill from *tar_path* into ``dest/<skill_name>/``.

    Supports both monorepo layout (``skills/{skill_id}/``) and standalone repos.
    Returns the skill name from ``SKILL.md`` frontmatter.
    """
    with tempfile.TemporaryDirectory() as extract_tmp:
        skill_content_dir = Path(extract_tmp) / "skill"
        skill_content_dir.mkdir()
        skill_name: str | None = None

        with tarfile.open(tar_path, "r:gz") as tar:
            members = tar.getmembers()
            root_dir = next((m.name.split("/")[0] for m in members if m.name.split("/")[0]), "")

            if skill_id and any(m.name == f"{root_dir}/skills/{skill_id}/SKILL.md" for m in members):
                target_prefix = f"{root_dir}/skills/{skill_id}/"
            else:
                target_prefix = f"{root_dir}/"

            for member in members:
                if not member.name.startswith(target_prefix):
                    continue
                rel_path = member.name[len(target_prefix) :]
                if not rel_path:
                    continue

                parts = Path(rel_path).parts
                if any(p in _EXCLUDE_NAMES for p in parts) or any(p == ".." for p in parts):
                    continue
                if member.issym() or member.islnk():
                    continue
                if member.isfile() and member.size > _MAX_FILE_BYTES:
                    raise RuntimeError(f"File too large (>25MB): {rel_path}")

                target_path = skill_content_dir / rel_path
                if member.isdir():
                    target_path.mkdir(parents=True, exist_ok=True)
                elif member.isfile():
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    file_obj = tar.extractfile(member)
                    if file_obj is not None:
                        target_path.write_bytes(file_obj.read())
                    if rel_path == "SKILL.md" and skill_name is None:
                        skill_name = _parse_frontmatter(target_path.read_text(encoding="utf-8")).get("name")

        if skill_name is None:
            raise RuntimeError(f"No SKILL.md found in archive (skill_id={skill_id!r})")

        final_dest = dest / skill_name
        if final_dest.exists():
            shutil.rmtree(final_dest)
        shutil.copytree(skill_content_dir, final_dest)
        return skill_name


def _parse_frontmatter(text: str) -> dict[str, str]:
    """Parse simple YAML frontmatter (``--- ... ---``) from SKILL.md."""
    if not text.startswith("---"):
        return {}
    end = text.find("\n---", 3)
    if end == -1:
        return {}
    result: dict[str, str] = {}
    for line in text[3:end].strip().splitlines():
        if ":" in line:
            key, _, value = line.partition(":")
            result[key.strip()] = value.strip()
    return result
