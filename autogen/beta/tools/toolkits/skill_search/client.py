# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import hashlib
import tempfile
from pathlib import Path

import httpx

from autogen.beta.exceptions import SkillDownloadError
from autogen.beta.tools.local_skills.loader import SkillMetadata

from .extractor import extract_skill


class SkillsClient:
    """HTTP client for skills.sh search and GitHub tarball downloads.

    Lazily creates a shared :class:`httpx.AsyncClient` for connection reuse
    across multiple search and download calls within a session.

    A ``github_token`` raises the GitHub API rate limit from 60 to 5,000
    requests per hour.
    """

    SKILLS_SH_API = "https://skills.sh/api"
    GITHUB_API = "https://api.github.com"

    def __init__(self, github_token: str | None = None, verify: bool | str = True) -> None:
        self._github_token = github_token
        self._verify = verify
        self._client: httpx.AsyncClient | None = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            headers: dict[str, str] = {
                "Accept": "application/vnd.github+json",
                "User-Agent": "ag2-skill-search/1.0",
            }
            if self._github_token:
                headers["Authorization"] = f"Bearer {self._github_token}"
            self._client = httpx.AsyncClient(
                follow_redirects=True,
                timeout=30,
                headers=headers,
                verify=self._verify,
            )
        return self._client

    async def close(self) -> None:
        """Close the underlying HTTP client (best-effort)."""
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def search(self, query: str, limit: int = 10) -> list[dict]:
        """Search skills.sh and return a list of skill records."""
        url = f"{self.SKILLS_SH_API}/search"
        client = self._get_client()
        response = await client.get(url, params={"q": query, "limit": limit})
        response.raise_for_status()
        return response.json().get("skills", [])

    async def download_skill(self, source: str, skill_id: str, dest: Path) -> tuple[SkillMetadata, str]:
        """Download a skill via the GitHub Tarball API and extract it to *dest*.

        Args:
            source:   ``"owner/repo"``
            skill_id: directory name inside the repo (e.g. ``"react-best-practices"``),
                      or empty string for a standalone repo.
            dest:     parent directory where the extracted skill folder is placed.

        Returns:
            A ``(metadata, sha256_hex)`` tuple.

        Raises:
            SkillDownloadError: On HTTP 403 (rate limit) or 404 (not found).
        """
        client = self._get_client()

        with tempfile.TemporaryDirectory() as tmp_dir:
            tar_path = Path(tmp_dir) / "skill.tar.gz"
            hash_obj = hashlib.sha256()

            async with client.stream(
                "GET",
                f"{self.GITHUB_API}/repos/{source}/tarball",
                timeout=120,
            ) as resp:
                if resp.status_code == 403:
                    raise SkillDownloadError(
                        "GitHub rate limit exceeded. Set GITHUB_TOKEN or pass github_token= to SkillSearchToolset."
                    )
                if resp.status_code == 404:
                    raise SkillDownloadError(
                        f"Skill not found: {source}. Check that the repository exists and is public."
                    )
                resp.raise_for_status()
                with tar_path.open("wb") as fh:
                    async for chunk in resp.aiter_bytes():
                        hash_obj.update(chunk)
                        fh.write(chunk)

            meta = extract_skill(tar_path, skill_id, dest)
            return meta, hash_obj.hexdigest()
