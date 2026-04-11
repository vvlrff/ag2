# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import hashlib
import os
import tempfile
from pathlib import Path

import httpx

from autogen.beta.exceptions import SkillDownloadError
from autogen.beta.tools.local_skills.loader import SkillMetadata
from autogen.beta.tools.local_skills.runtime import SkillRuntime

from .config import SkillsClientConfig
from .extractor import extract_skill


class SkillsClient:
    """HTTP client for skills.sh search and GitHub tarball downloads.

    Args:
        config: HTTP client configuration. Falls back to defaults when ``None``.
            Use :class:`SkillsClientConfig` to set a GitHub token, proxy, custom
            certificates, or other connection options.
    """

    SKILLS_SH_API = "https://skills.sh/api"
    GITHUB_API = "https://api.github.com"

    def __init__(self, config: SkillsClientConfig | None = None) -> None:
        cfg = config or SkillsClientConfig()

        headers: dict[str, str] = {
            "Accept": "application/vnd.github+json",
            "User-Agent": "ag2-skill-search/1.0",
        }
        token = cfg.github_token or os.environ.get("GITHUB_TOKEN")
        if token:
            headers["Authorization"] = f"Bearer {token}"
        if cfg.headers:
            headers.update(cfg.headers)

        self._client = httpx.AsyncClient(
            follow_redirects=True,
            timeout=cfg.timeout,
            headers=headers,
            proxy=cfg.proxy,
            verify=cfg.verify_ssl,
            cert=cfg.cert,
        )

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()

    async def search(self, query: str, limit: int = 10) -> list[dict]:
        """Search skills.sh and return a list of skill records."""
        url = f"{self.SKILLS_SH_API}/search"
        response = await self._client.get(url, params={"q": query, "limit": limit})
        response.raise_for_status()
        return response.json().get("skills", [])

    async def download_skill(self, source: str, skill_id: str, runtime: SkillRuntime) -> tuple[SkillMetadata, str]:
        """Download a skill via the GitHub Tarball API and install it via *runtime*.

        Args:
            source:   ``"owner/repo"``
            skill_id: Directory name inside the repo (e.g. ``"react-best-practices"``),
                      or empty string for a standalone repo.
            runtime:  Runtime that receives the extracted skill via ``runtime.install()``.

        Returns:
            A ``(metadata, sha256_hex)`` tuple.

        Raises:
            SkillDownloadError: On HTTP 403 (rate limit) or 404 (not found).
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            tar_path = Path(tmp_dir) / "skill.tar.gz"
            hash_obj = hashlib.sha256()

            async with self._client.stream(
                "GET",
                f"{self.GITHUB_API}/repos/{source}/tarball",
                timeout=120,
            ) as resp:
                if resp.status_code == 403:
                    raise SkillDownloadError(
                        "GitHub rate limit exceeded. Set GITHUB_TOKEN or pass github_token= to SkillsClientConfig."
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

            staging = Path(tmp_dir) / "staged"
            staging.mkdir()
            meta = extract_skill(tar_path, skill_id, staging)
            runtime.install(staging / meta.name, meta.name)
            return meta, hash_obj.hexdigest()
