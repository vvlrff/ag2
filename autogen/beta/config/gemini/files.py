# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from google import genai

from autogen.beta.files.types import FileContent, FileProvider, UploadedFile

if TYPE_CHECKING:
    from autogen.beta.config.gemini.config import GeminiConfig


class GeminiFilesClient:
    """Files API client for Google Gemini.

    Note: Gemini does not support downloading file content.
    Files are stored for 48 hours and then automatically deleted.
    """

    __slots__ = ("_client",)

    def __init__(self, config: "GeminiConfig") -> None:
        self._client = genai.Client(api_key=config.api_key)

    async def upload(self, data: bytes, filename: str, purpose: str | None = None) -> UploadedFile:
        suffix = Path(filename).suffix
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(data)
            tmp_path = tmp.name

        try:
            result = self._client.files.upload(file=tmp_path)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

        return UploadedFile(
            file_id=result.name,
            filename=filename,
            provider=FileProvider.GEMINI,
            bytes_count=result.size_bytes if hasattr(result, "size_bytes") else len(data),
            purpose=purpose,
            created_at=str(result.create_time) if hasattr(result, "create_time") else None,
        )

    async def read(self, file_id: str) -> FileContent:
        raise NotImplementedError(
            "Gemini Files API does not support downloading file content. "
            "Files can only be used as input to generate_content calls."
        )

    async def list(self) -> list[UploadedFile]:
        result = self._client.files.list()
        return [
            UploadedFile(
                file_id=f.name,
                filename=f.display_name if hasattr(f, "display_name") else None,
                provider=FileProvider.GEMINI,
                bytes_count=f.size_bytes if hasattr(f, "size_bytes") else None,
                purpose=None,
                created_at=str(f.create_time) if hasattr(f, "create_time") else None,
            )
            for f in result
        ]

    async def delete(self, file_id: str) -> None:
        self._client.files.delete(name=file_id)
