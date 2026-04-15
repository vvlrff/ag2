# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from autogen.beta.config.gemini.files import GeminiFilesClient
from autogen.beta.files.types import UploadedFile


@pytest.mark.asyncio
class TestGeminiFilesClient:
    @patch("autogen.beta.config.gemini.files.genai")
    async def test_upload(self, mock_genai: MagicMock, gemini_config: MagicMock) -> None:
        mock_client = MagicMock()
        mock_genai.Client.return_value = mock_client
        mock_client.files.upload.return_value = SimpleNamespace(
            name="files/abc123",
            size_bytes=512,
            create_time="2025-01-01T00:00:00Z",
        )

        client = GeminiFilesClient(gemini_config)
        result = await client.upload(b"audio-data", "recording.mp3")

        assert isinstance(result, UploadedFile)
        assert result.file_id == "files/abc123"
        assert result.filename == "recording.mp3"
        assert result.provider == "gemini"

    @patch("autogen.beta.config.gemini.files.genai")
    async def test_read_raises(self, mock_genai: MagicMock, gemini_config: MagicMock) -> None:
        mock_genai.Client.return_value = MagicMock()

        client = GeminiFilesClient(gemini_config)

        with pytest.raises(NotImplementedError, match="does not support downloading"):
            await client.read("files/abc123")

    @patch("autogen.beta.config.gemini.files.genai")
    async def test_list(self, mock_genai: MagicMock, gemini_config: MagicMock) -> None:
        mock_client = MagicMock()
        mock_genai.Client.return_value = mock_client
        mock_client.files.list.return_value = [
            SimpleNamespace(
                name="files/1",
                display_name="doc.pdf",
                size_bytes=100,
                create_time="2025-01-01",
            ),
        ]

        client = GeminiFilesClient(gemini_config)
        result = await client.list()

        assert len(result) == 1
        assert result[0].file_id == "files/1"
        assert result[0].filename == "doc.pdf"

    @patch("autogen.beta.config.gemini.files.genai")
    async def test_delete(self, mock_genai: MagicMock, gemini_config: MagicMock) -> None:
        mock_client = MagicMock()
        mock_genai.Client.return_value = mock_client

        client = GeminiFilesClient(gemini_config)
        await client.delete("files/abc123")

        mock_client.files.delete.assert_called_once_with(name="files/abc123")
