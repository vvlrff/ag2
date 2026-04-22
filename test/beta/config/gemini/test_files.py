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

        result = await GeminiFilesClient(gemini_config).upload(b"audio-data", "recording.mp3")

        assert result == UploadedFile(
            file_id="files/abc123",
            filename="recording.mp3",
            provider="gemini",
            bytes_count=512,
            purpose=None,
            created_at="2025-01-01T00:00:00Z",
        )

    @patch("autogen.beta.config.gemini.files.genai")
    async def test_read_raises(self, mock_genai: MagicMock, gemini_config: MagicMock) -> None:
        mock_genai.Client.return_value = MagicMock()

        with pytest.raises(NotImplementedError, match="does not support downloading"):
            await GeminiFilesClient(gemini_config).read("files/abc123")

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

        result = await GeminiFilesClient(gemini_config).list()

        assert result == [
            UploadedFile(
                file_id="files/1",
                filename="doc.pdf",
                provider="gemini",
                bytes_count=100,
                purpose=None,
                created_at="2025-01-01",
            ),
        ]

    @patch("autogen.beta.config.gemini.files.genai")
    async def test_delete(self, mock_genai: MagicMock, gemini_config: MagicMock) -> None:
        mock_client = MagicMock()
        mock_genai.Client.return_value = mock_client

        await GeminiFilesClient(gemini_config).delete("files/abc123")

        mock_client.files.delete.assert_called_once_with(name="files/abc123")
