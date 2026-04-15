# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from autogen.beta.config.anthropic.files import AnthropicFilesClient
from autogen.beta.files.types import FileContent, UploadedFile


@pytest.mark.asyncio
class TestAnthropicFilesClient:
    @patch("autogen.beta.config.anthropic.files.AsyncAnthropic")
    async def test_upload(self, mock_anthropic_cls: MagicMock, anthropic_config: MagicMock) -> None:
        mock_client = AsyncMock()
        mock_anthropic_cls.return_value = mock_client
        mock_client.beta.files.upload.return_value = SimpleNamespace(
            id="file-011CNha8",
            filename="document.pdf",
            size_bytes=1024000,
            created_at="2025-01-01T00:00:00Z",
        )

        client = AnthropicFilesClient(anthropic_config)
        result = await client.upload(b"pdf-data", "document.pdf")

        assert isinstance(result, UploadedFile)
        assert result.file_id == "file-011CNha8"
        assert result.filename == "document.pdf"
        assert result.provider == "anthropic"
        assert result.bytes_count == 1024000

    @patch("autogen.beta.config.anthropic.files.AsyncAnthropic")
    async def test_read(self, mock_anthropic_cls: MagicMock, anthropic_config: MagicMock) -> None:
        mock_client = AsyncMock()
        mock_anthropic_cls.return_value = mock_client
        mock_client.beta.files.download.return_value = SimpleNamespace(content=b"file-data")
        mock_client.beta.files.retrieve_metadata.return_value = SimpleNamespace(
            filename="output.csv",
            mime_type="text/csv",
        )

        client = AnthropicFilesClient(anthropic_config)
        result = await client.read("file-011CNha8")

        assert isinstance(result, FileContent)
        assert result.data == b"file-data"
        assert result.name == "output.csv"
        assert result.media_type == "text/csv"

    @patch("autogen.beta.config.anthropic.files.AsyncAnthropic")
    async def test_list(self, mock_anthropic_cls: MagicMock, anthropic_config: MagicMock) -> None:
        mock_client = AsyncMock()
        mock_anthropic_cls.return_value = mock_client
        mock_client.beta.files.list.return_value = SimpleNamespace(
            data=[
                SimpleNamespace(
                    id="file-1",
                    filename="a.pdf",
                    size_bytes=100,
                    created_at="2025-01-01T00:00:00Z",
                ),
            ]
        )

        client = AnthropicFilesClient(anthropic_config)
        result = await client.list()

        assert len(result) == 1
        assert result[0].file_id == "file-1"
        assert result[0].provider == "anthropic"

    @patch("autogen.beta.config.anthropic.files.AsyncAnthropic")
    async def test_delete(self, mock_anthropic_cls: MagicMock, anthropic_config: MagicMock) -> None:
        mock_client = AsyncMock()
        mock_anthropic_cls.return_value = mock_client

        client = AnthropicFilesClient(anthropic_config)
        await client.delete("file-011CNha8")

        mock_client.beta.files.delete.assert_awaited_once_with("file-011CNha8")
