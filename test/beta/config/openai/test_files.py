# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from autogen.beta.config.openai.files import OpenAIFilesClient
from autogen.beta.files.types import FileContent, UploadedFile


@pytest.mark.asyncio
class TestOpenAIFilesClient:
    @patch("autogen.beta.config.openai.files.AsyncOpenAI")
    async def test_upload(self, mock_openai_cls: MagicMock, openai_config: MagicMock) -> None:
        mock_client = AsyncMock()
        mock_openai_cls.return_value = mock_client
        mock_client.files.create.return_value = SimpleNamespace(
            id="file-abc",
            filename="test.pdf",
            bytes=2048,
            purpose="assistants",
            created_at=1700000000,
        )

        client = OpenAIFilesClient(openai_config)
        result = await client.upload(b"pdf-data", "test.pdf", "assistants")

        assert isinstance(result, UploadedFile)
        assert result.file_id == "file-abc"
        assert result.filename == "test.pdf"
        assert result.provider == "openai"
        assert result.bytes_count == 2048
        assert result.purpose == "assistants"

    @patch("autogen.beta.config.openai.files.AsyncOpenAI")
    async def test_read(self, mock_openai_cls: MagicMock, openai_config: MagicMock) -> None:
        mock_client = AsyncMock()
        mock_openai_cls.return_value = mock_client
        mock_client.files.content.return_value = SimpleNamespace(content=b"file-bytes")
        mock_client.files.retrieve.return_value = SimpleNamespace(filename="doc.pdf")

        client = OpenAIFilesClient(openai_config)
        result = await client.read("file-abc")

        assert isinstance(result, FileContent)
        assert result.data == b"file-bytes"
        assert result.name == "doc.pdf"

    @patch("autogen.beta.config.openai.files.AsyncOpenAI")
    async def test_list(self, mock_openai_cls: MagicMock, openai_config: MagicMock) -> None:
        mock_client = AsyncMock()
        mock_openai_cls.return_value = mock_client
        mock_client.files.list.return_value = SimpleNamespace(
            data=[
                SimpleNamespace(
                    id="file-1",
                    filename="a.txt",
                    bytes=100,
                    purpose="assistants",
                    created_at=None,
                ),
                SimpleNamespace(
                    id="file-2",
                    filename="b.csv",
                    bytes=200,
                    purpose="fine-tune",
                    created_at=1700000000,
                ),
            ]
        )

        client = OpenAIFilesClient(openai_config)
        result = await client.list()

        assert len(result) == 2
        assert result[0].file_id == "file-1"
        assert result[1].file_id == "file-2"
        assert result[1].purpose == "fine-tune"

    @patch("autogen.beta.config.openai.files.AsyncOpenAI")
    async def test_delete(self, mock_openai_cls: MagicMock, openai_config: MagicMock) -> None:
        mock_client = AsyncMock()
        mock_openai_cls.return_value = mock_client

        client = OpenAIFilesClient(openai_config)
        await client.delete("file-abc")

        mock_client.files.delete.assert_awaited_once_with("file-abc")
