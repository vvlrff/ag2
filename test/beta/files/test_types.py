# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from autogen.beta.events import BinaryResult
from autogen.beta.events.input_events import FileIdInput, Input
from autogen.beta.files import FileContent, UploadedFile


class TestUploadedFile:
    def test_is_file_id_input(self) -> None:
        f = UploadedFile(file_id="file-123", filename="test.pdf")
        assert isinstance(f, FileIdInput)

    def test_is_input(self) -> None:
        f = UploadedFile(file_id="file-123")
        assert isinstance(f, Input)

    def test_fields(self) -> None:
        f = UploadedFile(
            file_id="file-123",
            filename="doc.pdf",
            provider="openai",
            bytes_count=1024,
            purpose="assistants",
            created_at="2026-01-01T00:00:00Z",
        )
        assert f.file_id == "file-123"
        assert f.filename == "doc.pdf"
        assert f.provider == "openai"
        assert f.bytes_count == 1024
        assert f.purpose == "assistants"
        assert f.created_at == "2026-01-01T00:00:00Z"

    def test_minimal_construction(self) -> None:
        f = UploadedFile(file_id="file-456")
        assert f.file_id == "file-456"
        assert f.filename is None
        assert f.provider is None
        assert f.bytes_count is None


class TestFileContent:
    def test_frozen(self) -> None:
        fc = FileContent(name="test.txt", data=b"hello")
        with pytest.raises(AttributeError):
            fc.name = "other.txt"  # type: ignore[misc]

    def test_fields(self) -> None:
        fc = FileContent(name="doc.pdf", data=b"pdf-bytes", media_type="application/pdf")
        assert fc.name == "doc.pdf"
        assert fc.data == b"pdf-bytes"
        assert fc.media_type == "application/pdf"


class TestBinaryResult:
    def test_name_from_metadata(self) -> None:
        br = BinaryResult(data=b"image-bytes", metadata={"filename": "cat.png", "size": "512x512"})
        assert br.name == "cat.png"

    def test_name_default(self) -> None:
        br = BinaryResult(data=b"data")
        assert br.name == "generated_file"


@pytest.mark.asyncio
async def test_binary_result_content_returns_bytes() -> None:
    br = BinaryResult(data=b"raw-data", metadata={"filename": "out.bin"})
    assert await br.content() == b"raw-data"
