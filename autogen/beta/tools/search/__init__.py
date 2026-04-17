# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock


def _missing_optional_dependency(name: str, extra: str, error: ImportError) -> Mock:
    def _raise(*args: object, **kwargs: object) -> None:
        raise ImportError(
            f'{name} requires optional dependencies. Install with `pip install "ag2[{extra}]"`'
        ) from error

    return Mock(side_effect=_raise)


try:
    from .duckduckgo import DuckDuckGoSearchTool, SearchResponse, SearchResult
except ImportError as e:
    DuckDuckGoSearchTool = _missing_optional_dependency("DuckDuckGoSearchTool", "ddgs", e)  # type: ignore[misc]
    SearchResult = _missing_optional_dependency("SearchResult", "ddgs", e)  # type: ignore[misc]
    SearchResponse = _missing_optional_dependency("SearchResponse", "ddgs", e)  # type: ignore[misc]

__all__ = ("DuckDuckGoSearchTool", "SearchResponse", "SearchResult")
