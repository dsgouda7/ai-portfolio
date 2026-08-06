"""Pure parser and quarantine tests that require no Spark session."""

from pathlib import Path
import sys

import pytest


PHASE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PHASE_ROOT / "src"))

from remote.contracts import SourceObject  # noqa: E402
from remote.parsers import ParserRegistry, ParsingError  # noqa: E402


def _source(name: str, media_type: str) -> SourceObject:
    path = PHASE_ROOT / "tests" / "fixtures" / "source" / name
    return SourceObject(
        source_uri=f"abfss://raw@riverside.dfs.core.windows.net/{name}",
        storage_uri=f"abfss://raw@riverside.dfs.core.windows.net/{name}",
        source_name=name,
        media_type=media_type,
        content=path.read_bytes(),
    )


def test_markdown_parser_preserves_text_and_extracts_title() -> None:
    parsed = ParserRegistry.default().parse(_source("chapter-001.md", "text/markdown"))
    assert parsed.title == "Harbor Lights"
    assert "harbor lights" in parsed.text.lower()
    assert parsed.parser_name == "markdown-parser"


def test_malformed_json_has_bounded_error() -> None:
    with pytest.raises(ParsingError, match="malformed") as error:
        ParserRegistry.default().parse(_source("malformed-json.txt", "application/json"))
    assert error.value.code == "invalid_json"
    assert "Incomplete document" not in error.value.safe_message


def test_unsupported_format_is_quarantinable() -> None:
    source = SourceObject(
        source_uri="abfss://raw@riverside.dfs.core.windows.net/archive.bin",
        storage_uri="abfss://raw@riverside.dfs.core.windows.net/archive.bin",
        source_name="archive.bin",
        media_type="application/octet-stream",
        content=b"opaque",
    )
    with pytest.raises(ParsingError) as error:
        ParserRegistry.default().parse(source)
    assert error.value.code == "unsupported_media_type"
