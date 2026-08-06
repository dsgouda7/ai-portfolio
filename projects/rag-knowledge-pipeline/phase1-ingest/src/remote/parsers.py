"""Bounded, deterministic parsers for common unstructured document formats."""

from __future__ import annotations

from abc import ABC, abstractmethod
from io import BytesIO
import json
from pathlib import PurePosixPath
import re
from typing import Iterable

from .contracts import ParsedContent, SourceObject, normalize_text


PARSER_VERSION = "1.0.0"


class ParsingError(ValueError):
    def __init__(self, code: str, safe_message: str, *, retryable: bool = False) -> None:
        super().__init__(safe_message)
        self.code = code
        self.safe_message = safe_message
        self.retryable = retryable


class DocumentParser(ABC):
    name: str
    media_types: frozenset[str]
    extensions: frozenset[str]

    def supports(self, source: SourceObject) -> bool:
        extension = PurePosixPath(source.source_name).suffix.lower()
        return source.media_type.lower() in self.media_types or extension in self.extensions

    @abstractmethod
    def parse(self, source: SourceObject) -> ParsedContent:
        ...

    def result(self, source: SourceObject, text: str, title: str | None = None) -> ParsedContent:
        normalized = normalize_text(text)
        if not normalized:
            raise ParsingError("empty_document", "parser produced no document text")
        return ParsedContent(
            title=title or PurePosixPath(source.source_name).stem,
            text=normalized,
            language="en",
            parser_name=self.name,
            parser_version=PARSER_VERSION,
        )


class PlainTextParser(DocumentParser):
    name = "plain-text-parser"
    media_types = frozenset({"text/plain"})
    extensions = frozenset({".txt", ".text"})

    def parse(self, source: SourceObject) -> ParsedContent:
        if b"\x00" in source.content:
            raise ParsingError("binary_content", "plain-text input contains binary data")
        try:
            text = source.content.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise ParsingError("invalid_encoding", "plain-text input is not valid UTF-8") from exc
        return self.result(source, text)


class MarkdownParser(PlainTextParser):
    name = "markdown-parser"
    media_types = frozenset({"text/markdown", "text/x-markdown"})
    extensions = frozenset({".md", ".markdown"})

    def parse(self, source: SourceObject) -> ParsedContent:
        parsed = super().parse(source)
        heading = re.search(r"^#\s+(.+?)\s*$", parsed.text, flags=re.MULTILINE)
        title = heading.group(1).strip() if heading else PurePosixPath(source.source_name).stem
        return ParsedContent(title, parsed.text, "en", self.name, PARSER_VERSION)


class JsonParser(DocumentParser):
    name = "json-parser"
    media_types = frozenset({"application/json"})
    extensions = frozenset({".json"})

    def parse(self, source: SourceObject) -> ParsedContent:
        try:
            value = json.loads(source.content.decode("utf-8-sig"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ParsingError("invalid_json", "JSON input is malformed or not UTF-8") from exc
        strings = list(_json_strings(value))
        title = value.get("title") if isinstance(value, dict) and isinstance(value.get("title"), str) else None
        return self.result(source, "\n".join(strings), title)


def _json_strings(value: object) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for item in value:
            yield from _json_strings(item)
    elif isinstance(value, dict):
        for key in sorted(value):
            yield from _json_strings(value[key])


class HtmlParser(DocumentParser):
    name = "html-parser"
    media_types = frozenset({"text/html", "application/xhtml+xml"})
    extensions = frozenset({".html", ".htm", ".xhtml"})

    def parse(self, source: SourceObject) -> ParsedContent:
        try:
            from bs4 import BeautifulSoup
        except ImportError as exc:
            raise ParsingError("parser_dependency_missing", "HTML parser dependency is unavailable") from exc
        soup = BeautifulSoup(source.content, "html.parser")
        for element in soup(["script", "style", "noscript"]):
            element.decompose()
        title = soup.title.string.strip() if soup.title and soup.title.string else None
        return self.result(source, soup.get_text("\n"), title)


class PdfParser(DocumentParser):
    name = "pypdf-parser"
    media_types = frozenset({"application/pdf"})
    extensions = frozenset({".pdf"})

    def parse(self, source: SourceObject) -> ParsedContent:
        try:
            from pypdf import PdfReader
            reader = PdfReader(BytesIO(source.content), strict=True)
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
            title = reader.metadata.title if reader.metadata and reader.metadata.title else None
        except ImportError as exc:
            raise ParsingError("parser_dependency_missing", "PDF parser dependency is unavailable") from exc
        except Exception as exc:
            raise ParsingError("invalid_pdf", "PDF input could not be parsed") from exc
        return self.result(source, text, title)


class DocxParser(DocumentParser):
    name = "python-docx-parser"
    media_types = frozenset(
        {"application/vnd.openxmlformats-officedocument.wordprocessingml.document"}
    )
    extensions = frozenset({".docx"})

    def parse(self, source: SourceObject) -> ParsedContent:
        try:
            from docx import Document
            document = Document(BytesIO(source.content))
        except ImportError as exc:
            raise ParsingError("parser_dependency_missing", "DOCX parser dependency is unavailable") from exc
        except Exception as exc:
            raise ParsingError("invalid_docx", "DOCX input could not be parsed") from exc
        title = document.core_properties.title or None
        return self.result(source, "\n".join(paragraph.text for paragraph in document.paragraphs), title)


class ParserRegistry:
    def __init__(self, parsers: Iterable[DocumentParser]) -> None:
        self._parsers = tuple(parsers)

    @classmethod
    def default(cls) -> "ParserRegistry":
        return cls((PlainTextParser(), MarkdownParser(), JsonParser(), HtmlParser(), PdfParser(), DocxParser()))

    def parse(self, source: SourceObject) -> ParsedContent:
        media_type = source.media_type.lower()
        for parser in self._parsers:
            if media_type in parser.media_types:
                return parser.parse(source)

        extension = PurePosixPath(source.source_name).suffix.lower()
        for parser in self._parsers:
            if extension in parser.extensions:
                return parser.parse(source)
        raise ParsingError("unsupported_media_type", "no parser is registered for this document format")
