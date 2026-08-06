"""Deterministic, offset-preserving chunking for document-chunk v1 records."""

from __future__ import annotations

from typing import Any, Mapping

from .contracts import RemoteSettings, build_document_chunk, validate_chunk_invariants


def _preferred_end(text: str, start: int, hard_end: int, separators: tuple[str, ...]) -> int:
    if hard_end >= len(text):
        return len(text)
    minimum_useful_end = start + max(1, (hard_end - start) // 2)
    window = text[start:hard_end]
    for separator in separators:
        relative = window.rfind(separator)
        if relative < 0:
            continue
        candidate = start + relative + len(separator)
        if candidate >= minimum_useful_end:
            return candidate
    return hard_end


def chunk_document(
    document: Mapping[str, Any], settings: RemoteSettings
) -> list[dict[str, Any]]:
    """Split one active parsed document into stable, exact source spans."""
    text = document["text"]
    if not text:
        return []

    chunks: list[dict[str, Any]] = []
    start = 0
    ordinal = 0
    while start < len(text):
        hard_end = min(len(text), start + settings.chunking.size)
        end = _preferred_end(text, start, hard_end, settings.chunking.separators)
        if end <= start:
            end = hard_end
        content = text[start:end]
        chunk = build_document_chunk(
            document,
            content=content,
            ordinal=ordinal,
            start=start,
            end=end,
            settings=settings,
        )
        validate_chunk_invariants(chunk, text)
        chunks.append(chunk)
        if end == len(text):
            break
        next_start = max(start + 1, end - settings.chunking.overlap)
        start = next_start
        ordinal += 1
    return chunks
