"""Plain-text extractor (.txt, .rtf-stripped)."""

from __future__ import annotations

from pathlib import Path

from .base import BaseExtractor


class TxtExtractor(BaseExtractor):
    def extract(self, path: Path) -> str:
        return self._safe(self._read, path)

    def _read(self, path: Path) -> str:
        return path.read_text(encoding="utf-8", errors="replace")


class RtfExtractor(BaseExtractor):
    def extract(self, path: Path) -> str:
        return self._safe(self._read, path)

    def _read(self, path: Path) -> str:
        from striprtf.striprtf import rtf_to_text  # type: ignore[import]

        raw = path.read_text(encoding="utf-8", errors="replace")
        return rtf_to_text(raw)
