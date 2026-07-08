"""Markdown extractor — strips front-matter, code fences kept as text."""

from __future__ import annotations

import re
from pathlib import Path

from .base import BaseExtractor


class MarkdownExtractor(BaseExtractor):
    _FRONT_MATTER = re.compile(r"^---\s*\n.*?\n---\s*\n", re.DOTALL)

    def extract(self, path: Path) -> str:
        return self._safe(self._read, path)

    def _read(self, path: Path) -> str:
        text = path.read_text(encoding="utf-8", errors="replace")
        # Strip YAML/TOML front-matter
        text = self._FRONT_MATTER.sub("", text)
        # Strip HTML comments
        text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
        # Normalize image/link syntax to just the alt text
        text = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"\1", text)
        text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
        # Collapse multiple blank lines
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()
