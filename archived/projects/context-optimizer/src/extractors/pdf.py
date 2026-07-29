"""PDF extractor using pdfminer.six (text-layer PDFs; no OCR)."""

from __future__ import annotations

import io
from pathlib import Path

from .base import BaseExtractor


class PdfExtractor(BaseExtractor):
    def extract(self, path: Path) -> str:
        return self._safe(self._read, path)

    def _read(self, path: Path) -> str:
        from pdfminer.high_level import extract_text  # type: ignore[import]
        from pdfminer.layout import LAParams  # type: ignore[import]

        params = LAParams(line_margin=0.5, word_margin=0.1)
        text = extract_text(str(path), laparams=params)
        if not text:
            return ""
        import re

        # Collapse excessive whitespace while preserving paragraph breaks
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()
