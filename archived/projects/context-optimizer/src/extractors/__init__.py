"""
FormatRouter — detect file format, dispatch to the right extractor,
and resolve the compressor task name for task-based model selection.

Usage
-----
    from context_optimizer.extractors import FormatRouter

    router = FormatRouter()
    text = router.extract(Path("report.pdf"))
    task = router.task_for(Path("main.py"))   # -> "code"
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import BaseExtractor


# ── Extension to task mapping ─────────────────────────────────────────────────

_EXT_TO_TASK: dict[str, str] = {
    # Plain prose
    ".txt": "text_prose",
    ".rtf": "text_prose",
    # Markup / structured text
    ".md": "text_prose",
    ".mdx": "text_prose",
    ".rst": "text_prose",
    # Rich documents
    ".pdf": "document",
    ".docx": "document",
    ".doc": "document",
    # Tabular data
    ".xlsx": "tabular_data",
    ".xls": "tabular_data",
    ".csv": "tabular_data",
    # Structured markup
    ".xml": "markup",
    ".html": "markup",
    ".htm": "markup",
    ".xhtml": "markup",
    # Source code
    ".py": "code",
    ".c": "code",
    ".h": "code",
    ".cpp": "code",
    ".cc": "code",
    ".cxx": "code",
    ".hpp": "code",
    ".js": "code",
    ".ts": "code",
    ".jsx": "code",
    ".tsx": "code",
    ".go": "code",
    ".rs": "code",
    ".java": "code",
    ".kt": "code",
    ".rb": "code",
    ".swift": "code",
    ".cs": "code",
}

# Formats we can actually extract text from
_SUPPORTED_EXTS = set(_EXT_TO_TASK.keys())


class FormatRouter:
    """
    Detect file format, extract plain text, and resolve the compressor task.

    Parameters
    ----------
    extra_mappings:
        Additional ``{".ext": "task_name"}`` entries to merge into the
        default mapping.  Useful for domain-specific extensions.
    """

    def __init__(self, extra_mappings: dict[str, str] | None = None) -> None:
        self._map: dict[str, str] = dict(_EXT_TO_TASK)
        if extra_mappings:
            self._map.update(extra_mappings)

        # Lazy-load extractors
        self._extractors: dict[str, "BaseExtractor"] = {}

    # ── Public API ────────────────────────────────────────────────────────────

    def task_for(self, path: Path) -> str:
        """Return the task name for *path* based on its extension."""
        return self._map.get(path.suffix.lower(), "text_prose")

    def is_supported(self, path: Path) -> bool:
        """Return True if this router can extract text from *path*."""
        return path.suffix.lower() in self._map

    def extract(self, path: Path) -> str:
        """
        Extract plain UTF-8 text from *path*.

        Returns an empty string for unsupported formats rather than raising.
        """
        ext = path.suffix.lower()
        extractor = self._get_extractor(ext)
        if extractor is None:
            return path.read_text(encoding="utf-8", errors="replace")
        return extractor.extract(path)

    def scan_directory(
        self,
        directory: Path,
        recursive: bool = True,
    ) -> list[tuple[Path, str]]:
        """
        Return ``[(file_path, task_name)]`` for all supported files under
        *directory*.

        Parameters
        ----------
        directory:
            Root directory to scan.
        recursive:
            If True (default), scan sub-directories recursively.
        """
        pattern = "**/*" if recursive else "*"
        return [
            (p, self.task_for(p))
            for p in directory.glob(pattern)
            if p.is_file() and self.is_supported(p)
        ]

    # ── Private ───────────────────────────────────────────────────────────────

    def _get_extractor(self, ext: str) -> "BaseExtractor | None":
        if ext in self._extractors:
            return self._extractors[ext]

        extractor: "BaseExtractor | None" = None

        if ext in {".txt"}:
            from .txt import TxtExtractor

            extractor = TxtExtractor()
        elif ext in {".rtf"}:
            from .txt import RtfExtractor

            extractor = RtfExtractor()
        elif ext in {".md", ".mdx", ".rst"}:
            from .markdown import MarkdownExtractor

            extractor = MarkdownExtractor()
        elif ext == ".pdf":
            from .pdf import PdfExtractor

            extractor = PdfExtractor()
        elif ext in {".docx", ".doc"}:
            from .docx import DocxExtractor

            extractor = DocxExtractor()
        elif ext in {".xlsx", ".xls", ".csv"}:
            if ext == ".csv":
                from .txt import TxtExtractor

                extractor = TxtExtractor()
            else:
                from .xlsx import XlsxExtractor

                extractor = XlsxExtractor()
        elif ext in {".xml", ".html", ".htm", ".xhtml"}:
            from .xml_extractor import XmlExtractor

            extractor = XmlExtractor()
        elif self._map.get(ext) == "code":
            from .txt import TxtExtractor  # code files are plain text

            extractor = TxtExtractor()

        if extractor is not None:
            self._extractors[ext] = extractor
        return extractor
