"""Base extractor interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class BaseExtractor(ABC):
    """Extract plain UTF-8 text from a source file."""

    @abstractmethod
    def extract(self, path: Path) -> str:
        """Return normalized plain text. Never raises — returns '' on failure."""
        ...

    def _safe(self, fn, path: Path) -> str:
        try:
            return fn(path)
        except Exception as exc:
            import warnings

            warnings.warn(f"[{self.__class__.__name__}] {path.name}: {exc}")
            return ""
