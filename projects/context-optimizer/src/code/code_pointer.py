"""CodePointer — line-level file reference for source code chunks."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class CodePointer:
    """
    Points to a specific range of lines in a source file.

    Unlike BlockIndex (which stores byte offsets for arbitrary text),
    CodePointer stores 1-indexed line numbers and the symbol name
    (function/class) for human-readable citation.
    """

    file_path: str  # Absolute or relative path to the source file
    start_line: int  # First line of the symbol (1-indexed, inclusive)
    end_line: int  # Last line of the symbol (1-indexed, inclusive)
    symbol_name: str  # e.g. "tcp_retransmit_skb" or "class MyClass"
    language: str  # "c", "python", "javascript", etc.
    chunk_id: str = ""  # Unique ID used as the ChromaDB document ID

    def fetch(self) -> str:
        """Read the exact source lines from disk."""
        path = Path(self.file_path)
        if not path.exists():
            return ""
        with open(path, encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        return "".join(lines[self.start_line - 1 : self.end_line])

    def citation(self) -> str:
        """Human-readable citation: ``path:start_line-end_line``."""
        return f"{self.file_path}:{self.start_line}-{self.end_line}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "file_path": self.file_path,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "symbol_name": self.symbol_name,
            "language": self.language,
            "chunk_id": self.chunk_id,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CodePointer":
        return cls(**d)


@dataclass
class CodeChunk:
    """
    A single code unit (function, class, etc.) with source and optional summary.

    The *source* field contains the raw source code.
    The *summary* field is populated by the code summarizer (e.g. qwen2.5-coder:7b)
    and stored in ChromaDB for semantic search.
    """

    chunk_id: str
    source: str  # Raw source code
    pointer: CodePointer
    summary: str = ""  # Populated after LLM summarization
    summary_tokens: int = 0
