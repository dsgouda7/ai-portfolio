"""
CodeChunker — split source files into function/class-level chunks.

Strategy
--------
1. **tree-sitter** (preferred): parses the AST and extracts top-level
   function and class nodes with exact line numbers.
2. **Regex fallback**: used when tree-sitter grammar is unavailable.
   Captures function/class headers by heuristic patterns.

The two strategies produce identical ``CodeChunk`` objects; callers
need not distinguish between them.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .code_pointer import CodeChunk, CodePointer


# ── Language config ───────────────────────────────────────────────────────────

# Maps extension -> (language_name, tree-sitter grammar name if available)
_LANG_MAP: dict[str, tuple[str, str | None]] = {
    ".py": ("python", "python"),
    ".pyw": ("python", "python"),
    ".c": ("c", "c"),
    ".h": ("c", "c"),
    ".cpp": ("cpp", "cpp"),
    ".cc": ("cpp", "cpp"),
    ".cxx": ("cpp", "cpp"),
    ".hpp": ("cpp", "cpp"),
    ".js": ("javascript", "javascript"),
    ".jsx": ("javascript", "javascript"),
    ".ts": ("typescript", "typescript"),
    ".tsx": ("typescript", "tsx"),
    ".go": ("go", "go"),
    ".rs": ("rust", "rust"),
    ".java": ("java", "java"),
    ".kt": ("kotlin", None),
    ".rb": ("ruby", "ruby"),
    ".cs": ("c_sharp", "c_sharp"),
}

# Regex patterns for function/class detection per language (fallback)
_REGEX_PATTERNS: dict[str, list[str]] = {
    "python": [
        r"^(async\s+def\s+\w+|def\s+\w+|class\s+\w+)",
    ],
    "c": [
        # C function: return-type name(args) {  (top-level, non-indented)
        r"^[a-zA-Z_]\S.*\([^;]*\)\s*\{?\s*$",
    ],
    "cpp": [
        r"^[a-zA-Z_~]\S.*\([^;]*\)\s*(const\s*)?\{?\s*$",
        r"^class\s+\w+",
        r"^struct\s+\w+",
    ],
    "javascript": [
        r"^(async\s+)?function\s+\w+|^const\s+\w+\s*=\s*(async\s+)?\(",
        r"^class\s+\w+",
    ],
    "typescript": [
        r"^(export\s+)?(async\s+)?function\s+\w+",
        r"^(export\s+)?class\s+\w+",
        r"^(export\s+)?interface\s+\w+",
    ],
    "go": [
        r"^func\s+(\(\w+\s+\*?\w+\)\s+)?\w+\s*\(",
    ],
    "rust": [
        r"^(pub\s+)?(async\s+)?fn\s+\w+",
        r"^(pub\s+)?(struct|enum|impl|trait)\s+\w+",
    ],
    "java": [
        r"^\s*(public|private|protected|static|final|abstract|synchronized).*\w+\s*\(",
        r"^\s*(public|private|protected)?\s*class\s+\w+",
    ],
}


# ── Tree-sitter helper ────────────────────────────────────────────────────────


def _ts_grammar_available(grammar_name: str) -> bool:
    """Check if tree-sitter grammar is installed."""
    try:
        from tree_sitter_languages import get_language  # type: ignore[import]

        get_language(grammar_name)
        return True
    except Exception:
        return False


def _chunk_with_treesitter(
    path: Path,
    source: str,
    language: str,
    grammar_name: str,
) -> "list[tuple[int, int, str]]":
    """
    Return list of ``(start_line, end_line, symbol_name)`` via tree-sitter.
    Lines are 1-indexed.
    """
    from tree_sitter_languages import get_language, get_parser  # type: ignore[import]

    lang = get_language(grammar_name)
    parser = get_parser(grammar_name)
    tree = parser.parse(source.encode("utf-8"))

    _FUNC_TYPES = {
        "function_definition",
        "function_declaration",
        "method_definition",
        "class_definition",
        "class_declaration",
        "struct_item",
        "function_item",
        "impl_item",
    }

    results: list[tuple[int, int, str]] = []

    def _name(node) -> str:
        for child in node.children:
            if child.type in ("identifier", "name", "field_identifier"):
                return child.text.decode("utf-8", errors="replace")
        return node.type

    def _walk(node, depth: int = 0) -> None:
        if node.type in _FUNC_TYPES:
            # Only include top-level or class-level items (depth 0 or 1)
            if depth <= 1:
                start = node.start_point[0] + 1  # 0-indexed → 1-indexed
                end = node.end_point[0] + 1
                results.append((start, end, _name(node)))
                return  # don't recurse into nested functions
        for child in node.children:
            _walk(
                child,
                (
                    depth + 1
                    if node.type in {"class_definition", "class_declaration"}
                    else depth
                ),
            )

    _walk(tree.root_node)
    return results


# ── Regex fallback ────────────────────────────────────────────────────────────


def _chunk_with_regex(
    lines: list[str],
    language: str,
) -> "list[tuple[int, int, str]]":
    """
    Heuristic function/class boundary detection.
    Returns ``[(start_line, end_line, header)]`` (1-indexed).
    """
    patterns = _REGEX_PATTERNS.get(language, [r"^def |^class |^func |^function "])
    compiled = [re.compile(p) for p in patterns]

    starts: list[tuple[int, str]] = []
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if any(p.match(stripped) for p in compiled):
            starts.append((i, stripped[:80]))

    results: list[tuple[int, int, str]] = []
    for idx, (start, header) in enumerate(starts):
        end = starts[idx + 1][0] - 1 if idx + 1 < len(starts) else len(lines)
        results.append((start, end, header))
    return results


# ── Public API ────────────────────────────────────────────────────────────────


class CodeChunker:
    """
    Split source files into function/class-level :class:`CodeChunk` objects.

    Parameters
    ----------
    min_lines:
        Chunks shorter than this are merged into the previous chunk.
    max_lines:
        Chunks longer than this are split at the midpoint (keeps LLM
        context windows manageable).
    use_treesitter:
        Set False to force regex fallback (for testing or offline use).
    """

    def __init__(
        self,
        min_lines: int = 5,
        max_lines: int = 300,
        use_treesitter: bool = True,
    ) -> None:
        self._min = min_lines
        self._max = max_lines
        self._use_ts = use_treesitter

    def chunk_file(self, path: Path) -> "list[CodeChunk]":
        """
        Return chunks for one source file.
        Returns an empty list if the file is unrecognised or empty.
        """
        from .code_pointer import CodeChunk, CodePointer

        ext = path.suffix.lower()
        if ext not in _LANG_MAP:
            return []

        language, grammar_name = _LANG_MAP[ext]
        try:
            source = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return []

        if not source.strip():
            return []

        lines = source.splitlines(keepends=True)

        # Try tree-sitter, fall back to regex
        ranges: list[tuple[int, int, str]] = []
        if self._use_ts and grammar_name:
            try:
                if _ts_grammar_available(grammar_name):
                    ranges = _chunk_with_treesitter(
                        path, source, language, grammar_name
                    )
            except Exception:
                pass

        if not ranges:
            ranges = _chunk_with_regex(lines, language)

        # If no structure detected, treat whole file as one chunk
        if not ranges:
            ranges = [(1, len(lines), path.stem)]

        chunks: list[CodeChunk] = []
        for start, end, symbol in ranges:
            # Skip tiny stubs
            if end - start + 1 < self._min:
                continue
            # Split oversized chunks at midpoint
            if end - start + 1 > self._max:
                mid = (start + end) // 2
                sub_ranges = [
                    (start, mid, symbol + "_part1"),
                    (mid + 1, end, symbol + "_part2"),
                ]
            else:
                sub_ranges = [(start, end, symbol)]

            for s, e, sym in sub_ranges:
                chunk_source = "".join(lines[s - 1 : e])
                cid = hashlib.md5(f"{path}:{s}:{e}".encode()).hexdigest()[:16]
                ptr = CodePointer(
                    file_path=str(path),
                    start_line=s,
                    end_line=e,
                    symbol_name=sym,
                    language=language,
                    chunk_id=cid,
                )
                chunks.append(CodeChunk(chunk_id=cid, source=chunk_source, pointer=ptr))

        return chunks

    def chunk_directory(
        self,
        directory: Path,
        recursive: bool = True,
        include_exts: list[str] | None = None,
    ) -> "list[CodeChunk]":
        """Chunk all source files under *directory*."""
        pattern = "**/*" if recursive else "*"
        exts = {e.lower() for e in (include_exts or list(_LANG_MAP.keys()))}
        all_chunks: list[CodeChunk] = []

        files = [
            p
            for p in directory.glob(pattern)
            if p.is_file() and p.suffix.lower() in exts
        ]
        for path in files:
            all_chunks.extend(self.chunk_file(path))
        return all_chunks
