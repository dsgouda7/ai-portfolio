"""Validate metadata, syntax, and sequential execution for the Agentic AI track."""

from __future__ import annotations

import argparse
import ast
import json
import os
import sys
import traceback
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")

os.environ.setdefault("MPLBACKEND", "Agg")
TRACK_DIR = Path(__file__).resolve().parents[1]


def notebook_paths():
    return sorted(TRACK_DIR.glob("[0-9][0-9]*-*/*.ipynb"))


def validate_structure(path: Path):
    notebook = json.loads(path.read_text(encoding="utf-8"))
    if notebook.get("nbformat") != 4:
        raise ValueError(f"{path.name}: expected nbformat 4")
    kernelspec = notebook.get("metadata", {}).get("kernelspec", {})
    if kernelspec.get("name") != "agentic-ai":
        raise ValueError(f"{path.name}: expected agentic-ai kernel")
    for index, cell in enumerate(notebook["cells"], 1):
        language = cell.get("metadata", {}).get("language")
        if language != ("python" if cell["cell_type"] == "code" else "markdown"):
            raise ValueError(f"{path.name}: cell {index} missing correct metadata.language")
        if cell["cell_type"] == "code":
            ast.parse("".join(cell.get("source", [])), filename=f"{path}:cell-{index}")
            if cell.get("execution_count") is not None or cell.get("outputs"):
                raise ValueError(f"{path.name}: cell {index} contains persisted output")
    return notebook


def execute(path: Path, notebook: dict):
    os.chdir(path.parent)
    namespace = {
        "__name__": "__main__",
        "__file__": str(path),
        "__vsc_ipynb_file__": str(path),
    }
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    for index, cell in enumerate(code_cells, 1):
        source = "".join(cell.get("source", []))
        try:
            exec(
                compile(
                    source,
                    f"{path.name}:code-{index}",
                    "exec",
                    dont_inherit=True,
                ),
                namespace,
            )
        except BaseException:
            print(f"FAIL {path.name} code cell {index}", file=sys.stderr)
            traceback.print_exc()
            raise
    print(f"PASS {path.name}: {len(code_cells)} code cells")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--static-only", action="store_true")
    args = parser.parse_args()
    paths = notebook_paths()
    if len(paths) != 11:
        raise SystemExit(f"Expected 11 notebooks, found {len(paths)}")
    for path in paths:
        notebook = validate_structure(path)
        if args.static_only:
            print(f"PASS {path.name}: structure and syntax")
        else:
            execute(path, notebook)
    print("AGENTIC_AI_TRACK_VALIDATION_PASS")


if __name__ == "__main__":
    main()
