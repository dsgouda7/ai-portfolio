#!/usr/bin/env python
"""Set one Jupyter kernelspec on every notebook in a chapter directory."""

from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path


def update_notebook(path: Path, kernel_name: str, display_name: str) -> bool:
    original = path.read_text(encoding="utf-8")
    notebook = json.loads(original)
    target = {
        "display_name": display_name,
        "language": "python",
        "name": kernel_name,
    }

    changed = False
    metadata = notebook.setdefault("metadata", {})
    if metadata.get("kernelspec") != target:
        metadata["kernelspec"] = target
        changed = True
    metadata.setdefault("language_info", {}).setdefault("name", "python")

    for cell in notebook.get("cells", []):
        cell_id = cell.get("id")
        if not cell_id:
            cell_id = uuid.uuid4().hex[:8]
            cell["id"] = cell_id
            changed = True

        cell_metadata = cell.setdefault("metadata", {})
        language = "python" if cell.get("cell_type") == "code" else "markdown"
        if cell_metadata.get("id") != cell_id:
            cell_metadata["id"] = cell_id
            changed = True
        if cell_metadata.get("language") != language:
            cell_metadata["language"] = language
            changed = True

    if not changed:
        return False

    suffix = "\n" if original.endswith("\n") else ""
    updated = json.dumps(notebook, ensure_ascii=False, indent=1) + suffix
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(updated, encoding="utf-8", newline="\n")
    temporary.replace(path)
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--display-name", required=True)
    args = parser.parse_args()

    directory = args.directory.resolve()
    if not directory.is_dir():
        parser.error(f"notebook directory does not exist: {directory}")

    notebooks = sorted(
        path
        for path in directory.rglob("*.ipynb")
        if ".venv" not in path.parts and ".ipynb_checkpoints" not in path.parts
    )
    if not notebooks:
        parser.error(f"no notebooks found in: {directory}")

    changed = 0
    for notebook in notebooks:
        if update_notebook(notebook, args.name, args.display_name):
            print(f"Updated kernel: {notebook.name} -> {args.name}")
            changed += 1

    print(f"Kernel metadata ready for {len(notebooks)} notebook(s); {changed} updated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
