#!/usr/bin/env python3
"""
merge-exercise-solution.py

For every chapter under notes/, merges notebook-exercise.ipynb and
notebook-solution.ipynb into a single notebook.ipynb (and likewise for
notebook-supplement and notebook-pytorch variants). Solution cells are placed
immediately after their matching TODO cell, collapsed by default via
`"jupyter": {"source_hidden": true}`.

Usage:
    python scripts/merge-exercise-solution.py [--dry-run]
"""

import argparse
import json
import os
import re
import uuid
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
NOTES_DIR = REPO_ROOT / "notes"

PAIR_RULES = [
    # (exercise_name, solution_name, merged_name)
    ("notebook-exercise.ipynb", "notebook-solution.ipynb", "notebook.ipynb"),
    (
        "notebook-supplement-exercise.ipynb",
        "notebook-supplement-solution.ipynb",
        "notebook-supplement.ipynb",
    ),
    (
        "notebook-pytorch-exercise.ipynb",
        "notebook-pytorch-solution.ipynb",
        "notebook-pytorch.ipynb",
    ),
]

SOLUTION_LABEL_MD = (
    "<details>\n<summary>▶ <b>Solution</b> (click to expand)</summary>\n\n"
    "_Reveal only after attempting the exercise above._\n\n"
    "</details>"
)


def new_id() -> str:
    return uuid.uuid4().hex[:8]


def is_todo_cell(cell: dict) -> bool:
    """Return True if the cell is a code cell whose source starts with # TODO."""
    if cell.get("cell_type") != "code":
        return False
    src = cell.get("source", "")
    if isinstance(src, list):
        src = "".join(src)
    return src.lstrip().startswith("# TODO")


def hidden_cell(cell: dict) -> dict:
    """Return a copy of *cell* with source_hidden=True in its metadata."""
    c = json.loads(json.dumps(cell))  # deep copy
    c.setdefault("metadata", {})
    c["metadata"].setdefault("jupyter", {})
    c["metadata"]["jupyter"]["source_hidden"] = True
    # Give it a fresh unique id so it doesn't clash with the exercise cell
    c["id"] = new_id()
    return c


def make_solution_label_cell() -> dict:
    """Create a hidden markdown cell that acts as the expandable label."""
    return {
        "cell_type": "markdown",
        "id": new_id(),
        "metadata": {"jupyter": {"source_hidden": True}},
        "source": SOLUTION_LABEL_MD,
    }


def merge_notebooks(
    exercise_path: Path, solution_path: Path, out_path: Path, dry_run: bool
):
    with exercise_path.open(encoding="utf-8") as f:
        ex = json.load(f)
    with solution_path.open(encoding="utf-8") as f:
        sol = json.load(f)

    sol_cells = sol["cells"]

    merged_cells = []
    for idx, ex_cell in enumerate(ex["cells"]):
        merged_cells.append(ex_cell)

        if is_todo_cell(ex_cell):
            # Match by position (primary), validated by id when both present
            sol_cell = sol_cells[idx] if idx < len(sol_cells) else None
            if sol_cell is not None:
                ex_id = ex_cell.get("id")
                sol_id = sol_cell.get("id")
                if ex_id and sol_id and ex_id != sol_id:
                    # Position mismatch — skip rather than insert wrong solution
                    sol_cell = None
            if sol_cell is not None:
                sol_src = sol_cell.get("source", "")
                if isinstance(sol_src, list):
                    sol_src = "".join(sol_src)
                ex_src = ex_cell.get("source", "")
                if isinstance(ex_src, list):
                    ex_src = "".join(ex_src)
                # Only add a solution block if the content actually differs
                if sol_src.strip() != ex_src.strip():
                    merged_cells.append(make_solution_label_cell())
                    merged_cells.append(hidden_cell(sol_cell))

    result = dict(ex)  # carry over nbformat, metadata, kernelspec, etc.
    result["cells"] = merged_cells

    if dry_run:
        print(f"  [DRY-RUN] Would write {out_path} ({len(merged_cells)} cells)")
        return

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=1, ensure_ascii=False)
        f.write("\n")

    print(f"  Wrote {out_path.relative_to(REPO_ROOT)}  ({len(merged_cells)} cells)")

    # Remove originals
    exercise_path.unlink()
    print(f"  Removed {exercise_path.relative_to(REPO_ROOT)}")
    solution_path.unlink()
    print(f"  Removed {solution_path.relative_to(REPO_ROOT)}")


def process_dir(chapter_dir: Path, dry_run: bool):
    for ex_name, sol_name, out_name in PAIR_RULES:
        ex_path = chapter_dir / ex_name
        sol_path = chapter_dir / sol_name
        out_path = chapter_dir / out_name

        if not ex_path.exists() or not sol_path.exists():
            continue
        if ex_path.stat().st_size == 0 or sol_path.stat().st_size == 0:
            print(
                f"  [SKIP] Empty file in {chapter_dir.relative_to(NOTES_DIR)}: {ex_name} or {sol_name}"
            )
            continue

        print(
            f"\n[{chapter_dir.relative_to(NOTES_DIR)}]  {ex_name} + {sol_name} -> {out_name}"
        )
        merge_notebooks(ex_path, sol_path, out_path, dry_run)


def main():
    parser = argparse.ArgumentParser(description="Merge exercise/solution notebooks.")
    parser.add_argument(
        "--dry-run", action="store_true", help="Preview without writing files"
    )
    args = parser.parse_args()

    if args.dry_run:
        print("=== DRY-RUN mode — no files will be changed ===\n")

    # Walk every subdirectory inside notes/
    for root, dirs, files in os.walk(NOTES_DIR):
        dirs.sort()
        process_dir(Path(root), args.dry_run)

    print("\nDone.")


if __name__ == "__main__":
    main()
