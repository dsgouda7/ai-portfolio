#!/usr/bin/env python
"""Set the default kernel on every notebook under notes/.

Routing rules (by top-level folder under notes/):
  - notes/ML                → ml-notes
  - notes/05-AIInfrastructure  → ai-infrastructure
  - notes/MultiAgentAI      → multi-agent-ai
  - everything else         → ai-ml-dev

Idempotent: only rewrites the notebook if its kernelspec does not already
match the target. Safe to run repeatedly from both setup.ps1 and setup.sh.

Usage:
    python scripts/set_default_kernel.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

KERNEL_DISPLAY = {
    "ml-notes":          "ML Notes (venv)",
    "ai-infrastructure": "Python (AI Infrastructure)",
    "multi-agent-ai":    "Multi-Agent AI",
    "ai-ml-dev":         "AI/ML Dev (venv)",
}


def kernel_for(notebook_path: Path, notes_root: Path) -> str:
    rel = notebook_path.relative_to(notes_root)
    top = rel.parts[0] if rel.parts else ""
    return {
        "ML":               "ml-notes",
        "AIInfrastructure": "ai-infrastructure",
        "MultiAgentAI":     "multi-agent-ai",
    }.get(top, "ai-ml-dev")


def ensure_kernel(nb_path: Path, kernel_name: str) -> bool:
    """Return True if the notebook was modified."""
    try:
        with nb_path.open("r", encoding="utf-8") as f:
            nb = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"  ! skip {nb_path} ({e})")
        return False

    metadata = nb.setdefault("metadata", {})
    kernelspec = metadata.get("kernelspec", {})

    target = {
        "name":         kernel_name,
        "display_name": KERNEL_DISPLAY[kernel_name],
        "language":     "python",
    }

    if kernelspec == target:
        return False

    metadata["kernelspec"] = target
    # Also keep language_info consistent so VS Code/Jupyter pick Python 3
    metadata.setdefault("language_info", {}).setdefault("name", "python")

    tmp = nb_path.with_suffix(nb_path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write("\n")
    tmp.replace(nb_path)
    return True


def main() -> int:
    repo_root  = Path(__file__).resolve().parent.parent
    notes_root = repo_root / "notes"
    if not notes_root.is_dir():
        print(f"notes/ directory not found at {notes_root}")
        return 1

    notebooks = sorted(notes_root.rglob("*.ipynb"))
    # Exclude checkpoint files
    notebooks = [nb for nb in notebooks if ".ipynb_checkpoints" not in nb.parts]

    changed = unchanged = 0
    for nb in notebooks:
        kernel = kernel_for(nb, notes_root)
        if ensure_kernel(nb, kernel):
            print(f"  ✓ {nb.relative_to(repo_root)}  →  {kernel}")
            changed += 1
        else:
            unchanged += 1

    print(f"\n  {changed} notebook(s) updated, {unchanged} already correct "
          f"({len(notebooks)} total).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
