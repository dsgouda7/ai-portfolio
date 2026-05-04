#!/usr/bin/env python
"""Generate exercise/solution notebook pairs from existing notebooks.

Creates two versions from each notebook:
  - *_solution.ipynb: Read-only reference with full code
  - *_exercise.ipynb: Writable practice version with code removed

Usage:
    python scripts/generate_exercise_notebooks.py
    python scripts/generate_exercise_notebooks.py --filter "notes/05-multimodal_ai/**/*.ipynb"
    python scripts/generate_exercise_notebooks.py --dry-run

Strategy:
    - Solution notebooks: Exact copy with read-only metadata
    - Exercise notebooks: Markdown preserved, all Python code removed
    - Original notebooks: Left untouched (for backward compatibility)

Author: AI Portfolio Maintenance
Date: April 28, 2026
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple


def should_skip_notebook(notebook_path: Path) -> bool:
    """Check if notebook should be skipped."""
    # Skip checkpoint files
    if ".ipynb_checkpoints" in notebook_path.parts:
        return True
    
    # Skip already-generated solution/exercise notebooks
    stem = notebook_path.stem
    if stem.endswith("_solution") or stem.endswith("_exercise"):
        return True
    if stem.endswith("_reference"):  # grand_solution_reference.ipynb
        return True
    
    return False


def get_output_paths(notebook_path: Path) -> Tuple[Path, Path]:
    """Determine output paths for solution and exercise notebooks."""
    stem = notebook_path.stem
    parent = notebook_path.parent
    
    # Special case: grand_solution.ipynb → grand_solution_reference.ipynb
    if stem == "grand_solution":
        solution_path = parent / "grand_solution_reference.ipynb"
        exercise_path = parent / "grand_solution_exercise.ipynb"
    else:
        solution_path = parent / f"{stem}_solution.ipynb"
        exercise_path = parent / f"{stem}_exercise.ipynb"
    
    return solution_path, exercise_path


def create_solution_notebook(source_path: Path, dest_path: Path) -> bool:
    """Create read-only solution notebook (exact copy with metadata flag)."""
    try:
        with source_path.open("r", encoding="utf-8") as f:
            nb = json.load(f)
        
        # Set read-only metadata (Jupyter recognizes this)
        if "metadata" not in nb:
            nb["metadata"] = {}
        nb["metadata"]["editable"] = False
        
        # Write solution notebook
        with dest_path.open("w", encoding="utf-8", newline="\n") as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
            f.write("\n")
        
        # Set file system read-only
        try:
            if sys.platform == "win32":
                # Windows: use attrib command
                os.system(f'attrib +R "{dest_path}"')
            else:
                # Unix: chmod 444
                os.chmod(dest_path, 0o444)
        except Exception as e:
            print(f"    Warning: Could not set read-only permission: {e}")
        
        return True
    
    except Exception as e:
        print(f"    Error creating solution: {e}")
        return False


def create_exercise_notebook(source_path: Path, dest_path: Path) -> bool:
    """Create exercise notebook (markdown preserved, code removed)."""
    try:
        with source_path.open("r", encoding="utf-8") as f:
            nb = json.load(f)
        
        # Transform code cells
        for cell in nb.get("cells", []):
            if cell.get("cell_type") == "code":
                # Remove all code, outputs, execution count
                cell["source"] = ["# TODO: Implement this cell\n"]
                cell["outputs"] = []
                cell["execution_count"] = None
        
        # Keep editable (default, but explicit)
        if "metadata" not in nb:
            nb["metadata"] = {}
        nb["metadata"]["editable"] = True
        
        # Write exercise notebook
        with dest_path.open("w", encoding="utf-8", newline="\n") as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
            f.write("\n")
        
        # Ensure writable (remove read-only if present)
        try:
            if sys.platform == "win32":
                os.system(f'attrib -R "{dest_path}"')
            else:
                os.chmod(dest_path, 0o644)
        except Exception as e:
            print(f"    Warning: Could not set writable permission: {e}")
        
        return True
    
    except Exception as e:
        print(f"    Error creating exercise: {e}")
        return False


def process_notebook(notebook_path: Path, dry_run: bool = False) -> Tuple[bool, bool]:
    """Process a single notebook: create solution and exercise versions.
    
    Returns:
        (solution_success, exercise_success)
    """
    solution_path, exercise_path = get_output_paths(notebook_path)
    
    # Check if already exists
    solution_exists = solution_path.exists()
    exercise_exists = exercise_path.exists()
    
    if solution_exists and exercise_exists:
        print(f"  ⚡ Skip (already exists): {notebook_path.relative_to(notebook_path.parents[2])}")
        return (True, True)
    
    print(f"  🔄 Processing: {notebook_path.relative_to(notebook_path.parents[2])}")
    
    if dry_run:
        print(f"    [DRY RUN] Would create: {solution_path.name}")
        print(f"    [DRY RUN] Would create: {exercise_path.name}")
        return (True, True)
    
    # Create solution notebook
    solution_success = False
    if not solution_exists:
        solution_success = create_solution_notebook(notebook_path, solution_path)
        if solution_success:
            print(f"    ✓ Created: {solution_path.name}")
    else:
        solution_success = True
        print(f"    ⚡ Exists: {solution_path.name}")
    
    # Create exercise notebook
    exercise_success = False
    if not exercise_exists:
        exercise_success = create_exercise_notebook(notebook_path, exercise_path)
        if exercise_success:
            print(f"    ✓ Created: {exercise_path.name}")
    else:
        exercise_success = True
        print(f"    ⚡ Exists: {exercise_path.name}")
    
    return (solution_success, exercise_success)


def discover_notebooks(repo_root: Path, filter_pattern: str = None) -> List[Path]:
    """Discover all notebooks in the repository."""
    notes_root = repo_root / "notes"
    
    if filter_pattern:
        # Use filter pattern (e.g., "notes/05-multimodal_ai/**/*.ipynb")
        notebooks = list(repo_root.glob(filter_pattern))
    else:
        # Find all notebooks under notes/
        notebooks = list(notes_root.rglob("*.ipynb"))
    
    # Filter out checkpoints and already-generated notebooks
    notebooks = [nb for nb in notebooks if not should_skip_notebook(nb)]
    
    return sorted(notebooks)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate exercise/solution notebook pairs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all notebooks
  python scripts/generate_exercise_notebooks.py

  # Process only multimodal AI track
  python scripts/generate_exercise_notebooks.py --filter "notes/05-multimodal_ai/**/*.ipynb"

  # Dry run (show what would be created)
  python scripts/generate_exercise_notebooks.py --dry-run
        """
    )
    parser.add_argument(
        "--filter",
        type=str,
        help="Glob pattern to filter notebooks (e.g., 'notes/05-multimodal_ai/**/*.ipynb')"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be created without actually creating files"
    )
    args = parser.parse_args()
    
    # Determine repo root
    repo_root = Path(__file__).resolve().parent.parent
    notes_root = repo_root / "notes"
    
    if not notes_root.is_dir():
        print(f"Error: notes/ directory not found at {notes_root}")
        return 1
    
    print("\n" + "=" * 70)
    print("  Exercise/Solution Notebook Generator")
    print("=" * 70)
    
    # Discover notebooks
    print(f"\n🔍 Discovering notebooks...")
    notebooks = discover_notebooks(repo_root, args.filter)
    
    if not notebooks:
        print("  No notebooks found matching criteria.")
        return 0
    
    print(f"  Found {len(notebooks)} notebook(s) to process")
    
    if args.dry_run:
        print("\n⚠️  DRY RUN MODE - No files will be created")
    
    # Process notebooks
    print(f"\n📝 Processing notebooks...\n")
    
    solution_success_count = 0
    exercise_success_count = 0
    failed_notebooks = []
    
    for nb_path in notebooks:
        try:
            solution_ok, exercise_ok = process_notebook(nb_path, dry_run=args.dry_run)
            if solution_ok:
                solution_success_count += 1
            if exercise_ok:
                exercise_success_count += 1
            if not (solution_ok and exercise_ok):
                failed_notebooks.append(nb_path)
        except Exception as e:
            print(f"  ✗ Error processing {nb_path.name}: {e}")
            failed_notebooks.append(nb_path)
    
    # Summary
    print("\n" + "=" * 70)
    print("  Summary")
    print("=" * 70)
    print(f"  Total notebooks:      {len(notebooks)}")
    print(f"  Solution created:     {solution_success_count}")
    print(f"  Exercise created:     {exercise_success_count}")
    
    if failed_notebooks:
        print(f"\n  ⚠️  Failed: {len(failed_notebooks)} notebook(s)")
        for nb in failed_notebooks:
            print(f"    - {nb.relative_to(repo_root)}")
    else:
        print(f"\n  ✅ All notebooks processed successfully!")
    
    print("")
    
    return 0 if not failed_notebooks else 1


if __name__ == "__main__":
    sys.exit(main())
