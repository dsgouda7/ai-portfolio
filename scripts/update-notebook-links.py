#!/usr/bin/env python
"""Update markdown files to reference new exercise/solution notebook naming.

Transforms:
  - notebook.ipynb → notebook_solution.ipynb | notebook_exercise.ipynb
  - notebook_supplement.ipynb → notebook_supplement_solution.ipynb | notebook_supplement_exercise.ipynb
  - grand_solution.ipynb → grand_solution_reference.ipynb | grand_solution_exercise.ipynb

Usage:
    python scripts/update_notebook_links.py
    python scripts/update_notebook_links.py --filter "notes/05-multimodal_ai/**/*.md"
    python scripts/update_notebook_links.py --dry-run

Author: AI Portfolio Maintenance
Date: April 28, 2026
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path
from typing import List, Tuple


def backup_file(file_path: Path) -> Path:
    """Create backup of file before modification."""
    backup_path = file_path.with_suffix(file_path.suffix + ".backup")
    shutil.copy2(file_path, backup_path)
    return backup_path


def update_markdown_links(content: str) -> Tuple[str, int]:
    """Update notebook references in markdown content.
    
    Returns:
        (updated_content, num_replacements)
    """
    original = content
    replacements = 0
    
    # Pattern 1: [text](notebook.ipynb) → [text (solution)](notebook_solution.ipynb) | [text (exercise)](notebook_exercise.ipynb)
    pattern1 = r'\[([^\]]+)\]\((notebook(?:_supplement)?\.ipynb)\)'
    replacement1 = r'[\1 (solution)](\2_solution.ipynb) | [\1 (exercise)](\2_exercise.ipynb)'
    content, count1 = re.subn(pattern1, replacement1, content)
    replacements += count1
    
    # Pattern 2: [text](grand_solution.ipynb) → [text (reference)](grand_solution_reference.ipynb) | [text (exercise)](grand_solution_exercise.ipynb)
    pattern2 = r'\[([^\]]+)\]\(grand_solution\.ipynb\)'
    replacement2 = r'[\1 (reference)](grand_solution_reference.ipynb) | [\1 (exercise)](grand_solution_exercise.ipynb)'
    content, count2 = re.subn(pattern2, replacement2, content)
    replacements += count2
    
    # Pattern 3: `notebook.ipynb` → `notebook_solution.ipynb` (reference) or `notebook_exercise.ipynb` (practice)
    pattern3 = r'`(notebook(?:_supplement)?\.ipynb)`'
    replacement3 = r'`\1_solution.ipynb` (reference) or `\1_exercise.ipynb` (practice)'
    content, count3 = re.subn(pattern3, replacement3, content)
    replacements += count3
    
    # Pattern 4: `grand_solution.ipynb` → `grand_solution_reference.ipynb` (reference) or `grand_solution_exercise.ipynb` (practice)
    pattern4 = r'`grand_solution\.ipynb`'
    replacement4 = r'`grand_solution_reference.ipynb` (reference) or `grand_solution_exercise.ipynb` (practice)'
    content, count4 = re.subn(pattern4, replacement4, content)
    replacements += count4
    
    return content, replacements


def process_markdown_file(md_path: Path, dry_run: bool = False, create_backup: bool = True) -> int:
    """Process a single markdown file.
    
    Returns:
        Number of replacements made
    """
    try:
        content = md_path.read_text(encoding="utf-8")
        updated_content, num_replacements = update_markdown_links(content)
        
        if num_replacements > 0:
            if dry_run:
                print(f"  [DRY RUN] Would update: {md_path.relative_to(md_path.parents[2])} ({num_replacements} replacements)")
            else:
                if create_backup:
                    backup_file(md_path)
                md_path.write_text(updated_content, encoding="utf-8")
                print(f"  ✓ Updated: {md_path.relative_to(md_path.parents[2])} ({num_replacements} replacements)")
        
        return num_replacements
    
    except Exception as e:
        print(f"  ✗ Error processing {md_path.name}: {e}")
        return 0


def discover_markdown_files(repo_root: Path, filter_pattern: str = None) -> List[Path]:
    """Discover all markdown files to process."""
    if filter_pattern:
        # Use filter pattern
        md_files = list(repo_root.glob(filter_pattern))
    else:
        # Default: all markdown files in notes/, README.md, CONTRIBUTING.md
        md_files = list((repo_root / "notes").rglob("*.md"))
        md_files.append(repo_root / "README.md")
        md_files.append(repo_root / "CONTRIBUTING.md")
    
    # Filter out checkpoints and backups
    md_files = [
        f for f in md_files 
        if f.exists() and ".ipynb_checkpoints" not in f.parts and not f.suffix == ".backup"
    ]
    
    return sorted(md_files)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Update notebook references in markdown files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Update all markdown files
  python scripts/update_notebook_links.py

  # Update only multimodal AI track
  python scripts/update_notebook_links.py --filter "notes/05-multimodal_ai/**/*.md"

  # Dry run (show what would be changed)
  python scripts/update_notebook_links.py --dry-run

  # Skip backups (not recommended)
  python scripts/update_notebook_links.py --no-backup
        """
    )
    parser.add_argument(
        "--filter",
        type=str,
        help="Glob pattern to filter markdown files"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be changed without modifying files"
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Skip creating backup files (not recommended)"
    )
    args = parser.parse_args()
    
    # Determine repo root
    repo_root = Path(__file__).resolve().parent.parent
    
    print("\n" + "=" * 70)
    print("  Notebook Link Updater")
    print("=" * 70)
    
    # Discover markdown files
    print(f"\n[*] Discovering markdown files...")
    md_files = discover_markdown_files(repo_root, args.filter)
    
    if not md_files:
        print("  No markdown files found matching criteria.")
        return 0
    
    print(f"  Found {len(md_files)} markdown file(s) to process")
    
    if args.dry_run:
        print("\n[!] DRY RUN MODE - No files will be modified")
    
    if not args.no_backup and not args.dry_run:
        print("\n[*] Backup files will be created (.md.backup)")
    
    # Process markdown files
    print(f"\n[*] Processing markdown files...\n")
    
    total_replacements = 0
    files_modified = 0
    
    for md_path in md_files:
        replacements = process_markdown_file(
            md_path, 
            dry_run=args.dry_run,
            create_backup=not args.no_backup
        )
        total_replacements += replacements
        if replacements > 0:
            files_modified += 1
    
    # Summary
    print("\n" + "=" * 70)
    print("  Summary")
    print("=" * 70)
    print(f"  Total markdown files:     {len(md_files)}")
    print(f"  Files modified:           {files_modified}")
    print(f"  Total replacements:       {total_replacements}")
    
    if args.dry_run:
        print(f"\n  [i] Run without --dry-run to apply changes")
    elif files_modified > 0:
        print(f"\n  [+] All markdown files updated successfully!")
        if not args.no_backup:
            print(f"  [*] Backups saved as *.md.backup")
    else:
        print(f"\n  [i] No notebook references found to update")
    
    print("")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
