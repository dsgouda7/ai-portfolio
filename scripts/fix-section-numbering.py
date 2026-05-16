#!/usr/bin/env python3
"""
Fix section numbering in markdown files.

Ensures that:
- Main sections (##) are numbered sequentially: 0, 1, 2, 3...
- Subsections (###) match their parent: 1.1, 1.2, etc.
- No gaps or mismatches in numbering

Usage:
    python fix-section-numbering.py <path_to_readme.md> [--dry-run]
    python fix-section-numbering.py notes/03-ai --all [--dry-run]
"""

import argparse
import re
import sys
from pathlib import Path
from typing import List, Tuple


def extract_sections(content: str) -> List[Tuple[int, str, str, str]]:
    """
    Extract all section headers with their line positions.
    Returns: [(line_num, level, current_number, title), ...]
    """
    lines = content.split('\n')
    sections = []

    for i, line in enumerate(lines):
        # Match ## N · Title or ## § N · Title or ### N.M Subtitle
        match_main1 = re.match(r'^(##)\s+(\d+)\s+[·•]\s+(.+)$', line)
        match_main2 = re.match(r'^(##)\s+§\s+(\d+)\s+[·•]\s+(.+)$', line)
        match_sub = re.match(r'^(###)\s+(\d+\.\d+)\s+(.+)$', line)

        if match_main1:
            level = match_main1.group(1)
            number = match_main1.group(2)
            title = match_main1.group(3)
            sections.append((i, level, number, title, line))
        elif match_main2:
            level = match_main2.group(1)
            number = match_main2.group(2)
            title = match_main2.group(3)
            sections.append((i, level, number, title, line))
        elif match_sub:
            level = match_sub.group(1)
            number = match_sub.group(2)
            title = match_sub.group(3)
            sections.append((i, level, number, title, line))

    return sections


def check_numbering(sections: List[Tuple[int, str, str, str, str]]) -> List[dict]:
    """
    Check for numbering issues.
    Returns list of issues found.
    """
    issues = []
    expected_main = 0
    current_main = None
    expected_sub = 1

    for line_num, level, number, title, full_line in sections:
        if level == '##':
            # Main section
            try:
                actual = int(number)
            except ValueError:
                issues.append({
                    'line': line_num + 1,
                    'type': 'invalid_main',
                    'message': f"Invalid main section number: {number}",
                    'current': full_line
                })
                continue

            if actual != expected_main:
                issues.append({
                    'line': line_num + 1,
                    'type': 'main_mismatch',
                    'message': f"Expected § {expected_main}, found § {actual}",
                    'current': full_line,
                    'expected': expected_main,
                    'actual': actual
                })

            current_main = actual
            expected_main = actual + 1
            expected_sub = 1  # Reset subsection counter

        elif level == '###':
            # Subsection
            try:
                parent, sub = number.split('.')
                parent_num = int(parent)
                sub_num = int(sub)
            except (ValueError, AttributeError):
                issues.append({
                    'line': line_num + 1,
                    'type': 'invalid_sub',
                    'message': f"Invalid subsection number: {number}",
                    'current': full_line
                })
                continue

            if current_main is not None and parent_num != current_main:
                issues.append({
                    'line': line_num + 1,
                    'type': 'sub_parent_mismatch',
                    'message': f"Subsection {number} doesn't match parent § {current_main}",
                    'current': full_line,
                    'expected_parent': current_main,
                    'actual_parent': parent_num
                })

            if sub_num != expected_sub:
                issues.append({
                    'line': line_num + 1,
                    'type': 'sub_sequence',
                    'message': f"Expected {current_main}.{expected_sub}, found {number}",
                    'current': full_line,
                    'expected': f"{current_main}.{expected_sub}",
                    'actual': number
                })

            expected_sub = sub_num + 1

    return issues


def fix_numbering(content: str) -> Tuple[str, List[str]]:
    """
    Fix section numbering issues.
    Returns (fixed_content, list_of_changes)
    """
    lines = content.split('\n')
    changes = []

    expected_main = 0
    current_main = None
    current_sub = 0

    for i, line in enumerate(lines):
        # Match main sections (both "## N ·" and "## § N ·" formats)
        match_main1 = re.match(r'^(##)\s+(\d+)\s+([·•])\s+(.+)$', line)
        match_main2 = re.match(r'^(##)\s+(§)\s+(\d+)\s+([·•])\s+(.+)$', line)

        if match_main1:
            level = match_main1.group(1)
            old_num = int(match_main1.group(2))
            separator = match_main1.group(3)
            title = match_main1.group(4)

            # Fix main section numbering if needed
            if old_num != expected_main:
                new_line = f"{level} {expected_main} {separator} {title}"
                changes.append(f"Line {i+1}: § {old_num} → § {expected_main}")
                lines[i] = new_line
                current_main = expected_main
            else:
                current_main = old_num

            expected_main = current_main + 1
            current_sub = 0
            continue

        elif match_main2:
            level = match_main2.group(1)
            section_symbol = match_main2.group(2)  # "§"
            old_num = int(match_main2.group(3))
            separator = match_main2.group(4)
            title = match_main2.group(5)

            # Fix main section numbering if needed
            if old_num != expected_main:
                new_line = f"{level} {section_symbol} {expected_main} {separator} {title}"
                changes.append(f"Line {i+1}: § {old_num} → § {expected_main}")
                lines[i] = new_line
                current_main = expected_main
            else:
                current_main = old_num

            expected_main = current_main + 1
            current_sub = 0
            continue

        # Match subsections
        match_sub = re.match(r'^(###)\s+(\d+)\.(\d+)\s+(.+)$', line)
        if match_sub:
            level = match_sub.group(1)
            old_parent = int(match_sub.group(2))
            old_sub = int(match_sub.group(3))
            title = match_sub.group(4)

            if current_main is not None and old_parent != current_main:
                # Fix parent number
                current_sub += 1
                new_line = f"{level} {current_main}.{current_sub} {title}"
                changes.append(f"Line {i+1}: {old_parent}.{old_sub} → {current_main}.{current_sub}")
                lines[i] = new_line
            else:
                current_sub = old_sub

    return '\n'.join(lines), changes


def process_file(filepath: Path, dry_run: bool = False) -> bool:
    """Process a single markdown file. Returns True if issues found."""
    if not filepath.exists():
        print(f"❌ File not found: {filepath}")
        return False

    content = filepath.read_text(encoding='utf-8')
    sections = extract_sections(content)

    if not sections:
        print(f"ℹ️  No numbered sections found in {filepath.name}")
        return False

    issues = check_numbering(sections)

    if not issues:
        print(f"✅ {filepath.name}: All numbering correct")
        return False

    print(f"\n🔍 {filepath.name}: Found {len(issues)} issue(s)")
    for issue in issues:
        print(f"  Line {issue['line']}: {issue['message']}")
        print(f"    Current: {issue['current']}")

    if not dry_run:
        fixed_content, changes = fix_numbering(content)
        if changes:
            filepath.write_text(fixed_content, encoding='utf-8')
            print(f"✏️  Applied {len(changes)} fix(es):")
            for change in changes:
                print(f"    {change}")
        else:
            print(f"⚠️  Issues detected but couldn't auto-fix (manual review needed)")
    else:
        print(f"  [DRY RUN - no changes made]")

    return True


def main():
    parser = argparse.ArgumentParser(description='Fix section numbering in markdown files')
    parser.add_argument('path', help='Path to README.md or directory containing chapters')
    parser.add_argument('--all', action='store_true', help='Process all chapters in directory')
    parser.add_argument('--dry-run', action='store_true', help='Check only, do not modify files')

    args = parser.parse_args()
    path = Path(args.path)

    if args.all:
        if not path.is_dir():
            print(f"❌ --all requires a directory path")
            sys.exit(1)

        readme_files = sorted(path.glob('ch*/README.md'))
        if not readme_files:
            print(f"❌ No chapter READMEs found in {path}")
            sys.exit(1)

        print(f"📚 Processing {len(readme_files)} chapters...")
        issues_found = 0
        for readme in readme_files:
            if process_file(readme, args.dry_run):
                issues_found += 1

        print(f"\n{'='*60}")
        if issues_found == 0:
            print(f"✅ All {len(readme_files)} chapters have correct numbering")
        else:
            print(f"⚠️  Issues found in {issues_found}/{len(readme_files)} chapters")
            if args.dry_run:
                print(f"💡 Run without --dry-run to apply fixes")
    else:
        if path.is_dir():
            readme = path / 'README.md'
        else:
            readme = path

        process_file(readme, args.dry_run)


if __name__ == '__main__':
    main()
