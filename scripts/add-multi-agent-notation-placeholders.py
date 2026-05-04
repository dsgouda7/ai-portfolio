"""
add_multi_agent_notation_placeholders.py
Insert a notation placeholder comment at the end of the opening blockquote
in each of the 7 Multi-Agent AI chapter README.md files.

Inserts:  <!-- TODO: notation sentence — define symbols used in chapter -->
after the last line of the opening blockquote (the first > block in the file).
"""
import re
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MA_ROOT = os.path.join(REPO_ROOT, "notes", "04-multi_agent_ai")

CHAPTERS = [
    "ch01_message_formats", "ch02_mcp", "ch03_a2a", "ch04_event_driven_agents",
    "ch05_shared_memory", "ch06_trust_and_sandboxing", "ch07_agent_frameworks",
]

PLACEHOLDER = "<!-- TODO: notation sentence — define symbols used in chapter -->"

for chapter in CHAPTERS:
    filepath = os.path.join(MA_ROOT, chapter, "README.md")
    if not os.path.exists(filepath):
        print(f"SKIP (not found): {filepath}")
        continue

    with open(filepath, encoding="utf-8") as f:
        lines = f.readlines()

    # Already present?
    if any(PLACEHOLDER in line for line in lines):
        print(f"Placeholder already present in {chapter}/README.md")
        continue

    # Find the end of the first blockquote block (lines starting with ">")
    in_blockquote = False
    insert_after = None
    for i, line in enumerate(lines):
        if line.startswith(">"):
            in_blockquote = True
            insert_after = i
        elif in_blockquote:
            # First line NOT starting with ">" after the blockquote started
            break

    if insert_after is None:
        print(f"No blockquote found in {chapter}/README.md")
        continue

    lines.insert(insert_after + 1, PLACEHOLDER + "\n")
    with open(filepath, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print(f"Inserted notation placeholder in {chapter}/README.md after line {insert_after + 1}")
