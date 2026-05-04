"""
remove_ai_infra_duplicate_interview_table.py
Remove duplicate ## 12 · Interview Checklist section in gpu_architecture/gpu-architecture.md.
Keeps the one inside ## 11.5 · Progress Check, deletes the standalone heading that repeats the same content.
"""
import re
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TARGET = os.path.join(REPO_ROOT, "notes", "ai_infrastructure",
                      "gpu_architecture", "gpu-architecture.md")

with open(TARGET, encoding="utf-8") as f:
    content = f.read()

# Find all occurrences of the Interview Checklist heading
pattern = re.compile(r"(^## 12 · Interview Checklist\s*\n)", re.MULTILINE)
matches = list(pattern.finditer(content))

if len(matches) < 2:
    print(f"Found {len(matches)} occurrence(s) — nothing to remove.")
else:
    # Remove the second (and any further) duplicate occurrence
    # Keep the first, remove subsequent ones along with content until next ##-level heading
    # Strategy: split on all matches and remove from second occurrence onward
    first_end = matches[0].end()
    second_start = matches[1].start()

    # Find where the duplicate section ends (next ## heading or EOF)
    rest = content[matches[1].end():]
    next_heading = re.search(r"^## ", rest, re.MULTILINE)
    if next_heading:
        section_end = matches[1].end() + next_heading.start()
    else:
        section_end = len(content)

    new_content = content[:second_start] + content[section_end:]
    with open(TARGET, "w", encoding="utf-8") as f:
        f.write(new_content)
    print(f"Removed duplicate ## 12 · Interview Checklist section from {TARGET}")
