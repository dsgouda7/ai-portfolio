"""
insert_ai_infra_stub_sections.py
Insert missing stub sections into all 5 AI Infrastructure chapter .md files.
Idempotent — skips any section that already exists.

Sections inserted per file:
  ## Animation     (after ## 0 · The Challenge)
  ## 5 · Key Diagrams
  ## 6 · The Hyperparameter Dial  (already present in some; skip if found)
  ## 7 · Code Skeleton            (already present in some; skip if found)
  ## Where This Reappears         (already present in some; skip if found)
"""
import re
import os
import glob

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AI_INFRA_ROOT = os.path.join(REPO_ROOT, "notes", "ai_infrastructure")

CHAPTERS = [
    "gpu_architecture",
    "inference_optimization",
    "memory_and_compute_budgets",
    "parallelism_and_distributed_training",
    "quantization_and_precision",
]

def find_chapter_md(chapter_dir):
    """Return the primary .md file in a chapter dir (not README track-level)."""
    candidates = glob.glob(os.path.join(AI_INFRA_ROOT, chapter_dir, "*.md"))
    return candidates[0] if candidates else None


ANIMATION_STUB = """\n## Animation\n
> 🎬 *Animation placeholder — needle-builder agent will generate this.*\n
"""

KEY_DIAGRAMS_STUB = """\n## 5 · Key Diagrams\n
> Add 2–3 diagrams showing the key data flows or architectural boundaries here.\n
"""

HYPERPARAMETER_DIAL_STUB = """\n## 6 · The Hyperparameter Dial\n
> List 3–5 dials (batch size, precision, parallelism strategy, etc.) and their\n> effect on the latency/throughput/memory triangle.\n
"""

CODE_SKELETON_STUB = """\n## 7 · Code Skeleton\n
### Educational\n
```python\n# Educational: concept from scratch\npass\n```\n
### Production\n
```python\n# Production: optimized pipeline call\npass\n```\n
"""

WHERE_REAPPEARS_STUB = """\n## Where This Reappears\n
> Link to downstream chapters or tracks that build on this concept.\n
"""

# Anchors
CHALLENGE_HEADING = re.compile(r"(^## 0 · The Challenge[^\n]*\n)", re.MULTILINE)
# We look for the Progress Check to anchor section insertions before it
PC_PATTERN = re.compile(r"(^## \d+ · (?:Progress Check|Interview Checklist)[^\n]*\n)", re.MULTILINE)


def ensure_section_after(content, new_section_header, stub, anchor_pattern):
    """Insert stub after the first line matching anchor_pattern, unless header already in content."""
    if new_section_header in content:
        return content, False
    match = anchor_pattern.search(content)
    if not match:
        return content, False
    insert_pos = match.end()
    return content[:insert_pos] + stub + content[insert_pos:], True


def ensure_section_before_pc(content, new_section_header, stub):
    """Insert stub before the Progress Check or Interview Checklist section."""
    if new_section_header in content:
        return content, False
    match = PC_PATTERN.search(content)
    if not match:
        # Append at end
        return content.rstrip() + "\n\n" + stub, True
    insert_pos = match.start()
    return content[:insert_pos] + stub + content[insert_pos:], True


for chapter_dir in CHAPTERS:
    filepath = find_chapter_md(chapter_dir)
    if not filepath:
        print(f"SKIP (no .md file): {chapter_dir}")
        continue

    with open(filepath, encoding="utf-8") as f:
        content = f.read()

    changed = False

    content, c = ensure_section_after(content, "## Animation", ANIMATION_STUB, CHALLENGE_HEADING)
    changed |= c

    content, c = ensure_section_before_pc(content, "## 5 · Key Diagrams", KEY_DIAGRAMS_STUB)
    changed |= c

    content, c = ensure_section_before_pc(content, "## 6 · The Hyperparameter Dial", HYPERPARAMETER_DIAL_STUB)
    changed |= c

    content, c = ensure_section_before_pc(content, "## 7 · Code Skeleton", CODE_SKELETON_STUB)
    changed |= c

    content, c = ensure_section_before_pc(content, "## Where This Reappears", WHERE_REAPPEARS_STUB)
    changed |= c

    if changed:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Updated stubs in {os.path.relpath(filepath, REPO_ROOT)}")
    else:
        print(f"All stubs already present in {os.path.relpath(filepath, REPO_ROOT)}")

print("Done.")
