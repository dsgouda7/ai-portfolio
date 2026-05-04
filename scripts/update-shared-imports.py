#!/usr/bin/env python3
"""
Batch-update all notes/ gen_scripts that import from _shared.*
to use scripts/shared/ (the new centralised package location).

ML callers:        notes/01-ml/<topic>/<chapter>/gen_scripts/  — parents[3] → parents[5]
MultimodalAI:      notes/05-multimodal_ai/<chapter>/gen_scripts/ — parents[2] → parents[4]
"""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
NOTES_DIR = REPO_ROOT / "notes"

# ------------------------------------------------------------------
# ML pattern: parents[3] points to notes/01-ml/  →  need parents[5] (repo root)
# then sys.path.insert uses ROOT / "scripts"
# ------------------------------------------------------------------
ML_OLD = re.compile(
    r'ROOT = Path\(__file__\)\.resolve\(\)\.parents\[3\][^\n]*\n'
    r'if str\(ROOT\) not in sys\.path:\n'
    r'    sys\.path\.insert\(0, str\(ROOT\)\)\n'
    r'\n'
    r'from _shared\.ml_animation_renderer import render_metric_story',
)

ML_NEW = (
    'ROOT = Path(__file__).resolve().parents[5]\n'
    'if str(ROOT / "scripts") not in sys.path:\n'
    '    sys.path.insert(0, str(ROOT / "scripts"))\n'
    '\n'
    'from shared.animation_renderer import render_metric_story'
)

# ------------------------------------------------------------------
# MultimodalAI pattern: parents[2] points to notes/MultimodalAI/  →  need parents[4]
# ------------------------------------------------------------------
FLOW_OLD = re.compile(
    r'ROOT = Path\(__file__\)\.resolve\(\)\.parents\[2\][^\n]*\n'
    r'if str\(ROOT\) not in sys\.path:\n'
    r'    sys\.path\.insert\(0, str\(ROOT\)\)\n'
    r'\n'
    r'from _shared\.flow_animation import FlowStage, render_flow_animation',
)

FLOW_NEW = (
    'ROOT = Path(__file__).resolve().parents[4]\n'
    'if str(ROOT / "scripts") not in sys.path:\n'
    '    sys.path.insert(0, str(ROOT / "scripts"))\n'
    '\n'
    'from shared.flow_animation import FlowStage, render_flow_animation'
)

updated = 0
warnings = 0

for py_file in sorted(NOTES_DIR.rglob("*.py")):
    content = py_file.read_text(encoding="utf-8")
    new_content = content

    if "from _shared.ml_animation_renderer import render_metric_story" in content:
        new_content, count = ML_OLD.subn(ML_NEW, new_content)
        if count == 0:
            print(f"  WARN (no match): {py_file.relative_to(REPO_ROOT)}")
            warnings += 1

    if "from _shared.flow_animation import FlowStage, render_flow_animation" in content:
        new_content, count = FLOW_OLD.subn(FLOW_NEW, new_content)
        if count == 0:
            print(f"  WARN (no match): {py_file.relative_to(REPO_ROOT)}")
            warnings += 1

    if new_content != content:
        py_file.write_text(new_content, encoding="utf-8")
        print(f"  updated: {py_file.relative_to(REPO_ROOT)}")
        updated += 1

print(f"\n{'='*60}")
print(f"Done: {updated} files updated, {warnings} warnings.")
