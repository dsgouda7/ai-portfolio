"""
add_multimodal_stub_sections.py
Insert stub sections ## 10 · Further Reading and ## 11 · Notebook
between the Bridge section and Progress Check in all 13 Multimodal AI chapter READMEs.
"""
import re
import os
import glob

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MM_ROOT = os.path.join(REPO_ROOT, "notes", "multimodal_ai")

MD_FILES = [
    f for f in glob.glob(os.path.join(MM_ROOT, "**", "README.md"), recursive=True)
    if os.path.dirname(f) != MM_ROOT
]

FURTHER_READING_STUB = """\n## 10 · Further Reading\n
> *Add 3–5 key papers/resources with one-line annotations here.*\n
"""

NOTEBOOK_STUB = """\n## 11 · Notebook\n
> See `notebook.ipynb` in this chapter's folder for interactive exercises.\n
"""

# Anchor: insert before Progress Check (either 8.5 or 11.5 variant)
PC_PATTERN = re.compile(r"(^## (?:8\.5|11\.5) · Progress Check)", re.MULTILINE)

for filepath in MD_FILES:
    with open(filepath, encoding="utf-8") as f:
        content = f.read()

    further_missing = "## 10 · Further Reading" not in content
    notebook_missing = "## 11 · Notebook" not in content

    if not further_missing and not notebook_missing:
        print(f"Both stubs already present: {os.path.relpath(filepath, REPO_ROOT)}")
        continue

    match = PC_PATTERN.search(content)
    if not match:
        print(f"No Progress Check anchor found in {os.path.relpath(filepath, REPO_ROOT)}")
        continue

    insert_pos = match.start()
    stubs = ""
    if further_missing:
        stubs += FURTHER_READING_STUB
    if notebook_missing:
        stubs += NOTEBOOK_STUB

    new_content = content[:insert_pos] + stubs + content[insert_pos:]
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(new_content)
    print(f"Inserted stubs in {os.path.relpath(filepath, REPO_ROOT)}")

print("Done.")
