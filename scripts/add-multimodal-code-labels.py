"""
add_multimodal_code_labels.py
Prepend "# Educational: [concept] from scratch" or "# Production: pipeline call"
to the first line of each code block in all 13 Multimodal AI chapter READMEs.

Heuristic:
  - Block already has a label starting with "# Educational:" or "# Production:" -> skip
  - Block contains "pipe(" or "pipeline(" or "from diffusers" as the first/second line -> Production
  - Otherwise -> Educational
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

PRODUCTION_SIGNALS = ["pipe(", "pipeline(", "from diffusers", "StableDiffusionPipeline",
                       ".to(\"cuda\")", "DiffusionPipeline.from_pretrained"]


def label_for_block(code_body: str) -> str:
    first_lines = code_body.split("\n")[:5]
    joined = "\n".join(first_lines)
    for sig in PRODUCTION_SIGNALS:
        if sig in joined:
            return "# Production: diffusers pipeline"
    return "# Educational: concept from scratch"


def process_file(filepath: str) -> None:
    with open(filepath, encoding="utf-8") as f:
        content = f.read()

    # Match fenced code blocks: ```python ... ``` or ``` ... ```
    FENCE_RE = re.compile(r"(```(?:python)?\n)(.*?)(```)", re.DOTALL)

    def replacer(m):
        fence_open = m.group(1)
        body = m.group(2)
        fence_close = m.group(3)
        first_line = body.lstrip("\n").split("\n")[0] if body.strip() else ""
        if first_line.startswith("# Educational:") or first_line.startswith("# Production:"):
            return m.group(0)  # Already labeled
        label = label_for_block(body)
        return fence_open + label + "\n" + body + fence_close

    new_content = FENCE_RE.sub(replacer, content)
    if new_content != content:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(new_content)
        print(f"Labeled code blocks in {os.path.relpath(filepath, REPO_ROOT)}")
    else:
        print(f"No unlabeled code blocks found in {os.path.relpath(filepath, REPO_ROOT)}")


for filepath in MD_FILES:
    process_file(filepath)

print("Done.")
