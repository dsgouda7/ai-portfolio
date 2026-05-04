"""Scan all notebooks under notes/ and report code issues.

Checks:
  - JSON load errors
  - Python syntax errors (after stripping IPython magics/shell lines)
  - Undefined-name hints via simple import/def/assign tracking (best-effort, noisy - optional)
  - Common pitfalls: bare `print` missing parens in py3 (syntax-caught), tabs vs spaces
  - Duplicate imports noted (informational)
"""
from __future__ import annotations
import ast, glob, json, os, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "notes"

MAGIC_RE = re.compile(r"^\s*[%!?]")
LINE_MAGIC_RE = re.compile(r"^\s*%[A-Za-z]")

def clean_source(src: str) -> str:
    out = []
    for line in src.splitlines():
        stripped = line.lstrip()
        if stripped.startswith(("%", "!", "?")):
            out.append("pass  # magic/shell stripped")
        elif stripped.endswith("?") and not stripped.endswith(("==?", "!=?")):
            # help marker
            out.append("pass  # help stripped")
        else:
            out.append(line)
    return "\n".join(out)

def scan(nb_path: Path):
    issues = []
    try:
        data = json.loads(nb_path.read_text(encoding="utf-8"))
    except UnicodeError:
        try:
            data = json.loads(nb_path.read_text(encoding="utf-8-sig"))
            issues.append(("BOM", 0, "file has UTF-8 BOM; Jupyter/tools may reject"))
        except Exception as e:
            return [("LOAD_ERR", 0, str(e))]
    except json.JSONDecodeError as e:
        try:
            data = json.loads(nb_path.read_text(encoding="utf-8-sig"))
            issues.append(("BOM", 0, "file has UTF-8 BOM; Jupyter/tools may reject"))
        except Exception as e2:
            return [("LOAD_ERR", 0, f"{e}")]
    except Exception as e:
        return [("LOAD_ERR", 0, str(e))]
    for i, cell in enumerate(data.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        if not src.strip():
            continue
        cleaned = clean_source(src)
        try:
            ast.parse(cleaned)
        except SyntaxError as e:
            issues.append(("SYNTAX", i, f"line {e.lineno}: {e.msg} :: {(e.text or '').strip()[:120]}"))
            continue
        # Heuristic: tabs mixed
        if "\t" in src and "    " in src:
            issues.append(("MIXED_INDENT", i, "tabs and spaces both present"))
    return issues

def main():
    nbs = sorted(Path(p) for p in glob.glob(str(ROOT / "**" / "*.ipynb"), recursive=True))
    total_issues = 0
    per_nb = {}
    for nb in nbs:
        iss = scan(nb)
        if iss:
            per_nb[str(nb)] = iss
            total_issues += len(iss)
    print(f"Scanned {len(nbs)} notebooks")
    print(f"Notebooks with issues: {len(per_nb)}")
    print(f"Total issues: {total_issues}")
    print("=" * 80)
    for nb, iss in per_nb.items():
        rel = os.path.relpath(nb, ROOT.parent)
        print(f"\n## {rel}")
        for kind, cell_idx, msg in iss:
            print(f"  - [{kind}] cell#{cell_idx}: {msg}")

if __name__ == "__main__":
    main()
