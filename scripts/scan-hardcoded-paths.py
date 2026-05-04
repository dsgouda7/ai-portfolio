"""Scan gen_scripts Python files for hardcoded absolute output paths."""
import os, re

repo = r'C:\repos\ai-portfolio'
results = []

for root, dirs, files in os.walk(repo):
    # skip irrelevant subtrees
    if '.venv' in root or '__pycache__' in root:
        continue
    rel = root.replace(repo, '').replace('\\', '/')
    if '/gen_scripts' not in rel:
        continue
    for f in files:
        if not f.endswith('.py'):
            continue
        fp = os.path.join(root, f)
        try:
            text = open(fp, encoding='utf-8', errors='replace').read()
        except Exception:
            continue
        # Detect absolute Windows paths in any assignment or string
        abs_matches = re.findall(r'[rf]?"(C:[^"]+)"', text) + re.findall(r"[rf]?'(C:[^']+)'", text)
        if abs_matches:
            results.append((fp, abs_matches))

if results:
    print(f'FOUND {len(results)} file(s) with hardcoded absolute paths:')
    for fp, matches in results:
        rel = fp.replace(repo, '').replace('\\', '/')
        print(f'\n  FILE: {rel}')
        for m in matches:
            print(f'    PATH: {m}')
else:
    print('NO_HARDCODED_ABSOLUTE_PATHS_IN_GEN_SCRIPTS')
