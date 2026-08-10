"""Validate the legacy generated TF evaluation notebooks."""
import json, ast
from pathlib import Path

ROOT = Path(r"c:\r\ai-portfolio\learning\genai")

paths = [
    ROOT / "11-llm-evaluation" / "01-llm-evaluation-metrics-and-benchmarks.ipynb",
    ROOT / "11-llm-evaluation" / "02-llm-as-judge-safety-and-pipeline.ipynb",
    ROOT / "11-llm-evaluation" / "03-hallucination-detection.ipynb",
    ROOT / "11-llm-evaluation" / "04-calibration-and-confidence.ipynb",
]

all_ok = True
for p in paths:
    nb = json.load(open(p, encoding='utf-8'))
    errors = []

    # nbformat
    if nb['nbformat'] != 4 or nb['nbformat_minor'] != 5:
        errors.append(f'nbformat={nb["nbformat"]}.{nb["nbformat_minor"]}')

    # Unique cell IDs
    ids = [c['id'] for c in nb['cells']]
    if len(ids) != len(set(ids)):
        errors.append('duplicate cell IDs')

    # Syntax-check all code cells
    for i, c in enumerate(nb['cells']):
        if c['cell_type'] == 'code':
            src = ''.join(c['source'])
            try:
                ast.parse(src)
            except SyntaxError as e:
                errors.append(f'syntax error cell {i}: {e}')

    # No PyTorch artifacts
    all_code = '\n'.join(''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code')
    if 'import torch' in all_code:
        errors.append('has import torch')
    if 'TFTFGPT2' in all_code:
        errors.append('double-TF prefix TFTFGPT2')
    if "return_tensors='pt'" in all_code:
        errors.append("has return_tensors='pt'")
    if 'SentenceTransformer(' in all_code:
        errors.append('still has SentenceTransformer(')

    status = 'OK' if not errors else 'FAIL: ' + str(errors)
    if errors:
        all_ok = False
    print(f'{p.name}  ({len(nb["cells"])} cells): {status}')

print('\nAll OK' if all_ok else '\nSome notebooks failed validation')
