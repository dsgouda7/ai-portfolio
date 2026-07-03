"""Master script to generate all LLM Fundamentals diagrams.

Generates:
  1. LLM Fundamentals.png (tokenization, sampling, training, context window)
  2. multihead-attention-flow.png (Q/K/V, attention mechanism)
  3. encoder-decoder-comparison.png (attention patterns)
  4. kv-cache-inference.png (autoregressive generation)
  5. model-internals-breakdown.png (parameters, VRAM)
"""
import subprocess
import sys
from pathlib import Path

scripts = [
    "gen-llm-fundamentals.py",
    "gen-multihead-attention.py",
    "gen-encoder-decoder-comparison.py",
    "gen-kv-cache-inference.py",
    "gen-model-internals-breakdown.py",
]

print("=" * 60)
print("Generating all LLM Fundamentals diagrams...")
print("=" * 60)

gen_dir = Path(__file__).parent
failed = []

for script in scripts:
    script_path = gen_dir / script
    if not script_path.exists():
        print(f"⚠ Warning: {script} not found, skipping")
        failed.append(script)
        continue

    print(f"\n→ Running {script}...")
    result = subprocess.run([sys.executable, str(script_path)],
                          capture_output=True, text=True, cwd=gen_dir)

    if result.returncode == 0:
        print(result.stdout.strip())
    else:
        print(f"✗ Failed to generate {script}")
        print(result.stderr)
        failed.append(script)

print("\n" + "=" * 60)
if not failed:
    print(f"✓ Successfully generated all {len(scripts)} diagrams")
else:
    print(f"⚠ {len(failed)} script(s) failed:")
    for s in failed:
        print(f"  - {s}")
print("=" * 60)
