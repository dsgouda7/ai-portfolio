#!/usr/bin/env python3
"""
install.py — one-shot setup for context-optimizer (local, no Docker required).

What it does:
  1. Upgrades pip in the active environment
  2. Installs the context-optimizer package and all benchmark dependencies
  3. Pulls the three Ollama models used by the pipeline

Requirements:
  - Python 3.11+
  - Ollama installed and `ollama serve` running  (https://ollama.com)

Usage:
  # From the project root, with your venv activated:
  python install.py

  # Skip model pulls (e.g. you already have them):
  python install.py --skip-models

  # Pull a different set of models:
  python install.py --models llama3.2:3b,mistral:7b
"""

import argparse
import subprocess
import sys

# ── Default Ollama models used by the pipeline ──────────────────────────────
DEFAULT_MODELS = [
    "llama3.2:1b",  # compression / summarisation (fast, ~600 MB)
    "mistral:7b",  # reasoning + judge (~4 GB, Q4_K_M)
    # nomic-embed-text is optional; default embedding backend is sentence-transformers
    # uncomment to use Ollama embeddings instead:
    # "nomic-embed-text",
]


def run(cmd: list[str], **kwargs) -> int:
    """Run a subprocess, stream output, return exit code."""
    print(f"\n$ {' '.join(cmd)}")
    result = subprocess.run(cmd, **kwargs)
    return result.returncode


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Install context-optimizer dependencies"
    )
    parser.add_argument(
        "--skip-models",
        action="store_true",
        help="Skip ollama model pulls (useful if models are already downloaded)",
    )
    parser.add_argument(
        "--models",
        default=",".join(DEFAULT_MODELS),
        help=f"Comma-separated list of Ollama models to pull (default: {','.join(DEFAULT_MODELS)})",
    )
    args = parser.parse_args()

    python = sys.executable

    # ── Step 1: upgrade pip ─────────────────────────────────────────────────
    print("\n=== Step 1: Upgrade pip ===")
    rc = run([python, "-m", "pip", "install", "--upgrade", "pip"])
    if rc != 0:
        sys.exit(f"pip upgrade failed (exit {rc})")

    # ── Step 2: install package + dependencies ───────────────────────────────
    print("\n=== Step 2: Install context-optimizer ===")
    rc = run([python, "-m", "pip", "install", "-e", ".[evaluation]"])
    if rc != 0:
        sys.exit(f"Package install failed (exit {rc})")

    print("\n=== Step 2b: Install benchmark/runtime dependencies ===")
    rc = run([python, "-m", "pip", "install", "-r", "requirements.txt"])
    if rc != 0:
        sys.exit(f"requirements.txt install failed (exit {rc})")

    # ── Step 3: pull Ollama models ───────────────────────────────────────────
    if args.skip_models:
        print("\n=== Step 3: Skipping model pulls (--skip-models) ===")
    else:
        models = [m.strip() for m in args.models.split(",") if m.strip()]
        print(f"\n=== Step 3: Pull Ollama models: {', '.join(models)} ===")
        print("  (Requires 'ollama serve' to be running — https://ollama.com)")
        for model in models:
            rc = run(["ollama", "pull", model])
            if rc != 0:
                print(
                    f"\n  [WARN] 'ollama pull {model}' failed (exit {rc}).\n"
                    f"  Ensure Ollama is installed and running, then run:\n"
                    f"    ollama pull {model}"
                )

    # ── Done ─────────────────────────────────────────────────────────────────
    print("\n=== Setup complete ===")
    print("  Run benchmarks:")
    print("    python benchmarks/tot/run_experiments.py")
    print()
    print("  Override models via env vars (no code changes needed):")
    print("    CONTEXT_OPTIMIZER_COMPRESSOR_MODEL=llama3.2:1b")
    print("    CONTEXT_OPTIMIZER_REASONING_MODEL=mistral:7b")
    print("    CONTEXT_OPTIMIZER_EMBEDDING_BACKEND=sentence-transformers  # default")


if __name__ == "__main__":
    main()
