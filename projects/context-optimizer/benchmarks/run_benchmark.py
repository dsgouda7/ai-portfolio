"""
Context Optimizer Benchmark dispatcher.

Orchestrates text_corpus_benchmarks.py and/or image_corpus_benchmarks.py.

Usage
-----
    python run_benchmark.py --mode text   [--corpus small|medium|large]
    python run_benchmark.py --mode image  [--corpus small|medium|large]
    python run_benchmark.py --mode all    [--corpus small|medium|large]

Delegates entirely to the modality-specific scripts:
  - benchmarks/text_corpus_benchmarks.py
  - benchmarks/image_corpus_benchmarks.py
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

BENCH_DIR = Path(__file__).parent


def _run(script: str, corpus: str) -> None:
    cmd = [sys.executable, str(BENCH_DIR / script), "--corpus", corpus]
    print(f"\n  Dispatching → {' '.join(cmd)}\n")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"  [warn] {script} exited with code {result.returncode}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Context Optimizer benchmark dispatcher (text | image | all).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--mode",
        choices=["text", "image", "all"],
        default="text",
        help="Which modality benchmark(s) to run (default: text)",
    )
    parser.add_argument(
        "--corpus",
        choices=["small", "medium", "large"],
        default="small",
        help="Corpus size passed to the modality script (default: small)",
    )
    args = parser.parse_args()

    if args.mode in {"text", "all"}:
        _run("text_corpus_benchmarks.py", args.corpus)

    if args.mode in {"image", "all"}:
        _run("image_corpus_benchmarks.py", args.corpus)


if __name__ == "__main__":
    main()


