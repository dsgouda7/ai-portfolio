"""
Benchmark Orchestrator

Runs all benchmarks in the correct order and prints a unified PASS/FAIL report.

Usage:
    # Run everything (correctness + latency + accuracy; requires ChromaDB populated):
    python run_benchmarks.py

    # Populate ChromaDB first (runs quick_compress_and_save.py, needs Ollama):
    python run_benchmarks.py --compress-first

    # Also run the end-to-end retrieval benchmark (re-compresses every run):
    python run_benchmarks.py --with-retrieval

Prerequisites:
    - sentence-transformers installed:  pip install sentence-transformers
    - Ollama running for --compress-first / --with-retrieval:
        ollama serve && ollama pull qwen2.5-coder:7b

Benchmark pipeline:
    1. test_correctness.py     -- unit-style, synthetic data, no Azure (always)
    2. latency_comparison.py   -- cache hit vs miss on medium_corpus (needs ChromaDB)
    3. accuracy_benchmarks.py  -- F1/precision/recall on small+medium corpus (needs ChromaDB)
    4. retrieval_benchmarks.py -- full compress+store+query cycle (optional, needs Azure)
"""

import sys
import json
import argparse
import importlib
import subprocess
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent.parent
BENCH_DIR = Path(__file__).parent

# ── Helpers ────────────────────────────────────────────────────────────────────

def section(title: str) -> None:
    print()
    print("=" * 80)
    print(f"  {title}")
    print("=" * 80)


def run_module(module_file: str, label: str) -> dict:
    """Import and run a benchmark module's main function. Returns result dict."""
    module_path = BENCH_DIR / module_file
    spec = importlib.util.spec_from_file_location("_bench_mod", module_path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # Each module exposes a run_*() function
    for fn_name in dir(mod):
        if fn_name.startswith("run_"):
            fn = getattr(mod, fn_name)
            if callable(fn):
                try:
                    result = fn()
                    return result or {}
                except Exception as e:
                    print(f"  [ERROR] {label}: {e}")
                    return {"error": str(e)}
    return {}


# ── Summary table ──────────────────────────────────────────────────────────────

def print_summary(report: dict) -> None:
    section("BENCHMARK SUMMARY")
    print(f"  Run date: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")

    rows = []

    # Correctness
    c = report.get("correctness", {})
    if c:
        rows.append(("Correctness",
                     f"{c.get('passed', '?')}/{c.get('total', '?')} tests",
                     "[PASS]" if c.get("all_pass") else "[FAIL]"))

    # Latency comparison
    lat = report.get("latency", {})
    if lat:
        p = lat.get("pass", {})
        hit_ms  = lat.get("avg_hit_ms", float("inf"))
        miss_ms = lat.get("avg_miss_ms", float("inf"))
        hit_ok  = p.get("cache_hit", False)
        miss_ok = p.get("cache_miss", False)
        rows.append(("Latency: cache hit",
                     f"{hit_ms:.2f}ms (target <=5ms)", "[PASS]" if hit_ok else "[FAIL]"))
        rows.append(("Latency: cache miss",
                     f"{miss_ms:.2f}ms (target <=100ms)", "[PASS]" if miss_ok else "[FAIL]"))

    # Accuracy
    acc = report.get("accuracy", {})
    if acc:
        for corpus_result in acc:
            agg = corpus_result.get("aggregate", {})
            name = corpus_result.get("corpus_name", "corpus")
            avg_f1 = agg.get("avg_f1")
            rows.append((f"F1: {name}",
                         f"{avg_f1:.3f}" if avg_f1 is not None else "N/A",
                         "[PASS]" if avg_f1 and avg_f1 >= 0.60 else "[FAIL]"))

    # Retrieval
    ret = report.get("retrieval", {})
    if ret:
        for corpus_result in ret:
            agg = corpus_result.get("aggregate", {})
            name = corpus_result.get("corpus_name", "corpus")
            miss = agg.get("avg_miss_latency_ms")
            red  = agg.get("avg_context_reduction_pct")
            if miss is not None:
                rows.append((f"Retrieval miss: {name}",
                             f"{miss:.2f}ms", "[PASS]" if miss <= 100.0 else "[FAIL]"))
            if red is not None:
                rows.append((f"Context reduction: {name}",
                             f"{red:.2f}%", "[PASS]" if red >= 95.0 else "[FAIL]"))

    if not rows:
        print("  (no benchmark results to show)")
        return

    col0 = max(len(r[0]) for r in rows) + 2
    col1 = max(len(r[1]) for r in rows) + 2
    for label, value, status in rows:
        print(f"  {label:<{col0}} {value:<{col1}} {status}")

    total_pass = sum(1 for _, _, s in rows if s == "[PASS]")
    print(f"\n  Overall: {total_pass}/{len(rows)} checks PASS")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    import importlib.util  # ensure available inside function

    parser = argparse.ArgumentParser(description="Run all context-optimizer benchmarks")
    parser.add_argument("--compress-first", action="store_true",
                        help="Run quick_compress_and_save.py before tests (needs Ollama)")
    parser.add_argument("--with-retrieval", action="store_true",
                        help="Also run retrieval_benchmarks.py (re-compresses each run, needs Ollama)")
    parser.add_argument("--skip-chroma", action="store_true",
                        help="Skip latency + accuracy tests (don't need ChromaDB)")
    args = parser.parse_args()

    chroma_dir = BENCH_DIR / "chroma_db"
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    sys.path.insert(0, str(PROJECT_ROOT))

    report = {}

    # ── Step 0: Optionally populate ChromaDB ────────────────────────────────
    if args.compress_first:
        section("STEP 0: Compress corpora → ChromaDB (quick_compress_and_save)")
        run_module("quick_compress_and_save.py", "quick_compress_and_save")

    # ── Step 1: Correctness tests (no Azure) ────────────────────────────────
    section("STEP 1: Correctness Tests (synthetic data, no LLM needed)")
    import importlib.util
    spec = importlib.util.spec_from_file_location("tc", BENCH_DIR / "test_correctness.py")
    tc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tc)
    report["correctness"] = tc.run_correctness_tests()

    # ── Step 2: Latency comparison (needs ChromaDB) ──────────────────────────
    if not args.skip_chroma:
        if not chroma_dir.exists() or not any(chroma_dir.iterdir()):
            print(f"\n[WARN] ChromaDB not found at {chroma_dir}")
            print("[WARN] Skipping latency + accuracy tests.")
            print("[HINT] Run with --compress-first to populate ChromaDB first.")
        else:
            section("STEP 2: Latency Comparison (cache hit vs miss)")
            spec2 = importlib.util.spec_from_file_location("lc", BENCH_DIR / "latency_comparison.py")
            lc = importlib.util.module_from_spec(spec2)
            spec2.loader.exec_module(lc)
            report["latency"] = lc.run_latency_comparison() or {}

            section("STEP 3: Accuracy Benchmarks (F1 / Precision / Recall)")
            spec3 = importlib.util.spec_from_file_location("ab", BENCH_DIR / "accuracy_benchmarks.py")
            ab = importlib.util.module_from_spec(spec3)
            spec3.loader.exec_module(ab)
            acc_result = ab.run_accuracy_benchmarks()
            report["accuracy"] = acc_result or []

    # ── Step 4 (optional): Full retrieval benchmark ──────────────────────────
    if args.with_retrieval:
        section("STEP 4: Retrieval Benchmarks (compress + store + query, needs Ollama)")
        spec4 = importlib.util.spec_from_file_location("rb", BENCH_DIR / "retrieval_benchmarks.py")
        rb = importlib.util.module_from_spec(spec4)
        spec4.loader.exec_module(rb)
        rb_result = rb.run_retrieval_benchmarks()
        report["retrieval"] = rb_result.get("results", []) if isinstance(rb_result, dict) else []

    # ── Final summary ────────────────────────────────────────────────────────
    print_summary(report)

    # Save combined report
    report_file = BENCH_DIR / "BENCHMARK_REPORT.json"
    with open(report_file, "w") as f:
        # Strip non-serialisable objects before saving
        def _clean(obj):
            if isinstance(obj, dict):
                return {k: _clean(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_clean(v) for v in obj]
            try:
                json.dumps(obj)
                return obj
            except TypeError:
                return str(obj)

        json.dump({"run_date": datetime.now().isoformat(), **_clean(report)}, f, indent=2)
    print(f"\n[SUCCESS] Full report saved to: {report_file}")

    all_pass = (
        report.get("correctness", {}).get("all_pass", False)
    )
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
