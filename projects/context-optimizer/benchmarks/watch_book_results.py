#!/usr/bin/env python3
"""
Temp watcher: polls for BOOK_RESULTS.JSON and prints a formatted summary
when the benchmark completes.  Run this in a separate terminal alongside
the main benchmark process.

Usage:
    python watch_book_results.py [--interval 30]

Exits automatically once results are printed.
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

_RESULTS = Path(__file__).parent / "BOOK_RESULTS.json"
_LOCK    = Path(__file__).parent / "BENCH_RUNNING.lock"


def _fmt_pct(v: float) -> str:
    return f"{v * 100:.1f}%"


def _fmt_bar(v: float, width: int = 20) -> str:
    filled = round(v * width)
    return "[" + "█" * filled + "·" * (width - filled) + "]"


def report(data: dict) -> None:
    per_book: list[dict] = data.get("per_book", [])
    if not per_book:
        print("No per-book results found.")
        return

    print("\n" + "=" * 65)
    print("  BOOK BENCHMARK RESULTS")
    print("=" * 65)
    print(f"  Run date     : {data.get('run_date', 'unknown')}")
    print(f"  Books        : {data['books']}")
    print(f"  Questions    : {data['total_q']:,}")
    print(f"  Avg KW recall: {_fmt_pct(data['avg_kw_recall'])}  "
          f"{_fmt_bar(data['avg_kw_recall'])}")
    print(f"  Token reduc. : {data['reduction_pct']:.1f}%")
    print()

    # Sort by avg_kw_recall descending
    ranked = sorted(per_book, key=lambda r: -r["avg_kw_recall"])

    print(f"  {'Book':<40} {'Chunks':>6}  {'Qs':>4}  {'KW Recall':>9}  {'Compress':>8}")
    print("  " + "-" * 63)
    for r in ranked:
        title = r["title"][:40]
        bar = _fmt_bar(r["avg_kw_recall"], width=10)
        print(
            f"  {title:<40} {r['chunks']:>6}  {r['questions_run']:>4}  "
            f"{_fmt_pct(r['avg_kw_recall']):>9}  {r['compress_sec']:>7.0f}s"
        )

    print("=" * 65)

    # Top/bottom 5
    print("\n  TOP 5 BY KW RECALL:")
    for r in ranked[:5]:
        print(f"    {r['title'][:50]:<50}  {_fmt_pct(r['avg_kw_recall'])}")

    worst = sorted(per_book, key=lambda r: r["avg_kw_recall"])
    print("\n  BOTTOM 5 BY KW RECALL:")
    for r in worst[:5]:
        print(f"    {r['title'][:50]:<50}  {_fmt_pct(r['avg_kw_recall'])}")

    print()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--interval", type=int, default=30,
                   help="Poll interval in seconds (default: 30)")
    args = p.parse_args()

    print(f"[watcher] Watching for {_RESULTS.name}  (poll every {args.interval}s)")
    print("[watcher] Press Ctrl+C to stop.\n")

    start = time.time()
    while True:
        if _RESULTS.exists():
            try:
                data = json.loads(_RESULTS.read_text(encoding="utf-8"))
                report(data)
                elapsed = time.time() - start
                print(f"[watcher] Results ready after {elapsed / 60:.1f} min.  Exiting.")
                sys.exit(0)
            except json.JSONDecodeError:
                print("[watcher] Results file exists but is not complete yet...")
        else:
            elapsed = time.time() - start
            print(f"[watcher] Waiting... ({elapsed / 60:.1f} min elapsed)")

        time.sleep(args.interval)


if __name__ == "__main__":
    main()
