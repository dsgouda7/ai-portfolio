"""
Unified ToT Benchmark Runner

Runs all ToT benchmarks (text + multimedia) and saves results to temp markdown files.
Results will be consolidated later into final reports.
"""

import time
from pathlib import Path
from datetime import datetime


TEMP_DIR = Path(__file__).parent / "temp_results"
TEMP_DIR.mkdir(exist_ok=True)


def run_all_tot_benchmarks():
    """Run all ToT benchmarks and generate temp markdown files."""

    print("=" * 100)
    print("UNIFIED ToT BENCHMARK SUITE")
    print("=" * 100)
    print(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Results directory: {TEMP_DIR}")
    print()

    master_log = []

    # Phase 1: Text benchmarks
    print("\n" + "▓" * 100)
    print("PHASE 1: TEXT CORPUS BENCHMARKS")
    print("▓" * 100)

    try:
        from run_fast_tot_benchmarks import run_fast_tot_benchmarks

        start = time.time()
        text_results = run_fast_tot_benchmarks()
        duration = time.time() - start

        master_log.append({
            "phase": "text",
            "status": "success" if text_results else "failed",
            "duration_sec": duration,
            "test_count": len(text_results) if text_results else 0
        })

        print(f"\n✓ Text benchmarks completed in {duration:.1f}s")

    except Exception as e:
        print(f"\n✗ Text benchmarks failed: {e}")
        master_log.append({
            "phase": "text",
            "status": "error",
            "error": str(e)
        })

    # Phase 2: Multimedia benchmarks
    print("\n" + "▓" * 100)
    print("PHASE 2: MULTIMEDIA CORPUS BENCHMARKS")
    print("▓" * 100)

    try:
        from run_multimedia_tot_benchmarks import run_multimedia_tot_benchmarks

        start = time.time()
        multimedia_results = run_multimedia_tot_benchmarks()
        duration = time.time() - start

        test_count = sum(len(results) for results in multimedia_results.values()) if multimedia_results else 0

        master_log.append({
            "phase": "multimedia",
            "status": "success" if multimedia_results else "failed",
            "duration_sec": duration,
            "test_count": test_count
        })

        print(f"\n✓ Multimedia benchmarks completed in {duration:.1f}s")

    except Exception as e:
        print(f"\n✗ Multimedia benchmarks failed: {e}")
        master_log.append({
            "phase": "multimedia",
            "status": "error",
            "error": str(e)
        })

    # Generate master summary
    print("\n" + "=" * 100)
    print("GENERATING MASTER SUMMARY")
    print("=" * 100)

    summary_file = TEMP_DIR / f"MASTER_SUMMARY_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"

    with open(summary_file, 'w') as f:
        f.write("# ToT Benchmark Master Summary\n\n")
        f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## Execution Log\n\n")
        f.write("| Phase | Status | Duration | Tests |\n")
        f.write("|-------|--------|----------|-------|\n")

        total_duration = 0
        total_tests = 0

        for log in master_log:
            status_icon = "✅" if log["status"] == "success" else "❌"
            duration = log.get("duration_sec", 0)
            tests = log.get("test_count", 0)

            total_duration += duration
            total_tests += tests

            f.write(f"| {log['phase'].title()} | {status_icon} {log['status']} | {duration:.1f}s | {tests} |\n")

            if "error" in log:
                f.write(f"| | Error: {log['error']} | | |\n")

        f.write(f"| **TOTAL** | | **{total_duration:.1f}s** | **{total_tests}** |\n\n")

        # List temp files
        f.write("## Generated Temp Files\n\n")
        temp_files = sorted(TEMP_DIR.glob("*.md"))

        for temp_file in temp_files:
            if temp_file != summary_file:
                f.write(f"- `{temp_file.name}`\n")

        json_files = sorted(TEMP_DIR.glob("*.json"))
        if json_files:
            f.write("\n**JSON Data Files**:\n")
            for json_file in json_files:
                f.write(f"- `{json_file.name}`\n")

        f.write("\n## Next Steps\n\n")
        f.write("1. Review individual temp markdown files for detailed results\n")
        f.write("2. Consolidate results into final experiment reports\n")
        f.write("3. Update EXPERIMENTS_GUIDE.md with ToT benchmark findings\n")
        f.write("4. Add ToT results to ARCHITECTURE_EVOLUTION.md\n")
        f.write("5. Consider production integration if hypothesis validated\n\n")

        f.write("## Success Criteria Check\n\n")
        f.write("ToT should meet the following criteria:\n\n")
        f.write("- [ ] F1 improvement > +0.05 (5% better quality)\n")
        f.write("- [ ] Token ratio < 3x (acceptable overhead)\n")
        f.write("- [ ] Deduplication > 20% (proves multi-perspective efficiency)\n")
        f.write("- [ ] Scaling: quality improvement increases with corpus size\n\n")

    print(f"  ✓ Master summary: {summary_file}")

    # Final report
    print("\n" + "=" * 100)
    print("BENCHMARK SUITE COMPLETE")
    print("=" * 100)
    print(f"\nTotal duration: {total_duration:.1f}s ({total_duration/60:.1f} min)")
    print(f"Total tests: {total_tests}")
    print(f"\nResults location: {TEMP_DIR}")
    print("\nNext: Review temp markdown files and consolidate into final reports")

    return master_log


if __name__ == "__main__":
    run_all_tot_benchmarks()
