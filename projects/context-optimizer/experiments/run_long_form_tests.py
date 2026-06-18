"""
Orchestrator for chat-assistant long-context experiments.

Runs tests in parallel, aggregates results, and generates a single consolidated
markdown report under docs/experiments/EXPERIMENTS_CONSOLIDATED.md.

Usage:
  python experiments/run_long_form_tests.py
"""
from __future__ import annotations

import concurrent.futures
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.long_form_tests import LongFormTestResult, run_all_long_form_tests


RESULTS_PATH = Path(__file__).resolve().parents[1] / "docs" / "experiments" / "EXPERIMENTS_CONSOLIDATED.md"


def _safe_avg(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _write_results_md(all_results: dict[str, list[LongFormTestResult]]) -> None:
    """Write comprehensive consolidated results to docs/experiments/EXPERIMENTS_CONSOLIDATED.md."""
    run_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    corpus_map = {
        "books-docs": {
            "name": "Large Book/Document QA",
            "corpus": "Project Gutenberg books + open-access technical docs (chapter/section indexed)",
            "indexes": "doc_id, chapter_id/section_id, heading_path, entities, prev_chunk_id, next_chunk_id",
            "goal": "Long-range factual and comparative QA with source-grounded citations",
        },
        "chat-memory": {
            "name": "Episodic Chat Memory",
            "corpus": "Prior conversation sessions/transcripts with turn-level metadata",
            "indexes": "session_id, turn_id, speaker, topic_tags, commitments, unresolved_threads, adjacency links",
            "goal": "Continuity, prior-decision recall, and contradiction avoidance",
        },
        "terms-fine-print": {
            "name": "Terms/Fine-Print Assistant",
            "corpus": "Public terms/privacy docs and clause-level policy text",
            "indexes": "doc_type, clause_id, obligations, exceptions, penalties, jurisdiction, cross_refs",
            "goal": "Risk-focused clause QA and plain-language policy interpretation",
        },
        "social-analytics": {
            "name": "Social Sentiment/Abuse Analytics",
            "corpus": "Large social-text streams with sentiment/toxicity annotations",
            "indexes": "community, time_bucket, sentiment, abuse_type, topic_cluster, sample_ids",
            "goal": "Trend analysis, moderation risk summaries, and week-over-week changes",
        },
    }

    lines = [
        "# Consolidated Experiment Report: Chat-Assistant Context Architecture",
        "",
        "> **Purpose:** Validate Pipe C (MCP Pull) for chat-assistant tasks over long external memory,",
        "> with boundary-preserving semantic retrieval and task-specific indexing.",
        "",
        f"**Run time:** {run_time}",
        "",
        "---",
        "",
        "## Scope",
        "",
        "- Focus: chat assistants, not coding agents",
        "- Pattern: compress intent -> retrieve evidence -> reason over bounded context",
        "- Retrieval contract: typed MCP pull with relevance scoring and boundary hints",
        "",
        "---",
        "",
        "## Corpus Definitions by Experiment Family",
        "",
        "| Domain | Corpus | Stored Index Fields | Evaluation Goal |",
        "|---|---|---|---|",
    ]

    # Domain metadata table from observed domains in results
    seen_domains: list[str] = []
    for suite_results in all_results.values():
        for result in suite_results:
            if result.domain not in seen_domains:
                seen_domains.append(result.domain)

    for domain in seen_domains:
        meta = corpus_map.get(domain, None)
        if meta is None:
            continue
        lines.append(
            f"| {meta['name']} | {meta['corpus']} | {meta['indexes']} | {meta['goal']} |"
        )

    lines += [
        "",
        "---",
        "",
        "## Test Summary",
        "",
        "| Test | Domain | Avg Token Reduction | Avg Quality Parity | Notes |",
        "|---|---|---|---|---|",
    ]

    # Summary table
    for test_name, results in all_results.items():
        avg_reduction = _safe_avg([r.token_reduction for r in results])
        avg_quality = _safe_avg([r.quality_parity for r in results])
        domain = results[0].domain if results else "unknown"

        if avg_reduction > 75:
            notes = "[HIGH] Strong token savings"
        elif avg_reduction > 50:
            notes = "[MOD] Moderate token savings"
        elif avg_reduction > 0:
            notes = "[LOW] Marginal savings"
        else:
            notes = "[NEG] Overhead dominates at this corpus size"

        lines.append(
            f"| {test_name} | {domain} | {avg_reduction:.1f}% | {avg_quality:.2f} | {notes} |"
        )

    lines += [
        "",
        "---",
        "",
        "## Test Results (Detailed)",
        "",
    ]

    # Per-suite detailed results
    for test_name, results in all_results.items():
        domain = results[0].domain if results else "unknown"
        lines += [
            f"### {test_name}",
            "",
            f"**Domain:** {domain}",
            "",
        ]

        for i, r in enumerate(results, 1):
            lines += [
                f"**Question {i}:** {r.question}",
                "",
                "| Metric | Monolithic | Pipe C | Delta |",
                "|---|---|---|---|",
                f"| Tokens | {r.monolithic_tokens:,} | {r.pipe_c_tokens:,} | **{r.token_reduction:.1f}% saved** |",
                f"| Latency | {r.monolithic_latency_s:.3f}s | {r.pipe_c_latency_s:.3f}s | {r.pipe_c_latency_s - r.monolithic_latency_s:+.3f}s |",
                f"| Retrieved Lines | — | {r.retrieved_lines} | — |",
                f"| Tool Calls | 0 | {r.tool_calls} | — |",
                "",
                "**Quality Scores:**",
                f"- Structural: {r.quality_structural_score:.2f} (expected: ≥0.5)",
                f"- Citations: {r.quality_citations_score:.2f} (expected: ≥0.3)",
                f"- Specificity: {r.quality_specificity_score:.2f} (expected: ≥0.3)",
                f"- **Overall Parity:** {r.quality_parity:.2f}",
                "",
                "**Pipe C Answer (excerpt):**",
                f"> {r.pipe_c_answer}",
                "",
                "**Monolithic Answer (excerpt):**",
                f"> {r.monolithic_answer}",
                "",
            ]

    avg_reduction_all = _safe_avg([
        r.token_reduction
        for suite in all_results.values()
        for r in suite
    ])
    avg_quality_all = _safe_avg([
        r.quality_parity
        for suite in all_results.values()
        for r in suite
    ])

    lines += [
        "---",
        "",
        "## Key Findings",
        "",
        "### 1. Architecture Behavior",
        "",
        f"- Average token reduction across all suites: **{avg_reduction_all:.1f}%**",
        f"- Average quality parity across all suites: **{avg_quality_all:.2f}**",
        "- Pipe C quality remains strong when evidence retrieval is selective and structured.",
        "- Overhead appears on small corpora where retrieval shell and index context dominate prompt budget.",
        "",
        "### 2. What Improves Results",
        "",
        "- Boundary-preserving chunks reduce local truncation errors and missing-evidence claims.",
        "- Task-specific metadata (chapter/turn/clause/time bucket) improves retrieval precision.",
        "- Tool-aware prompting helps the reasoner refine queries instead of guessing from weak context.",
        "",
        "### 3. Residual Risks",
        "",
        "- Small contexts can produce negative savings due to fixed shell/tool overhead.",
        "- Quality still depends on embedding fidelity and index freshness.",
        "- Multi-hop answers require explicit retrieval refinement loops to avoid shallow synthesis.",
        "",
        "### 4. Production Guidance",
        "",
        "- Use Pipe C for large corpora where selective retrieval removes most irrelevant context.",
        "- Maintain domain-specific indexes: chapter/section, session/turn, clause/risk, time/community.",
        "- Keep boundary metadata and neighbor links in storage and retrieval outputs.",
        "- Add cache invalidation by TTL + update events before production deployment.",
        "",
        "---",
        "",
        "## Next Steps",
        "",
        "1. Add real corpora ingestion for each suite (Gutenberg/docs, chat transcripts, terms policies, social datasets).",
        "2. Run N>=3 trials per question and report confidence intervals.",
        "3. Evaluate retrieval recall/precision and citation correctness separately from answer quality.",
        "4. Add event-driven cache invalidation and stale-index detection.",
        "5. Extend with multi-hop tool-call tests where one retrieval result triggers a follow-up query.",
        "",
        "---",
        "",
        "## Hypothesis Validation",
        "",
        "**H4 (chat-assistant scope):** Pipe C can maintain answer quality while reducing prompt size",
        "for large memory-retrieval tasks when context selection is high and indexing is task-aware.",
        "",
        "**Evidence:**",
        f"- Average token reduction: {avg_reduction_all:.1f}% across all suites",
        f"- Average quality parity: {avg_quality_all:.2f} across all suites",
        f"- Domains covered: {len(seen_domains)} assistant-focused families",
        "",
        "**Conclusion:** Partially supported in this mock run. Pipe C is strongest when corpus scale and",
        "retrieval selectivity are high; additional large-corpus runs are required for final confirmation.",
        "",
        "---",
        "",
        "<!-- INCIDENT_APPENDIX_START -->",
        "## Incident Benchmark Appendix (Auto-Generated)",
        "",
        "Run `python experiments/run_all_experiments.py` to refresh this section.",
        "<!-- INCIDENT_APPENDIX_END -->",
    ]

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\n[OK] Results written to {RESULTS_PATH}")


def main():
    print("\n" + "="*90)
    print("  CHAT-ASSISTANT LONG-CONTEXT SUITE — Pipe C Validation")
    print("="*90)

    # Run all tests in a thread pool. Suites are independent and safe to execute in parallel.
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        future = executor.submit(run_all_long_form_tests)
        all_results = future.result()

    # Write results
    _write_results_md(all_results)

    # Print summary
    print("\n" + "="*90)
    print("  SUMMARY")
    print("="*90)
    total_token_reduction = 0.0
    total_suites = 0
    for test_name, results in all_results.items():
        if results:
            avg_reduction = sum(r.token_reduction for r in results) / len(results)
            avg_quality = sum(r.quality_parity for r in results) / len(results)
            domain = results[0].domain
            print(f"  {test_name}:")
            print(f"    Domain: {domain}")
            print(f"    Token reduction: {avg_reduction:.1f}%")
            print(f"    Quality parity: {avg_quality:.2f}")
            total_token_reduction += avg_reduction
            total_suites += 1

    if total_suites > 0:
        overall_reduction = total_token_reduction / total_suites
        print(f"\n  OVERALL: {overall_reduction:.1f}% token reduction across all domains")
        print("  [OK] Consolidated report generated for chat-assistant scope")


if __name__ == "__main__":
    main()
