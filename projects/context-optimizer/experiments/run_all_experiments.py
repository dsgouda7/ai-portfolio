"""
Master experiment runner.

Runs Pipe A (monolithic baseline), Pipe OOTB (standard RAG), and Pipe C (MCP Pull) on the
SAME incident input and SAME log corpus, then measures:
  - token counts sent to the reasoning model
  - end-to-end latency
  - structural quality score (deterministic, no LLM)
  - LLM-as-judge quality score (when Ollama is available)

Pipe C is the proposed solution; Pipes A & OOTB are baselines for comparison.

Usage:
  # mock mode (no Ollama needed — illustrative token + structure comparison)
  python experiments/run_all_experiments.py --provider mock

  # Ollama mode (requires: ollama pull phi4:mini && ollama pull qwen3)
  python experiments/run_all_experiments.py --provider ollama

  # Groq mode
  python experiments/run_all_experiments.py --provider groq

Results are written to: docs/experiments/EXPERIMENTS_CONSOLIDATED.md (incident appendix section)
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib import error as urlerror
from urllib import request as urlrequest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from context_optimizer_benchmark import init_chat_model, default_model_names
from experiments.pipes import PipeResult, run_pipe_a, run_pipe_ootb, run_pipe_c
from experiments.quality import QualityReport, structural_score, llm_judge_score
from experiments.shared_inputs import (
    INCIDENT_PROMPT,
    get_log_corpus,
    estimate_tokens,
    GROUND_TRUTH,
)

RESULTS_PATH = Path(__file__).resolve().parents[1] / "docs" / "experiments" / "EXPERIMENTS_CONSOLIDATED.md"
APPENDIX_START = "<!-- INCIDENT_APPENDIX_START -->"
APPENDIX_END = "<!-- INCIDENT_APPENDIX_END -->"


def _preflight_ollama_models(small_model: str, reasoning_model: str) -> None:
    """Fail fast with a clear message if Ollama server/models are unavailable."""
    base_url = "http://localhost:11434"
    tags_url = f"{base_url}/api/tags"

    try:
        with urlrequest.urlopen(tags_url, timeout=5) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urlerror.URLError as exc:
        raise RuntimeError(
            "Ollama server is not reachable at http://localhost:11434. "
            "Start Ollama first, then retry."
        ) from exc
    except Exception as exc:
        raise RuntimeError(
            "Failed to read Ollama model tags from /api/tags. "
            "Check Ollama health and retry."
        ) from exc

    available = {
        m.get("name", "") for m in payload.get("models", []) if isinstance(m, dict)
    }
    missing = [m for m in [small_model, reasoning_model] if m not in available]
    if missing:
        missing_csv = ", ".join(missing)
        raise RuntimeError(
            "Missing Ollama model(s): "
            f"{missing_csv}. Pull them first, e.g. `ollama pull {missing[0]}`."
        )


def _sep(title: str = "") -> None:
    print("\n" + "=" * 90)
    if title:
        print(f"  {title}")
        print("=" * 90)


def _run_all_pipes(
    small_model_name: str,
    reasoning_model_name: str,
    provider: str,
    log_corpus: list[str],
    pipe_timeout_s: int,
) -> list[PipeResult]:
    small_llm = init_chat_model(provider, small_model_name)
    reasoning_llm = init_chat_model(provider, reasoning_model_name)

    results: list[PipeResult] = []

    def run_with_timeout(label: str, fn, *args, **kwargs) -> PipeResult:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(fn, *args, **kwargs)
            try:
                return fut.result(timeout=pipe_timeout_s)
            except concurrent.futures.TimeoutError:
                return PipeResult(
                    pipe_name=label,
                    answer=(
                        f"[Timed out after {pipe_timeout_s}s] "
                        "Pipeline execution exceeded timeout budget on this hardware."
                    ),
                    latency_s=float(pipe_timeout_s),
                    prompt_tokens_sent=0,
                    tool_call_count=0,
                    retrieved_lines=0,
                    extra={"timed_out": True, "timeout_s": pipe_timeout_s},
                )

    _sep("Pipe A — Monolithic Baseline")
    if provider == "ollama":
        r_a = PipeResult(
            pipe_name="Pipe A — Monolithic (baseline)",
            answer=(
                "[Skipped] Monolithic baseline can stall on local 7B Ollama models due to "
                "large raw-context prompt volume. Use provider=mock or a larger hosted model "
                "to include this baseline."
            ),
            latency_s=0.0,
            prompt_tokens_sent=0,
            tool_call_count=0,
            retrieved_lines=0,
            extra={"skipped": True, "reason": "local_ollama_monolithic_stall"},
        )
    else:
        r_a = run_with_timeout(
            "Pipe A — Monolithic (baseline)",
            run_pipe_a,
            reasoning_llm,
            INCIDENT_PROMPT,
            log_corpus,
            provider,
        )
    results.append(r_a)
    _print_pipe(r_a)

    _sep("Pipe OOTB — Standard LangChain RAG")
    r_ootb = run_with_timeout(
        "Pipe OOTB — Standard LangChain RAG",
        run_pipe_ootb,
        reasoning_llm,
        INCIDENT_PROMPT,
        log_corpus,
        provider,
    )
    results.append(r_ootb)
    _print_pipe(r_ootb)

    _sep("Pipe C — MCP Pull (structured shell) [PROPOSED SOLUTION]")
    r_c = run_with_timeout(
        "Pipe C — MCP Pull (structured shell)",
        run_pipe_c,
        small_llm,
        reasoning_llm,
        INCIDENT_PROMPT,
        log_corpus,
        provider,
    )
    results.append(r_c)
    _print_pipe(r_c)

    return results


def _print_pipe(r: PipeResult) -> None:
    print(f"  Answer preview: {r.answer[:200].replace(chr(10), ' ')}...")
    print(f"  Tokens sent:    {r.prompt_tokens_sent:,}")
    print(f"  Latency:        {r.total_latency_s:.3f}s  (reasoning {r.latency_s:.3f}s + compress {r.compression_latency_s:.3f}s)")
    print(f"  Tool calls:     {r.tool_call_count}   Retrieved lines: {r.retrieved_lines}")


def _build_quality_report(
    results: list[PipeResult],
    judge_llm: object,
    provider: str,
) -> QualityReport:
    report = QualityReport()
    for r in results:
        report.add_structural(structural_score(r.pipe_name, r.answer))
        if provider != "mock" and judge_llm is not None:
            report.add_llm_judge(llm_judge_score(judge_llm, r.pipe_name, r.answer))
        else:
            from experiments.quality import LLMJudgeScore
            report.add_llm_judge(
                LLMJudgeScore(
                    pipe_name=r.pipe_name,
                    error="Skipped in mock mode — run with --provider ollama for LLM judge scores",
                )
            )
    return report


def _write_results_md(
    results: list[PipeResult],
    report: QualityReport,
    provider: str,
    small_model: str,
    reasoning_model: str,
    log_lines: int,
    run_time_utc: str,
) -> None:
    """Write incident benchmark appendix content into docs/experiments/EXPERIMENTS_CONSOLIDATED.md."""
    incident_tokens = estimate_tokens(INCIDENT_PROMPT)
    corpus_tokens = estimate_tokens(" ".join(get_log_corpus()))

    lines: list[str] = [
        APPENDIX_START,
        "## Incident Benchmark Appendix (Auto-Generated)",
        "",
        "> **Important:** Results produced in mock mode are illustrative estimates.",
        "> Structural quality scores reflect the design intent of each pipeline.",
        "> Run with `--provider ollama` for real LLM inference and LLM-as-judge scoring.",
        "",
        "---",
        "",
        "## Run Metadata",
        "",
        f"| Key | Value |",
        f"|---|---|",
        f"| Run time (UTC) | {run_time_utc} |",
        f"| Provider | `{provider}` |",
        f"| Small model (compression) | `{small_model}` |",
        f"| Reasoning model | `{reasoning_model}` |",
        f"| Log corpus size | {log_lines:,} lines |",
        f"| Incident prompt tokens (est.) | {incident_tokens:,} |",
        f"| Full corpus tokens (est.) | {corpus_tokens:,} |",
        "",
        "---",
        "",
        "## Input: Same Incident Prompt (All Pipes)",
        "",
        "```",
        INCIDENT_PROMPT,
        "```",
        "",
        "---",
        "",
        "## 1. Token Efficiency Comparison",
        "",
        "| Pipeline | Prompt Tokens Sent | Tool Calls | Retrieved Lines | Total Latency (s) | vs Pipe A |",
        "|---|---|---|---|---|---|",
    ]

    pipe_a_tokens = next((r.prompt_tokens_sent for r in results if "Pipe A" in r.pipe_name), 1)
    if pipe_a_tokens <= 0:
        pipe_a_tokens = 1
    for r in results:
        vs_a = f"{r.prompt_tokens_sent / pipe_a_tokens:.1%}" if "Pipe A" not in r.pipe_name else "—"
        lines.append(
            f"| {r.pipe_name} "
            f"| **{r.prompt_tokens_sent:,}** "
            f"| {r.tool_call_count} "
            f"| {r.retrieved_lines} "
            f"| {r.total_latency_s:.3f} "
            f"| {vs_a} |"
        )

    lines += [
        "",
        "> Token counts are estimated at 4 chars/token. Real counts will vary by model tokeniser.",
        "",
        "---",
        "",
        "## 2. Quality Evaluation",
        "",
        report.summary_table(),
        "",
        "---",
        "",
        "## 3. Per-Pipeline Answers",
        "",
    ]

    for r in results:
        lines += [
            f"### {r.pipe_name}",
            "",
            "```",
            r.answer.strip(),
            "```",
            "",
            f"**Token budget:** {r.prompt_tokens_sent:,} tokens  |  "
            f"**Latency:** {r.total_latency_s:.3f}s  |  "
            f"**Tool calls:** {r.tool_call_count}",
            "",
        ]
        if r.extra:
            lines.append(f"_Extra telemetry: {json.dumps(r.extra)}_")
            lines.append("")

    lines += [
        "---",
        "",
        "## 4. Key Observations",
        "",
        "### Token Efficiency",
        "",
        "- **Pipe A** sends the full log corpus to the reasoning model: O(corpus size) tokens.",
        "- **Pipe OOTB** uses TF-IDF retrieval to reduce token load, but without compression the",
        "  query quality depends on raw user phrasing and retrieval may miss precise error codes.",
        "- **Pipe C** maintains a fixed structured shell (~1.7k token contract) and pulls context",
        "  only when the reasoning model requests it via typed MCP tool calls.",
        "",
        "### Quality",
        "",
        "- In mock mode, quality differences reflect deliberate calibration of canned responses",
        "  to illustrate what each architecture would likely produce in practice.",
        "- In Ollama mode, structural and LLM-judge scores reflect actual model outputs.",
        "- Pipe C is expected to produce more specific answers because the retrieval queries",
        "  are derived from the compressed identifiers, targeting precise error codes rather",
        "  than the raw user prose.",
        "",
        "### Trade-off Summary",
        "",
        "| Pipe | Token cost | Quality potential | Latency overhead | Complexity |",
        "|---|---|---|---|---|",
        "| A  | Highest (O corpus) | Baseline | Lowest | Lowest |",
        "| OOTB | Medium (top-k chunks) | Similar to A | Low | Low |",
        "| C  | Bounded (shell + MCP) | Best (typed pull) | +1 compress + N MCP RTTs | Highest |",
        "",
        "---",
        "",
        "## 5. Open Problems and Next Steps",
        "",
        "1. **Cache invalidation strategy** — session-persisted semantic cache has no TTL policy yet.",
        "2. **Real embedding quality test** — compare retrieval recall on compressed-before-index",
        "   vs raw-indexed corpus.",
        "3. **MCP server process** — replace in-process simulation with a real FastMCP server.",
        "4. **Domain transfer** — run same pipelines on chat-assistant datasets (books/docs, episodic memory, terms, social analytics).",
        "5. **LLM judge calibration** — run human eval alongside LLM judge to validate correlation.",
        "",
        f"_Generated: {run_time_utc} | Provider: {provider}_",
        APPENDIX_END,
    ]

    appendix_block = "\n".join(lines)

    if RESULTS_PATH.exists():
        existing = RESULTS_PATH.read_text(encoding="utf-8")
        if APPENDIX_START in existing and APPENDIX_END in existing:
            start_idx = existing.index(APPENDIX_START)
            end_idx = existing.index(APPENDIX_END) + len(APPENDIX_END)
            updated = existing[:start_idx].rstrip() + "\n\n" + appendix_block + "\n"
            if end_idx < len(existing):
                updated += existing[end_idx:].lstrip("\n")
        else:
            updated = existing.rstrip() + "\n\n---\n\n" + appendix_block + "\n"
    else:
        updated = "# Consolidated Experiment Report: Chat-Assistant Context Architecture\n\n" + appendix_block + "\n"

    RESULTS_PATH.write_text(updated, encoding="utf-8")
    print(f"\n✔  Incident appendix updated in: {RESULTS_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all context-optimizer pipeline experiments.")
    parser.add_argument("--provider", default="mock", choices=["mock", "ollama", "groq"])
    parser.add_argument("--small-model", default=None)
    parser.add_argument("--reasoning-model", default=None)
    parser.add_argument("--log-lines", type=int, default=1050)
    parser.add_argument("--pipe-timeout-s", type=int, default=45)
    args = parser.parse_args()

    small_model, reasoning_model = default_model_names(args.provider)
    if args.small_model:
        small_model = args.small_model
    if args.reasoning_model:
        reasoning_model = args.reasoning_model

    log_corpus = get_log_corpus(args.log_lines)

    if args.provider == "ollama":
        _preflight_ollama_models(small_model, reasoning_model)

    _sep(f"Context Optimizer — Experiment Suite  [{args.provider.upper()}]")
    print(f"  Small model:     {small_model}")
    print(f"  Reasoning model: {reasoning_model}")
    print(f"  Log corpus:      {len(log_corpus):,} lines")
    print(f"  Input:           {len(INCIDENT_PROMPT)} chars / ~{estimate_tokens(INCIDENT_PROMPT)} tokens")

    run_time_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    t_total_start = time.perf_counter()

    results = _run_all_pipes(
        small_model,
        reasoning_model,
        args.provider,
        log_corpus,
        args.pipe_timeout_s,
    )

    # Quality evaluation
    _sep("Quality Evaluation")
    judge_llm = init_chat_model(args.provider, reasoning_model) if args.provider != "mock" else None
    report = _build_quality_report(results, judge_llm, args.provider)

    for s in report.structural:
        print(
            f"  {s.pipe_name:<50} structural={s.overall:.3f}  "
            f"keywords={s.keyword_coverage:.0%}  specificity={s.specificity:.0%}"
        )

    _sep("Writing Results")
    _write_results_md(
        results=results,
        report=report,
        provider=args.provider,
        small_model=small_model,
        reasoning_model=reasoning_model,
        log_lines=len(log_corpus),
        run_time_utc=run_time_utc,
    )

    total_elapsed = time.perf_counter() - t_total_start
    _sep(f"Done in {total_elapsed:.2f}s")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("\n[ERROR] Experiment run failed:")
        print(f"  {exc}")
        sys.exit(1)
