"""
Post-hoc LLM-as-judge pass over an existing EXPERIMENT_RESULTS.json.

Reads the saved answer_snippets, scores each with llama3.2:3b, writes
judge_score back into the JSON, and regenerates experiment_results.md.

Usage:
    python judge_existing_results.py                          # default path
    python judge_existing_results.py --results path/to.json
"""

import sys
import os
import re
import json
import argparse
from pathlib import Path

_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_root / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from llm_provider import build_compression_llm, get_embedding_config

# ── Questions registry (same as run_experiments.py) ───────────────────────────
QUESTIONS = {
    "q001": {
        "question": "Who is Elizabeth Bennet and what are her main character traits?",
        "keywords": ["elizabeth", "bennet", "protagonist", "witty", "intelligent", "darcy", "independent"],
    },
    "q002": {
        "question": "Who is Mr. Bingley and where does he settle?",
        "keywords": ["bingley", "netherfield", "wealthy", "gentleman", "jane"],
    },
    "q003": {
        "question": "What social themes are central to the story?",
        "keywords": ["marriage", "class", "society", "wealth", "reputation", "family"],
    },
    "q004": {
        "question": "Describe the Bennet family members",
        "keywords": ["jane", "elizabeth", "lydia", "kitty", "mary", "mrs bennet", "mr bennet"],
    },
    "q005": {
        "question": "What is Mr. Darcy's initial attitude towards Elizabeth and how does it change?",
        "keywords": ["darcy", "proud", "tolerable", "inferior", "love", "admire", "respect", "change"],
    },
    "q006": {
        "question": "How does the theme of first impressions affect relationships in the novel?",
        "keywords": ["impression", "prejudice", "pride", "misjudge", "misunderstand", "reveal", "character"],
    },
}

_JUDGE_SYSTEM = (
    "You are a strict but fair evaluation judge. "
    "Given a question, the key concepts that should appear in a correct answer, "
    "and an actual answer, rate how well the answer addresses the question and covers "
    "the key concepts. Reply with ONLY a single decimal number between 0.0 and 1.0. "
    "1.0 = perfect coverage and accuracy, 0.0 = completely wrong or empty. "
    "No explanation, no other text — just the number."
)


def _judge(llm, question: str, keywords: list[str], answer: str) -> float:
    """Score answer semantically. Falls back to recall on LLM failure."""
    if not answer or answer.startswith("[ERROR]"):
        found = sum(1 for kw in keywords if kw.lower() in answer.lower())
        return found / len(keywords) if keywords else 0.0

    key_concepts = ", ".join(keywords)
    prompt = (
        f"Question: {question}\n\n"
        f"Key concepts expected in a correct answer: {key_concepts}\n\n"
        f"Actual answer: {answer}\n\n"
        "Score (0.0 – 1.0):"
    )
    try:
        from langchain_core.messages import HumanMessage, SystemMessage
        response = llm.invoke(
            [SystemMessage(content=_JUDGE_SYSTEM), HumanMessage(content=prompt)]
        )
        raw = response.content if hasattr(response, "content") else str(response)
        m = re.search(r"\b([01]?\.\d+|[01])\b", raw.strip())
        if m:
            return max(0.0, min(1.0, float(m.group(1))))
    except Exception as exc:
        print(f"    [WARN] Judge LLM failed: {exc}")
    # fallback: recall
    found = sum(1 for kw in keywords if kw.lower() in answer.lower())
    return found / len(keywords) if keywords else 0.0


def _score_list(llm, results: list[dict], label: str) -> None:
    """Add judge_score to each result dict in-place."""
    for r in results:
        qid     = r.get("question_id", "")
        snippet = r.get("answer_snippet", "")
        q       = QUESTIONS.get(qid, {})
        if not q:
            r["judge_score"] = 0.0
            continue
        score = _judge(llm, q["question"], q["keywords"], snippet)
        r["judge_score"] = score
        print(f"    [{qid}] {label}  snippet_len:{len(snippet)}  judge:{score:.2f}")


def _avg(lst: list[dict], key: str) -> float:
    vals = [r[key] for r in lst if key in r]
    return sum(vals) / len(vals) if vals else 0.0


def _pass_fail(value: float, target: float, higher_is_better: bool = True) -> str:
    if higher_is_better:
        return "[PASS]" if value >= target else "[FAIL]"
    return "[PASS]" if value <= target else "[FAIL]"


def regenerate_report(payload: dict) -> str:
    """Rebuild the markdown summary from updated payload."""
    exp1    = payload["experiment_1"]
    exp2    = payload["experiment_2"]
    models  = payload.get("models", {})
    embed   = payload.get("embedding", {})
    thr     = payload.get("thresholds", {"latency_delta_pct": 10, "f1_delta_pct": 20, "token_reduction_pct": 90})
    run_date = payload.get("run_date", "unknown")
    corpus   = payload.get("corpus_lines", 0)

    e2a = exp2["exp_2a"]
    e2b = exp2["exp_2b"]

    e1_lat  = _avg(exp1, "latency_sec");  e1_tok  = _avg(exp1, "prompt_tokens")
    e1_kw   = _avg(exp1, "f1");           e1_j    = _avg(exp1, "judge_score")

    e2a_lat = _avg(e2a, "latency_sec");   e2a_tok = _avg(e2a, "prompt_tokens")
    e2a_kw  = _avg(e2a, "f1");            e2a_j   = _avg(e2a, "judge_score")
    e2a_ret = _avg(e2a, "retrieval_ms");  e2a_hit = _avg(e2a, "cache_hit_ms")

    e2b_lat = _avg(e2b, "latency_sec");   e2b_tok = _avg(e2b, "prompt_tokens")
    e2b_kw  = _avg(e2b, "f1");            e2b_j   = _avg(e2b, "judge_score")

    tok_red_2a = (1 - e2a_tok / e1_tok) * 100 if e1_tok else 0
    tok_red_2b = (1 - e2b_tok / e1_tok) * 100 if e1_tok else 0
    lat_d_2a   = (e2a_lat - e1_lat) / e1_lat * 100 if e1_lat else 0
    lat_d_2b   = (e2b_lat - e1_lat) / e1_lat * 100 if e1_lat else 0
    jdg_d_2a   = (e2a_j - e1_j) / e1_j * 100 if e1_j else 0
    jdg_d_2b   = (e2b_j - e1_j) / e1_j * 100 if e1_j else 0
    kw_d_2a    = (e2a_kw - e1_kw) / e1_kw * 100 if e1_kw else 0
    kw_d_2b    = (e2b_kw - e1_kw) / e1_kw * 100 if e1_kw else 0

    tok_pf_2a = _pass_fail(tok_red_2a, thr["token_reduction_pct"])
    tok_pf_2b = _pass_fail(tok_red_2b, thr["token_reduction_pct"])
    lat_pf_2a = _pass_fail(abs(lat_d_2a), thr["latency_delta_pct"], False)
    lat_pf_2b = _pass_fail(abs(lat_d_2b), thr["latency_delta_pct"], False)
    jdg_pf_2a = _pass_fail(abs(jdg_d_2a), thr["f1_delta_pct"], False)
    jdg_pf_2b = _pass_fail(abs(jdg_d_2b), thr["f1_delta_pct"], False)
    kw_pf_2a  = _pass_fail(abs(kw_d_2a),  thr["f1_delta_pct"], False)
    kw_pf_2b  = _pass_fail(abs(kw_d_2b),  thr["f1_delta_pct"], False)

    md = f"""# Experiment Results: Standard LLM vs Compressed Architecture

> **Run date:** {run_date}
> **Corpus:** {corpus:,} lines (Pride & Prejudice excerpt)
> **Judge:** `{models.get('compression', 'llama3.2:3b')}` (LLM-as-judge, 0–1 scale)
> **Note:** Judge scores added by post-hoc pass on saved answer snippets (200 chars).
> Status: Local-only execution — no API keys, no cloud.

---

## Model Configuration

| Role | Model | Backend |
|------|-------|---------|
| Compression / Summarisation | `{models.get('compression', '?')}` | Ollama (local) |
| Embeddings | `{embed.get('model', '?')}` | {embed.get('backend', '?')} |
| Reasoning | `{models.get('reasoning', '?')}` | Ollama (local) |
| Judge (evaluator) | `{models.get('compression', 'llama3.2:3b')}` | reused from compression role |

---

## Accuracy Method: LLM-as-Judge vs Keyword-Overlap F1

| Method | How it works | Bias | Use as |
|--------|-------------|------|--------|
| **Judge score** | LLM rates answer 0–1 on concept coverage | None — model understands paraphrasing | **Primary** |
| **KW-F1** | keyword overlap precision × recall | Penalises verbose answers (precision = found/all_words) | Secondary / sanity check |

---

## Pass/Fail Thresholds

| Metric | Threshold |
|--------|-----------|
| Latency delta vs baseline | ±{thr['latency_delta_pct']:.0f}% |
| Judge-score delta vs baseline | ±{thr['f1_delta_pct']:.0f}% |
| Token reduction vs baseline | ≥{thr['token_reduction_pct']:.0f}% |

---

## Experiment 1 — Standard LLM Baseline

| Q | Difficulty | Tokens | Latency (s) | KW-F1 | Judge |
|---|------------|--------|-------------|-------|-------|
"""
    for r in exp1:
        md += (f"| {r['question_id']} | {r['difficulty']} | {r['prompt_tokens']:,} | "
               f"{r['latency_sec']:.2f} | {r.get('f1', 0):.3f} | {r.get('judge_score', 0):.2f} |\n")
    md += (f"| **Avg** | — | **{e1_tok:,.0f}** | **{e1_lat:.2f}** | "
           f"**{e1_kw:.3f}** | **{e1_j:.2f}** |\n")

    comp = exp2.get("compression", {})
    md += f"""
---

## Experiment 2a — Compressed Architecture (Summaries Only)

### Compression

| Metric | Value |
|--------|-------|
| Chunks | {comp.get('chunks', '?'):,} |
| Time | {comp.get('time_sec', 0):.1f}s (one-time) |
| Ratio | {comp.get('ratio', 0):.3f} |
| Original tokens | {comp.get('original_tokens', 0):,} |
| Compressed tokens | {comp.get('compressed_tokens', 0):,} |

### Query Results

| Q | Difficulty | Tokens | Ret (ms) | Hit (ms) | Latency (s) | KW-F1 | Judge | Token Δ |
|---|------------|--------|----------|----------|-------------|-------|-------|---------|
"""
    for r, b in zip(e2a, exp1):
        td = (1 - r["prompt_tokens"] / b["prompt_tokens"]) * 100 if b["prompt_tokens"] else 0
        md += (f"| {r['question_id']} | {r['difficulty']} | {r['prompt_tokens']:,} | "
               f"{r['retrieval_ms']:.1f} | {r['cache_hit_ms']:.1f} | {r['latency_sec']:.2f} | "
               f"{r.get('f1', 0):.3f} | {r.get('judge_score', 0):.2f} | -{td:.1f}% |\n")
    md += (f"| **Avg** | — | **{e2a_tok:,.0f}** | **{e2a_ret:.1f}** | **{e2a_hit:.1f}** | "
           f"**{e2a_lat:.2f}** | **{e2a_kw:.3f}** | **{e2a_j:.2f}** | **-{tok_red_2a:.1f}%** |\n")

    md += f"""
---

## Experiment 2b — Compressed Architecture (Summaries + Raw Detail)

| Q | Difficulty | Tokens | Latency (s) | KW-F1 | Judge | Token Δ |
|---|------------|--------|-------------|-------|-------|---------|
"""
    for r, b in zip(e2b, exp1):
        td = (1 - r["prompt_tokens"] / b["prompt_tokens"]) * 100 if b["prompt_tokens"] else 0
        md += (f"| {r['question_id']} | {r['difficulty']} | {r['prompt_tokens']:,} | "
               f"{r['latency_sec']:.2f} | {r.get('f1', 0):.3f} | {r.get('judge_score', 0):.2f} | -{td:.1f}% |\n")
    md += (f"| **Avg** | — | **{e2b_tok:,.0f}** | **{e2b_lat:.2f}** | "
           f"**{e2b_kw:.3f}** | **{e2b_j:.2f}** | **-{tok_red_2b:.1f}%** |\n")

    md += f"""
---

## Cross-Experiment Comparison

> **Primary accuracy metric**: Judge score (LLM-as-judge).
> KW-F1 is secondary — it under-scores verbose-but-correct answers.

| Metric | Baseline (Exp 1) | Exp 2a (Summary) | Exp 2b (Summary+Raw) |
|--------|-----------------|-----------------|---------------------|
| Avg prompt tokens | {e1_tok:,.0f} | {e2a_tok:,.0f} | {e2b_tok:,.0f} |
| Token reduction | — | **-{tok_red_2a:.1f}%** {tok_pf_2a} | **-{tok_red_2b:.1f}%** {tok_pf_2b} |
| Avg reasoning latency (s) | {e1_lat:.2f} | {e2a_lat:.2f} ({lat_d_2a:+.1f}%) {lat_pf_2a} | {e2b_lat:.2f} ({lat_d_2b:+.1f}%) {lat_pf_2b} |
| Avg retrieval latency (ms) | N/A | {e2a_ret:.1f} (miss) / {e2a_hit:.1f} (hit) | same |
| Avg Judge score (0–1) | {e1_j:.2f} | {e2a_j:.2f} ({jdg_d_2a:+.1f}%) {jdg_pf_2a} | {e2b_j:.2f} ({jdg_d_2b:+.1f}%) {jdg_pf_2b} |
| Avg KW-F1 (secondary) | {e1_kw:.3f} | {e2a_kw:.3f} ({kw_d_2a:+.1f}%) {kw_pf_2a} | {e2b_kw:.3f} ({kw_d_2b:+.1f}%) {kw_pf_2b} |

### Threshold Summary

| Threshold | Target | Exp 2a | Exp 2b |
|-----------|--------|--------|--------|
| Token reduction ≥{thr['token_reduction_pct']:.0f}% | ≥{thr['token_reduction_pct']:.0f}% | {tok_red_2a:.1f}% {tok_pf_2a} | {tok_red_2b:.1f}% {tok_pf_2b} |
| Latency delta ≤±{thr['latency_delta_pct']:.0f}% | ≤±{thr['latency_delta_pct']:.0f}% | {lat_d_2a:+.1f}% {lat_pf_2a} | {lat_d_2b:+.1f}% {lat_pf_2b} |
| Judge-score delta ≤±{thr['f1_delta_pct']:.0f}% | ≤±{thr['f1_delta_pct']:.0f}% | {jdg_d_2a:+.1f}% {jdg_pf_2a} | {jdg_d_2b:+.1f}% {jdg_pf_2b} |

---

## Key Observations

- **Token efficiency**: Exp 2a delivers {tok_red_2a:.0f}% token reduction — {"above" if tok_red_2a >= thr['token_reduction_pct'] else "below"} the {thr['token_reduction_pct']:.0f}% target.
- **Latency**: Reasoning latency {"improved" if lat_d_2a < 0 else "increased"} by {abs(lat_d_2a):.0f}% in Exp 2a. Retrieval adds {e2a_ret:.0f}ms (miss) / {e2a_hit:.1f}ms (cache hit).
- **Quality**: Judge score {"held" if abs(jdg_d_2a) <= thr['f1_delta_pct'] else "dropped"} within {abs(jdg_d_2a):.0f}% of baseline (threshold: ±{thr['f1_delta_pct']:.0f}%).
- **Raw detail (2b)**: Pointer-model fetch gives {jdg_d_2b - jdg_d_2a:+.0f}% judge-score delta vs 2a at +{e2b_tok - e2a_tok:,.0f} tokens.
- **Why KW-F1 is low**: Keyword precision = matched_keywords / all_answer_words. A verbose-but-correct 200-word answer mentioning 2 of 7 keywords scores 2/200 = 0.01 precision regardless of factual quality. Judge score does not have this flaw.
"""
    return md


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default=None,
                        help="Path to EXPERIMENT_RESULTS.json (default: auto-find)")
    args = parser.parse_args()

    # ── Find results file ─────────────────────────────────────────────
    if args.results:
        results_path = Path(args.results)
    else:
        results_path = Path(__file__).parent / "EXPERIMENT_RESULTS.json"

    if not results_path.exists():
        print(f"[ERROR] Results file not found: {results_path}")
        print("  Run run_experiments.py first, then re-run this script.")
        sys.exit(1)

    print(f"[Judge] Reading results from: {results_path}")
    with open(results_path, encoding="utf-8") as f:
        payload = json.load(f)

    # ── Init judge LLM ────────────────────────────────────────────────
    print("[Judge] Initialising judge LLM (llama3.2:3b)...")
    judge_llm = build_compression_llm()
    if judge_llm is None:
        print("[WARN] Judge LLM unavailable — will use recall-only fallback.")

    # ── Score Exp 1 ───────────────────────────────────────────────────
    print("\n[Judge] Scoring Experiment 1 (baseline)...")
    _score_list(judge_llm, payload["experiment_1"], "Exp1")

    # ── Score Exp 2a ──────────────────────────────────────────────────
    print("\n[Judge] Scoring Experiment 2a (summaries)...")
    _score_list(judge_llm, payload["experiment_2"]["exp_2a"], "Exp2a")

    # ── Score Exp 2b ──────────────────────────────────────────────────
    print("\n[Judge] Scoring Experiment 2b (summaries+raw)...")
    _score_list(judge_llm, payload["experiment_2"]["exp_2b"], "Exp2b")

    # ── Save updated JSON ─────────────────────────────────────────────
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[OK] Updated JSON: {results_path}")

    # ── Regenerate markdown ───────────────────────────────────────────
    report_path = (Path(__file__).parent.parent.parent
                   / "docs" / "experiments" / "experiment_results.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    md = regenerate_report(payload)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"[OK] Markdown report: {report_path}")
    print("\nDONE")


if __name__ == "__main__":
    main()
