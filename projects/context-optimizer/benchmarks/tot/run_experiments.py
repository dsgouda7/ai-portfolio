"""
End-to-End Experiment Runner: Standard LLM Baseline vs Compressed Architecture

Implements the two experiments from EXPERIMENTS_CONSOLIDATED.md:
  Exp 1  Raw corpus injected directly into reasoning LLM (no architecture).
  Exp 2a Compress -> retrieve compressed summaries -> reason (cache + vector DB).
  Exp 2b Compress -> retrieve compressed summaries + raw detail -> reason.

All results are relative to the Exp 1 baseline:
  Latency delta   : +/- 10%  of baseline  (PASS threshold)
  F1 delta        : +/- 20%  of baseline  (PASS threshold)
  Token reduction :  >= 90%  vs baseline  (PASS threshold)

Models (all local via Ollama):
  Compression  : CONTEXT_OPTIMIZER_COMPRESSOR_MODEL   (default: llama3.2:3b)
  Embedding    : CONTEXT_OPTIMIZER_EMBEDDING_BACKEND  (default: sentence-transformers)
                 Set =ollama to use CONTEXT_OPTIMIZER_EMBEDDING_MODEL (nomic-embed-text)
  Reasoning    : CONTEXT_OPTIMIZER_REASONING_MODEL    (default: qwen2.5-coder:7b)

Usage:
  python run_experiments.py                # mini corpus, 500 lines, quick E2E
  python run_experiments.py --lines 2000   # larger sample (slower)
  python run_experiments.py --full         # full 25K medium corpus (hours with Ollama)
"""

import sys
import os
import json
import time
import tempfile
import argparse
from pathlib import Path
from datetime import datetime

_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_root / "src"))
sys.path.insert(0, str(_root))

from context_optimizer.compressor import compress_corpus_rolling
from context_optimizer.cached_retriever import CachedChromaRetriever
from llm_provider import build_compression_llm, build_reasoning_llm, get_embedding_config

# ── Pass/Fail thresholds ──────────────────────────────────────────────────────
THRESHOLDS = {
    "latency_delta_pct":  10.0,   # +/- 10% vs baseline latency
    "f1_delta_pct":       20.0,   # +/- 20% vs baseline F1
    "token_reduction_pct": 90.0,  # >= 90% fewer tokens than baseline
}

# ── Questions with expected-keyword ground truth ──────────────────────────────
QUESTIONS = [
    {
        "id": "q001", "difficulty": "easy",
        "question": "Who is Elizabeth Bennet and what are her main character traits?",
        "keywords": ["elizabeth", "bennet", "protagonist", "witty", "intelligent", "darcy", "independent"],
    },
    {
        "id": "q002", "difficulty": "easy",
        "question": "Who is Mr. Bingley and where does he settle?",
        "keywords": ["bingley", "netherfield", "wealthy", "gentleman", "jane"],
    },
    {
        "id": "q003", "difficulty": "medium",
        "question": "What social themes are central to the story?",
        "keywords": ["marriage", "class", "society", "wealth", "reputation", "family"],
    },
    {
        "id": "q004", "difficulty": "medium",
        "question": "Describe the Bennet family members",
        "keywords": ["jane", "elizabeth", "lydia", "kitty", "mary", "mrs bennet", "mr bennet"],
    },
    {
        "id": "q005", "difficulty": "hard",
        "question": "What is Mr. Darcy's initial attitude towards Elizabeth and how does it change?",
        "keywords": ["darcy", "proud", "tolerable", "inferior", "love", "admire", "respect", "change"],
    },
    {
        "id": "q006", "difficulty": "hard",
        "question": "How does the theme of first impressions affect relationships in the novel?",
        "keywords": ["impression", "prejudice", "pride", "misjudge", "misunderstand", "reveal", "character"],
    },
]

SYSTEM_PROMPT = (
    "You are a helpful literary assistant. Answer the question concisely and accurately "
    "based only on the provided text. Be specific and use details from the text."
)


# ── Utilities ─────────────────────────────────────────────────────────────────

def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _score_answer(answer: str, keywords: list[str]) -> tuple[float, float, float]:
    """Keyword-overlap F1: precision=keywords_found/answer_words, recall=keywords_found/total_keywords."""
    answer_lower = answer.lower()
    answer_words = set(answer_lower.split())
    found = sum(1 for kw in keywords if kw.lower() in answer_lower)
    recall    = found / len(keywords) if keywords else 0.0
    precision = found / len(answer_words) if answer_words else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def _call_llm(llm, prompt: str, system: str = SYSTEM_PROMPT) -> tuple[str, float]:
    """Invoke LLM and return (answer_text, elapsed_seconds)."""
    from langchain_core.messages import HumanMessage, SystemMessage
    t0 = time.time()
    try:
        response = llm.invoke([SystemMessage(content=system), HumanMessage(content=prompt)])
        elapsed  = time.time() - t0
        text = response.content if hasattr(response, "content") else str(response)
        return text.strip(), elapsed
    except Exception as exc:
        elapsed = time.time() - t0
        return f"[ERROR] {exc}", elapsed


# ── Experiment 1 — Standard LLM baseline ─────────────────────────────────────

def run_experiment1(corpus_lines: list[str], llm) -> list[dict]:
    """Inject the full raw corpus into the reasoning LLM for each question."""
    corpus_text  = "\n".join(corpus_lines)
    prompt_base  = _estimate_tokens(corpus_text)

    print("\n" + "=" * 80)
    print("EXPERIMENT 1 — Standard LLM Baseline (raw corpus injected)")
    print("=" * 80)
    print(f"  Corpus: {len(corpus_lines):,} lines  |  ~{prompt_base:,} tokens (raw)")

    results = []
    for q in QUESTIONS:
        prompt = (
            f"The following is a text corpus:\n\n{corpus_text}\n\n"
            f"Question: {q['question']}\n\nAnswer:"
        )
        total_tokens = _estimate_tokens(prompt)

        print(f"\n  [{q['id']}] {q['difficulty'].upper()}: {q['question'][:60]}")
        answer, elapsed = _call_llm(llm, prompt)
        _, _, f1 = _score_answer(answer, q["keywords"])

        print(f"    Tokens: {total_tokens:,}   Latency: {elapsed:.2f}s   F1: {f1:.3f}")
        print(f"    Answer (first 120 chars): {answer[:120]}")

        results.append({
            "question_id":    q["id"],
            "difficulty":     q["difficulty"],
            "prompt_tokens":  total_tokens,
            "latency_sec":    elapsed,
            "f1":             f1,
            "answer_snippet": answer[:200],
        })

    return results


# ── Experiment 2 — Compressed Architecture ───────────────────────────────────

def run_experiment2(corpus_lines: list[str], compress_llm, reason_llm, embed_cfg: dict) -> dict:
    """
    Compress corpus, build index, then run:
      2a — answer from compressed summaries only
      2b — answer from compressed summaries + raw-detail fetch
    """
    print("\n" + "=" * 80)
    print("EXPERIMENT 2 — Compressed Architecture")
    print("=" * 80)

    # ── Compress ────────────────────────────────────────────────────────
    print(f"\n[1/3] Compressing {len(corpus_lines):,} lines...")
    t_compress = time.time()
    chunks = compress_corpus_rolling(
        corpus_lines=corpus_lines,
        chunk_size_threshold=512,
        chunk_overlap_tokens=128,
        llm=compress_llm,
    )
    compress_sec   = time.time() - t_compress
    total_orig     = sum(c.original_tokens  for c in chunks)
    total_comp     = sum(c.compressed_tokens for c in chunks)
    compress_ratio = total_comp / total_orig if total_orig else 0
    print(f"  [OK] {len(chunks):,} chunks in {compress_sec:.1f}s")
    print(f"  Ratio: {compress_ratio:.3f}  ({total_orig:,} -> {total_comp:,} tokens)")

    results_2a = []
    results_2b = []

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_dir:

        # ── Index ────────────────────────────────────────────────────────
        print(f"\n[2/3] Building ChromaDB index ({embed_cfg['backend']})...")
        retriever = CachedChromaRetriever(
            collection_name="exp_corpus",
            persist_directory=tmp_dir,
            embedding_model_name=embed_cfg["model"] if embed_cfg["backend"] == "sentence-transformers" else None,
            embedding_backend=embed_cfg["backend"],
            cache_size=500,
            cache_threshold=0.85,
        )
        retriever.add_chunks(chunks)
        print(f"  [OK] {retriever.collection.count():,} chunks indexed")

        # ── Query ────────────────────────────────────────────────────────
        print(f"\n[3/3] Running queries...")

        for q in QUESTIONS:
            print(f"\n  [{q['id']}] {q['difficulty'].upper()}: {q['question'][:60]}")

            # -- 2a: compressed summaries only --
            retriever.cache.clear()
            t_ret = time.time()
            hits  = retriever.search(q["question"], top_k=5, use_cache=True)
            ret_ms = (time.time() - t_ret) * 1000

            context_text = "\n".join(
                f"[chunk {h['chunk_id']}] {h['compressed_summary']}" for h in hits
            )
            prompt_2a = (
                f"Relevant compressed passages from the corpus:\n\n{context_text}\n\n"
                f"Question: {q['question']}\n\nAnswer:"
            )
            tokens_2a = _estimate_tokens(prompt_2a)
            answer_2a, latency_2a = _call_llm(reason_llm, prompt_2a)
            _, _, f1_2a = _score_answer(answer_2a, q["keywords"])

            # cache hit pass
            t_hit = time.time()
            retriever.search(q["question"], top_k=5, use_cache=True)
            hit_ms = (time.time() - t_hit) * 1000

            print(f"    2a (summary)  | tokens:{tokens_2a:,}  ret:{ret_ms:.1f}ms  "
                  f"llm:{latency_2a:.2f}s  hit:{hit_ms:.1f}ms  F1:{f1_2a:.3f}")

            results_2a.append({
                "question_id":     q["id"],
                "difficulty":      q["difficulty"],
                "prompt_tokens":   tokens_2a,
                "retrieval_ms":    ret_ms,
                "cache_hit_ms":    hit_ms,
                "latency_sec":     latency_2a,
                "f1":              f1_2a,
                "chunks_retrieved": len(hits),
                "answer_snippet":  answer_2a[:200],
            })

            # -- 2b: compressed summaries + raw detail for top chunk --
            raw_text = ""
            if hits:
                top_chunk = retriever.get_chunk_by_id(hits[0]["chunk_id"])
                if top_chunk:
                    raw_text = top_chunk.get("raw_text", "")

            context_2b = context_text + (
                f"\n\n[Full text of most relevant chunk]:\n{raw_text}" if raw_text else ""
            )
            prompt_2b = (
                f"Relevant passages from the corpus:\n\n{context_2b}\n\n"
                f"Question: {q['question']}\n\nAnswer:"
            )
            tokens_2b = _estimate_tokens(prompt_2b)
            answer_2b, latency_2b = _call_llm(reason_llm, prompt_2b)
            _, _, f1_2b = _score_answer(answer_2b, q["keywords"])

            print(f"    2b (raw+sum)  | tokens:{tokens_2b:,}  llm:{latency_2b:.2f}s  F1:{f1_2b:.3f}")

            results_2b.append({
                "question_id":    q["id"],
                "difficulty":     q["difficulty"],
                "prompt_tokens":  tokens_2b,
                "latency_sec":    latency_2b,
                "f1":             f1_2b,
                "answer_snippet": answer_2b[:200],
            })

        # cleanup
        try:
            if hasattr(retriever, "client"):
                retriever.client.clear_system_cache()
            del retriever
        except Exception:
            pass

    return {
        "compression": {
            "chunks": len(chunks),
            "time_sec": compress_sec,
            "ratio": compress_ratio,
            "original_tokens": total_orig,
            "compressed_tokens": total_comp,
        },
        "exp_2a": results_2a,
        "exp_2b": results_2b,
    }


# ── Comparison & Markdown generation ─────────────────────────────────────────

def _pass_fail(value: float, target: float, higher_is_better: bool = True) -> str:
    if higher_is_better:
        return "[PASS]" if value >= target else "[FAIL]"
    return "[PASS]" if value <= target else "[FAIL]"


def generate_report(exp1: list[dict], exp2: dict, corpus_lines: int,
                    models: dict, embed_cfg: dict, run_date: str) -> str:
    """Build the experiment_results.md markdown string."""

    def avg(lst, key): return sum(r[key] for r in lst) / len(lst) if lst else 0

    e1_avg_tokens    = avg(exp1, "prompt_tokens")
    e1_avg_latency   = avg(exp1, "latency_sec")
    e1_avg_f1        = avg(exp1, "f1")

    e2a = exp2["exp_2a"]
    e2b = exp2["exp_2b"]
    e2a_avg_tokens   = avg(e2a, "prompt_tokens")
    e2a_avg_latency  = avg(e2a, "latency_sec")
    e2a_avg_f1       = avg(e2a, "f1")
    e2a_avg_ret_ms   = avg(e2a, "retrieval_ms")
    e2a_avg_hit_ms   = avg(e2a, "cache_hit_ms")

    e2b_avg_tokens   = avg(e2b, "prompt_tokens")
    e2b_avg_latency  = avg(e2b, "latency_sec")
    e2b_avg_f1       = avg(e2b, "f1")

    token_red_2a = (1 - e2a_avg_tokens / e1_avg_tokens) * 100 if e1_avg_tokens else 0
    token_red_2b = (1 - e2b_avg_tokens / e1_avg_tokens) * 100 if e1_avg_tokens else 0

    lat_delta_2a = ((e2a_avg_latency - e1_avg_latency) / e1_avg_latency * 100) if e1_avg_latency else 0
    lat_delta_2b = ((e2b_avg_latency - e1_avg_latency) / e1_avg_latency * 100) if e1_avg_latency else 0

    f1_delta_2a  = ((e2a_avg_f1 - e1_avg_f1) / e1_avg_f1 * 100) if e1_avg_f1 else 0
    f1_delta_2b  = ((e2b_avg_f1 - e1_avg_f1) / e1_avg_f1 * 100) if e1_avg_f1 else 0

    thr = THRESHOLDS

    md = f"""# Experiment Results: Standard LLM vs Compressed Architecture

> **Run date:** {run_date}
> **Corpus:** {corpus_lines:,} lines (Pride & Prejudice excerpt)
> **Status:** Local-only execution (no API keys, no cloud)

---

## Model Configuration

| Role | Model | Backend |
|------|-------|---------|
| Compression / Summarisation | `{models['compression']}` | Ollama (local) |
| Embeddings | `{embed_cfg['model']}` | {embed_cfg['backend']} |
| Reasoning | `{models['reasoning']}` | Ollama (local) |

---

## Pass/Fail Thresholds

| Metric | Threshold | Rationale |
|--------|-----------|-----------|
| Latency delta vs baseline | ±{thr['latency_delta_pct']:.0f}% | Architectural overhead must be within 10% |
| F1 accuracy delta vs baseline | ±{thr['f1_delta_pct']:.0f}% | Answer quality within 20% of full-corpus baseline |
| Token reduction vs baseline | ≥{thr['token_reduction_pct']:.0f}% | Core efficiency target |

---

## Experiment 1 — Standard LLM Baseline (Raw Corpus Injected)

Full raw corpus injected into the reasoning LLM on every query. No preprocessing,
no retrieval, no compression. This is the cost/latency ceiling we are beating.

| Question | Difficulty | Prompt Tokens | Latency (s) | F1 |
|----------|------------|--------------|-------------|-----|
"""
    for r in exp1:
        md += f"| {r['question_id']} | {r['difficulty']} | {r['prompt_tokens']:,} | {r['latency_sec']:.2f} | {r['f1']:.3f} |\n"
    md += f"| **Average** | — | **{e1_avg_tokens:,.0f}** | **{e1_avg_latency:.2f}** | **{e1_avg_f1:.3f}** |\n"

    md += f"""
---

## Experiment 2a — Compressed Architecture (Summaries Only)

Corpus compressed with `{models['compression']}`, indexed in ChromaDB with `{embed_cfg['model']}`.
Reasoning LLM receives only the top-5 retrieved compressed summaries (~50 tokens each).

### Compression Stats

| Metric | Value |
|--------|-------|
| Chunks produced | {exp2['compression']['chunks']:,} |
| Compression time | {exp2['compression']['time_sec']:.1f}s (one-time) |
| Compression ratio | {exp2['compression']['ratio']:.3f} |
| Original tokens | {exp2['compression']['original_tokens']:,} |
| Compressed tokens | {exp2['compression']['compressed_tokens']:,} |

### Query Results

| Question | Difficulty | Prompt Tokens | Ret (ms) | Hit (ms) | Latency (s) | F1 | Token Δ |
|----------|------------|--------------|----------|----------|-------------|-----|---------|
"""
    for r, b in zip(e2a, exp1):
        tok_delta = (1 - r["prompt_tokens"] / b["prompt_tokens"]) * 100 if b["prompt_tokens"] else 0
        md += (f"| {r['question_id']} | {r['difficulty']} | {r['prompt_tokens']:,} | "
               f"{r['retrieval_ms']:.1f} | {r['cache_hit_ms']:.1f} | {r['latency_sec']:.2f} | "
               f"{r['f1']:.3f} | -{tok_delta:.1f}% |\n")
    md += (f"| **Average** | — | **{e2a_avg_tokens:,.0f}** | **{e2a_avg_ret_ms:.1f}** | "
           f"**{e2a_avg_hit_ms:.1f}** | **{e2a_avg_latency:.2f}** | **{e2a_avg_f1:.3f}** | "
           f"**-{token_red_2a:.1f}%** |\n")

    md += f"""
---

## Experiment 2b — Compressed Architecture (Summaries + Raw Detail)

Same pipeline as 2a but the reasoning LLM also receives the full raw text of the
most relevant chunk via the pointer model (`get_chunk_by_id`).

| Question | Difficulty | Prompt Tokens | Latency (s) | F1 | Token Δ |
|----------|------------|--------------|-------------|-----|---------|
"""
    for r, b in zip(e2b, exp1):
        tok_delta = (1 - r["prompt_tokens"] / b["prompt_tokens"]) * 100 if b["prompt_tokens"] else 0
        md += (f"| {r['question_id']} | {r['difficulty']} | {r['prompt_tokens']:,} | "
               f"{r['latency_sec']:.2f} | {r['f1']:.3f} | -{tok_delta:.1f}% |\n")
    md += (f"| **Average** | — | **{e2b_avg_tokens:,.0f}** | **{e2b_avg_latency:.2f}** | "
           f"**{e2b_avg_f1:.3f}** | **-{token_red_2b:.1f}%** |\n")

    lat_pf_2a  = _pass_fail(abs(lat_delta_2a),  thr["latency_delta_pct"],  higher_is_better=False)
    lat_pf_2b  = _pass_fail(abs(lat_delta_2b),  thr["latency_delta_pct"],  higher_is_better=False)
    f1_pf_2a   = _pass_fail(abs(f1_delta_2a),   thr["f1_delta_pct"],       higher_is_better=False)
    f1_pf_2b   = _pass_fail(abs(f1_delta_2b),   thr["f1_delta_pct"],       higher_is_better=False)
    tok_pf_2a  = _pass_fail(token_red_2a,        thr["token_reduction_pct"], higher_is_better=True)
    tok_pf_2b  = _pass_fail(token_red_2b,        thr["token_reduction_pct"], higher_is_better=True)

    md += f"""
---

## Cross-Experiment Comparison

| Metric | Baseline (Exp 1) | Exp 2a (Summary) | Exp 2b (Summary+Raw) |
|--------|-----------------|-----------------|---------------------|
| Avg prompt tokens | {e1_avg_tokens:,.0f} | {e2a_avg_tokens:,.0f} | {e2b_avg_tokens:,.0f} |
| Token reduction | — | **-{token_red_2a:.1f}%** {tok_pf_2a} | **-{token_red_2b:.1f}%** {tok_pf_2b} |
| Avg reasoning latency (s) | {e1_avg_latency:.2f} | {e2a_avg_latency:.2f} ({lat_delta_2a:+.1f}%) {lat_pf_2a} | {e2b_avg_latency:.2f} ({lat_delta_2b:+.1f}%) {lat_pf_2b} |
| Avg retrieval latency (ms) | N/A | {e2a_avg_ret_ms:.1f} (miss) / {e2a_avg_hit_ms:.1f} (hit) | same |
| Avg F1 | {e1_avg_f1:.3f} | {e2a_avg_f1:.3f} ({f1_delta_2a:+.1f}%) {f1_pf_2a} | {e2b_avg_f1:.3f} ({f1_delta_2b:+.1f}%) {f1_pf_2b} |

### Threshold Summary

| Threshold | Target | Exp 2a | Exp 2b |
|-----------|--------|--------|--------|
| Token reduction ≥{thr['token_reduction_pct']:.0f}% | ≥{thr['token_reduction_pct']:.0f}% | {token_red_2a:.1f}% {tok_pf_2a} | {token_red_2b:.1f}% {tok_pf_2b} |
| Latency delta ≤±{thr['latency_delta_pct']:.0f}% | ≤±{thr['latency_delta_pct']:.0f}% | {lat_delta_2a:+.1f}% {lat_pf_2a} | {lat_delta_2b:+.1f}% {lat_pf_2b} |
| F1 delta ≤±{thr['f1_delta_pct']:.0f}% | ≤±{thr['f1_delta_pct']:.0f}% | {f1_delta_2a:+.1f}% {f1_pf_2a} | {f1_delta_2b:+.1f}% {f1_pf_2b} |

---

## Key Observations

- **Token efficiency**: Exp 2a delivers {token_red_2a:.0f}% token reduction vs the full-corpus
  baseline, well {"above" if token_red_2a >= 90 else "below"} the 90% target.
- **Latency**: Reasoning latency {"improved" if lat_delta_2a < 0 else "increased"} by
  {abs(lat_delta_2a):.0f}% in Exp 2a (fewer tokens = faster LLM). Retrieval adds
  {e2a_avg_ret_ms:.0f}ms (miss) / {e2a_avg_hit_ms:.1f}ms (cache hit).
- **F1 quality**: Exp 2a F1 {"matched" if abs(f1_delta_2a) <= 10 else "diverged from"} the
  baseline within {abs(f1_delta_2a):.0f}% (threshold: ±20%).
- **Raw detail (2b)**: Adding the pointer-model raw-text fetch gives
  {f1_delta_2b - f1_delta_2a:+.0f}% F1 delta vs 2a at the cost of
  {e2b_avg_tokens - e2a_avg_tokens:+,.0f} extra tokens.
- **Cache benefit**: Repeated / similar queries drop from {e2a_avg_ret_ms:.0f}ms to
  {e2a_avg_hit_ms:.1f}ms ({e2a_avg_ret_ms / max(e2a_avg_hit_ms, 0.1):.0f}x speedup).

---

## Next Steps

- Run with `--full` flag (25K lines) to validate results at production corpus scale.
- Populate persistent ChromaDB with `quick_compress_and_save.py` then run
  `accuracy_benchmarks.py` for full F1 + precision/recall metrics.
- Switch embedding backend to Ollama:
  `$env:CONTEXT_OPTIMIZER_EMBEDDING_BACKEND = "ollama"` then re-run.
"""
    return md


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run end-to-end experiments")
    parser.add_argument("--lines", type=int, default=500,
                        help="Number of corpus lines (default: 500 for quick run)")
    parser.add_argument("--full", action="store_true",
                        help="Use full 25K medium corpus (hours with Ollama)")
    args = parser.parse_args()

    corpus_lines_limit = 25_000 if args.full else args.lines
    run_date = datetime.now().strftime("%Y-%m-%d %H:%M")

    print("=" * 80)
    print("CONTEXT OPTIMIZER — End-to-End Experiment Runner")
    print("=" * 80)
    print(f"  Corpus size : {corpus_lines_limit:,} lines")
    print(f"  Run date    : {run_date}\n")

    # ── Models ─────────────────────────────────────────────────────────
    print("[Setup] Initialising models...")
    embed_cfg      = get_embedding_config()
    compress_llm   = build_compression_llm()
    reason_llm     = build_reasoning_llm()

    models = {
        "compression": os.getenv("CONTEXT_OPTIMIZER_COMPRESSOR_MODEL", "llama3.2:3b"),
        "reasoning":   os.getenv("CONTEXT_OPTIMIZER_REASONING_MODEL",  "qwen2.5-coder:7b"),
    }

    if reason_llm is None:
        print("\n[ERROR] Reasoning LLM unavailable — cannot run experiments.")
        sys.exit(1)

    # ── Corpus ─────────────────────────────────────────────────────────
    print("\n[Corpus] Loading Pride and Prejudice corpus...")
    pride_file = Path(__file__).parent / "test_data" / "books_pride-prejudice.txt"
    if not pride_file.exists():
        print(f"[ERROR] {pride_file} not found. Run download_test_data.py first.")
        sys.exit(1)

    with open(pride_file, encoding="utf-8", errors="ignore") as f:
        all_lines = [ln.strip() for ln in f if ln.strip()]

    corpus = all_lines[:corpus_lines_limit]
    print(f"  Loaded {len(corpus):,} lines from Pride & Prejudice")

    # ── Run Exp 1 (baseline) ───────────────────────────────────────────
    exp1_results = run_experiment1(corpus, reason_llm)

    # ── Run Exp 2 ──────────────────────────────────────────────────────
    exp2_results = run_experiment2(corpus, compress_llm, reason_llm, embed_cfg)

    # ── JSON output ────────────────────────────────────────────────────
    bench_dir  = Path(__file__).parent
    json_out   = bench_dir / "EXPERIMENT_RESULTS.json"
    report_out = bench_dir.parent.parent / "docs" / "experiments" / "experiment_results.md"

    payload = {
        "run_date":        run_date,
        "corpus_lines":    len(corpus),
        "models":          models,
        "embedding":       embed_cfg,
        "thresholds":      THRESHOLDS,
        "experiment_1":    exp1_results,
        "experiment_2":    exp2_results,
    }
    with open(json_out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[OK] JSON results saved: {json_out}")

    # ── Markdown report ────────────────────────────────────────────────
    md = generate_report(exp1_results, exp2_results, len(corpus),
                         models, embed_cfg, run_date)
    report_out.parent.mkdir(parents=True, exist_ok=True)
    with open(report_out, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"[OK] Markdown report: {report_out}")

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()
