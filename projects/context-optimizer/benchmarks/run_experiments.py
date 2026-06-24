"""
End-to-End Experiment Runner: Standard LLM Baseline vs Compressed Architecture

Implements the two experiments from EXPERIMENTS_CONSOLIDATED.md:
  Exp 1  Raw corpus injected directly into reasoning LLM (no architecture).
  Exp 2  Compress -> retrieve summaries -> adaptive raw-fetch if needed (targeted)
         or ALL summaries + Tree-of-Thought reasoning (aggregated).
         get_context_details() -> reason.  The LLM sees both tools and makes
         the raw-detail fetch only when it judges summaries insufficient.

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

import argparse
import json
import os
import re
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root / "src"))
sys.path.insert(0, str(_root))

from context_optimizer.cached_retriever import CachedChromaRetriever
from context_optimizer.compressor import compress_corpus_rolling
from llm_provider import (
    build_compression_llm,
    build_reasoning_llm,
    get_embedding_config,
)

# ── Pass/Fail thresholds ──────────────────────────────────────────────────────
THRESHOLDS = {
    "latency_delta_pct": 10.0,  # +/- 10% vs baseline latency
    "f1_delta_pct": 20.0,  # +/- 20% vs baseline F1
    "token_reduction_pct": 90.0,  # >= 90% fewer tokens than baseline
}

# ── Questions with expected-keyword ground truth ──────────────────────────────
QUESTIONS = [
    {
        # TARGETED: answer lives in a handful of specific passages about one character.
        "id": "q001",
        "difficulty": "easy",
        "query_type": "targeted",
        "question": "Who is Elizabeth Bennet and what are her main character traits?",
        "keywords": [
            "elizabeth",
            "bennet",
            "protagonist",
            "witty",
            "intelligent",
            "darcy",
            "independent",
        ],
    },
    {
        # TARGETED: a single, locatable fact (character + place name).
        "id": "q002",
        "difficulty": "easy",
        "query_type": "targeted",
        "question": "Who is Mr. Bingley and where does he settle?",
        "keywords": ["bingley", "netherfield", "wealthy", "gentleman", "jane"],
    },
    {
        # AGGREGATED: thematic question — evidence is spread across the whole corpus.
        "id": "q003",
        "difficulty": "medium",
        "query_type": "aggregated",
        "question": "What social themes are central to the story?",
        "keywords": ["marriage", "class", "society", "wealth", "reputation", "family"],
    },
    {
        # AGGREGATED: requires synthesising descriptions of multiple characters.
        "id": "q004",
        "difficulty": "medium",
        "query_type": "aggregated",
        "question": "Describe the Bennet family members",
        "keywords": [
            "jane",
            "elizabeth",
            "lydia",
            "kitty",
            "mary",
            "mrs bennet",
            "mr bennet",
        ],
    },
    {
        # TARGETED: a defined character arc with a start-state, trigger, and end-state.
        "id": "q005",
        "difficulty": "hard",
        "query_type": "targeted",
        "question": "What is Mr. Darcy's initial attitude towards Elizabeth and how does it change?",
        "keywords": [
            "darcy",
            "proud",
            "tolerable",
            "inferior",
            "love",
            "admire",
            "respect",
            "change",
        ],
    },
    {
        # AGGREGATED: a recurring structural theme — evidence spans every chapter.
        "id": "q006",
        "difficulty": "hard",
        "query_type": "aggregated",
        "question": "How does the theme of first impressions affect relationships in the novel?",
        "keywords": [
            "impression",
            "prejudice",
            "pride",
            "misjudge",
            "misunderstand",
            "reveal",
            "character",
        ],
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
    """
    Keyword-overlap F1.
    NOTE: precision is penalised for verbosity (found / total_answer_words).
    Use alongside judge_score for a fairer picture.
    """
    answer_lower = answer.lower()
    answer_words = set(answer_lower.split())
    found = sum(1 for kw in keywords if kw.lower() in answer_lower)
    recall = found / len(keywords) if keywords else 0.0
    precision = found / len(answer_words) if answer_words else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return precision, recall, f1


_JUDGE_SYSTEM = (
    "You are a strict but fair evaluation judge. "
    "Given a question, the key concepts that should appear in a correct answer, "
    "and an actual answer, rate how well the answer addresses the question and covers "
    "the key concepts. Reply with ONLY a single decimal number between 0.0 and 1.0. "
    "1.0 = perfect coverage and accuracy, 0.0 = completely wrong or empty. "
    "No explanation, no other text — just the number."
)


def _judge_answer(llm, question: str, keywords: list[str], answer: str) -> float:
    """
    LLM-as-judge: use the compression LLM (llama3.2:3b) to score the answer
    semantically rather than via keyword overlap.

    Returns a float in [0.0, 1.0].  Falls back to recall-only keyword score
    if the LLM call fails or returns an unparseable response.
    """
    if llm is None or answer.startswith("[ERROR]"):
        # fallback: recall only (not penalised for verbosity)
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
        # Extract first float-like token from the response
        import re

        m = re.search(r"\b([01]?\.\d+|[01])\b", raw.strip())
        if m:
            score = float(m.group(1))
            return max(0.0, min(1.0, score))
    except Exception:
        pass
    # fallback: recall only
    found = sum(1 for kw in keywords if kw.lower() in answer.lower())
    return found / len(keywords) if keywords else 0.0


def _call_llm(llm, prompt: str, system: str = SYSTEM_PROMPT) -> tuple[str, float]:
    """Invoke LLM and return (answer_text, elapsed_seconds)."""
    from langchain_core.messages import HumanMessage, SystemMessage

    t0 = time.time()
    try:
        response = llm.invoke(
            [SystemMessage(content=system), HumanMessage(content=prompt)]
        )
        elapsed = time.time() - t0
        text = response.content if hasattr(response, "content") else str(response)
        return text.strip(), elapsed
    except Exception as exc:
        elapsed = time.time() - t0
        return f"[ERROR] {exc}", elapsed


# ── Exp 2 — Adaptive agentic query (LLM decides if raw detail is needed) ─────

_ADAPTIVE_T1_SYSTEM = (
    "You are a helpful literary assistant with access to compressed passage summaries. "
    "Decide whether the summaries are sufficient to answer the question fully.\n\n"
    "Respond ONLY with a JSON object — no other text:\n"
    '  Summaries sufficient  → {"answer": "your complete answer", "needs_raw": false}\n'
    '  Need one chunk\'s full text → {"answer": null, "needs_raw": true, "chunk_id": "<id>"}\n\n'
    "Choose needs_raw=true only when the summary is clearly truncated and the missing "
    "detail would materially change your answer (exact wording, precise number, "
    "multi-sentence passage). Available chunk IDs are listed in the context."
)

_ADAPTIVE_T2_SYSTEM = (
    "You are a helpful literary assistant. You now have both compressed summaries and "
    "the full raw text of the chunk you requested. Answer the question concisely and "
    "accurately based on the provided text."
)

# ── ToT system prompt for aggregated (corpus-wide) queries ────────────────────
# Used when a question's answer is distributed across many chunks and cannot be
# recovered by top-k vector retrieval alone.  The model sees ALL compressed
# summaries and is required to walk three reasoning paths before answering.

_TOT_AGGREGATED_SYSTEM = (
    "You are a literary analyst. For broad thematic questions that span an entire corpus "
    "you MUST reason through three distinct paths before producing a final answer.\n\n"
    "PATH A — Direct Evidence: cite explicit statements from the summaries that directly "
    "address the question.\n"
    "PATH B — Contextual Patterns: identify recurring themes, motifs, or character "
    "behaviours that appear across multiple passages.\n"
    "PATH C — Synthesis: combine the evidence from A and B into a complete, nuanced answer "
    "that covers all key aspects.\n\n"
    "You MUST use EXACTLY this format (do not omit any heading):\n"
    "PATH A: [your direct-evidence reasoning]\n"
    "PATH B: [your contextual-patterns reasoning]\n"
    "PATH C: [your synthesis reasoning]\n"
    "FINAL ANSWER: [complete synthesised answer — this is what will be evaluated]"
)


def _parse_turn1_json(raw: str) -> dict:
    """
    Robustly extract the Turn-1 JSON decision from a model response.

    Tries in order:
      1. Fenced code block  (```json ... ```)
      2. Bare JSON object   ({...})
      3. Key-value regex    (needs_raw / chunk_id / answer anywhere in text)
      4. Plain-text heuristic (contains 'need', 'raw', 'fetch' → needs_raw=True)
      5. Hard fallback      (needs_raw=False, answer=raw text)

    Returns a dict with keys: needs_raw (bool), chunk_id (str|None), answer (str|None).
    """
    import json as _json
    import re as _re

    # Strip common model thinking tags (e.g. <think>...</think>)
    cleaned = _re.sub(r"<think>[\s\S]*?</think>", "", raw, flags=_re.IGNORECASE).strip()

    def _try_parse(s: str) -> dict | None:
        try:
            obj = _json.loads(s)
            if isinstance(obj, dict):
                return {
                    "needs_raw": bool(obj.get("needs_raw", False)),
                    "chunk_id": obj.get("chunk_id")
                    or obj.get("chunkId")
                    or obj.get("chunk_id"),
                    "answer": str(obj["answer"]) if obj.get("answer") else None,
                }
        except Exception:
            pass
        return None

    # 1. Fenced block
    fence_m = _re.search(r"```(?:json)?\s*([\s\S]+?)```", cleaned)
    if fence_m:
        result = _try_parse(fence_m.group(1).strip())
        if result:
            return result

    # 2. Bare JSON object (outermost {...})
    obj_m = _re.search(r"\{[\s\S]*\}", cleaned)
    if obj_m:
        result = _try_parse(obj_m.group(0))
        if result:
            return result

    # 3. Key-value regex fallback — tolerates single quotes, missing quotes, etc.
    needs_raw = False
    chunk_id = None
    answer = None

    nr_m = _re.search(r'"?needs_raw"?\s*:\s*(true|false)', cleaned, _re.IGNORECASE)
    if nr_m:
        needs_raw = nr_m.group(1).lower() == "true"

    cid_m = _re.search(r'"?chunk_id"?\s*:\s*["\']?([a-zA-Z0-9_\-]+)["\']?', cleaned)
    if cid_m:
        chunk_id = cid_m.group(1)

    ans_m = _re.search(r'"answer"\s*:\s*"((?:[^"\\]|\\.)*)"', cleaned)
    if not ans_m:
        ans_m = _re.search(r"'answer'\s*:\s*'((?:[^'\\]|\\.)*)'", cleaned)
    if ans_m:
        answer = ans_m.group(1)

    if nr_m or cid_m or ans_m:
        return {"needs_raw": needs_raw, "chunk_id": chunk_id, "answer": answer}

    # 4. Plain-text heuristic: model said it needs more detail
    if _re.search(
        r"\b(need(s)?( the)? raw|fetch|get.{0,20}chunk|insufficient)\b",
        cleaned,
        _re.IGNORECASE,
    ):
        cid_heuristic = _re.search(r"\b(chunk[-_]?\d+)\b", cleaned, _re.IGNORECASE)
        return {
            "needs_raw": True,
            "chunk_id": cid_heuristic.group(1) if cid_heuristic else None,
            "answer": None,
        }

    # 5. Hard fallback: treat entire response as the answer
    return {"needs_raw": False, "chunk_id": None, "answer": cleaned or None}


def _run_tot_aggregated_query(
    reason_llm,
    chunks: list,  # all CompressedChunk objects
    q: dict,
    judge_llm=None,
) -> tuple:
    """
    Tree-of-Thought query for aggregated (corpus-wide) questions.

    Injects ALL compressed summaries and instructs the reasoning model to walk
    three analytical paths (direct evidence → patterns → synthesis) before
    producing a FINAL ANSWER.  Only that final section is scored.

    Returns: answer, latency_sec, prompt_tokens, f1, judge_score
    """
    context_text = "\n".join(
        f"[chunk {c.chunk_id}] {c.compressed_summary}" for c in chunks
    )
    prompt = (
        f"Corpus summaries ({len(chunks):,} passages — full corpus coverage):\n\n"
        f"{context_text}\n\n"
        f"Question: {q['question']}\n\n"
        "Walk all three reasoning paths as instructed, then give your FINAL ANSWER:"
    )
    tokens = _estimate_tokens(prompt)
    answer_raw, latency = _call_llm(reason_llm, prompt, system=_TOT_AGGREGATED_SYSTEM)

    # Extract only the FINAL ANSWER section for scoring
    final_m = re.search(
        r"FINAL ANSWER\s*:\s*(.+)", answer_raw, re.DOTALL | re.IGNORECASE
    )
    answer = final_m.group(1).strip() if final_m else answer_raw

    _, _, f1 = _score_answer(answer, q["keywords"])
    judge = _judge_answer(judge_llm, q["question"], q["keywords"], answer)
    return answer, latency, tokens, f1, judge


def _run_adaptive_query(
    reason_llm,
    hits: list,
    q: dict,
    retriever,
    judge_llm=None,
) -> tuple:
    """
    Two-turn adaptive query.

    Turn 1: Give the LLM compressed summaries.  Ask it to either answer or
            identify a chunk whose full text it needs.
    Turn 2: (only if needs_raw=true) Fetch that chunk, re-prompt with full text.

    Returns:
        answer, latency_sec, prompt_tokens, raw_fetched, raw_fetch_tokens, f1, judge_score
    """
    chunk_ids = [h["chunk_id"] for h in hits]
    context_text = "\n".join(
        f"[chunk {h['chunk_id']}] {h['compressed_summary']}" for h in hits
    )
    prompt_t1 = (
        f"Compressed summaries (chunk IDs available for raw-detail fetch: "
        f"{', '.join(chunk_ids)}):\n\n{context_text}\n\n"
        f"Question: {q['question']}\n\n"
        "Respond with JSON as instructed:"
    )
    tokens_used = _estimate_tokens(prompt_t1)

    t0 = time.time()
    raw_t1, _ = _call_llm(reason_llm, prompt_t1, system=_ADAPTIVE_T1_SYSTEM)

    # Debug: show turn-1 raw response so we can see if adaptive fetch is working
    print(f"      [T1 raw] {raw_t1[:200].replace(chr(10), ' ')}")

    # --- parse Turn-1 response (robust, multi-strategy) ---
    parsed = _parse_turn1_json(raw_t1)
    needs_raw = parsed["needs_raw"]
    requested_chunk = parsed["chunk_id"]
    answer = parsed["answer"]

    raw_fetched = False
    raw_fetch_tokens = 0

    if needs_raw and requested_chunk:
        detail = retriever.get_chunk_by_id(requested_chunk)
        raw_text = detail.get("raw_text", "") if detail else ""
        if raw_text:
            raw_fetched = True
            raw_fetch_tokens = _estimate_tokens(raw_text)
            prompt_t2 = (
                f"Compressed summaries:\n\n{context_text}\n\n"
                f"Full raw text of chunk '{requested_chunk}' (as requested):\n\n{raw_text}\n\n"
                f"Question: {q['question']}\n\nAnswer:"
            )
            tokens_used += _estimate_tokens(prompt_t2)
            answer, _ = _call_llm(reason_llm, prompt_t2, system=_ADAPTIVE_T2_SYSTEM)
        else:
            answer = raw_t1 if answer is None else answer

    latency = time.time() - t0
    _, _, f1 = _score_answer(answer or "", q["keywords"])
    judge = _judge_answer(judge_llm, q["question"], q["keywords"], answer or "")
    return answer or "", latency, tokens_used, raw_fetched, raw_fetch_tokens, f1, judge


# ── Experiment 1 — Standard LLM baseline ─────────────────────────────────────


def run_experiment1(corpus_lines: list[str], llm, judge_llm=None) -> list[dict]:
    """Inject the full raw corpus into the reasoning LLM for each question."""
    corpus_text = "\n".join(corpus_lines)
    prompt_base = _estimate_tokens(corpus_text)

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
        _, _, kw_f1 = _score_answer(answer, q["keywords"])
        judge_score = _judge_answer(judge_llm, q["question"], q["keywords"], answer)

        print(
            f"    Tokens: {total_tokens:,}   Latency: {elapsed:.2f}s   "
            f"KW-F1: {kw_f1:.3f}   Judge: {judge_score:.2f}"
        )
        print(f"    Answer (first 120 chars): {answer[:120]}")

        results.append(
            {
                "question_id": q["id"],
                "difficulty": q["difficulty"],
                "prompt_tokens": total_tokens,
                "latency_sec": elapsed,
                "f1": kw_f1,
                "judge_score": judge_score,
                "answer_snippet": answer[:200],
            }
        )

    return results


# ── Experiment 2 — Compressed Architecture ───────────────────────────────────


def run_experiment2(
    corpus_lines: list[str], compress_llm, reason_llm, embed_cfg: dict, judge_llm=None
) -> dict:
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
    compress_sec = time.time() - t_compress
    total_orig = sum(c.original_tokens for c in chunks)
    total_comp = sum(c.compressed_tokens for c in chunks)
    compress_ratio = total_comp / total_orig if total_orig else 0
    print(f"  [OK] {len(chunks):,} chunks in {compress_sec:.1f}s")
    print(f"  Ratio: {compress_ratio:.3f}  ({total_orig:,} -> {total_comp:,} tokens)")

    results_2 = []

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_dir:

        # ── Index ────────────────────────────────────────────────────────
        print(f"\n[2/3] Building ChromaDB index ({embed_cfg['backend']})...")
        retriever = CachedChromaRetriever(
            collection_name="exp_corpus",
            persist_directory=tmp_dir,
            embedding_model_name=(
                embed_cfg["model"]
                if embed_cfg["backend"] == "sentence-transformers"
                else None
            ),
            embedding_backend=embed_cfg["backend"],
            cache_size=500,
            cache_threshold=0.85,
        )
        retriever.add_chunks(chunks)
        print(f"  [OK] {retriever.collection.count():,} chunks indexed")

        # ── Query ────────────────────────────────────────────────────────
        print(f"\n[3/3] Running queries...")

        for q in QUESTIONS:
            q_type = q.get("query_type", "targeted")
            print(
                f"\n  [{q['id']}] {q['difficulty'].upper()} [{q_type}]: {q['question'][:55]}"
            )

            if q_type == "aggregated":
                # ── Aggregated: ALL summaries + Tree-of-Thought (no vector retrieval) ──
                answer, latency, tokens, f1, judge = _run_tot_aggregated_query(
                    reason_llm, chunks, q, judge_llm
                )
                print(
                    f"    [ToT/all]   | tokens:{tokens:,}  llm:{latency:.2f}s  "
                    f"KW-F1:{f1:.3f}  Judge:{judge:.2f}"
                )
                results_2.append(
                    {
                        "question_id": q["id"],
                        "difficulty": q["difficulty"],
                        "query_strategy": "tot_aggregated",
                        "prompt_tokens": tokens,
                        "retrieval_ms": 0.0,
                        "cache_hit_ms": 0.0,
                        "latency_sec": latency,
                        "raw_fetched": False,
                        "raw_fetch_tokens": 0,
                        "f1": f1,
                        "judge_score": judge,
                        "chunks_retrieved": len(chunks),
                        "answer_snippet": answer[:200],
                    }
                )

            else:
                # ── Targeted: top-5 retrieval → adaptive raw-fetch if needed ──
                retriever.cache.clear()
                t_ret = time.time()
                hits = retriever.search(q["question"], top_k=5, use_cache=True)
                ret_ms = (time.time() - t_ret) * 1000

                answer, latency, tokens, raw_fetched, raw_tok, f1, judge = (
                    _run_adaptive_query(reason_llm, hits, q, retriever, judge_llm)
                )
                raw_flag = "Y" if raw_fetched else "N"
                print(
                    f"    [adaptive]  | tokens:{tokens:,}  ret:{ret_ms:.1f}ms  "
                    f"llm:{latency:.2f}s  raw:{raw_flag}  KW-F1:{f1:.3f}  Judge:{judge:.2f}"
                )
                results_2.append(
                    {
                        "question_id": q["id"],
                        "difficulty": q["difficulty"],
                        "query_strategy": "targeted_adaptive",
                        "prompt_tokens": tokens,
                        "retrieval_ms": ret_ms,
                        "cache_hit_ms": 0.0,
                        "latency_sec": latency,
                        "raw_fetched": raw_fetched,
                        "raw_fetch_tokens": raw_tok,
                        "f1": f1,
                        "judge_score": judge,
                        "chunks_retrieved": len(hits),
                        "answer_snippet": answer[:200],
                    }
                )

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
        "exp_2": results_2,
    }


# ── Comparison & Markdown generation ─────────────────────────────────────────


def _pass_fail(value: float, target: float, higher_is_better: bool = True) -> str:
    if higher_is_better:
        return "[PASS]" if value >= target else "[FAIL]"
    return "[PASS]" if value <= target else "[FAIL]"


def generate_report(
    exp1: list[dict],
    exp2: dict,
    corpus_lines: int,
    models: dict,
    embed_cfg: dict,
    run_date: str,
) -> str:
    """Build the experiment_results.md markdown string."""

    def avg(lst, key):
        return sum(r[key] for r in lst) / len(lst) if lst else 0

    e1_avg_tokens = avg(exp1, "prompt_tokens")
    e1_avg_latency = avg(exp1, "latency_sec")
    e1_avg_f1 = avg(exp1, "f1")

    e2 = exp2["exp_2"]
    e2_avg_tokens = avg(e2, "prompt_tokens")
    e2_avg_latency = avg(e2, "latency_sec")
    e2_avg_f1 = avg(e2, "f1")
    e2_avg_judge = avg(e2, "judge_score")
    e2_avg_ret_ms = avg(e2, "retrieval_ms")
    raw_fetch_count = sum(1 for r in e2 if r.get("raw_fetched", False))

    e1_avg_judge = avg(exp1, "judge_score")

    token_red = (1 - e2_avg_tokens / e1_avg_tokens) * 100 if e1_avg_tokens else 0
    lat_delta = (
        ((e2_avg_latency - e1_avg_latency) / e1_avg_latency * 100)
        if e1_avg_latency
        else 0
    )
    f1_delta = ((e2_avg_f1 - e1_avg_f1) / e1_avg_f1 * 100) if e1_avg_f1 else 0
    judge_delta = (
        ((e2_avg_judge - e1_avg_judge) / e1_avg_judge * 100) if e1_avg_judge else 0
    )

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
| Judge (LLM-as-judge) | `{models.get('judge', models['reasoning'])}` | Ollama (local) |

---

## Pass/Fail Thresholds

| Metric | Threshold | Rationale |
|--------|-----------|-----------|
| Latency regression vs baseline | ≤+{thr['latency_delta_pct']:.0f}% | Slowdowns >10% fail; improvements always pass |
| Judge-score drop vs baseline | ≤-{thr['f1_delta_pct']:.0f}% | Quality drops >20% fail; improvements always pass |
| KW-F1 drop vs baseline | ≤-{thr['f1_delta_pct']:.0f}% | KW-F1 drops >20% fail; improvements always pass |
| Token reduction vs baseline | ≥{thr['token_reduction_pct']:.0f}% | Core efficiency target |

---

## Experiment 1 — Standard LLM Baseline (Raw Corpus Injected)

Full raw corpus injected into the reasoning LLM on every query. No preprocessing,
no retrieval, no compression. This is the cost/latency ceiling we are beating.

| Question | Difficulty | Prompt Tokens | Latency (s) | KW-F1 | Judge |
|----------|------------|--------------|-------------|-------|-------|
"""  # noqa: E501
    for r in exp1:
        md += (
            f"| {r['question_id']} | {r['difficulty']} | {r['prompt_tokens']:,} | "
            f"{r['latency_sec']:.2f} | {r['f1']:.3f} | {r.get('judge_score', 0):.2f} |\n"
        )
    md += (
        f"| **Average** | — | **{e1_avg_tokens:,.0f}** | **{e1_avg_latency:.2f}** | "
        f"**{e1_avg_f1:.3f}** | **{e1_avg_judge:.2f}** |\n"
    )

    md += f"""
---

## Experiment 2 — Compressed Architecture

Corpus compressed with `{models['compression']}`, indexed in ChromaDB with `{embed_cfg['model']}`.

- **Targeted queries**: top-5 vector retrieval → adaptive raw-fetch if the model judges summaries insufficient.
- **Aggregated queries**: ALL compressed summaries injected + Tree-of-Thought reasoning
  (three analytical paths → synthesised FINAL ANSWER), bypassing vector retrieval.

### Compression Stats

| Metric | Value |
|--------|-------|
| Chunks produced | {exp2['compression']['chunks']:,} |
| Compression time | {exp2['compression']['time_sec']:.1f}s (one-time) |
| Compression ratio | {exp2['compression']['ratio']:.3f} |
| Original tokens | {exp2['compression']['original_tokens']:,} |
| Compressed tokens | {exp2['compression']['compressed_tokens']:,} |

### Query Results

| Question | Difficulty | Strategy | Prompt Tokens | Ret (ms) | Latency (s) | Raw? | KW-F1 | Judge | Token Δ |
|----------|------------|----------|--------------|----------|-------------|------|-------|-------|---------|
"""  # noqa: E501
    for r, b in zip(e2, exp1):
        tok_delta = (
            (1 - r["prompt_tokens"] / b["prompt_tokens"]) * 100
            if b["prompt_tokens"]
            else 0
        )
        raw_flag = "Y" if r.get("raw_fetched") else "N"
        md += (
            f"| {r['question_id']} | {r['difficulty']} | {r.get('query_strategy','—')} | "
            f"{r['prompt_tokens']:,} | {r['retrieval_ms']:.1f} | {r['latency_sec']:.2f} | "
            f"{raw_flag} | {r['f1']:.3f} | {r.get('judge_score', 0):.2f} | -{tok_delta:.1f}% |\n"
        )
    md += (
        f"| **Average** | — | — | **{e2_avg_tokens:,.0f}** | **{e2_avg_ret_ms:.1f}** | "
        f"**{e2_avg_latency:.2f}** | {raw_fetch_count}/{len(e2)} | **{e2_avg_f1:.3f}** | "
        f"**{e2_avg_judge:.2f}** | **-{token_red:.1f}%** |\n"
    )

    # Directional thresholds: only flag regressions, never flag improvements.
    # Latency: FAIL only if latency INCREASED by more than threshold.
    # F1/Judge: FAIL only if quality DROPPED by more than threshold.
    lat_pf = _pass_fail(lat_delta, thr["latency_delta_pct"], higher_is_better=False)
    f1_pf = _pass_fail(f1_delta, -thr["f1_delta_pct"], higher_is_better=True)
    jdg_pf = _pass_fail(judge_delta, -thr["f1_delta_pct"], higher_is_better=True)
    tok_pf = _pass_fail(token_red, thr["token_reduction_pct"], higher_is_better=True)

    md += f"""
---

## Cross-Experiment Comparison

> **Accuracy note**: *Judge score* (LLM-as-judge, 0–1) is the primary quality metric.
> *KW-F1* (keyword-overlap) is secondary — it under-reports quality for verbose answers
> because precision is penalised by answer word count.

| Metric | Baseline (Exp 1) | Exp 2 (Adaptive) |
|--------|-----------------|------------------|
| Avg prompt tokens | {e1_avg_tokens:,.0f} | {e2_avg_tokens:,.0f} |
| Token reduction | — | **-{token_red:.1f}%** {tok_pf} |
| Avg reasoning latency (s) | {e1_avg_latency:.2f} | {e2_avg_latency:.2f} ({lat_delta:+.1f}%) {lat_pf} |
| Avg retrieval latency (ms) | N/A | {e2_avg_ret_ms:.1f} |
| Avg Judge score (0–1) | {e1_avg_judge:.2f} | {e2_avg_judge:.2f} ({judge_delta:+.1f}%) {jdg_pf} |
| Avg KW-F1 (secondary) | {e1_avg_f1:.3f} | {e2_avg_f1:.3f} ({f1_delta:+.1f}%) {f1_pf} |

### Threshold Summary

| Threshold | Target | Exp 2 |
|-----------|--------|-------|
| Token reduction ≥{thr['token_reduction_pct']:.0f}% | ≥{thr['token_reduction_pct']:.0f}% | {token_red:.1f}% {tok_pf} |
| Latency regression ≤+{thr['latency_delta_pct']:.0f}% | ≤+{thr['latency_delta_pct']:.0f}% | {lat_delta:+.1f}% {lat_pf} |
| Judge-score drop ≤-{thr['f1_delta_pct']:.0f}% | ≤-{thr['f1_delta_pct']:.0f}% | {judge_delta:+.1f}% {jdg_pf} |
| KW-F1 drop ≤-{thr['f1_delta_pct']:.0f}% | ≤-{thr['f1_delta_pct']:.0f}% | {f1_delta:+.1f}% {f1_pf} |

---

## Key Observations

- **Token efficiency**: Exp 2 delivers {token_red:.0f}% token reduction vs the full-corpus
  baseline, {"above" if token_red >= 90 else "below"} the 90% target.
- **Latency**: Reasoning latency {"improved" if lat_delta < 0 else "increased"} by
  {abs(lat_delta):.0f}% (fewer tokens = faster LLM). Retrieval adds {e2_avg_ret_ms:.0f}ms.
- **Quality**: Judge score delta {judge_delta:+.1f}% vs baseline (threshold: ±{thr['f1_delta_pct']:.0f}%) {jdg_pf}.
- **Adaptive raw fetch**: triggered for {raw_fetch_count} of {len(e2)} targeted queries
  ({raw_fetch_count / max(len(e2), 1) * 100:.0f}%).
  Raw fetch added ~{avg(e2, 'raw_fetch_tokens'):.0f} extra tokens where used.

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
    parser.add_argument(
        "--lines",
        type=int,
        default=500,
        help="Number of corpus lines (default: 500 for quick run)",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Use full 25K medium corpus (hours with Ollama)",
    )
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
    embed_cfg = get_embedding_config()
    compress_llm = build_compression_llm()
    reason_llm = build_reasoning_llm()
    # Judge uses the reasoning LLM (mistral:7b) for reliable semantic scoring.
    # llama3.2:1b is too small to consistently evaluate answer quality.
    judge_llm = reason_llm
    print(
        f"  [Judge LLM]      Using reasoning model ({os.getenv('CONTEXT_OPTIMIZER_REASONING_MODEL', 'mistral:7b')}) as evaluator"
    )
    models = {
        "compression": os.getenv("CONTEXT_OPTIMIZER_COMPRESSOR_MODEL", "llama3.2:1b"),
        "reasoning": os.getenv("CONTEXT_OPTIMIZER_REASONING_MODEL", "mistral:7b"),
        "judge": os.getenv("CONTEXT_OPTIMIZER_REASONING_MODEL", "mistral:7b"),
    }

    if reason_llm is None:
        print("\n[ERROR] Reasoning LLM unavailable — cannot run experiments.")
        sys.exit(1)

    # ── Corpus ─────────────────────────────────────────────────────────
    print("\n[Corpus] Loading Pride and Prejudice corpus...")
    pride_file = Path(__file__).parent / "data" / "books_pride-prejudice.txt"
    if not pride_file.exists():
        print(f"[ERROR] {pride_file} not found. Run download_test_data.py first.")
        sys.exit(1)

    with open(pride_file, encoding="utf-8", errors="ignore") as f:
        all_lines = [ln.strip() for ln in f if ln.strip()]

    corpus = all_lines[:corpus_lines_limit]
    print(f"  Loaded {len(corpus):,} lines from Pride & Prejudice")

    # ── Run Exp 1 (baseline) ───────────────────────────────────────────
    exp1_results = run_experiment1(corpus, reason_llm, judge_llm=judge_llm)

    # ── Run Exp 2 ───────────────────────────────────────────────
    exp2_results = run_experiment2(
        corpus, compress_llm, reason_llm, embed_cfg, judge_llm=judge_llm
    )

    # ── JSON output ────────────────────────────────────────────────────
    bench_dir = Path(__file__).parent
    json_out = bench_dir / "EXPERIMENT_RESULTS.json"
    report_out = (
        bench_dir.parent.parent / "docs" / "experiments" / "experiment_results.md"
    )

    payload = {
        "run_date": run_date,
        "corpus_lines": len(corpus),
        "models": models,
        "embedding": embed_cfg,
        "thresholds": THRESHOLDS,
        "experiment_1": exp1_results,
        "experiment_2": exp2_results,
    }
    with open(json_out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[OK] JSON results saved: {json_out}")

    # ── Markdown report ────────────────────────────────────────────────
    md = generate_report(
        exp1_results, exp2_results, len(corpus), models, embed_cfg, run_date
    )
    report_out.parent.mkdir(parents=True, exist_ok=True)
    with open(report_out, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"[OK] Markdown report: {report_out}")

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()
