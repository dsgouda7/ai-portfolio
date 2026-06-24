# Experiment Results: Standard LLM vs Compressed Architecture

> **Run date:** 2026-06-23 16:24
> **Corpus:** 11,574 lines (Pride & Prejudice excerpt)
> **Status:** Local-only execution (no API keys, no cloud)

---

## Model Configuration

| Role | Model | Backend |
|------|-------|---------|
| Compression / Summarisation | `llama3.2:1b` | Ollama (local) |
| Embeddings | `all-MiniLM-L6-v2` | sentence-transformers |
| Reasoning | `mistral:7b` | Ollama (local) |
| Judge (LLM-as-judge) | `mistral:7b` | Ollama (local) |

---

## Pass/Fail Thresholds

| Metric | Threshold | Rationale |
|--------|-----------|-----------|
| Latency regression vs baseline | ≤+10% | Slowdowns >10% fail; improvements always pass |
| Judge-score drop vs baseline | ≤-20% | Quality drops >20% fail; improvements always pass |
| KW-F1 drop vs baseline | ≤-20% | KW-F1 drops >20% fail; improvements always pass |
| Token reduction vs baseline | ≥90% | Core efficiency target |

---

## Experiment 1 — Standard LLM Baseline (Raw Corpus Injected)

Full raw corpus injected into the reasoning LLM on every query. No preprocessing,
no retrieval, no compression. This is the cost/latency ceiling we are beating.

| Question | Difficulty | Prompt Tokens | Latency (s) | KW-F1 | Judge |
|----------|------------|--------------|-------------|-------|-------|
| q001 | easy | 180,736 | 131.81 | 0.107 | 1.00 |
| q002 | easy | 180,731 | 122.92 | 0.098 | 0.85 |
| q003 | medium | 180,731 | 196.29 | 0.058 | 1.00 |
| q004 | medium | 180,728 | 193.28 | 0.057 | 1.00 |
| q005 | hard | 180,739 | 125.38 | 0.027 | 1.00 |
| q006 | hard | 180,738 | 165.51 | 0.059 | 1.00 |
| **Average** | — | **180,734** | **155.86** | **0.068** | **0.97** |

---

## Experiment 2 — Compressed Architecture

Corpus compressed with `llama3.2:1b`, indexed in ChromaDB with `all-MiniLM-L6-v2`.

- **Targeted queries**: top-5 vector retrieval → adaptive raw-fetch if the model judges summaries insufficient.
- **Aggregated queries**: ALL compressed summaries injected + Tree-of-Thought reasoning
  (three analytical paths → synthesised FINAL ANSWER), bypassing vector retrieval.

### Compression Stats

| Metric | Value |
|--------|-------|
| Chunks produced | 435 |
| Compression time | 2799.4s (one-time) |
| Compression ratio | 0.122 |
| Original tokens | 234,623 |
| Compressed tokens | 28,722 |

### Query Results

| Question | Difficulty | Strategy | Prompt Tokens | Ret (ms) | Latency (s) | Raw? | KW-F1 | Judge | Token Δ |
|----------|------------|----------|--------------|----------|-------------|------|-------|-------|---------|
| q001 | easy | targeted_adaptive | 269 | 30.0 | 30.07 | N | 0.286 | 0.85 | -99.9% |
| q002 | easy | targeted_adaptive | 327 | 34.4 | 12.59 | N | 0.267 | 1.00 | -99.8% |
| q003 | medium | tot_aggregated | 31,319 | 0.0 | 154.46 | N | 0.060 | 1.00 | -82.7% |
| q004 | medium | tot_aggregated | 31,316 | 0.0 | 119.82 | N | 0.112 | 1.00 | -82.7% |
| q005 | hard | targeted_adaptive | 231 | 26.7 | 9.74 | N | 0.182 | 1.00 | -99.9% |
| q006 | hard | tot_aggregated | 31,326 | 0.0 | 150.28 | N | 0.053 | 1.00 | -82.7% |
| **Average** | — | — | **15,798** | **15.2** | **79.49** | 0/6 | **0.160** | **0.97** | **-91.3%** |

---

## Cross-Experiment Comparison

> **Accuracy note**: *Judge score* (LLM-as-judge, 0–1) is the primary quality metric.
> *KW-F1* (keyword-overlap) is secondary — it under-reports quality for verbose answers
> because precision is penalised by answer word count.

| Metric | Baseline (Exp 1) | Exp 2 (Adaptive) |
|--------|-----------------|------------------|
| Avg prompt tokens | 180,734 | 15,798 |
| Token reduction | — | **-91.3%** [PASS] |
| Avg reasoning latency (s) | 155.86 | 79.49 (-49.0%) [PASS] |
| Avg retrieval latency (ms) | N/A | 15.2 |
| Avg Judge score (0–1) | 0.97 | 0.97 (+0.0%) [PASS] |
| Avg KW-F1 (secondary) | 0.068 | 0.160 (+136.8%) [PASS] |

### Threshold Summary

| Threshold | Target | Exp 2 |
|-----------|--------|-------|
| Token reduction ≥90% | ≥90% | 91.3% [PASS] |
| Latency regression ≤+10% | ≤+10% | -49.0% [PASS] |
| Judge-score drop ≤-20% | ≤-20% | +0.0% [PASS] |
| KW-F1 drop ≤-20% | ≤-20% | +136.8% [PASS] |

---

## Key Observations

- **Token efficiency**: Exp 2 delivers 91% token reduction vs the full-corpus
  baseline, above the 90% target.
- **Latency**: Reasoning latency improved by
  49% (fewer tokens = faster LLM). Retrieval adds 15ms.
- **Quality**: Judge score delta +0.0% vs baseline (threshold: ±20%) [PASS].
- **Adaptive raw fetch**: triggered for 0 of 6 targeted queries
  (0%).
  Raw fetch added ~0 extra tokens where used.

---

## Next Steps

- Run with `--full` flag (25K lines) to validate results at production corpus scale.
- Populate persistent ChromaDB with `quick_compress_and_save.py` then run
  `accuracy_benchmarks.py` for full F1 + precision/recall metrics.
- Switch embedding backend to Ollama:
  `$env:CONTEXT_OPTIMIZER_EMBEDDING_BACKEND = "ollama"` then re-run.
