# Experiment Results: Standard LLM vs Compressed Architecture

> **Run date:** 2026-06-27 18:43
> **Corpus:** 500 lines (Pride & Prejudice excerpt)
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
| q001 | easy | 8,157 | 174.99 | 0.111 | 0.95 |
| q002 | easy | 8,153 | 174.75 | 0.244 | 1.00 |
| q003 | medium | 8,153 | 200.56 | 0.056 | 1.00 |
| q004 | medium | 8,150 | 235.48 | 0.062 | 1.00 |
| q005 | hard | 8,161 | 159.61 | 0.078 | 1.00 |
| q006 | hard | 8,160 | 178.52 | 0.079 | 1.00 |
| **Average** | — | **8,156** | **187.32** | **0.105** | **0.99** |

---

## Experiment 2 — Compressed Architecture

Corpus compressed with `llama3.2:1b`, indexed in ChromaDB with `all-MiniLM-L6-v2`.

- **Targeted queries**: top-5 vector retrieval → adaptive raw-fetch if the model judges summaries insufficient.
- **Aggregated queries**: ALL compressed summaries injected + Tree-of-Thought reasoning
  (three analytical paths → synthesised FINAL ANSWER), bypassing vector retrieval.

### Compression Stats

| Metric | Value |
|--------|-------|
| Chunks produced | 17 |
| Compression time | 185.4s (one-time) |
| Compression ratio | 0.157 |
| Original tokens | 8,978 |
| Compressed tokens | 1,412 |

### Query Results

| Question | Difficulty | Strategy | Prompt Tokens | Ret (ms) | Latency (s) | Raw? | KW-F1 | Judge | Token Δ |
|----------|------------|----------|--------------|----------|-------------|------|-------|-------|---------|
| q001 | easy | targeted_adaptive | 1,718 | 89.4 | 78.93 | Y | 0.062 | 0.85 | -78.9% |
| q002 | easy | targeted_adaptive | 1,787 | 38.3 | 52.53 | Y | 0.118 | 0.50 | -78.1% |
| q003 | medium | tot_aggregated | 1,556 | 0.0 | 93.21 | N | 0.065 | 0.65 | -80.9% |
| q004 | medium | tot_aggregated | 1,554 | 0.0 | 43.43 | N | 0.000 | 0.60 | -80.9% |
| q005 | hard | targeted_adaptive | 1,807 | 55.4 | 60.10 | Y | 0.102 | 1.00 | -77.9% |
| q006 | hard | tot_aggregated | 1,564 | 0.0 | 86.46 | N | 0.047 | 0.95 | -80.8% |
| **Average** | — | — | **1,664** | **30.5** | **69.11** | 3/6 | **0.065** | **0.76** | **-79.6%** |

---

## Cross-Experiment Comparison

> **Accuracy note**: *Judge score* (LLM-as-judge, 0–1) is the primary quality metric.
> *KW-F1* (keyword-overlap) is secondary — it under-reports quality for verbose answers
> because precision is penalised by answer word count.

| Metric | Baseline (Exp 1) | Exp 2 (Adaptive) |
|--------|-----------------|------------------|
| Avg prompt tokens | 8,156 | 1,664 |
| Token reduction | — | **-79.6%** [FAIL] |
| Avg reasoning latency (s) | 187.32 | 69.11 (-63.1%) [PASS] |
| Avg retrieval latency (ms) | N/A | 30.5 |
| Avg Judge score (0–1) | 0.99 | 0.76 (-23.5%) [FAIL] |
| Avg KW-F1 (secondary) | 0.105 | 0.065 (-37.8%) [FAIL] |

### Threshold Summary

| Threshold | Target | Exp 2 |
|-----------|--------|-------|
| Token reduction ≥90% | ≥90% | 79.6% [FAIL] |
| Latency regression ≤+10% | ≤+10% | -63.1% [PASS] |
| Judge-score drop ≤-20% | ≤-20% | -23.5% [FAIL] |
| KW-F1 drop ≤-20% | ≤-20% | -37.8% [FAIL] |

---

## Key Observations

- **Token efficiency**: Exp 2 delivers 80% token reduction vs the full-corpus
  baseline, below the 90% target.
- **Latency**: Reasoning latency improved by
  63% (fewer tokens = faster LLM). Retrieval adds 31ms.
- **Quality**: Judge score delta -23.5% vs baseline (threshold: ±20%) [FAIL].
- **Adaptive raw fetch**: triggered for 3 of 6 targeted queries
  (50%).
  Raw fetch added ~238 extra tokens where used.

---

## Next Steps

- Run with `--full` flag (25K lines) to validate results at production corpus scale.
- Populate persistent ChromaDB with `quick_compress_and_save.py` then run
  `accuracy_benchmarks.py` for full F1 + precision/recall metrics.
- Switch embedding backend to Ollama:
  `$env:CONTEXT_OPTIMIZER_EMBEDDING_BACKEND = "ollama"` then re-run.
