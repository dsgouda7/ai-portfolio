# Experiment Results: Standard LLM vs Compressed Architecture

> **Run date:** 2026-06-21 21:06
> **Corpus:** 500 lines (Pride & Prejudice excerpt)
> **Status:** Local-only execution (no API keys, no cloud)

---

## Model Configuration

| Role | Model | Backend |
|------|-------|---------|
| Compression / Summarisation | `llama3.2:3b` | Ollama (local) |
| Embeddings | `all-MiniLM-L6-v2` | sentence-transformers |
| Reasoning | `qwen2.5-coder:7b` | Ollama (local) |

---

## Pass/Fail Thresholds

| Metric | Threshold | Rationale |
|--------|-----------|-----------|
| Latency delta vs baseline | ±10% | Architectural overhead must be within 10% |
| Judge-score delta vs baseline | ±20% | Semantic quality (LLM-as-judge 0–1) within 20% of full-corpus baseline |
| KW-F1 delta vs baseline | ±20% | Keyword-overlap F1 delta (secondary, penalised for verbosity) |
| Token reduction vs baseline | ≥90% | Core efficiency target |

---

## Experiment 1 — Standard LLM Baseline (Raw Corpus Injected)

Full raw corpus injected into the reasoning LLM on every query. No preprocessing,
no retrieval, no compression. This is the cost/latency ceiling we are beating.

| Question | Difficulty | Prompt Tokens | Latency (s) | KW-F1 | Judge |
|----------|------------|--------------|-------------|-------|-------|
| q001 | easy | 8,157 | 110.64 | 0.080 | 0.80 |
| q002 | easy | 8,153 | 98.94 | 0.114 | 0.80 |
| q003 | medium | 8,153 | 123.83 | 0.106 | 0.80 |
| q004 | medium | 8,150 | 112.89 | 0.110 | 0.80 |
| q005 | hard | 8,161 | 114.91 | 0.038 | 0.80 |
| q006 | hard | 8,160 | 130.16 | 0.062 | 0.80 |
| **Average** | — | **8,156** | **115.23** | **0.085** | **0.80** |

---

## Experiment 2a — Compressed Architecture (Summaries Only)

Corpus compressed with `llama3.2:3b`, indexed in ChromaDB with `all-MiniLM-L6-v2`.
Reasoning LLM receives only the top-5 retrieved compressed summaries (~50 tokens each).

### Compression Stats

| Metric | Value |
|--------|-------|
| Chunks produced | 20 |
| Compression time | 250.4s (one-time) |
| Compression ratio | 0.143 |
| Original tokens | 10,467 |
| Compressed tokens | 1,500 |

### Query Results

| Question | Difficulty | Prompt Tokens | Ret (ms) | Hit (ms) | Latency (s) | KW-F1 | Judge | Token Δ |
|----------|------------|--------------|----------|----------|-------------|-------|-------|---------|
| q001 | easy | 484 | 34.0 | 0.0 | 21.19 | 0.109 | 0.80 | -94.1% |
| q002 | easy | 430 | 29.7 | 0.6 | 10.25 | 0.143 | 0.80 | -94.7% |
| q003 | medium | 446 | 26.7 | 0.0 | 10.44 | 0.105 | 0.40 | -94.5% |
| q004 | medium | 449 | 25.6 | 0.0 | 45.20 | 0.050 | 0.80 | -94.5% |
| q005 | hard | 395 | 28.9 | 0.6 | 11.70 | 0.098 | 0.80 | -95.2% |
| q006 | hard | 418 | 46.1 | 0.0 | 21.49 | 0.122 | 0.80 | -94.9% |
| **Average** | — | **437** | **31.8** | **0.2** | **20.05** | **0.104** | **0.73** | **-94.6%** |

---

## Experiment 2b — Compressed Architecture (Summaries + Raw Detail)

Same pipeline as 2a but the reasoning LLM also receives the full raw text of the
most relevant chunk via the pointer model (`get_chunk_by_id`).

| Question | Difficulty | Prompt Tokens | Latency (s) | KW-F1 | Judge | Token Δ |
|----------|------------|--------------|-------------|-------|-------|---------|
| q001 | easy | 1,026 | 28.12 | 0.133 | 0.80 | -87.4% |
| q002 | easy | 965 | 24.11 | 0.044 | 0.80 | -88.2% |
| q003 | medium | 982 | 18.02 | 0.100 | 0.40 | -88.0% |
| q004 | medium | 984 | 26.21 | 0.213 | 0.80 | -87.9% |
| q005 | hard | 937 | 23.96 | 0.091 | 0.40 | -88.5% |
| q006 | hard | 964 | 26.99 | 0.182 | 0.80 | -88.2% |
| **Average** | — | **976** | **24.57** | **0.127** | **0.67** | **-88.0%** |

---

## Cross-Experiment Comparison

> **Accuracy note**: *Judge score* (LLM-as-judge, 0–1) is the primary quality metric.
> *KW-F1* (keyword-overlap) is secondary — it under-reports quality for verbose answers
> because precision is penalised by answer word count.

| Metric | Baseline (Exp 1) | Exp 2a (Summary) | Exp 2b (Summary+Raw) |
|--------|-----------------|-----------------|---------------------|
| Avg prompt tokens | 8,156 | 437 | 976 |
| Token reduction | — | **-94.6%** [PASS] | **-88.0%** [FAIL] |
| Avg reasoning latency (s) | 115.23 | 20.05 (-82.6%) [FAIL] | 24.57 (-78.7%) [FAIL] |
| Avg retrieval latency (ms) | N/A | 31.8 (miss) / 0.2 (hit) | same |
| Avg Judge score (0–1) | 0.80 | 0.73 (-8.3%) [PASS] | 0.67 (-16.7%) [PASS] |
| Avg KW-F1 (secondary) | 0.085 | 0.104 (+22.8%) [FAIL] | 0.127 (+49.6%) [FAIL] |

### Threshold Summary

| Threshold | Target | Exp 2a | Exp 2b |
|-----------|--------|--------|--------|
| Token reduction ≥90% | ≥90% | 94.6% [PASS] | 88.0% [FAIL] |
| Latency delta ≤±10% | ≤±10% | -82.6% [FAIL] | -78.7% [FAIL] |
| Judge-score delta ≤±20% | ≤±20% | -8.3% [PASS] | -16.7% [PASS] |
| KW-F1 delta ≤±20% | ≤±20% | +22.8% [FAIL] | +49.6% [FAIL] |

---

## Key Observations

- **Token efficiency**: Exp 2a delivers 95% token reduction vs the full-corpus
  baseline, well above the 90% target.
- **Latency**: Reasoning latency improved by
  83% in Exp 2a (fewer tokens = faster LLM). Retrieval adds
  32ms (miss) / 0.2ms (cache hit).
- **F1 quality**: Exp 2a F1 diverged from the
  baseline within 23% (threshold: ±20%).
- **Raw detail (2b)**: Adding the pointer-model raw-text fetch gives
  +27% F1 delta vs 2a at the cost of
  +539 extra tokens.
- **Cache benefit**: Repeated / similar queries drop from 32ms to
  0.2ms (172x speedup).

---

## Next Steps

- Run with `--full` flag (25K lines) to validate results at production corpus scale.
- Populate persistent ChromaDB with `quick_compress_and_save.py` then run
  `accuracy_benchmarks.py` for full F1 + precision/recall metrics.
- Switch embedding backend to Ollama:
  `$env:CONTEXT_OPTIMIZER_EMBEDDING_BACKEND = "ollama"` then re-run.
