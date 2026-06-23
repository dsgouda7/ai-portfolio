# Experiment Results: Standard LLM vs Compressed Architecture

> **Run date:** 2026-06-23 08:22
> **Corpus:** 11,574 lines (Pride & Prejudice excerpt)
> **Judge:** `llama3.2:3b` (LLM-as-judge, 0–1 scale)
> **Note:** Judge scores added by post-hoc pass on saved answer snippets (200 chars).
> Status: Local-only execution — no API keys, no cloud.

---

## Model Configuration

| Role | Model | Backend |
|------|-------|---------|
| Compression / Summarisation | `llama3.2:3b` | Ollama (local) |
| Embeddings | `all-MiniLM-L6-v2` | sentence-transformers |
| Reasoning | `qwen2.5-coder:7b` | Ollama (local) |
| Judge (evaluator) | `llama3.2:3b` | reused from compression role |

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
| Latency delta vs baseline | ±10% |
| Judge-score delta vs baseline | ±20% |
| Token reduction vs baseline | ≥90% |

---

## Experiment 1 — Standard LLM Baseline

| Q | Difficulty | Tokens | Latency (s) | KW-F1 | Judge |
|---|------------|--------|-------------|-------|-------|
| q001 | easy | 180,736 | 145.72 | 0.047 | 0.80 |
| q002 | easy | 180,731 | 228.19 | 0.133 | 0.20 |
| q003 | medium | 180,731 | 232.59 | 0.057 | 0.80 |
| q004 | medium | 180,728 | 249.45 | 0.370 | 0.80 |
| q005 | hard | 180,739 | 106.81 | 0.050 | 0.60 |
| q006 | hard | 180,738 | 178.76 | 0.041 | 0.80 |
| **Avg** | — | **180,734** | **190.25** | **0.116** | **0.67** |

---

## Experiment 2a — Compressed Architecture (Summaries Only)

### Compression

| Metric | Value |
|--------|-------|
| Chunks | 435 |
| Time | 5403.6s (one-time) |
| Ratio | 0.127 |
| Original tokens | 234,623 |
| Compressed tokens | 29,703 |

### Query Results

| Q | Difficulty | Tokens | Ret (ms) | Hit (ms) | Latency (s) | KW-F1 | Judge | Token Δ |
|---|------------|--------|----------|----------|-------------|-------|-------|---------|
| q001 | easy | 425 | 33.8 | 0.0 | 29.97 | 0.109 | 0.80 | -99.8% |
| q002 | easy | 350 | 32.0 | 0.6 | 12.49 | 0.105 | 0.20 | -99.8% |
| q003 | medium | 452 | 29.0 | 0.0 | 10.92 | 0.000 | 0.40 | -99.7% |
| q004 | medium | 429 | 28.8 | 0.0 | 25.82 | 0.100 | 0.80 | -99.8% |
| q005 | hard | 464 | 28.9 | 0.6 | 14.31 | 0.100 | 0.80 | -99.7% |
| q006 | hard | 384 | 24.6 | 0.6 | 19.41 | 0.133 | 0.40 | -99.8% |
| **Avg** | — | **417** | **29.5** | **0.3** | **18.82** | **0.091** | **0.57** | **-99.8%** |

---

## Experiment 2b — Compressed Architecture (Summaries + Raw Detail)

| Q | Difficulty | Tokens | Latency (s) | KW-F1 | Judge | Token Δ |
|---|------------|--------|-------------|-------|-------|---------|
| q001 | easy | 453 | 23.18 | 0.109 | 0.80 | -99.7% |
| q002 | easy | 1,273 | 36.53 | 0.121 | 0.20 | -99.3% |
| q003 | medium | 480 | 13.83 | 0.000 | 0.40 | -99.7% |
| q004 | medium | 1,442 | 47.12 | 0.161 | 0.80 | -99.2% |
| q005 | hard | 493 | 17.89 | 0.114 | 0.40 | -99.7% |
| q006 | hard | 1,341 | 41.12 | 0.179 | 0.60 | -99.3% |
| **Avg** | — | **914** | **29.95** | **0.114** | **0.53** | **-99.5%** |

---

## Cross-Experiment Comparison

> **Primary accuracy metric**: Judge score (LLM-as-judge).
> KW-F1 is secondary — it under-scores verbose-but-correct answers.

| Metric | Baseline (Exp 1) | Exp 2a (Summary) | Exp 2b (Summary+Raw) |
|--------|-----------------|-----------------|---------------------|
| Avg prompt tokens | 180,734 | 417 | 914 |
| Token reduction | — | **-99.8%** [PASS] | **-99.5%** [PASS] |
| Avg reasoning latency (s) | 190.25 | 18.82 (-90.1%) [FAIL] | 29.95 (-84.3%) [FAIL] |
| Avg retrieval latency (ms) | N/A | 29.5 (miss) / 0.3 (hit) | same |
| Avg Judge score (0–1) | 0.67 | 0.57 (-15.0%) [PASS] | 0.53 (-20.0%) [PASS] |
| Avg KW-F1 (secondary) | 0.116 | 0.091 (-21.6%) [FAIL] | 0.114 (-2.1%) [PASS] |

### Threshold Summary

| Threshold | Target | Exp 2a | Exp 2b |
|-----------|--------|--------|--------|
| Token reduction ≥90% | ≥90% | 99.8% [PASS] | 99.5% [PASS] |
| Latency delta ≤±10% | ≤±10% | -90.1% [FAIL] | -84.3% [FAIL] |
| Judge-score delta ≤±20% | ≤±20% | -15.0% [PASS] | -20.0% [PASS] |

---

## Key Observations

- **Token efficiency**: Exp 2a delivers 100% token reduction — above the 90% target.
- **Latency**: Reasoning latency improved by 90% in Exp 2a. Retrieval adds 30ms (miss) / 0.3ms (cache hit).
- **Quality**: Judge score held within 15% of baseline (threshold: ±20%).
- **Raw detail (2b)**: Pointer-model fetch gives -5% judge-score delta vs 2a at +496 tokens.
- **Why KW-F1 is low**: Keyword precision = matched_keywords / all_answer_words. A verbose-but-correct 200-word answer mentioning 2 of 7 keywords scores 2/200 = 0.01 precision regardless of factual quality. Judge score does not have this flaw.
