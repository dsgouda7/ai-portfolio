# Corpus Benchmark: Vanilla RAG vs Optimized RAG vs Tree RAG

**Run date**: 2026-07-08 10:57  |  **Corpus**: enwik9_clean.txt (676 MB)  |  **Questions**: 43  |  **top-k**: 5  |  **Block size**: 2.0 MB  |  **Corpus cap**: 400 MB

---

## Results Summary

| Metric | Vanilla RAG | Optimized RAG | Delta |
|--------|-------------|---------------|-------|
| Avg retrieval recall (context has keywords) | — | 5.2% | — |
| Avg tokens per query | — | 396 | — |
| Avg query latency (ms) | — | 9918.8 | — |
| Index ingestion time (s) | — | 0.0 | — |
| Index size (MB, excl. corpus) | — | 3.5 | — |
| Index entries | — | 200 | — |
| Raw block fallback rate | — | 0% | — |

### Reasoning Evaluation

Can the reasoning model synthesize correct answers from the retrieved context?
- **Reasoning recall**: keyword overlap between LLM-generated answer and expected answer.
- **Faithfulness**: fraction of answer content words traceable back to the retrieved context.
- **Reasoning gap**: retrieval_recall − reasoning_recall  (positive = info lost in reasoning; negative = hallucination).

| Metric | Vanilla RAG | Optimized RAG | Delta |
|--------|-------------|---------------|-------|
| Reasoning recall (LLM answer) | — | 56.4% | — |
| Faithfulness (grounded in context) | — | 6.4% | — |
| Reasoning gap (ret - reason) | — | -51.3% | — |
| Avg reasoning latency (ms) | — | 9902 | — |

**Delta** is Optimized relative to Vanilla.  ✓ = improvement, ✗ = regression.

---

## Architecture Contrast

```
VANILLA RAG                            OPTIMIZED RAG
────────────────────────────────────   ────────────────────────────────────
Corpus split into 512-token chunks      Corpus split into 2 MB blocks

Raw text embedded directly              Compressed summaries embedded
No fallback                             Raw block fetched from disk on demand

```

---

## Per-Question Breakdown


---

## Tree-of-Summaries Results

Two-level hierarchical index: L1 block summaries + L2 cluster super-summaries.
The reasoning agent navigates the tree autonomously (search_cluster / fetch_raw_block).

| Metric | Vanilla RAG | Tree RAG | Delta (vs Vanilla) |
|--------|-------------|----------|--------------------|
| Avg retrieval recall | — | 58.3% | — |
| Avg tokens per query | — | 404 | — |
| Avg query latency (ms) | — | 30631.1 | — |
| Index ingestion time (s) | — | 3784.1 | — |
| Index size (MB) | — | 3.0 | — |
| L1 block entries | — | 200 | — |
| Raw block fallback rate | — | 0% | — |

### Tree Reasoning Evaluation

| Metric | Vanilla RAG | Tree RAG | Delta |
|--------|-------------|----------|-------|
| Reasoning recall | — | 58.3% | — |
| Faithfulness | — | 100.0% | — |
| Reasoning gap | — | 0.0% | — |
| Avg reasoning latency (ms) | — | 30631 | — |

---

## How to Re-run

```bash
# Full run (prepare corpus + build indexes + evaluate)
python corpus_benchmark.py all

# Build indexes only (corpus already prepared)
python corpus_benchmark.py run

# Use a custom corpus
python corpus_benchmark.py prepare --corpus-path /path/to/corpus.txt
python corpus_benchmark.py run
```

*Generated 2026-07-08 10:57 — do not edit manually.*