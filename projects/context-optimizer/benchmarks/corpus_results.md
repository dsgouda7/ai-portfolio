# Corpus Benchmark: Vanilla RAG vs Optimized RAG

**Run date**: 2026-07-06 13:17  |  **Corpus**: enwik9 (954 MB)  |  **Questions**: 50  |  **top-k**: 5  |  **Block size**: 0.5 MB  |  **Corpus cap**: 200 MB

---

## Results Summary

| Metric | Vanilla RAG | Optimized RAG | Delta |
|--------|-------------|---------------|-------|
| Avg answer accuracy (KW recall) | — | 86.7% | — |
| Avg tokens per query | — | 322,679 | — |
| Avg query latency (ms) | — | 35.3 | — |
| Index ingestion time (s) | — | 129.6 | — |
| Index size (MB, excl. corpus) | — | 608.2 | — |
| Index entries | — | 400 | — |
| Raw block fallback rate | — | 0% | — |

**Delta** is Optimized relative to Vanilla.  ✓ = improvement, ✗ = regression.

---

## Architecture Contrast

```
VANILLA RAG                            OPTIMIZED RAG
────────────────────────────────────   ────────────────────────────────────
Corpus split into 512-token chunks      Corpus split into 0 MB blocks

Raw text embedded directly              Compressed summaries embedded
No fallback                             Raw block fetched from disk on demand

```

---

## Per-Question Breakdown


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

*Generated 2026-07-06 13:17 — do not edit manually.*