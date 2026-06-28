# Context Optimizer: Architecture Design

> **Canonical reference** for the Context Optimizer system.
> Related: [plan.md](../plan.md) · [Experiments](experiments/EXPERIMENTS_CONSOLIDATED.md) · [Benchmark results](experiments/experiment_results.md)

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Data Ingestion Pipeline](#2-data-ingestion-pipeline)
3. [Rolling Window Compression](#3-rolling-window-compression)
4. [Dual Storage Architecture](#4-dual-storage-architecture)
5. [Retrieval Layer](#5-retrieval-layer)
6. [Semantic Cache Decision](#6-semantic-cache-decision)
7. [MCP Tool Contract](#7-mcp-tool-contract)
8. [Swappable Embedding Backend](#8-swappable-embedding-backend)
9. [Integrated Query Flow](#9-integrated-query-flow)
10. [Performance & Benchmarks](#10-performance--benchmarks)
11. [Implementation Files](#11-implementation-files)
12. [End-to-End Local Test Setup](#12-end-to-end-local-test-setup)

---

## 1. System Overview

Context Optimizer is a **token-efficient RAG pipeline with corpus-wide reasoning**.
It compresses an arbitrarily large corpus once at write-time, then answers queries
using only the relevant compressed summaries — achieving **91.3% token reduction
with 0% quality loss** vs raw-injection baseline (measured, Pride & Prejudice corpus).

### Standard LLM vs Compressed Architecture

```
╔══════════════════════════════════════════════════════════════════════════════╗
║              STANDARD LLM  (Monolithic / RAG Baseline)                     ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  User Query                                                                  ║
║      ▼                                                                       ║
║  Raw corpus injected directly into context window                            ║
║  (full text → context window exhaustion at scale)                           ║
║      ▼                                                                       ║
║  Reasoning LLM  ──→  Answer                                                  ║
║  Problems:  context exhaustion · latency scales linearly with corpus size   ║
╚══════════════════════════════════════════════════════════════════════════════╝

╔══════════════════════════════════════════════════════════════════════════════╗
║              CONTEXT OPTIMIZER  (Compress → Retrieve → Reason)             ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  ┌─────────────────────────────────────────────────────────────────────┐   ║
║  │  WRITE-TIME  (once, offline)                                        │   ║
║  │  Raw Corpus ──rolling-window──→ Compression LLM                    │   ║
║  │                                → CompressedChunk (~50 tokens each) │   ║
║  │                                → ChromaDB (summaries + raw vault)  │   ║
║  └─────────────────────────────────────────────────────────────────────┘   ║
║                                                                              ║
║  ┌─────────────────────────────────────────────────────────────────────┐   ║
║  │  QUERY-TIME  (every request, no re-compression)                     │   ║
║  │  User Query ──→ Route (targeted | aggregated)                       │   ║
║  │    Targeted:    top-k summary retrieval via ChromaDB HNSW           │   ║
║  │    Aggregated:  all summaries injected + ToT multi-path reasoning   │   ║
║  │      ▼                                                              │   ║
║  │  Reasoning LLM  ──→  Answer  (91.3% fewer tokens vs baseline)      │   ║
║  └─────────────────────────────────────────────────────────────────────┘   ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

### Pipeline at a Glance

```
Raw Corpus (any size)
    │
    ▼  [WRITE-TIME — once]
Rolling Window Compression
  └─ Compression LLM · 512-token batches · entity/keyword extraction
    │
    ▼
Dual Storage (ChromaDB)
  ├─ Compressed Index   (embedded summaries — fast semantic search)
  └─ Raw Vault          (original text in metadata — fetched on demand)
    │
    ▼  [QUERY-TIME — every request]
Query Router
  ├─ targeted   → top-k summary retrieval (2-turn adaptive: summaries first, raw if needed)
  └─ aggregated → all summaries + Tree-of-Thought reasoning
    │
    ▼
Reasoning LLM  →  Answer
(token budget: ~650 targeted / full-summary-set for aggregated)
```

---

## 2. Data Ingestion Pipeline

### Stage 1 — Raw Data → Semantic Chunks

**Input**: Any text corpus — documents, transcripts, code, logs.

```python
chunks = semantic_chunk(
    raw_data,
    strategy="boundary-aware",
    target_size=256,
    overlap=50,
)
```

### Stage 2 — Chunk → Compressed Summary + Embedding

Compression happens at **write-time** using a cheap local LLM (default: `llama3.2:1b`
via Ollama). The embedding backend is **swappable** (see [Section 8](#8-swappable-embedding-backend)).

```python
summary = compress_chunk(chunk, max_tokens=150, extract_entities=True)
embedding = embed(chunk["text"])
# Default: sentence-transformers all-MiniLM-L6-v2 (local CPU, 90 MB)
```

**Output per chunk:**
```json
{
  "chunk_id": "doc-001-seg-003",
  "summary": "CosmosDB timeout (21012) in order-service → payment-service cascade",
  "entities": ["CosmosDB", "order-service", "payment-service", "21012"],
  "original_tokens": 87,
  "summary_tokens": 18,
  "compression_ratio": 0.21
}
```

**Compression enrichment pipeline** (applied before ChromaDB storage):
1. **Retrieval-optimized prompt** — the LLM produces entity-dense noun phrases joined by semicolons, not narrative prose (e.g. `"CosmosDB RU limit exceeded; request timeout 21012; AKS ingress 504 upstream"`).
2. **Entity appending** — `entities` list is appended to `compressed_summary`: `summary += "; " + "; ".join(entities)`, bridging the gap between ChromaDB metadata (where entities are stored) and the embedded vector.
3. **`_normalise_for_index()`** — 44 English stopwords (the/a/an/is/are/was/and/or/but …) are stripped token-by-token from the enriched summary before the vector is computed. Original casing is preserved (`CosmosDB` stays `CosmosDB`).

### Stage 3 — Indexed Storage

```sql
-- ChromaDB collection (cosine HNSW)
chunks (
  id TEXT PRIMARY KEY,
  embedding VECTOR(384),        -- all-MiniLM-L6-v2, local CPU
  summary TEXT,                 -- compressed, indexed
  raw_text TEXT,                -- stored in metadata, NOT indexed (pointer model)
  entities TEXT[],
  source TEXT
)
```

**Key design decisions:**
- Raw text stored as metadata, **not indexed** — keeps the index 10× smaller; fetched on demand.
- Embeddings computed once at write-time.

**Retriever selection** (see [Section 5](#5-retrieval-layer)):

| Class | File | When to use |
|-------|------|-------------|
| `DualStorageRetriever` | `src/context_optimizer/retriever.py` | Zero-dependency fallback (keyword/entity) |
| `CachedChromaRetriever` | `src/context_optimizer/cached_retriever.py` | **Recommended** — production and benchmarking |

---

## 3. Rolling Window Compression

### The Problem: Context Exhaustion

```python
# ❌ Naive — explodes on large corpora
for chunk in million_chunks:
    summary = llm.compress(full_document)  # context window exhausted

# ✅ Rolling window — one chunk at a time, no accumulation
for chunk in million_chunks:
    summary = llm.compress(chunk)   # only ~512 tokens per LLM call
    store(compressed=summary, raw=chunk)
```

### Implementation

```python
from context_optimizer.compressor import compress_corpus_rolling

compressed_chunks = compress_corpus_rolling(
    corpus_lines,
    chunk_size_threshold=512,
    chunk_overlap_tokens=64,
    llm=None,   # auto-resolved from env vars; see llm_provider.py
)
```

**LLM backend selection (env vars):**

```bash
# Ollama (local, default)
CONTEXT_OPTIMIZER_COMPRESSOR_PROVIDER=ollama
CONTEXT_OPTIMIZER_COMPRESSOR_MODEL=llama3.2:1b

# Groq (cloud, faster)
CONTEXT_OPTIMIZER_COMPRESSOR_PROVIDER=groq
CONTEXT_OPTIMIZER_COMPRESSOR_MODEL=llama-3.3-70b-versatile
GROQ_API_KEY=...
```

**Quality properties:**
- No context exhaustion: each LLM call ≤ 512 tokens.
- Parallel-safe: each chunk is independent.
- Fallback: if no LLM available, truncates to first 200 chars.
- Measured ratio: **12.24%** (8× reduction) on Pride & Prejudice full corpus.
- Retrieval-optimized compression prompt: produces entity-dense noun phrases, not narrative prose.
- Entity enrichment: `entities` list appended to `compressed_summary` before ChromaDB storage.
- Index normalisation: `_normalise_for_index()` strips 44 English stopwords before embedding.
- Reduced overlap: **64-token** boundary window (~12% of chunk) — entity enrichment covers the boundary-entity problem that larger overlap was solving.

---

## 4. Dual Storage Architecture

```
┌─────────────────────────────────────────┐
│         Semantic Index (Fast)           │
│  chunk_001: "CosmosDB timeout 21012..."│  ← Compressed (~50 tokens), indexed
│  chunk_002: "Payment cascade fail..."  │
└─────────────────────────────────────────┘
           ↓ Search returns compressed summaries
┌─────────────────────────────────────────┐
│        Raw Vault (On-Demand Only)       │
│  chunk_001: "System.TimeoutException..." │  ← Raw (~500 tokens)
│                                          │    stored in metadata, NOT indexed
└─────────────────────────────────────────┘
           ↑ Fetched via get_chunk_by_id(chunk_id)
```

### Token Economics

| Call | Tokens |
|------|--------|
| `get_context(top_k=6)` — compressed summaries | ~300 |
| `get_context_details(["chunk-042"])` — raw on demand | ~500 |
| **Total (two-call flow)** | **~800** |
| Monolithic `retrieve_raw(top_k=6)` | ~3,000 |

**Savings: 73% reduction** in the two-call pattern.

---

## 5. Retrieval Layer

### CachedChromaRetriever (Recommended)

Two-tier query path:
1. **Semantic cache** (in-memory LRU, exact-string + cosine similarity) — 0–2 ms
2. **ChromaDB HNSW** (sentence-transformers embeddings, local CPU) — 10–50 ms on cache miss

```python
from context_optimizer.cached_retriever import CachedChromaRetriever

retriever = CachedChromaRetriever(
    collection_name="my_corpus",
    persist_directory="./chroma_db",
    embedding_model_name="all-MiniLM-L6-v2",
    cache_size=1000,
    cache_threshold=0.85,
)

retriever.add_chunks(compressed_chunks)
results = retriever.search("query", top_k=5)      # compressed summaries (~250 tokens)
details = retriever.get_chunk_by_id(chunk_id)     # raw text on demand
```

### Query Routing

The benchmark runner (`run_experiments.py`) routes each question based on `query_type`:

| Route | Trigger | Context injected | Use case |
|-------|---------|-----------------|---------|
| `targeted` | `query_type="targeted"` | top-k summaries (adaptive: raw if needed) | Specific fact lookup |
| `aggregated` | `query_type="aggregated"` | all summaries + Tree-of-Thought prompt | Thematic / cross-corpus analysis |

**Adaptive raw-fetch (targeted path):** Turn 1 asks the LLM if it needs raw text
(`needs_raw: bool`). If `true`, the specific chunk is fetched and reasoning re-runs.
In practice on the current corpus, summaries are sufficient and raw is never requested.

**Tree-of-Thought (aggregated path):** System prompt forces three parallel reasoning
paths (A/B/C) over all 435 compressed summaries, then synthesises to a FINAL ANSWER.
Ensures complete corpus coverage for thematic questions. Branch retrieval issues one
**composite sentence query per branch** (all branch entities joined into a single query),
cutting ChromaDB calls 3× vs per-entity queries. Branch ranking uses **mean cosine
similarity** (gradient signal, e.g. 0.72 vs 0.68) rather than binary hit counts.

### DualStorageRetriever (Fallback)

```python
from context_optimizer.retriever import DualStorageRetriever
retriever = DualStorageRetriever(compressed_chunks)
hits = retriever.search_compressed("query", top_k=5)
```

Zero dependencies — keyword/entity scoring only. Useful for unit tests and
zero-dependency environments.

---

## 6. Semantic Cache Decision

```
Query → Semantic Cache (0–2 ms) ────────────────→ Results  (~70% hit rate)
              │ (on miss)
              ▼
         ChromaDB (10–50 ms) → Cache result → Results
```

### Cache Configuration

```python
CachedChromaRetriever(
    cache_size=1000,       # ~10–20 MB memory for 1 000 queries
    cache_threshold=0.85,  # 85% cosine similarity threshold for semantic hit
)
```

**Average latency at 70% hit rate:** `(0.70 × 2ms) + (0.30 × 30ms) = 10.4ms`
vs 30ms ChromaDB-only → **2.9× speedup**.

---

## 7. MCP Tool Contract

### Tool Schema

```json
{
  "name": "retrieve_context",
  "description": "Retrieve compressed evidence matching a query.",
  "parameters": {
    "query": "string — natural language or keyword query",
    "depth": "enum — brief (top-3) | detailed (top-6) | exhaustive (top-12)"
  }
}
```

```json
{
  "name": "get_context_details",
  "description": "Return full raw text for specific chunk IDs (pointer model).",
  "parameters": {
    "chunk_ids": ["chunk_001", "chunk_002"]
  }
}
```

### Reasoning LLM Workflow

```
User: "Analyse the CosmosDB timeout cascade"
    ↓
LLM: calls retrieve_context("CosmosDB timeout cascade", depth="detailed")
    ↓
MCP: returns 6 compressed summaries (~300 tokens)
    ↓
LLM: answers directly.  Or if summaries insufficient:
    calls get_context_details(["chunk_003"])
    ↓
MCP: returns raw text (~800 tokens) from metadata (no re-embedding)
```

---

## 8. Swappable Embedding Backend

The embedding backend is a **swappable interface** — all other stages are unaffected.

| Deployment | Backend | Model | Size | Cost |
|------------|---------|-------|------|------|
| **Local (default)** | `sentence-transformers` | `all-MiniLM-L6-v2` | 90 MB | Free |
| **Air-gapped** | Ollama | `nomic-embed-text` | 274 MB | Free |

**Swap via environment variables only:**

```bash
# Default — no Ollama needed for embeddings
CONTEXT_OPTIMIZER_EMBEDDING_BACKEND=sentence-transformers
CONTEXT_OPTIMIZER_EMBEDDING_MODEL=all-MiniLM-L6-v2

# Ollama backend
CONTEXT_OPTIMIZER_EMBEDDING_BACKEND=ollama
CONTEXT_OPTIMIZER_EMBEDDING_MODEL=nomic-embed-text
```

Reference implementation: `src/context_optimizer/cached_retriever.py`.

---

## 9. Integrated Query Flow

**Adaptive targeted query (2-turn):**

```
Turn 1:
  User question → summaries injected → Reasoning LLM asked:
    "Can you answer from these summaries? If not, set needs_raw=true + chunk_id."
  → LLM returns JSON {"needs_raw": false, "answer": "..."}
  → Answer returned directly

Turn 2 (only if needs_raw=true):
  Raw chunk fetched → re-injected → Reasoning LLM produces final answer
```

**Aggregated Tree-of-Thought query:**

```
All compressed summaries injected once
    ↓
System prompt forces PATH A / PATH B / PATH C analysis
  Each branch → ONE composite sentence query (all branch entities joined)
    ↓
Branch ranking by mean cosine similarity (gradient signal, not binary hit counts)
    ↓
FINAL ANSWER synthesises three paths
    ↓
Judge LLM (mistral:7b) scores quality vs baseline
```

### Token Accounting (targeted)

```
System prompt:            ~200 tokens  (fixed)
Compressed summaries:     ~300 tokens  (top-5 retrieval)
Reasoning instruction:    ~150 tokens  (fixed)
─────────────────────────────────────────────────────
Total input to reasoning: ~650 tokens

vs. raw-injection baseline: ~7 500 tokens
Token reduction: 91.3%
```

---

## 10. Performance & Benchmarks

### Latest E2E Results (2026-06-23, local CPU, no GPU)

Full corpus run via `benchmarks/tot/run_experiments.py`.
Corpus: Pride & Prejudice (11,574 lines). Models: `llama3.2:1b` (compression),
`mistral:7b` (reasoning + judge), `all-MiniLM-L6-v2` (embeddings).

**Compression (one-time write):**

| Metric | Measured |
|--------|----------|
| Chunks produced | 435 |
| Compression time | 2,799 s (one-time) |
| Compression ratio | **12.24%** |

**Cross-experiment comparison (avg across 6 questions):**

| Metric | Exp 1 — Baseline | Exp 2 — Compressed |
|--------|-----------------|-------------------|
| Avg prompt tokens | ~7,500 | **~650 (−91.3%) ✅** |
| Avg reasoning latency | baseline | −49% improvement ✅ |
| Judge score (mistral:7b, 0–1) | 0.975 | **0.975 (0% delta) ✅** |
| KW-F1 | baseline | +135% improvement ✅ |

**Pass/Fail summary:**

| Threshold | Target | Result |
|-----------|--------|--------|
| Token reduction ≥ 90% | ≥ 90% | **91.3% PASS** |
| Latency regression | no regression | **PASS** (−49% improvement) |
| Judge-score regression | no regression | **PASS** (0% delta) |
| KW-F1 regression | no regression | **PASS** (+135%) |

### Retrieval Latency

| Metric | Measured |
|--------|----------|
| Cache miss | ~10–50 ms |
| Cache hit | < 1 ms |

---

## 11. Implementation Files

### Core Source (`src/context_optimizer/`)

| File | Class | Purpose |
|------|-------|---------|
| `compressor.py` | `CompressedChunk`, `compress_corpus_rolling()` | Rolling window LLM compression pipeline |
| `retriever.py` | `DualStorageRetriever` | Keyword/entity fallback retriever (zero dependencies) |
| `cached_retriever.py` | `CachedChromaRetriever`, `SemanticCache` | **Recommended** — two-tier semantic cache + ChromaDB |

### Benchmark Harness (`benchmarks/tot/`)

| File | Purpose |
|------|---------|
| `run_experiments.py` | Main E2E runner — Exp 1 (baseline) vs Exp 2 (compressed) |
| `llm_provider.py` | **Callable stubs** — env-var-driven model selection with Ollama defaults |
| `download_test_data.py` | Downloads Project Gutenberg test corpus (cached locally) |

### Running the Benchmark

```powershell
# Prerequisites: Ollama running, models pulled
cd c:\repos\ai-portfolio\projects\context-optimizer

# Full corpus run (hours on CPU)
.venv\Scripts\python.exe benchmarks\tot\run_experiments.py

# Quick smoke test (500 lines, ~30–90 min)
.venv\Scripts\python.exe benchmarks\tot\run_experiments.py --lines 500
```

---

## 12. End-to-End Local Test Setup

All results in [Section 10](#10-performance--benchmarks) are produced by a fully local,
no-cloud run. No API keys, no paid subscriptions, no GPU required.

### 12.1 Hardware & Environment

| Item | Specification |
|------|---------------|
| OS | Windows 11 (PowerShell 7) |
| CPU | Stock consumer CPU (no GPU) |
| Python | 3.11 · virtual environment at `.venv` |
| Ollama | Local inference server · `http://localhost:11434` |

### 12.2 Model Role Assignment

```
┌──────────────────────┬───────────────────┬──────────────────────────────────────┐
│ Role                 │ Model (default)   │ Rationale                            │
├──────────────────────┼───────────────────┼──────────────────────────────────────┤
│ Compression          │ llama3.2:1b       │ Fast, ~600 MB — good at              │
│                      │ (Ollama)          │ chunk-level summarisation            │
├──────────────────────┼───────────────────┼──────────────────────────────────────┤
│ Embeddings (default) │ all-MiniLM-L6-v2  │ 90 MB · 384-dim · no Ollama needed   │
│                      │ (sentence-xformers)│                                     │
│ Embeddings (alt)     │ nomic-embed-text  │ 274 MB · 768-dim · CPU-optimised     │
│                      │ (Ollama)          │ swap via env var                     │
├──────────────────────┼───────────────────┼──────────────────────────────────────┤
│ Reasoning + Judge    │ mistral:7b        │ ~4 GB Q4_K_M · strong general        │
│                      │ (Ollama)          │ reasoning · used for both query      │
│                      │                   │ answering and quality judging         │
└──────────────────────┴───────────────────┴──────────────────────────────────────┘
```

**One-time setup:**
```powershell
python install.py        # installs deps + pulls llama3.2:1b and mistral:7b

# Or manually:
ollama pull llama3.2:1b
ollama pull mistral:7b
```

### 12.3 Test Corpus

| Flag | Lines | Use case |
|------|-------|---------|
| `--lines 500` (default) | 500 | Quick E2E smoke test, ~minutes |
| `--lines 2000` | 2,000 | Broader coverage, ~30–60 min |
| _(no flag)_ | 11,574 (full) | Full corpus validation (hours) |

All books cached locally under `benchmarks/tot/test_data/`.

### 12.4 Provider Selection Reference

All roles controlled via env vars, resolved in `benchmarks/tot/llm_provider.py`:

```bash
# Compression LLM
CONTEXT_OPTIMIZER_COMPRESSOR_PROVIDER   ollama (default) | groq
CONTEXT_OPTIMIZER_COMPRESSOR_MODEL      llama3.2:1b (default)
OLLAMA_BASE_URL                         http://localhost:11434 (default)

# Reasoning + Judge LLM
CONTEXT_OPTIMIZER_REASONING_MODEL       mistral:7b (default)

# Embedding backend
CONTEXT_OPTIMIZER_EMBEDDING_BACKEND     sentence-transformers (default) | ollama
CONTEXT_OPTIMIZER_EMBEDDING_MODEL       all-MiniLM-L6-v2 (s-t default)
                                        nomic-embed-text  (ollama default)
```

Swapping any of these requires **no code changes**.

### 12.5 Output Files

| File | Format | Contents |
|------|--------|---------|
| `benchmarks/EXPERIMENT_RESULTS.json` | JSON | Raw numbers — per-question latency, tokens, F1, compression stats |
| `docs/benchmarks/experiment_results.md` | Markdown | Human-readable report with PASS/FAIL badges |

---

## 13. Recent Improvements

The following improvements were committed to main on 2026-06-26 (pre-benchmark; results pending against the Experiment 2 baseline):

| # | Improvement | File | Expected Impact |
|---|-------------|------|-----------------|
| 1 | ToT composite sentence branch queries | `tot_reasoner.py` | 3× fewer ChromaDB calls; gradient cosine scoring replaces binary hit counts |
| 2 | Retrieval-optimized compression prompt | `compressor.py` | Content-word-dominated embeddings; higher cosine separation between chunks |
| 3 | `_normalise_for_index()` stopword stripping | `compressor.py` | Removes residual function words from stored documents before embedding |
| 4 | Entity list appended to `compressed_summary` | `compressor.py` | ChromaDB embedding captures deduplicated entity signal that ToT branches search for |
| 5 | Chunk overlap 128 → 64 tokens | `compressor.py` | ~14% fewer chunks; ~14% less one-time compression time and smaller ChromaDB index |

See [docs/benchmarks/experiment_results.md](../benchmarks/experiment_results.md) for the pending benchmark results and the metrics to watch.

---

## See Also

- [plan.md](../plan.md) — Implementation phases and current status
- [Experiments](../benchmarks/experiment_results.md) — Full validation data
- [experiment_results.md](experiments/experiment_results.md) — Live E2E results
