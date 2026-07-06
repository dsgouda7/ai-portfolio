# Context Optimizer: Architecture Design

> **Canonical reference** for the Context Optimizer system.
> Related: [Benchmark results](../benchmarks/experiment_results.md) · [README](../../README.md)

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Data Ingestion Pipeline](#2-data-ingestion-pipeline)
3. [Rolling Window Compression](#3-rolling-window-compression)
4. [K-Means Cluster-then-Compress](#4-k-means-cluster-then-compress)
5. [Dual Storage Architecture](#5-dual-storage-architecture)
6. [Retrieval Layer](#6-retrieval-layer)
7. [Parent-Child Multi-Vector Retrieval](#7-parent-child-multi-vector-retrieval)
8. [Semantic Cache](#8-semantic-cache)
9. [Tree-of-Thought Reasoner](#9-tree-of-thought-reasoner)
10. [Agentic Raw-Text Fallback](#10-agentic-raw-text-fallback)
11. [Swappable Embedding Backend](#11-swappable-embedding-backend)
12. [Integrated Query Flow](#12-integrated-query-flow)
13. [Performance & Benchmarks](#13-performance--benchmarks)
14. [Implementation Files](#14-implementation-files)

---

## 1. System Overview

Context Optimizer is a **token-efficient hierarchical RAG pipeline** that makes
large-corpus retrieval economically viable. It compresses an arbitrarily large corpus
once at write-time, then answers queries by searching compressed summaries first and
fetching raw text on demand only when the reasoning model asks for it.

**Measured results** (11,574-line corpus, CPU-only, no GPU):

| Metric | Baseline | Compressed | Threshold |
|---|---|---|---|
| Token reduction | — | **91.3%** | ≥ 90% |
| Reasoning latency | 155.9 s | 79.5 s (−49%) | ≤ +10% |
| Judge score (0–1) | 0.97 | 0.97 (+0%) | ≤ −20% |
| KW-F1 | 0.068 | 0.160 (+136%) | ≤ −20% |

**Three failure modes addressed:**

| Failure mode | Solution | Where |
|---|---|---|
| Context exhaustion | Rolling-window compression | §3 |
| Ingestion bottleneck (O(N) LLM calls) | K-Means cluster-then-compress | §4 |
| Summary blurring (lost keywords) | Parent-child multi-vector retrieval | §7 |

### Full Pipeline

```
Raw Corpus (any size)
    │
    ├──[OPTIONAL]── K-Means clustering (§4)
    │               Groups related sub-chunks before compression
    │               Reduces LLM calls by 90–98%
    │
    ▼  [WRITE-TIME — once, offline]
Rolling Window Compression (§3)
  LLM (cheap: llama3.2:3b) · 512-token batches · entity/keyword extraction
    │
    ▼
Three storage tiers:
  ┌─────────────────────────────────────────────────────────────────┐
  │  ChromaDB  parent collection                                    │
  │  compressed summaries (~50 tokens) · cosine HNSW · semantic     │
  ├─────────────────────────────────────────────────────────────────┤
  │  ChromaDB  children collection  (§7)                            │
  │  raw 200-token sub-chunks · exact vocabulary · parent pointer   │
  ├─────────────────────────────────────────────────────────────────┤
  │  SQLite + FTS5  raw vault                                       │
  │  original text · O(1) lookup · BM25 keyword search              │
  └─────────────────────────────────────────────────────────────────┘
    │
    ▼  [QUERY-TIME — every request]
Parent-child retrieval (§7)
  1. Query → children collection (exact vocabulary hit)
  2. Map child → parent_chunk_id → fetch parent summary
  (fallback: semantic search on parent collection)
    │
    ▼
Tree-of-Thought reasoner (§9)
  N branches, each retrieving evidence; winner ranked by cosine score
    │
    ├── Sufficient evidence → answer directly (compressed summaries only)
    └── Insufficient (score < threshold) → fetch raw text on demand (§10)
    │
    ▼
Answer  (91.3% fewer tokens vs baseline)
```


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

## 4. K-Means Cluster-then-Compress

### The Ingestion Bottleneck

Compressing every 200-token sub-chunk individually makes O(N) LLM calls. For a 2 GB
corpus (~500 M tokens → ~2.5 M sub-chunks), that is millions of LLM inference calls.
Even at 30 tokens/second this would take months on a local SLM.

### The Solution: Cluster First, Summarise Once Per Cluster

```python
from context_optimizer.compressor import cluster_and_compress_corpus

# 55 sub-chunks grouped into 2 clusters = 96% fewer LLM calls
chunks = cluster_and_compress_corpus(
    corpus_lines,
    target_cluster_size=25,  # ~25 sub-chunks per cluster
    sub_chunk_tokens=200,    # split corpus into 200-token windows first
    strategy="extractive",   # or "llm" when Ollama is running
)
```

### Algorithm

```
1. split_into_sub_chunks(text, 200 tokens)
       ↓
2. TfidfVectorizer(max_features=5000, stop_words="english")
   Vectorise all sub-chunks — no LLM, no embeddings, < 1 s for 10K sub-chunks
       ↓
3. KMeans(n_clusters = len(sub_chunks) // target_cluster_size)
   Group semantically related sub-chunks
       ↓
4. For each cluster:
   Concatenate all sub-chunk texts → one LLM call per cluster
   (vs. one LLM call per sub-chunk in naive approach)
       ↓
5. Return list[CompressedChunk]  (one per cluster)
```

### Measured Savings

| Target cluster size | Sub-chunks | Clusters | LLM call reduction |
|---|---|---|---|
| 10 | 55 | 5 | **90.9%** |
| 25 | 55 | 2 | **96.4%** |
| 50 | 55 | 1 | **98.2%** |

Intra-cluster Jaccard coherence at `target=10`: **0.635** (sentences grouped by topic,
not randomly). At `target=50` coherence drops to 0.065 as fewer, larger clusters
inevitably contain more diverse content.

**Practical guidance:** `target_cluster_size=25` balances LLM call savings (~96%)
against summary coherence. For domain-specific corpora (all logs, all legal text)
larger clusters work well. For mixed corpora, prefer smaller clusters.

---

## 5. Dual Storage Architecture

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

## 6. Retrieval Layer

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

## 7. Parent-Child Multi-Vector Retrieval

### The Summary-Blurring Problem

An LLM compressing "The detective found a gold pocket watch in the drawer" might
produce "detective found evidence." The specific noun *pocket watch* vanishes.
A user query for "pocket watch" returns zero hits from the summary index — even
though the raw text contains the answer.

This is not a contrived example. In the offline benchmark:
- 3 of 20 specific-detail queries missed on the summary-only index: `21012 connection limit`,
  `runbook #RT-1042`, `3 NADH 1 FADH2 GTP`
- All 3 were recovered by the child index
- Summary-only Recall@3: **85%** → Parent-child Recall@3: **100%**

### Design

```
ChromaDB  parent collection     ChromaDB  children collection
──────────────────────────────  ──────────────────────────────────────────
chunk_000: "CosmosDB timeout…"  chunk_000__child_0000: "System.Timeout…"
                                chunk_000__child_0001: "Error code 21012…"
                                chunk_000__child_0002: "retry 3/3 failed…"
                                  │
                                  └── metadata: { parent_chunk_id: "chunk_000" }
```

Children embed raw text, not summaries. Exact vocabulary (including `21012`,
`pocket watch`, `3 NADH`) is preserved in the child vector space.

On retrieval:
1. Query → search children collection (exact vocab match)
2. Map each child → `parent_chunk_id`
3. Deduplicate by parent; rank by best child cosine distance
4. Fetch parent summaries from main collection
5. Return parent summaries (coherent context for reasoning model)

### Usage

```python
retriever = CachedChromaRetriever(collection_name="my_corpus")

# Build both indexes
retriever.add_chunks(compressed_chunks)               # parent summaries
n = retriever.add_raw_sub_chunks(
    compressed_chunks,
    sub_chunk_tokens=200,  # ~800 chars per child
)
print(f"{n} child sub-chunks indexed")

# Query via parent-child (recommended for specific-detail queries)
results = retriever.search_with_child_index("pocket watch gold chain", top_k=5)

# Query via summary-only (faster, sufficient for broad thematic queries)
results = retriever.search("CosmosDB timeout cascade", top_k=5)
```

### When to use each mode

| Query type | Recommended mode | Why |
|---|---|---|
| Specific facts, exact values, error codes | `search_with_child_index` | Child index preserves granular vocabulary |
| Thematic / conceptual questions | `search` | Summary embeddings capture broader semantics |
| Unknown query type | `search_with_child_index` | Slightly higher latency (~17 ms vs ~12 ms) but higher recall |

---

## 8. Semantic Cache

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

## 9. Tree-of-Thought Reasoner

The Tree-of-Thought (ToT) reasoner explores multiple analytical hypotheses in parallel
before committing to an answer, modelled after the [Tree of Thoughts paper (Yao et al., 2023)].

### Algorithm

```
1. Derive N branch specs from the compressed context
   (entity list → one branch per entity cluster)
       ↓
2. For each branch:
   - Issue composite sentence query against ChromaDB
   - Retrieve top-k evidence snippets
   - Accumulate cosine similarity scores
       ↓
3. Score branches by mean cosine similarity (gradient signal)
   (e.g. 0.72 vs 0.68 — quantitative, not binary hit/miss)
       ↓
4. Return winner: selected_branch + evidence + synthesised answer
```

### Usage

```python
from context_optimizer.tot_reasoner import ToTReasoner

reasoner = ToTReasoner(
    retriever=cached_chroma_retriever,
    top_k_per_term=3,
    raw_fallback_threshold=0.40,   # score < 0.40 → fetch raw text
)

result = reasoner.reason(compressed_chunk)
print(result.synthesized_answer)
print(result.winner.score)         # winning branch cosine score
```

---

## 10. Agentic Raw-Text Fallback

The reasoning pipeline implements lazy loading: summaries are sent to the model first;
raw text is fetched only when the evidence is insufficient.

```
Query
  ↓
Retrieve compressed summaries (§6)
  ↓
ToT reasoning pass
  ├── winner.score >= raw_fallback_threshold (0.40)
  │     → answer directly from summaries
  │
  └── winner.score < threshold
        → re-retrieve using FTS5 keyword search on RawIndex
        → exact source vocabulary available for precise answer
        → raw text tokens added only for this query
```

**Why this is significant:**
- Summaries handle ~80% of queries → 91.3% average token reduction
- Raw text available for the remaining ~20% where precision matters
- The model never "loses" information — it can always drill down
- Token cost is **per-query-on-demand**, not upfront for all queries

```python
# Raw fallback via RawIndex BM25 keyword search
hits = raw_index.search("21012 connection limit", top_k=3)
for hit in hits:
    print(hit.chunk_id, hit.raw_text[:80])
```

---

## 11. MCP Tool Contract

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

## 12. Swappable Embedding Backend

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

## 13. Integrated Query Flow

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

## 14. Performance & Benchmarks

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

## 15. Implementation Files

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

## 16. End-to-End Local Test Setup

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
