# Context Optimizer: Hierarchical RAG at Scale

> **Independently designed and benchmarked.** A production-grade context engineering
> pipeline that makes large-corpus retrieval economically viable by replacing naive
> context injection with a two-tier compress-then-retrieve architecture.

---

## The Engineering Problem

Standard RAG breaks down at scale. A 2 GB unstructured corpus (~500 M tokens) injected
naively into an LLM context window would cost ~$185 per query at GPT-4 pricing — and
exceeds every available context window. Even "smart" chunking still sends hundreds of
raw text passages that overwhelm the reasoning model.

This project solves that with three interlocking mechanisms, each addressing a specific
failure mode the naive approach cannot handle:

| Failure mode | Mechanism | Measured result |
|---|---|---|
| Context exhaustion | Rolling-window compression at write-time | 91.3% token reduction |
| Summary blurring | Parent-child multi-vector retrieval | 85% → 100% recall on specific queries |
| Ingestion bottleneck | K-Means cluster-then-compress | 90–98% fewer LLM calls |
| Cost at query time | Agentic raw-text fallback (lazy loading) | Raw text fetched only when summaries are insufficient |

These are **not theoretical improvements** — each has a benchmark that measures it.

---

## Architecture: Three-Stage Pipeline

```
[ Raw Corpus ] ──────────────────────────────────────────────────
      │                  WRITE-TIME (once, offline)
      ▼
  K-Means clustering (TF-IDF, no LLM)
  Groups related content → one LLM call per cluster, not per chunk
      │
      ▼
  Rolling-window compression (cheap SLM, e.g. llama3.2:3b)
  → CompressedChunk: ~50-token summary + entity list + keywords
      │
  ┌───┴───────────────────────────────────────────────────┐
  │   ChromaDB (parent index)    SQLite FTS5 (raw vault)  │
  │   compressed summaries       original text            │
  │   embedded via MiniLM        BM25 keyword search      │
  │   ──────────────────         ─────────────────────    │
  │   ChromaDB (child index)                              │
  │   raw 200-token sub-chunks                            │
  │   maps each child → parent                           │
  └───────────────────────────────────────────────────────┘
      │
      │                  QUERY-TIME (every request)
      ▼
  Parent-child retrieval (search_with_child_index)
  1. Query hits child index → raw sub-chunks preserve exact vocabulary
  2. Child result maps back to parent summary
  3. Reasoning model gets coherent summary + raw text pointer on demand
      │
      ▼
  Tree-of-Thought reasoner (multi-branch, scores by evidence density)
      │
      ▼
  [ Answer ]  (91.3% fewer tokens vs raw-injection baseline)
```

### Why parent-child retrieval matters

An LLM summariser compresses "The detective found a gold pocket watch in the drawer"
into "detective found clue in drawer." The word *pocket watch* is gone. A query for
"pocket watch" now misses the chunk entirely.

The parent-child index keeps the raw sub-chunk (with *pocket watch* intact) in a
separate ChromaDB collection. The query hits the child, which carries a `parent_chunk_id`
pointer, and the parent summary is returned. Summary-only recall on specific-detail
queries in the benchmark: **85% → 100%** after enabling the child index.

---

## Benchmark Results

All benchmarks run locally — no cloud API keys, no GPU required.

### Core metrics (11,574-line corpus, Pride & Prejudice)

| Metric | Baseline (raw corpus) | Compressed architecture | Pass/fail |
|---|---|---|---|
| Avg prompt tokens | 180,734 | 15,798 | — |
| **Token reduction** | — | **91.3%** | ≥ 90% ✅ |
| Reasoning latency | 155.9 s | 79.5 s (−49%) | ≤ +10% ✅ |
| Judge score (0–1) | 0.97 | 0.97 (+0%) | ≤ −20% ✅ |
| KW-F1 | 0.068 | 0.160 (+136%) | ≤ −20% ✅ |

### Retrieval quality (offline benchmark, no LLM needed)

| Mode | Recall@3 (granular queries) | Avg query latency |
|---|---|---|
| Summary-only | 85% | 11.5 ms |
| **Parent-child index** | **100%** | 17.4 ms |

Granular queries test specific low-salience details (`21012 connection limit`,
`runbook #RT-1042`, `3 NADH`) that summarisers predictably drop. The child index
recovers all of them.

### K-Means ingestion savings (500-sentence corpus, offline)

| Target cluster size | Sub-chunks (naive) | Clusters | LLM call savings |
|---|---|---|---|
| 10 | 55 | 5 | 90.9% |
| 25 | 55 | 2 | 96.4% |
| 50 | 55 | 1 | 98.2% |

**Reproduce all benchmarks** (no Ollama required):
```powershell
python benchmarks\retrieval_benchmark.py          # all three experiments
python benchmarks\retrieval_benchmark.py --exp 2  # recall comparison only
```

**Reproduce the full corpus benchmark** (requires Ollama):
```powershell
ollama pull llama3.2:3b
python benchmarks\run_experiments.py --full
```

---

## Cost Model

```
Without compression:  $0.37 per query  (50 KB context, GPT-4 pricing)
With compression:     $0.007 per query (97.8% reduction)
Savings:              $0.363/query

1,000 queries/day:    $370 → $7  ($363/day savings)
Annual:               ~$132,000 saved
Break-even point:     2.4 queries (compression amortised)
```

---

## Project Structure

```
context-optimizer/
├── src/context_optimizer/       # Core library
│   ├── compressor.py            # Rolling-window compression
│   │                            # + split_into_sub_chunks()
│   │                            # + cluster_and_compress_corpus() (K-Means)
│   ├── cached_retriever.py      # Semantic cache + ChromaDB
│   │                            # + add_raw_sub_chunks() (child index)
│   │                            # + search_with_child_index() (parent-child retrieval)
│   ├── raw_index.py             # SQLite+FTS5 raw content store
│   ├── index.py                 # CorpusIndex — high-level public API
│   ├── tot_reasoner.py          # Tree-of-Thought multi-branch reasoner
│   ├── protocols.py             # Retriever protocol
│   └── __init__.py
│
├── benchmarks/
│   ├── retrieval_benchmark.py   # Offline benchmark (no LLM/internet)
│   │                            # Exp 1: compression ratio
│   │                            # Exp 2: summary-blurring recall
│   │                            # Exp 3: K-Means ingestion savings
│   ├── book_benchmark.py        # 100-book Gutenberg parallel benchmark
│   ├── run_experiments.py       # Full corpus benchmark (requires Ollama)
│   └── incident_benchmark.py   # Incident-log domain benchmark
│
├── tests/                       # 112 unit tests
│   ├── test_compressor.py
│   ├── test_cached_retriever.py
│   ├── test_raw_index.py
│   ├── test_corpus_index.py
│   └── test_tot_reasoner.py
│
└── docs/
    ├── design/ARCHITECTURE.md   # Full system design with diagrams
    └── benchmarks/experiment_results.md
```

---

## Quick Start

```powershell
# Install
pip install -e .

# Run offline benchmark (verifies the architecture, no Ollama needed)
python benchmarks\retrieval_benchmark.py

# High-level API
python - <<'EOF'
from context_optimizer import CorpusIndex

index = CorpusIndex(compression_model="llama3.2:3b")
index.ingest(open("my_corpus.txt").readlines())
result = index.query("what caused the CosmosDB timeout?")
print(result.answer)
EOF
```

**Use parent-child retrieval directly:**
```python
from context_optimizer.compressor import compress_corpus_rolling, split_into_sub_chunks
from context_optimizer.cached_retriever import CachedChromaRetriever

# Compress corpus
chunks = compress_corpus_rolling(corpus_lines, strategy="extractive")

# Build index with child sub-chunks for granular keyword recall
retriever = CachedChromaRetriever(collection_name="my_corpus")
retriever.add_chunks(chunks)                         # parent summaries
retriever.add_raw_sub_chunks(chunks, sub_chunk_tokens=200)  # child index

# Query — child hits route back to parent summaries automatically
results = retriever.search_with_child_index("pocket watch gold", top_k=5)
```

**Use K-Means clustering to cut ingestion cost:**
```python
from context_optimizer.compressor import cluster_and_compress_corpus

# 50 raw sub-chunks → 2 cluster summaries (96% fewer LLM calls)
chunks = cluster_and_compress_corpus(
    corpus_lines,
    target_cluster_size=25,   # ~25 sub-chunks per cluster
    strategy="extractive",    # or "llm" with Ollama running
)
```

---

## Provider Support

| Provider | Role | Config |
|---|---|---|
| Ollama (`llama3.2:3b`, `qwen2.5-coder:7b`) | Compression + reasoning (local) | Default; `OLLAMA_BASE_URL` |
| Groq (`llama-3.3-70b-versatile`) | Cloud compression/reasoning | `GROQ_API_KEY` |
| Azure OpenAI (`gpt-4o-mini`, `gpt-4o`) | Enterprise deployment | `AZURE_OPENAI_ENDPOINT` + `AZURE_OPENAI_API_KEY` |
| sentence-transformers (`all-MiniLM-L6-v2`) | Embeddings (local CPU) | Default |
| Ollama (`nomic-embed-text`) | Embeddings (local, higher quality) | `CONTEXT_OPTIMIZER_EMBEDDING_BACKEND=ollama` |

---

## Tests

112 tests across all modules:

```powershell
python -m pytest tests/ -q
python -m pytest tests/ --cov=context_optimizer --cov-report=term-missing
```

| Module | Tests | What is covered |
|---|---|---|
| `test_compressor.py` | 28 | Rolling window, extractive, sub-chunk splitting, K-Means clustering |
| `test_cached_retriever.py` | 31 | Semantic cache, ChromaDB, parent-child index |
| `test_raw_index.py` | 17 | SQLite+FTS5 CRUD, BM25 search, WAL-mode thread safety |
| `test_corpus_index.py` | 12 | CorpusIndex high-level API |
| `test_tot_reasoner.py` | 14 | Branch generation, scoring, raw fallback |
| `test_protocols.py` | 10 | Protocol conformance |

---

## Design Documentation

| Document | What it covers |
|---|---|
| [docs/design/ARCHITECTURE.md](docs/design/ARCHITECTURE.md) | Full system design: rolling compression, dual storage, parent-child retrieval, K-Means ingestion, Tree-of-Thought |
| [docs/benchmarks/experiment_results.md](docs/benchmarks/experiment_results.md) | All benchmark runs with raw numbers |

---

## Engineering Decisions

**Why not just use a bigger context window?**
Context windows are finite and expensive. Longer windows degrade reasoning quality
("lost in the middle" effect). This architecture keeps the reasoning model's context
tight and evidence-dense regardless of corpus size.

**Why rolling-window compression instead of full-document summarisation?**
Full-document summarisation hits context limits immediately on large corpora. Rolling
window processes one chunk at a time — O(1) memory, parallelisable, and failures are
isolated to one chunk.

**Why two ChromaDB collections (parent + child)?**
Semantic similarity over compressed summaries works well for broad thematic queries.
But specific low-salience details (exact error codes, character-level facts, specific
measurements) get dropped by LLM summarisers. The child collection embeds raw 200-token
windows that preserve exact vocabulary. On retrieval, the child hit routes back to the
parent summary so the reasoning model gets coherent context, not a fragment.

**Why K-Means before summarising?**
Summarising every chunk independently ignores semantic structure. Related content
clustered together produces denser, more coherent summaries and drastically reduces
LLM calls — from O(N chunks) to O(N/cluster_size), typically a 95%+ reduction.

**Why SQLite + FTS5 as a second store?**
ChromaDB is optimised for vector similarity. BM25 keyword search over raw text
requires a different engine. SQLite's FTS5 provides sub-millisecond exact-match
lookup and ranked keyword search without any additional infrastructure. The WAL
journal mode allows parallel writes during ingestion at essentially zero cost
(SQLite write ~1 ms vs LLM call ~500 ms — fully overlapped).

- `benchmarks/run_experiments.py` updates the incident benchmark appendix section inside `docs/benchmarks/experiment_results.md`.

## Project files

- `context_optimizer_benchmark.py`: end-to-end benchmark runner with both pipelines
- `context_optimizer/`: installable package namespace and CLI entry point
- `pyproject.toml`: build metadata for packaging and deployment
- `requirements.txt`: runtime dependencies
- `README.md`: benchmark design and operating guide
- `Dockerfile.raw`: baseline CPU image (full raw context mode)
- `Dockerfile.optimized`: optimized CPU image (compression + retrieval mode)
- `evaluation/run_dual_benchmark.py`: single orchestration script for container runs + report generation
- `run_evaluation.ps1` and `run_evaluation.sh`: cross-platform launch scripts
- `setup.ps1` and `setup.sh`: project setup with dependency install + optional model downloads
- `tests/test_context_optimizer.py`: CPU-safe unit tests for cache/search/compression/pipelines

## Limitations

- The log backend is in-memory and single-process; this isolates behavior but is not a distributed cache.
- Token counts are estimated via character counts in this baseline version.
- Model quality and tool-calling reliability vary by provider/model choice.

## Next extensions

- Add real token accounting (provider tokenizer API or `tiktoken` where applicable).
- Persist benchmark runs to CSV/SQLite for trend analysis.
- Expand retrieval tooling (time-range filters, service filters, regex mode).
