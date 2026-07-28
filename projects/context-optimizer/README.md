# Arbor: Hierarchical RAG Retrieval

> A retrieval library that builds a multi-level summary tree over your corpus at
> index time so every query navigates from a global overview down to the exact
> raw passage — routing to the right document first, drilling to the right
> granularity second.

---

## The Problem

Standard RAG retrieves the *k* nearest text chunks to a query vector. That sounds right,
but it has three systematic failure modes:

| Failure mode | What happens | Example |
|---|---|---|
| **Domain collision** | Semantically similar words in unrelated domains confuse the retriever | Query "trust" hits both a Python auth module *and* a Jane Austen passage |
| **Context window waste** | Top-k chunks may all come from the wrong document, spending the entire context budget on irrelevant content | A codebase query retrieves 5 novel excerpts |
| **Granularity mismatch** | A broad question needs a document-level answer; a narrow question needs a single sentence — flat RAG has no concept of level | "What does this library do?" and "What does `HTTPAdapter.send()` return?" both query the same flat index |

Arbor addresses all three by building a **summary tree** over your corpus at index time:
a hierarchy of progressively coarser abstractions (raw block → section → document →
corpus). Every query descends the tree from coarse to fine, routing to the right domain
at the top and arriving at the right passage at the bottom.

---

## Architecture

```
 INDEX TIME (once, offline)
 ─────────────────────────────────────────────────────────────────────
 Raw files
     │
     ▼
 Block extraction  ←── configurable block size (default ~800 tokens)
     │
     ▼
 BART summarization (facebook/bart-large-cnn)
 + [Document: filename (type)] prefix baked into each summary
 so BART knows which document a block comes from
     │
     ├── L1 ChromaDB collection  (one entry per raw block)
     │   summary embedded with all-MiniLM-L6-v2
     │
     ├── L2 ChromaDB collection  (cluster summaries of L1 groups)
     ├── L3 ChromaDB collection  (cluster summaries of L2 groups)
     └── L4 ChromaDB collection  (root — summary of summaries)
         │
         └── SQLite blocks.db  (block_id → file_path, byte_start, byte_end)
             raw text read on demand via byte-range seek

 QUERY TIME (every request)
 ─────────────────────────────────────────────────────────────────────
 Query
     │
     ▼
 L1 top-k cosine search  (domain routing via BART summaries)
     │
     ├── score excerpts against query keywords
     │   score ≥ 0.25 → return top-3 sentence-ranked excerpts
     │   score  < 0.25 → expand to top-8, re-score, return best 3
     │
     └── [optional] TreeReasoningAgent (requires Ollama or OpenAI-compat)
         LLM tool loop: search_cluster | fetch_raw_block | answer
         iterates until context is sufficient or max_rounds reached
```

**Key design choices:**

- **Document context prefix** — each block summary begins with `[Document: pride and
  prejudice (text)]` or `[Document: auth (Python source)]`. BART sees this at
  summarization time, so the L1 vectors encode domain identity alongside content.
  This is what makes cosine similarity route correctly across mixed-domain corpora.

- **Byte-range raw text** — the SQLite index stores `(file_path, byte_start, byte_end)`
  rather than copying raw text. Raw content is read on demand with a single seek.
  The index stays small regardless of corpus size.

- **Two retrieval modes** — without a local LLM the server returns sentence-ranked
  excerpts (fast, offline, no API keys). With Ollama running, `TreeReasoningAgent`
  takes over: it calls tools in a loop and synthesizes a coherent answer from
  whatever blocks it fetches.

---

## Quick Start

```powershell
# 1. Install into a virtual environment
python -m venv .venv && .venv\Scripts\Activate.ps1
pip install -e "projects/context-optimizer[dev]"

# 2. Build the index (downloads BART ~400 MB on first run)
cd projects/context-optimizer/demo
python setup_demo.py

# 3. Start the demo server
uvicorn app.server:app --reload
# Open http://127.0.0.1:8000
```

**Optional — enable the LLM reasoning path:**
```powershell
ollama pull llama3.2:3b          # ~2 GB, runs on CPU
$env:DEMO_REASONING_MODEL = "llama3.2:3b"
uvicorn app.server:app --reload  # TreeReasoningAgent now active
```

---

## Benchmark Scenarios

Three scenarios correspond to the three failure modes Arbor is designed to fix.
Run them against a live index:

```powershell
python demo/benchmarks/tree_retrieval_benchmark.py
```

### Scenario 1 — Domain routing accuracy

Queries that have an unambiguous home domain. Measures whether the top-1 retrieved
block comes from the correct source document.

| Query | Expected domain | Pass condition |
|---|---|---|
| "Describe the character of Mr Darcy" | pride-and-prejudice.txt | source_file contains `pride` |
| "How does session-level authentication work?" | auth.py | source_file contains `auth` |
| "What is RAG and how does it work?" | prose corpus | source_file not in novel or requests |
| "How does the cookie jar persist state?" | cookies.py | source_file contains `cookies` |

**Flat RAG failure mode**: without document-context prefixes in summaries, cosine
similarity for "trust" or "session" cross-contaminates between the auth module and
P&P passages about social trust.

### Scenario 2 — Multi-granularity retrieval

Measures whether the retrieval depth matches the question's scope. High-level questions
should be answered from cluster summaries (fewer tokens); low-level questions need
raw block text.

| Query type | Example | Expected behaviour |
|---|---|---|
| High-level | "What HTTP features does requests provide?" | Cluster summary sufficient; low distance at L3/L4 |
| Low-level | "How does `HTTPAdapter.send()` handle SSL?" | Drills to L1 raw block; specific function present |

### Scenario 3 — Iterative expansion

Queries where initial top-3 keyword overlap scores below 0.25, triggering second-pass
expansion to top-8. Measures: (a) expansion triggered when expected, (b) the
expanded result scores higher.

| Query | Why initial pass is weak | Expected step in response |
|---|---|---|
| "Is Mr Darcy proud or humble?" | "proud" not in any L1 summary verbatim | `expand_retrieval` step logged |
| "How does requests handle redirects end to end?" | Answer spans multiple blocks | `expand_retrieval` step logged |

---

## Demo API Reference

The demo server exposes these endpoints:

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/` | Single-page demo app |
| `GET` | `/api/status` | Index stats: block count, depth, ready flag |
| `GET` | `/api/tree` | Full tree as `{nodes, edges}` for D3 visualization |
| `GET` | `/api/files` | Indexed file listing |
| `GET` | `/api/file?path=…` | Raw content of a specific file |
| `POST` | `/api/query` | Run a query; returns `answer`, `cluster_hits`, `steps` |
| `GET` | `/api/block?block_id=…` | Raw text of a specific block |
| `GET` | `/api/debug?query=…` | Diagnostic: L1 query with error exposure |
| `POST` | `/api/repair` | Rebuild a broken L1 HNSW index from metadata |

**`POST /api/query` request body:**
```json
{
  "query": "How does session-level authentication work?",
  "top_clusters": 4,
  "top_blocks_per_cluster": 4,
  "max_rounds": 3,
  "gap": 2.0
}
```

**Response:**
```json
{
  "query": "…",
  "answer": "[Retrieved from: auth.py]\n\n…",
  "cluster_hits": […],
  "steps": [
    {"action": "search_clusters", "detail": "top 4 clusters"},
    {"action": "retrieval_sufficient", "detail": "initial retrieval: best_score=0.67"}
  ],
  "fetched_blocks": [],
  "latency_ms": 89.4,
  "reasoning_model": null
}
```

---

## Project Structure

```
context-optimizer/
├── src/context_optimizer/
│   ├── tree_index.py        # TreeIndex — N-level ChromaDB hierarchy
│   ├── raw_index.py         # BlockIndex — SQLite byte-range store
│   ├── compressor.py        # BART summarization + document context prefix
│   ├── ingest_corpus.py     # Multi-file ingestion pipeline
│   ├── tree_reasoner.py     # TreeReasoningAgent — LLM tool-calling loop
│   ├── providers/           # Ollama, OpenAI-compat, Azure, HuggingFace
│   ├── extractors/          # Per-filetype text extraction
│   └── adapters/            # Protocol adapters
│
├── demo/
│   ├── setup_demo.py        # Build the Arbor index for the demo corpus
│   ├── app/
│   │   ├── server.py        # FastAPI backend
│   │   └── static/          # Single-page app (D3 tree visualization)
│   ├── benchmarks/
│   │   └── tree_retrieval_benchmark.py   # Domain routing + granularity + expansion
│   └── corpus/              # Demo corpus (prose, requests source, P&P)
│
└── checkpoints/             # Pre-built model adapters (LoRA, DPO, PEFT)
```

---

## Known Limitations

**No generation step in the default path.** Without a local LLM configured, Arbor
returns sentence-ranked excerpts — relevant passages from the right document, but
not a synthesized prose answer. Wire in `TreeReasoningAgent` (Ollama) or any
OpenAI-compatible endpoint for the full RAG experience.

**BART at the root level.** `facebook/bart-large-cnn` was trained on CNN/DailyMail
news. When the L4 root summarizes a mixed corpus (novel + Python code), it will
favour the technical content (closer to news prose) and lose the literary content.
The domain-context prefix mitigates this at L1; at L4 a proper instruction-following
model would do better.

**Static index.** The summary tree is built once. Adding new documents requires a
full or partial rebuild. Incremental update support is a planned extension.


