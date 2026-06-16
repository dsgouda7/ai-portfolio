# RAG Knowledge Pipeline

## Problem statement

**Can a fully local, containerised pipeline ingest an arbitrary text corpus, build a vector index, and serve accurate retrieval-augmented answers — with each stage independently deployable and replaceable?**

Most RAG demos are single-script prototypes: one file, one vector store, one LLM call. Real production systems need the ingest, vectorisation, and serving stages to be independently scalable, observable, and replaceable. This project builds that separation explicitly, using Delta Lake as the durable intermediate store between ingest and vectorisation, and ChromaDB as the vector store backing a FastAPI RAG server.

**Constraints we set for ourselves:**
- Each pipeline phase must be independently runnable (local venv or Docker) with no dependency on the other phases at runtime
- Durable intermediate storage between stages — not in-memory hand-offs
- The serving layer must not know anything about how the corpus was ingested or embedded

**Result:** A three-phase containerised pipeline (Wikipedia corpus → Delta Lake → ChromaDB → FastAPI RAG server) where each phase can be developed, tested, and deployed independently.

## Architecture

```
┌─────────────┐      ┌──────────────┐      ┌─────────────┐
│  Raw corpus │──1──▶│  Delta Lake  │──2──▶│  ChromaDB   │
│ (Wikipedia) │      │   (ACID)     │      │  (Vectors)  │
└─────────────┘      └──────────────┘      └──────┬──────┘
                                                   │ 3
                                                   ▼
                                            ┌─────────────┐
                                            │ RAG Server  │
                                            │  (FastAPI)  │
                                            └─────────────┘
```

| Phase | Input | Output | Key tech |
|---|---|---|---|
| **1 — Ingest** | Raw Wikipedia articles | Delta Lake parquet | PySpark, delta-spark |
| **2 — Vectorise** | Delta Lake | ChromaDB collection | sentence-transformers, chromadb |
| **3 — Serve** | ChromaDB + LLM | HTTP RAG responses | FastAPI, LangChain |

## Quick start

### Local (no Docker)

```bash
make local-setup    # creates venvs for all three phases
make local-ingest   # Phase 1: corpus → Delta Lake
make local-vectorize # Phase 2: Delta Lake → ChromaDB
make local-serve    # Phase 3: start FastAPI server
# or run all at once:
make local-full
```

### Docker

```bash
make docker-build   # build all three images
make docker-run     # run the complete pipeline
# or individually:
make docker-ingest && make docker-vectorize && make docker-serve
```

All phases share a mounted data volume. Delta Lake → ChromaDB communication happens through persisted storage, not in-memory.

## Project structure

```
rag-knowledge-pipeline/
├── phase1-ingest/          # corpus → Delta Lake
│   ├── Dockerfile
│   ├── requirements.txt
│   └── src/ingest.py
├── phase2-vectorize/       # Delta Lake → ChromaDB embeddings
│   ├── Dockerfile
│   ├── requirements.txt
│   └── src/vectorize.py
├── phase3-serve/           # FastAPI RAG query server
│   ├── Dockerfile
│   ├── requirements.txt
│   └── src/server.py
├── shared/                 # cross-phase config utilities
├── data/                   # shared volume mount point
├── docker-compose.yml
└── Makefile
```

## Limitations

The corpus is a static Wikipedia snapshot — there is no incremental update mechanism. Embedding quality is bounded by the chosen sentence-transformer model (no fine-tuning). The LLM used for generation is a local model via LangChain; answer quality depends entirely on what fits in the available compute budget.
