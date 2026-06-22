# Context Optimizer — Deployment

Multi-container packaging of the full Context Optimizer pipeline.

```
deployment/
  docker-compose.yml        ← orchestrates all four services
  README.md                 ← this file
  ingestion/                ← Service 1: rolling-window LLM compression
  embedding/                ← Service 2: sentence-transformers vectorization
  vector-store/             ← Service 3: ChromaDB + semantic cache + raw-text pointers
  reasoning-gateway/        ← Service 4: LiteLLM + MCP tools → remote reasoning LLM
```

---

## Architecture

```
                          ┌──────────────────────────────────────────────┐
WRITE-TIME (once)         │  Client / CLI                                │
                          └──────────────┬───────────────────────────────┘
                                         │  POST /ingest  {corpus: [...]}
                                         ▼
                          ┌──────────────────────────────────────────────┐
                          │  ingestion  :8001                            │
                          │  Rolling-window LLM compression              │
                          │  (compressor.py → CompressedChunk[])         │
                          └──────────────┬───────────────────────────────┘
                                         │  POST /chunks  {chunks: [...]}
                                         ▼
                          ┌──────────────────────────────────────────────┐
                          │  embedding  :8002                            │
                          │  sentence-transformers vectorization         │
                          │  (all-MiniLM-L6-v2, local CPU)              │
                          │  POST /embed → float[][]                     │
                          └──────────────┬───────────────────────────────┘
                                         │  POST /store  {chunk, vector}
                                         ▼
                          ┌──────────────────────────────────────────────┐
                          │  vector-store  :8003                         │
                          │  ChromaDB HNSW + in-memory semantic cache    │
                          │  Stores: compressed summaries + raw-text     │
                          │  Serves:  GET /search   GET /chunks/{id}     │
                          └──────────────────────────────────────────────┘

QUERY-TIME (every request)

  User ──→ reasoning-gateway :8080
              │
              ├── calls retrieve_context → vector-store /search
              ├── calls get_context_details → vector-store /chunks/{id}
              └── routes completion to remote reasoning LLM (Ollama / Groq / Azure)
```

---

## Quick Start

### Prerequisites

- Docker + Docker Compose v2
- Ollama running locally (for compression + reasoning with local models):

  ```powershell
  ollama serve
  ollama pull llama3.2:3b          # compression model
  ollama pull qwen3                 # reasoning model
  ollama pull nomic-embed-text      # fallback embeddings
  ```

### Start the stack

```powershell
cd projects/context-optimizer/deployment
docker compose up --build
```

Services will be available at:
| Service | URL |
|---------|-----|
| ingestion | http://localhost:8001/docs |
| embedding | http://localhost:8002/docs |
| vector-store | http://localhost:8003/docs |
| reasoning-gateway | http://localhost:8080/docs |

### Ingest a corpus

```bash
# Ingest raw corpus lines
curl -X POST http://localhost:8001/ingest \
  -H "Content-Type: application/json" \
  -d '{"corpus": ["line 1", "line 2", "..."]}'
```

### Query via reasoning gateway

```bash
# OpenAI-compatible chat completions
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3",
    "messages": [{"role": "user", "content": "Diagnose the CosmosDB timeout"}]
  }'
```

---

## Environment Variables

### Shared

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_BASE_URL` | `http://host.docker.internal:11434` | Local Ollama endpoint |

### ingestion

| Variable | Default | Description |
|----------|---------|-------------|
| `CONTEXT_OPTIMIZER_COMPRESSOR_PROVIDER` | `ollama` | `ollama` / `groq` / `mock` |
| `CONTEXT_OPTIMIZER_COMPRESSOR_MODEL` | `llama3.2:3b` | Model for rolling-window compression |
| `VECTOR_STORE_URL` | `http://vector-store:8003` | URL of vector-store service |
| `EMBEDDING_SERVICE_URL` | `http://embedding:8002` | URL of embedding service |

### embedding

| Variable | Default | Description |
|----------|---------|-------------|
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | sentence-transformers model name |

### vector-store

| Variable | Default | Description |
|----------|---------|-------------|
| `EMBEDDING_SERVICE_URL` | `http://embedding:8002` | URL of embedding service |
| `CHROMA_PERSIST_DIR` | `/data/chroma_db` | ChromaDB persistence path (volume-mounted) |

### reasoning-gateway

| Variable | Default | Description |
|----------|---------|-------------|
| `VECTOR_STORE_URL` | `http://vector-store:8003` | URL of vector-store service |
| `CONTEXT_OPTIMIZER_COMPRESSOR_PROVIDER` | `ollama` | Compression provider (for middleware) |
| `GROQ_API_KEY` | — | Required when routing to Groq |

---

## Volumes

| Volume | Mounted in | Purpose |
|--------|-----------|---------|
| `chroma-data` | `vector-store:/data/chroma_db` | Persistent ChromaDB index |
| `corpus-data` | `ingestion:/data/corpus`, `vector-store:/data/corpus` | Shared raw corpus for pointer model |

The **pointer model** means raw text is stored on disk (corpus-data volume) and
ChromaDB metadata holds a pointer (`raw_text` field).  The vector index only
stores compressed summaries (~50 tokens each) — keeping the index 10× smaller.
