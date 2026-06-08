# Databricks RAG Pipeline

A production-ready retrieval-augmented generation (RAG) pipeline demonstrating the integration of Delta Lake, vector embeddings, and large language models. This project showcases a microservices architecture where each pipeline phase can be developed, tested, and deployed independently.

## Architecture Overview

The pipeline is separated into three independent phases:

1. **Phase 1: Ingestion** - Raw corpus → Delta Lake storage
2. **Phase 2: Vectorization** - Delta Lake → Vector embeddings in ChromaDB
3. **Phase 3: Serving** - FastAPI-based RAG query interface

```
┌─────────────┐      ┌──────────────┐      ┌─────────────┐
│   Raw Data  │──1──▶│  Delta Lake  │──2──▶│  ChromaDB   │
│ (Wikipedia) │      │   (ACID)     │      │  (Vectors)  │
└─────────────┘      └──────────────┘      └──────┬──────┘
                                                   │
                                                   │ 3
                                                   ▼
                                            ┌─────────────┐
                                            │ RAG Server  │
                                            │  (FastAPI)  │
                                            └─────────────┘
```

Each phase has:
- Independent Dockerfile for container deployment
- Isolated requirements.txt for minimal dependencies
- Standalone setup scripts for local development
- Dedicated test suite

## Execution Modes

### Local Development (No Docker)

For rapid iteration and debugging:

```bash
# Setup all phases
make local-setup

# Run individual phases
make local-ingest      # Phase 1 only
make local-vectorize   # Phase 2 only
make local-serve       # Phase 3 only

# Or run the complete pipeline
make local-full
```

Each phase runs in its own virtual environment with isolated dependencies.

### Docker Deployment (Production-Ready)

For reproducible deployments and cluster readiness:

```bash
# Build all Docker images
make docker-build

# Run the complete orchestrated pipeline
make docker-run

# Or run individual phases
make docker-ingest
make docker-vectorize
make docker-serve
```

All phases share a mounted data volume and communicate through persisted storage layers (Delta Lake → ChromaDB).

## Project Structure

```
databricks_rag/
├── phase1-ingest/          # Corpus → Delta Lake
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── setup.sh / .ps1
│   └── src/
│       ├── ingest.py
│       └── loaders/
├── phase2-vectorize/       # Delta Lake → ChromaDB
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── setup.sh / .ps1
│   └── src/
│       ├── vectorize.py
│       └── embeddings/
├── phase3-serve/           # RAG query server
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── setup.sh / .ps1
│   └── src/
│       ├── server.py
│       ├── rag_pipeline.py
│       └── models.py
├── shared/                 # Cross-phase utilities
│   ├── config_loader.py
│   ├── logging_config.py
│   └── constants.py
└── data/                   # Shared volume
    ├── raw/
    ├── delta_lake/
    └── chroma_db/
```

## Configuration

The `config.yaml` file controls execution mode and parameters:

```yaml
mode: local  # or "remote" for Databricks workspace

local:
  dataset: wikipedia
  sample_size: 1000
  delta_path: ./data/delta_lake
  vector_store: ./data/chroma_db
  embedding_model: sentence-transformers/all-MiniLM-L6-v2

remote:
  workspace_url: ""
  token_env_var: DATABRICKS_TOKEN
  vector_search_endpoint: ""
  catalog: main
  schema: rag_demo
```

Credentials are managed through environment variables (see `.env.example`).

## Dataset

The default corpus is Wikipedia Simple English (1000 articles, ~10MB). This dataset:
- Runs efficiently on stock CPUs without GPU requirements
- Provides diverse general-knowledge content for testing retrieval quality
- Completes full pipeline (ingest → vectorize → serve) in under 5 minutes

Alternative datasets can be configured in `phase1-ingest/src/loaders/`.

## API Usage

Once Phase 3 is running:

```bash
# Query the RAG endpoint
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is machine learning?"}'

# Interactive API documentation
open http://localhost:8000/docs
```

## Why This Architecture

**Microservices pattern**: Each phase can scale independently. In production, you might run vectorization as a batch job (scale up for large corpora) while keeping the serving layer lightweight (scale out for query throughput).

**Technology demonstration**: Shows understanding of:
- Delta Lake for data versioning and ACID guarantees
- Vector embeddings and semantic search
- LangChain orchestration patterns
- FastAPI for production serving
- Docker multi-stage builds and compose orchestration

**Deployment flexibility**: The same codebase supports local development (bare metal Python) and cloud deployment (containerized microservices). The remote mode demonstrates Databricks integration without requiring a paid account for local testing.

## Testing

Each phase includes unit tests:

```bash
# Test individual phases
cd phase1-ingest && pytest tests/
cd phase2-vectorize && pytest tests/
cd phase3-serve && pytest tests/
```

## Cleanup

```bash
# Remove generated data
make clean

# Remove Docker images and containers
make docker-clean
```

## Next Steps

1. **Add monitoring**: Integrate Prometheus metrics and Grafana dashboards
2. **Implement retrieval evaluation**: Add NDCG, MRR, and precision@k metrics
3. **Support additional vector stores**: Extend to FAISS, Pinecone, or Weaviate
4. **Enable Databricks remote mode**: Configure Unity Catalog and Vector Search integration

## License

Part of the ai-portfolio repository. See root LICENSE for details.
