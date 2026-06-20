# Context Optimizer - Docker Deployment Guide

## Quick Start

### 1. Reorganize the Codebase (One-Time Setup)

```powershell
# Run reorganization script
cd c:\repos\ai-portfolio\projects\context-optimizer
.\reorganize.ps1

# Or dry-run to preview changes
.\reorganize.ps1 -DryRun
```

### 2. Build Docker Images

```bash
# Build service image
docker compose -f docker/docker-compose.yml build context-optimizer

# Build test image
docker compose -f docker/docker-compose.yml build test-runner
```

### 3. Run the Service

```bash
# Start compression service
docker compose -f docker/docker-compose.yml up context-optimizer

# Or with Ollama included
docker compose -f docker/docker-compose.yml --profile llm up
```

### 4. Run Tests

```bash
# Run unit tests
docker compose -f docker/docker-compose.yml --profile testing run test-runner

# Run specific benchmark
docker compose -f docker/docker-compose.yml --profile testing run test-runner \
    python benchmarks/tot/run_fast_tot_benchmarks.py

# Interactive shell
docker compose -f docker/docker-compose.yml --profile testing run test-runner /bin/bash
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    context-optimizer:latest                  │
│  ┌──────────────┐  ┌─────────────┐  ┌──────────────────┐  │
│  │   FastAPI    │  │  Compressor │  │    Retriever     │  │
│  │   Gateway    │──│   (Core)    │──│   (DualStorage)  │  │
│  └──────────────┘  └─────────────┘  └──────────────────┘  │
│         ↓                                                    │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              LLM Backend (Ollama/Groq)               │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│               context-optimizer-tests:latest                 │
│  ┌──────────┐  ┌────────────┐  ┌──────────────────────┐   │
│  │  pytest  │  │ Benchmarks │  │  Coverage Reports    │   │
│  └──────────┘  └────────────┘  └──────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## Directory Structure (After Reorganization)

```
context-optimizer/
├── docker/
│   ├── Dockerfile              # Production service image
│   ├── Dockerfile.tests        # Test runner image
│   ├── docker-compose.yml      # Orchestration
│   ├── .dockerignore           # Build exclusions
│   └── gateway.py              # FastAPI service template
│
├── src/
│   └── context_optimizer/      # Core package
│       ├── __init__.py
│       ├── cli.py
│       ├── compressor.py       # Compression logic
│       ├── retriever.py        # Dual-storage retrieval
│       └── api/                # API gateway (future)
│
├── pipeline/                   # Data processing
│   ├── domain_corpus_generators.py
│   ├── shared_inputs.py
│   └── quality.py
│
├── benchmarks/                 # All benchmarks
│   ├── text/                   # Text corpus benchmarks
│   ├── tot/                    # Tree-of-Thought benchmarks
│   ├── reasoning/              # Advanced reasoning
│   └── evaluation/             # Visualization tools
│
├── tests/                      # Unit/integration tests
│   ├── test_compressor.py
│   ├── test_retriever.py
│   └── test_integration.py
│
└── docs/                       # Documentation
    └── experiments/            # Experiment reports
```

## Usage Examples

### As a Service (API Gateway)

```bash
# Start service
docker compose -f docker/docker-compose.yml up context-optimizer

# In another terminal, call API
curl -X POST http://localhost:8000/compress \
  -H "Content-Type: application/json" \
  -d '{
    "lines": ["Line 1", "Line 2", "Line 3"],
    "chunk_threshold": 512,
    "max_summary_tokens": 150
  }'

# Retrieve compressed chunks
curl -X POST http://localhost:8000/retrieve \
  -H "Content-Type: application/json" \
  -d '{
    "query": "authentication",
    "top_k": 5
  }'
```

### As a CLI Tool

```bash
# Run compression via CLI
docker run --rm -v $(pwd)/data:/app/data \
  context-optimizer \
  python -m context_optimizer.cli compress \
  --input /app/data/corpus.txt \
  --output /app/results/compressed.json
```

### Running Benchmarks

```bash
# Text compression benchmarks
docker compose -f docker/docker-compose.yml --profile testing run test-runner \
  python benchmarks/text/run_compression_benchmark.py

# ToT multi-perspective benchmarks
docker compose -f docker/docker-compose.yml --profile testing run test-runner \
  python benchmarks/tot/run_fast_tot_benchmarks.py

# All benchmarks
docker compose -f docker/docker-compose.yml --profile testing run test-runner \
  python benchmarks/tot/run_all_tot_benchmarks.py
```

## Environment Variables

Create `.env` file in docker/ directory:

```bash
# LLM Configuration
OLLAMA_BASE_URL=http://host.docker.internal:11434
GROQ_API_KEY=your_api_key_here

# Service Configuration
LOG_LEVEL=INFO
API_PORT=8000

# Compression Defaults
DEFAULT_CHUNK_THRESHOLD=512
DEFAULT_MAX_SUMMARY_TOKENS=150
DEFAULT_CHUNK_OVERLAP=128
```

## Volume Mounts

```yaml
# Data (read-only)
- ./data:/app/data:ro

# Results (read-write)
- ./results:/app/results:rw

# Test outputs
- ./test_results:/app/test_results:rw
- ./benchmark_results:/app/benchmark_results:rw
- ./coverage:/app/coverage:rw
```

## Health Checks

```bash
# Service health
curl http://localhost:8000/health

# Metrics
curl http://localhost:8000/metrics
```

## Development Workflow

### 1. Make Code Changes
Edit files in `src/`, `pipeline/`, or `benchmarks/`

### 2. Rebuild Image
```bash
docker compose -f docker/docker-compose.yml build
```

### 3. Run Tests
```bash
docker compose -f docker/docker-compose.yml --profile testing run test-runner
```

### 4. Deploy
```bash
docker compose -f docker/docker-compose.yml up -d
```

## Production Deployment

### Using Docker Compose

```bash
# Production mode
docker compose -f docker/docker-compose.yml -f docker/docker-compose.prod.yml up -d
```

### Using Kubernetes (future)

```bash
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
```

## Troubleshooting

### Build Failures
```bash
# Check logs
docker compose -f docker/docker-compose.yml logs

# Rebuild from scratch
docker compose -f docker/docker-compose.yml build --no-cache
```

### Test Failures
```bash
# Run tests with verbose output
docker compose -f docker/docker-compose.yml --profile testing run test-runner \
  pytest tests/ -vv

# Check coverage
docker compose -f docker/docker-compose.yml --profile testing run test-runner \
  pytest tests/ --cov=context_optimizer --cov-report=term-missing
```

### Service Not Responding
```bash
# Check container status
docker ps

# View logs
docker logs context-optimizer-service

# Restart service
docker compose -f docker/docker-compose.yml restart context-optimizer
```

## Cleanup

```bash
# Stop containers
docker compose -f docker/docker-compose.yml down

# Remove volumes
docker compose -f docker/docker-compose.yml down -v

# Remove images
docker rmi context-optimizer:latest context-optimizer-tests:latest
```

## Next Steps

1. ✅ Run reorganization script: `.\reorganize.ps1`
2. ✅ Build Docker images
3. ✅ Run tests to verify
4. ⬜ Customize API gateway (`docker/gateway.py`)
5. ⬜ Add semantic caching layer
6. ⬜ Integrate cost tracking
7. ⬜ Add Kubernetes manifests
8. ⬜ Set up CI/CD pipeline
