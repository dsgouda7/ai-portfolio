# Context Optimizer: LLM Context Engineering at Scale

> **Core innovation:** A two-stage context architecture that decouples problem understanding (compression) from evidence gathering (retrieval), reducing token consumption from O(corpus size) to O(1) while improving failure observability.

## Design Scope

The project maintains three core design documents:
- **[docs/design/TECHNICAL_DESIGN.md](docs/design/TECHNICAL_DESIGN.md)** - Implementation architecture, contracts, and engineering decisions
- **[docs/design/COMPRESSION_ARCHITECTURE.md](docs/design/COMPRESSION_ARCHITECTURE.md)** - Rolling window compression pipeline design
- **[docs/whitepaper/proposed-whitepaper.md](docs/whitepaper/proposed-whitepaper.md)** - Hypothesis framing, scientific positioning, and research narrative

---

## Documentation Map

**Read this to understand the full arc — from concept to implementation to evidence:**

| Document | Focus | Audience | Key Takeaway |
|---|---|---|---|
| **[docs/design/TECHNICAL_DESIGN.md](docs/design/TECHNICAL_DESIGN.md)** | How? | Engineers, implementers | System contracts, data model, retrieval path, operations, implementation details |
| **[docs/design/COMPRESSION_ARCHITECTURE.md](docs/design/COMPRESSION_ARCHITECTURE.md)** | How? | Engineers, implementers | Rolling window compression, dual storage, no context exhaustion |
| **[docs/design/ARCHITECTURE_DIAGRAMS.md](docs/design/ARCHITECTURE_DIAGRAMS.md)** | Visuals? | Engineers, architects | Visual architecture, data flow, evolution timeline, deployment topologies |
| **[docs/whitepaper/proposed-whitepaper.md](docs/whitepaper/proposed-whitepaper.md)** | Why / What? | Researchers, technical leads | Hypothesis-driven tri-stage architecture and modality-transfer framing |
| **[docs/benchmarks/experiment_results.md](docs/benchmarks/experiment_results.md)** | Evidence? | Performance engineers, reviewers | Benchmark results across domains with quality metrics |

**Quick navigation**:
- **New to the project?** Start with [docs/design/TECHNICAL_DESIGN.md](docs/design/TECHNICAL_DESIGN.md), then read [docs/design/ARCHITECTURE_DIAGRAMS.md](docs/design/ARCHITECTURE_DIAGRAMS.md)
- **Building it?** Use [docs/design/TECHNICAL_DESIGN.md](docs/design/TECHNICAL_DESIGN.md) and [docs/design/COMPRESSION_ARCHITECTURE.md](docs/design/COMPRESSION_ARCHITECTURE.md) as implementation guides
- **Evaluating it?** Read [docs/benchmarks/experiment_results.md](docs/benchmarks/experiment_results.md) for benchmark results
- **Understanding compression?** See [docs/design/COMPRESSION_ARCHITECTURE.md](docs/design/COMPRESSION_ARCHITECTURE.md) for rolling window design
- **Running tests?** See [tests/](tests/) for unit tests
- **Running benchmarks?** See [benchmarks/](benchmarks/) for benchmark scripts

---

## Problem Statement

**Can we quantify latency and token savings from combining intent compression with on-demand retrieval, instead of sending monolithic context directly to a reasoning model?**

Chat-assistant tasks often mix noisy user requests with large external memory corpora (documents, prior chats, policy text, and social streams). Sending all context up front increases cost and weakens relevance.

This project implements two benchmarkable components:
- **Token Compression Engine (The Edge Filter)**: rewrites raw user prompts into strict structured anchors for downstream reasoning.
- **Semantic MCP Retrieval Layer**: semantically chunks growing context, indexes it in a vector store, and returns ranked evidence packs instead of raw corpus dumps.

The benchmark scripts run two baselines plus the proposed solution:
- **Pipe A (baseline)**: raw prompt + full corpus slice
- **Pipe OOTB**: standard RAG over the same corpus
- **Pipe C (solution)**: compressed prompt + MCP tool-driven semantic retrieval

Pipelines are timed so you can compare behavior, throughput, and payload efficiency in a repeatable way across incident and chat-assistant domains.

## Architecture: MCP Pull Context Pipeline

```
User Input (rambling)
    ↓
[STAGE 1: Compression Engine]
  LLM extracts: core_issue, symptoms, technical_identifiers
  Output: 412-char Pydantic schema (99.8% reduction)
    ↓
[STAGE 2: Semantic Chunking + Vector Index]
  Chunk by paragraph/log windows → preserve boundaries → embed → store with metadata
  Output: searchable evidence store with stable retrieval contract + continuation hints
    ↓
[STAGE 3: MCP Retrieval + Reasoning]
  LLM issues retrieve_context(query, depth, service, severity)
  Input: structured shell + ranked evidence pack
  Output: Diagnosis
```

**Why this matters:**
- **Token cost is O(1)**, not O(corpus size)—constant even at 100K logs
- **Failures are observable**—compression validates schema, MCP retrieval returns scored evidence and explicit empty-set responses
- **Stages are decoupled**—optimize compression independently from retrieval
- **Inversion principle**—use cheap operation (compression) to optimize expensive operation (reasoning)
- **Tool-aware reasoning**—the model is explicitly taught how retrieval works, what scores mean, and when to refine its query
- **Boundary-preserving storage**—stored chunks retain original span metadata and signal when adjacent context may be required

See [docs/design/ARCHITECTURE_DIAGRAMS.md](docs/design/ARCHITECTURE_DIAGRAMS.md) for visual architecture documentation.

---

## Production Deployment Options

This project now offers **three deployment modes**:

### 1. Python Library (Core Engine)
Direct integration into Python applications:
```python
from context_optimizer.compressor import compress_corpus_rolling
from context_optimizer.retriever import DualStorageRetriever

compressed = compress_corpus_rolling(corpus_lines)
retriever = DualStorageRetriever(compressed)
```
**Best for:** Research, experimentation, direct integration

### 2. LiteLLM Wrapper Package (Multi-Provider Support)
Pip-installable package with automatic compression for 100+ LLM providers:
```bash
cd ai-gateway/wrapper
pip install -e .
```
```python
from context_optimizer_gateway import CompressedLiteLLM

client = CompressedLiteLLM()
response = client.completion(model="gpt-4", messages=[...])
```
**Best for:** Python apps needing multi-provider support with compression

### 3. Docker AI Gateway (Production Service)
OpenAI-compatible REST API with Redis caching:
```bash
cd ai-gateway/service
docker compose up -d

curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-4","messages":[...]}'
```
**Best for:** Microservices, multi-language clients, team deployments

See [ai-gateway/README.md](ai-gateway/README.md) for complete LiteLLM integration documentation.

---

## Production Metrics (Updated)

**Quality Achievement:** ✅ 0.83 F1 average (all 7 domains >0.80)
**Token Reduction:** ✅ 97.8% (was 99.9% before quality optimization)
**ROI:** ✅ 51.6x average (break-even at 2.4 queries)
**Production Status:** ✅ Ready for deployment

### Quality by Domain (F1 Scores)

| Domain | F1 Score | Status |
|--------|----------|--------|
| **Log Analysis** | 0.86 | ✅ Production Ready (113.8x ROI) |
| **Support Tickets** | 0.85 | ✅ Production Ready (60.4x ROI) |
| **Code Search** | 0.84 | ✅ Production Ready (30.2x ROI) |
| **Research Papers** | 0.84 | ✅ Production Ready (45.3x ROI) |
| **Clinical Notes** | 0.82 | ✅ Production Ready (30.8x ROI) |
| **Multilingual Docs** | 0.81 | ✅ Production Ready (20.1x ROI) |
| **Legal Discovery** | 0.80 | ✅ Production Ready (60.4x ROI) |

All domains exceed the 0.80 F1 production threshold.

### Cost Savings Example (GPT-4)

```
Without Compression:  $0.37 per query (50KB context)
With Compression:     $0.007 per query (97.8% reduction)
Savings:              $0.363 (98.1% cost reduction)

1,000 queries/day:    $370 → $7 ($363/day savings)
Annual savings:       $135,000+
```

---

## Project Structure (Updated)

```
context-optimizer/
├── src/context_optimizer/     # Core compression engine
│   ├── compressor.py          # Rolling window compression
│   ├── raw_index.py           # SQLite+FTS5 raw content store (NEW)
│   ├── cached_retriever.py    # Semantic cache + ChromaDB retrieval
│   ├── index.py               # CorpusIndex — high-level public API
│   ├── tot_reasoner.py        # Tree-of-Thought multi-branch reasoner
│   ├── benchmark.py           # Comparison harness and metrics
│   ├── protocols.py           # Retriever protocol definition
│   └── __init__.py
│
├── pipeline/                  # Data processing utilities
│   ├── domain_corpus_generators.py
│   ├── quality.py
│   └── shared_inputs.py
│
├── benchmarks/                # Test suites (reorganized)
│   ├── text/                  # Text compression tests
│   ├── tot/                   # Tree-of-Thought tests
│   ├── reasoning/             # Advanced reasoning tests
│   └── evaluation/            # Quality evaluation tools
│
├── ai-gateway/                # LiteLLM integration (NEW)
│   ├── wrapper/              # Pip-installable package
│   │   └── context_optimizer_gateway/
│   │       ├── litellm_wrapper.py (359 lines)
│   │       ├── middleware.py
│   │       └── cache.py
│   └── service/              # Docker-deployable gateway
│       ├── gateway_service.py (324 lines)
│       ├── Dockerfile
│       └── docker-compose.yml
│
├── docs/                      # Documentation
│   ├── design/               # Technical architecture
│   │   ├── TECHNICAL_DESIGN.md
│   │   ├── COMPRESSION_ARCHITECTURE.md
│   │   └── ARCHITECTURE_DIAGRAMS.md (NEW)
│   ├── experiments/          # Benchmark results
│   └── whitepaper/           # Research paper
│
└── experiments/               # Historical documentation
    └── README.md             # Experiments guide
```

**Key Changes:**
- ✅ Reorganized benchmarks by category (text, tot, reasoning, evaluation)
- ✅ Added LiteLLM gateway integration (wrapper + service)
- ✅ Consolidated documentation (temp files removed)
- ✅ Added architecture diagrams document

---

> **Engineering benchmark project** - this is designed as a practical harness for comparing context optimization strategies, not a toy notebook.
>
> **What is included:** provider abstraction for Ollama vs Groq, strict schema output for compression, deterministic corpora for incident + chat-assistant suites, LangChain tool wiring, and run-level telemetry.
>
> **What is intentionally mocked:** no external observability backend is required; the cache is in-process to isolate prompt strategy effects.

## Quick start

```powershell
# 1. create and activate a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. install the package
pip install -e .[evaluation]

# 3a. run with local Ollama models (default)
context-optimizer --provider ollama --small-model phi4:mini --reasoning-model qwen3

# 3b. run with Groq models
$env:GROQ_API_KEY = "<your_key_here>"
context-optimizer --provider groq --small-model llama-3.1-8b-instant --reasoning-model llama-3.3-70b-versatile

# 3c. run benchmarks against Azure OpenAI (20-30x faster than Ollama)
# See deployments/azure/ for Azure Key Vault–backed setup
cd benchmarks/tot
python run_benchmarks.py

# 3d. module execution
python -m context_optimizer --provider ollama --pipeline both
```

Deploy locally (host Ollama) or to Azure via [deployments/](deployments/) — `bash deployments/deploy.sh local up` or `deployments\deploy.ps1 azure up`.

## Package and deployment

This project is now installable as a Python package via [pyproject.toml](pyproject.toml).

Primary package surfaces:
- console script: `context-optimizer`
- module entry point: `python -m context_optimizer`
- import namespace: `context_optimizer`

## Setup scripts (dependencies + model pulls)

```powershell
# installs Python dependencies only (default lightweight setup)
./setup.ps1

# install deps and pull default Ollama models (phi4:mini, qwen3)
./setup.ps1 -EnableModelPull
```

```bash
# installs Python dependencies only (default lightweight setup)
chmod +x setup.sh
./setup.sh

# install deps and pull default Ollama models
./setup.sh --enable-model-pull
```

Both scripts install:
- the package itself in editable mode via `pip install -e .[evaluation]`

And optionally pull Ollama models to support local CPU inference benchmarks.

## Tests

112 tests across all modules. Run with pytest:

```powershell
# Run full suite
.venv\Scripts\python.exe -m pytest tests/ -q

# Run with coverage
.venv\Scripts\python.exe -m pytest tests/ --cov=context_optimizer --cov-report=term-missing
```

Test modules:
- `test_raw_index.py` — SQLite+FTS5 raw content store (17 tests)
- `test_compressor.py` — rolling window compression pipeline
- `test_cached_retriever.py` — semantic cache and ChromaDB retrieval
- `test_corpus_index.py` — high-level `CorpusIndex` API
- `test_tot_reasoner.py` — Tree-of-Thought multi-branch reasoning
- `test_benchmark.py` — benchmark utilities and comparison harness
- `test_protocols.py` — protocol/interface conformance

## Large-Corpus Parallel Benchmarks

Run parallel dataset tracks (Gutenberg text + large XLSX analytics corpus) to stress test token behavior at higher scale.

```powershell
# installs openpyxl dependency used by XLSX generation/reading
pip install -r requirements.txt

# run both tracks in parallel; appends results to docs/benchmarks/experiment_results.md
.\.venv\Scripts\python.exe experiments/run_large_corpus_benchmarks.py --target-mb 120

# heavier run (few hundred MB target per track)
.\.venv\Scripts\python.exe experiments/run_large_corpus_benchmarks.py --target-mb 300
```

Generated artifacts:
- `data/large_corpus/gutenberg/combined_gutenberg.txt`
- `data/large_corpus/excel/mock_<target>mb.xlsx`
- report section appended to `docs/benchmarks/experiment_results.md`

## CPU-only Docker benchmark (raw vs optimized)

```powershell
# build images
docker build -f Dockerfile.raw -t context-optimizer-raw:cpu .
docker build -f Dockerfile.optimized -t context-optimizer-optimized:cpu .

# run the full evaluation harness (downloads/assembles logs, runs both containers, writes report)
./run_evaluation.ps1
```

```bash
# Linux/macOS
chmod +x run_evaluation.sh
./run_evaluation.sh
```

Evaluation outputs are written to `evaluation/out/`:
- `raw_metrics.json`
- `optimized_metrics.json`
- `metrics_animation.gif`
- `architecture_differences.gif`
- `evaluation_report.md`

## Environment and model routing

| Provider | Class used | Required env vars | Example models |
|---|---|---|---|
| Ollama | `ChatOllama` | Optional: `OLLAMA_BASE_URL` (defaults to `http://localhost:11434`) | `phi4:mini`, `qwen3` |
| Groq | `ChatGroq` | `GROQ_API_KEY` | `llama-3.1-8b-instant`, `llama-3.3-70b-versatile` |

Model defaults are provider-aware and can be overridden via CLI flags or env vars (`SMALL_MODEL`, `REASONING_MODEL`).

## Dual-Storage Architecture (RawIndex + ChromaDB)

Version 2 of the retrieval layer adds a second storage tier alongside ChromaDB:

| Store | What is stored | Lookup type | Typical latency |
|-------|---------------|-------------|-----------------|
| **ChromaDB** | Compressed summaries (LLM-generated) | Vector similarity (semantic) | ~5–50 ms |
| **RawIndex** (SQLite+FTS5) | Original un-truncated chunk text | Exact chunk ID lookup + BM25 keyword search | ~0.1 ms (ID), ~1 ms (FTS5) |

**Why two stores?** ChromaDB is optimised for *semantic* search over summaries. RawIndex fills the complementary roles:
- **O(1) exact lookup** when the chunk ID is already known (e.g., `get_chunk_by_id`)
- **Keyword search** over the original text without touching the embedding layer
- **Thread-safe parallel writes** during ingestion — SQLite WAL mode lets background threads write raw chunks (~1 ms) while the main thread blocks on LLM compression (~500 ms), giving essentially free parallelism

```python
from context_optimizer import CorpusIndex, RawIndex

idx = CorpusIndex()
stats = idx.ingest(lines, collection="my_corpus")

# Fast O(1) lookup by chunk ID
text = idx.raw_lookup("chunk::0000", collection="my_corpus")

# BM25 keyword search over original text
hits = idx.raw_search("machine learning", collection="my_corpus", top_k=5)
for hit in hits:
    print(hit.chunk_id, hit.rank, hit.raw_text[:80])
```

---

## Benchmark Setup & Results

Benchmarks run entirely locally — no external APIs, no GPU required.

**Hardware:**

| Component | Spec |
|-----------|------|
| CPU | AMD64 Family 25 Model 1 (Zen 3), 8 physical / 16 logical cores |
| RAM | 64 GB |
| GPU | None (CPU-only inference) |
| OS | Windows 10 |

**Software:**

| Component | Version / Model |
|-----------|-----------------|
| Python | 3.11.9 |
| LLM (compression) | `llama3.2:3b` via Ollama |
| LLM (reasoning) | `qwen2.5-coder:7b` via Ollama |
| Embeddings | `all-MiniLM-L6-v2` via sentence-transformers |

**Reproduce the full run:**

```powershell
# 1. Start Ollama and pull required models
ollama pull llama3.2:3b
ollama pull qwen2.5-coder:7b
ollama pull mistral:7b

# 2. Run default 500-line quick benchmark
cd benchmarks
python run_experiments.py

# 3. Run full 11K-line corpus benchmark (takes ~1 hour on CPU)
python run_experiments.py --full
```

Results are written to:
- `benchmarks/EXPERIMENT_RESULTS.json` — machine-readable metrics
- `docs/benchmarks/experiment_results.md` — human-readable report

**Latest results** (11,574-line corpus, see [docs/benchmarks/experiment_results.md](docs/benchmarks/experiment_results.md)):

| Metric | Baseline (raw corpus) | Compressed Architecture | Threshold |
|--------|----------------------|------------------------|-----------|
| Avg prompt tokens | 180,734 | 15,798 | — |
| **Token reduction** | — | **91.3%** | ≥90% ✅ |
| Avg reasoning latency (s) | 155.9 | 79.5 | ≤+10% ✅ |
| Avg judge score (0–1) | 0.97 | 0.97 | ≤-20% ✅ |
| Avg KW-F1 | 0.068 | 0.160 | ≤-20% ✅ |

All four production thresholds pass. Compression was a one-time 2,799 s cost; subsequent queries pay only the retrieval + reasoning latency.

---

## Architecture

### Component 1: Token Compression Engine (Edge Filter)

- Uses the exact system prompt specified in the design request.
- Enforces a strict structured output schema:
  - `core_issue: str`
  - `observed_symptoms: list[str]`
  - `technical_identifiers: list[str]`
- Primary path: `with_structured_output(PydanticModel)`
- Fallback path: explicit `PydanticOutputParser`

### Component 2: Semantic MCP retrieval tool

- Builds a deterministic mock log cache with 1,050 lines.
- Semantically chunks the corpus and indexes chunks in a vector backend.
- Preserves original chunk boundaries and stores `prev_chunk_id`, `next_chunk_id`, and continuation flags.
- Uses Chroma for local persistence when available, with an in-memory fallback for deterministic mock runs.
- Injects realistic patterns:
  - CosmosDB timeout events (`substatus=21012`)
  - AKS ingress warnings (`upstream timed out while reading response header`)
  - stack traces (`CosmosClient.ReadItemAsync`, `PaymentConnector.SubmitAsync`)
- Exposes a tool-oriented retrieval contract:
  - `retrieve_context(query: str, depth: str = "brief", service: str | None = None, severity: str | None = None)`
  - Returned chunks include boundary metadata (`boundary_preserved`, `needs_prev_chunk`, `needs_next_chunk`) so the reasoning model can detect truncated local evidence.

## Telemetry emitted each run

| Metric | Description |
|---|---|
| `raw_char_count` | Character count of original user incident prompt |
| `compressed_char_count` | Character count of compressed structured payload |
| `char_savings` | Absolute and percentage reduction |
| `compression_latency_s` | Compression step latency using `time.perf_counter()` |
| `pipe_a_reasoning_s` | Baseline reasoning latency |
| `pipe_c_reasoning_s` | Optimized reasoning latency (includes MCP tool calls) |
| `pipe_c_tool_calls` | Number of retrieval calls made by the optimized pipeline |

## How to compare results

1. Run the script with the same provider/model pair for both pipelines.
2. Inspect the final telemetry block.
3. Validate:
   - Compression savings are meaningful (raw vs compressed chars).
  - Pipe C reaches similar or better diagnosis quality while consuming bounded retrieved context.
   - Tool calls are focused (few, high-signal retrievals).

## Experiment Reports

- `docs/benchmarks/experiment_results.md` is the canonical experiment report.
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
