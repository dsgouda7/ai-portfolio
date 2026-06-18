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
| **[docs/whitepaper/proposed-whitepaper.md](docs/whitepaper/proposed-whitepaper.md)** | Why / What? | Researchers, technical leads | Hypothesis-driven tri-stage architecture and modality-transfer framing |
| **[docs/experiments/EXPERIMENTS_CONSOLIDATED.md](docs/experiments/EXPERIMENTS_CONSOLIDATED.md)** | Evidence? | Performance engineers, reviewers | Chat-assistant benchmarks, latency measurements (10-17x speedup) |
| **[experiments/EXPERIMENTS_GUIDE.md](experiments/EXPERIMENTS_GUIDE.md)** | Results? | Engineers, reviewers | GB-scale compression validation, architecture diagrams, performance tables |
| **[experiments/README.md](experiments/README.md)** | What's tested? | Developers, QA | Quick start guide to running experiments |

**Quick navigation**:
- **New to the project?** Start with [docs/design/TECHNICAL_DESIGN.md](docs/design/TECHNICAL_DESIGN.md), then read [docs/whitepaper/proposed-whitepaper.md](docs/whitepaper/proposed-whitepaper.md)
- **Building it?** Use [docs/design/TECHNICAL_DESIGN.md](docs/design/TECHNICAL_DESIGN.md) and [docs/design/COMPRESSION_ARCHITECTURE.md](docs/design/COMPRESSION_ARCHITECTURE.md) as implementation guides
- **Evaluating it?** Read [experiments/EXPERIMENTS_GUIDE.md](experiments/EXPERIMENTS_GUIDE.md) for GB-scale results and [docs/experiments/EXPERIMENTS_CONSOLIDATED.md](docs/experiments/EXPERIMENTS_CONSOLIDATED.md) for chat-assistant benchmarks
- **Understanding compression?** See [docs/design/COMPRESSION_ARCHITECTURE.md](docs/design/COMPRESSION_ARCHITECTURE.md) for rolling window design
- **Running tests?** See [experiments/README.md](experiments/README.md) for quick start

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

See [docs/design/ARCHITECTURE_DIAGRAMS.txt](docs/design/ARCHITECTURE_DIAGRAMS.txt) for visual comparisons with monolithic approaches.

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

# 3c. module execution also works
python -m context_optimizer --provider mock --pipeline both
```

## Package and deployment

This project is now installable as a Python package via [pyproject.toml](c:/repos/ai-portfolio/playground/context-optimizer/pyproject.toml).

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

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## Large-Corpus Parallel Benchmarks

Run parallel dataset tracks (Gutenberg text + large XLSX analytics corpus) to stress test token behavior at higher scale.

```powershell
# installs openpyxl dependency used by XLSX generation/reading
pip install -r requirements.txt

# run both tracks in parallel; appends results to docs/experiments/EXPERIMENTS_CONSOLIDATED.md
.\.venv\Scripts\python.exe experiments/run_large_corpus_benchmarks.py --target-mb 120

# heavier run (few hundred MB target per track)
.\.venv\Scripts\python.exe experiments/run_large_corpus_benchmarks.py --target-mb 300
```

Generated artifacts:
- `data/large_corpus/gutenberg/combined_gutenberg.txt`
- `data/large_corpus/excel/mock_<target>mb.xlsx`
- report section appended to `docs/experiments/EXPERIMENTS_CONSOLIDATED.md`

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

- `docs/experiments/EXPERIMENTS_CONSOLIDATED.md` is the single canonical experiment report.
- `experiments/run_all_experiments.py` updates the incident benchmark appendix section inside `docs/experiments/EXPERIMENTS_CONSOLIDATED.md`.

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
