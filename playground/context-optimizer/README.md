# Context Optimizer: LLM Context Engineering at Scale

> **Core innovation:** A two-stage context architecture that decouples problem understanding (compression) from evidence gathering (retrieval), reducing token consumption from O(corpus size) to O(1) while improving failure observability.

## Design Sophistication

This is not "compression + retrieval" as a tactic—it's **context engineering as an architectural principle**. See [DESIGN_SOPHISTICATION.md](DESIGN_SOPHISTICATION.md) for:
- **The inversion principle:** Use cheap compression to constrain expensive reasoning
- **Failure mode cascade analysis:** How compression quality propagates to retrieval and diagnosis
- **Tradeoff matrix:** Cost vs latency vs debuggability vs adaptability
- **Scalability proof:** Token cost stays constant as corpus grows 100x (1K → 100K logs)

## Problem Statement

**Can we quantify how much latency and prompt bloat we remove by combining input compression with on-demand log retrieval instead of sending raw incident text plus full logs to a reasoning model?**

Incident triage prompts in production are usually verbose and emotional. They include useful identifiers (IPs, service names, metrics, error codes), but those details are mixed with non-technical narration. At the same time, teams often pass entire log files to LLMs up front, which drives token cost and noise.

This project implements two benchmarkable components:
- **Token Compression Engine (The Edge Filter)**: rewrites raw user incident reports into a strict structured payload for downstream reasoning.
- **Mock Logs In-Memory Cache Tool**: lets an agent fetch only relevant log fragments (`query_log_cache`) instead of ingesting the full log corpus.

The script runs two pipelines side by side:
- **Pipe A (baseline)**: raw prompt + full logs
- **Pipe B (optimized)**: compressed prompt + dynamic tool-based log retrieval

Both pipelines are timed so you can compare behavior, throughput, and payload efficiency in a repeatable way.

## Architecture: Two-Stage Context Pipeline

```
User Input (rambling)
    ↓
[STAGE 1: Compression Engine]
  LLM extracts: core_issue, symptoms, technical_identifiers
  Output: 412-char Pydantic schema (99.8% reduction)
    ↓
[STAGE 2: Targeted Retrieval]
  Extract keywords → Query log corpus → Return context-windowed results
  Output: 64-82 relevant log lines (93-99.9% reduction)
    ↓
[STAGE 3: Reasoning]
  LLM processes compressed schema + curated logs
  Input: 1.4K-1.7K tokens (vs 44K raw)
  Output: Diagnosis
```

**Why this matters:**
- **Token cost is O(1)**, not O(corpus size)—constant even at 100K logs
- **Failures are observable**—compression validates schema, retrieval shows no matches
- **Stages are decoupled**—optimize compression independently from retrieval
- **Inversion principle**—use cheap operation (compression) to optimize expensive operation (reasoning)

See [ARCHITECTURE_DIAGRAMS.txt](ARCHITECTURE_DIAGRAMS.txt) for visual comparisons with monolithic approaches.

> **Engineering benchmark project** - this is designed as a practical harness for comparing context optimization strategies, not a toy notebook.
>
> **What is included:** provider abstraction for Ollama vs Groq, strict schema output for compression, realistic in-memory log corpus (~1,000 lines), LangChain tool wiring, and run-level telemetry.
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

### Component 2: In-memory log retrieval tool

- Builds a deterministic mock log cache with 1,050 lines.
- Injects realistic patterns:
  - CosmosDB timeout events (`substatus=21012`)
  - AKS ingress warnings (`upstream timed out while reading response header`)
  - stack traces (`CosmosClient.ReadItemAsync`, `PaymentConnector.SubmitAsync`)
- Exposes a LangChain standard tool:
  - `query_log_cache(keyword: str, lines_context: int = 5)`

## Telemetry emitted each run

| Metric | Description |
|---|---|
| `raw_char_count` | Character count of original user incident prompt |
| `compressed_char_count` | Character count of compressed structured payload |
| `char_savings` | Absolute and percentage reduction |
| `compression_latency_s` | Compression step latency using `time.perf_counter()` |
| `pipe_a_reasoning_s` | Baseline reasoning latency |
| `pipe_b_reasoning_s` | Optimized reasoning latency (includes tool calls) |
| `pipe_b_tool_calls` | Number of retrieval calls made by the optimized pipeline |

## How to compare results

1. Run the script with the same provider/model pair for both pipelines.
2. Inspect the final telemetry block.
3. Validate:
   - Compression savings are meaningful (raw vs compressed chars).
   - Pipe B reaches similar or better diagnosis quality while consuming less upfront context.
   - Tool calls are focused (few, high-signal retrievals).

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
