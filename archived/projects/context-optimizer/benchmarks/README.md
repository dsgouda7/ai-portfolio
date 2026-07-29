# Book Benchmark

Evaluates the full context-optimizer pipeline on Gutenberg public-domain books
using Wikipedia-sourced Q&A pairs as ground truth.

---

## Quick-start

```bash
# From the project root:
cd projects/context-optimizer

# 1. Build Wikipedia question banks once (downloads Q&A from Wikipedia)
python benchmarks/book_benchmark.py build-banks --books 25 --qpb 20

# 2. Run extractive benchmark (no LLM required, fast)
python benchmarks/book_benchmark.py run \
  --books 25 --lines 3000 --strategy extractive

# 3. Run with LLM compressor (Ollama must be running)
python benchmarks/book_benchmark.py run \
  --books 25 --lines 3000 --strategy llm

# 4. Run with Azure compressor + Azure reasoner (see provider config below)
python benchmarks/book_benchmark.py run \
  --books 25 --lines 3000 --strategy llm \
  --compressor-provider azure --compressor-model gpt-4o-mini \
  --reasoner-provider azure --reasoner-model gpt-4o
```

---

## Subcommands

| Command | Description |
|---|---|
| `build-banks` | Fetch Wikipedia Q&A banks (one-time setup, results cached to `data/question_banks/`) |
| `chunk-banks` | Build Q&A banks from cached compressed chunks (no internet needed) |
| `run` | Run compression + retrieval + Q&A scoring |
| `all` | `build-banks` then `run` in one shot |

### Key flags (all subcommands that support them)

| Flag | Default | Description |
|---|---|---|
| `--books N` | `25` | Number of Gutenberg books |
| `--lines N` | `3000` | Lines of text per book (0 = unlimited) |
| `--qpb N` | `20 / 100` | Questions per book |
| `--strategy` | `llm` | `llm`, `extractive`, or `raw_only` |
| `--force` | off | Ignore all caches, rerun from scratch |

---

## Two-model pipeline

The benchmark supports a **cheap compressor** + **heavy reasoner** split:

| Role | Flag | Env override | Purpose |
|---|---|---|---|
| Compressor | `--compressor-provider` / `--compressor-model` | `CONTEXT_OPTIMIZER_COMPRESSOR_PROVIDER` / `CONTEXT_OPTIMIZER_COMPRESSOR_MODEL` | Summarises raw chunks into dense representations |
| Reasoner | `--reasoner-provider` / `--reasoner-model` | `CONTEXT_OPTIMIZER_REASONER_PROVIDER` / `CONTEXT_OPTIMIZER_REASONER_MODEL` | Synthesises a natural-language answer from retrieved evidence |

When no `--reasoner-provider` is given, the reasoner is off and the benchmark
aggregates the top-6 evidence snippets directly (the default and fastest path).

---

## Provider configuration

### Ollama (default, local, free)

```bash
# Start Ollama before running
ollama serve
ollama pull llama3.2:3b          # compressor
# No extra env vars needed
python benchmarks/book_benchmark.py run --strategy llm
```

### Groq (fast cloud inference, free tier)

```bash
export CONTEXT_OPTIMIZER_COMPRESSOR_PROVIDER=groq
export GROQ_API_KEY=<your-key>
python benchmarks/book_benchmark.py run --strategy llm --compressor-provider groq --compressor-model llama3-8b-8192
```

### Azure OpenAI

```bash
export AZURE_OPENAI_ENDPOINT=https://<resource>.openai.azure.com/
export AZURE_OPENAI_API_KEY=<your-key>
export AZURE_OPENAI_API_VERSION=2024-02-01          # optional, default shown

# Compressor deployment (cheap, fast)
export AZURE_COMPRESSOR_DEPLOYMENT=gpt-4o-mini      # optional, default shown

# Reasoner deployment (heavier, more accurate)
export AZURE_REASONER_DEPLOYMENT=gpt-4o             # optional, default shown

python benchmarks/book_benchmark.py run \
  --strategy llm \
  --compressor-provider azure \
  --reasoner-provider azure
```

---

## Strategy comparison (25 books, 3000 lines, 20 Q/book)

| Strategy | Avg Recall | Avg F1 | Token savings |
|---|---|---|---|
| `extractive` | 0.351 | 0.231 | **68.6 %** (1.21M → 382K) |
| `raw_only` | 0.216 | 0.119 | 0 % |

Extractive compression consistently beats raw FTS5 on both accuracy **and** cost.

---

## Data layout

```
benchmarks/
  book_benchmark.py      ← main runner
  show_f1.py             ← P / R / F1 per strategy (side-by-side)
  show_compression.py    ← token reduction analysis
  watch_book_results.py  ← live tail of in-progress results
  BOOK_RESULTS_*.json    ← machine-readable results per strategy

data/                    ← generated, git-ignored
  book_cache/            ← raw downloaded book text
  chunks/                ← compressed chunk JSONL caches
  question_banks/        ← Wikipedia Q&A banks (JSON)
```

---

## Running the analysis scripts

```bash
# Side-by-side F1 comparison of two strategies
python benchmarks/show_f1.py extractive raw_only

# Token-reduction breakdown
python benchmarks/show_compression.py BOOK_RESULTS_extractive.json

# Live progress watcher (while a run is in flight)
python benchmarks/watch_book_results.py
```
