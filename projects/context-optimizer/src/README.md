# Context Optimizer — Source Module Reference

This README covers the Python packages under `src/`, how they fit together,
and how every `bench_config.yaml` flag maps to behaviour.

---

## Quickstart

```bash
# 1. Edit benchmarks/bench_config.yaml — set corpus_path at minimum
# 2. Run everything:
python benchmarks/corpus_benchmark.py run
```

That is the entire command. The config file is auto-loaded from `benchmarks/`.
No CLI flags are required.

---

## Bare-minimum config

```yaml
compressor:
  provider: hf          # or ollama if you have Ollama running

benchmark:
  corpus_path: path/to/your/corpus.txt
```

Every other key has a sensible default and can be omitted.

---

## Config file reference (`bench_config.yaml`)

### `compressor` section

Controls how the corpus is turned into compressed block summaries.

| Key | Values | Default | What it does |
|---|---|---|---|
| `provider` | `hf` / `ollama` / `azure_foundry` | `hf` | Which LLM backend to use for compression |
| `hf.model` | HuggingFace Hub model ID | `facebook/bart-large-cnn` | BART/T5 model (5-15x faster than Ollama on CPU) |
| `hf.device` | `-1` / `0` / `1`... | `-1` (CPU) | `-1` = CPU, `0` = first GPU |
| `ollama.model` | Ollama model name | `qwen2.5:3b` | Model for prose, docs, spreadsheets, markup |
| `ollama.code_model` | Ollama model name | `qwen2.5-coder:3b` | Model for source code (same size, better quality) |
| `ollama.base_url` | URL | `http://localhost:11434` | Ollama server address |
| `azure_foundry.endpoint` | URL | — | Azure AI Foundry project endpoint |
| `azure_foundry.model` | Model name | `phi-4-mini` | Model for prose/docs (set `AZURE_AI_FOUNDRY_API_KEY` env var) |

**Which provider to choose:**

| Provider | Best for | Speed (CPU) | Cost |
|---|---|---|---|
| `hf` | Text-only corpora, no GPU, offline | ~12 s/block | Free |
| `ollama` | Mixed content (docs + code), better quality | ~25 s/block | Free |
| `azure_foundry` | Production, <1 s/block, no local GPU | <1 s/block | ~$0.0003–0.001/1K tokens |

---

### `reasoning` section

The LLM that navigates the tree index at query time and synthesizes answers.
Set `model: ""` to disable reasoning (retrieval metrics only, no Ollama needed).

| Key | Values | Default | What it does |
|---|---|---|---|
| `reasoning.ollama.model` | Ollama model name | `mistral:7b` | Reasoning model (larger = better answers) |
| `reasoning.ollama.base_url` | URL | `http://localhost:11434` | Ollama server address |

---

### `benchmark` section

#### Paths

| Key | Default | What it does |
|---|---|---|
| `corpus_path` | — | **Required.** Path to the text file to index. Relative paths resolve from the project root (`context-optimizer/`). |
| `index_dir` | temp dir | Where the built index is saved (ChromaDB + SQLite). Omit = temp dir, lost on restart. Set this to avoid re-building on every run. |

#### Block / tree parameters

| Key | Default | What it does |
|---|---|---|
| `block_mb` | `0.5` | Corpus block size in MB. Smaller = more precise summaries but more LLM calls. `0.5` is good for text; `2.0` is usable with tree navigation. |
| `cluster_size` | `4` | Children per tree node. **The primary accuracy knob.** Lower = more specific summaries = better recall. 4 is recommended; try 2–3 for max accuracy. |
| `overlap_pct` | `10.0` | % of each block prepended to the next as context. Prevents boundary concepts from being missed. |
| `tree_depth` | `0` (auto) | Number of summary levels. `0` = compute automatically after Pass 1 using `depth = ceil(log(n/k) / log(k)) + 1`. `2` = L1+L2 only, `3` = L1+L2+L3. |

#### Evaluation parameters

| Key | Default | What it does |
|---|---|---|
| `questions` | `20` | Number of eval questions to run (from the question bank). |
| `top_k` | `5` | Chunks/blocks retrieved per query. |
| `fallback_threshold` | `0.30` | Cosine distance threshold for raw-block fallback. If best match score < threshold the full raw block is fetched. `0.0` = never fall back, `1.0` = always. |
| `opt_strategy` | `llm` | `llm` = BART/Ollama compresses blocks. `raw_only` = store first 800 chars verbatim (instant build, worse recall). |

#### Run mode flags

| Key | Default | What it does |
|---|---|---|
| `tree` | `true` | Build and evaluate Tree-of-Summaries. Set `false` to skip. |
| `optimized_only` | `true` | Skip vanilla RAG (raw 512-token chunks). Vanilla takes ~80 min at 400 MB; only disable when you need a fresh baseline. |
| `vanilla_only` | `false` | Run only vanilla RAG; skip optimized and tree. |
| `build_only` | `false` | Build index then exit. Pair with `eval_only: true` on the next run. |
| `eval_only` | `false` | Skip building; load existing index from `index_dir` and run eval only. |
| `force_fallback` | `false` | Always fetch raw blocks (diagnostic flag — measures BlockIndex recall gain). |
| `max_mb` | `0` | Cap corpus at N MB before indexing. `0` = use the full file. |

---

## Module map

```
src/
  compressor.py         ingest_file_blocks · _build_local_llm · rolling-window compression
  tree_index.py         TreeIndex (N-level) · _auto_tree_depth · build_from_chunks · expand_cluster
  tree_reasoner.py      TreeReasoningAgent · tool-calling loop (search_cluster / fetch_raw_block)
  raw_index.py          BlockIndex (SQLite) · byte-offset file pointers · get_text()
  cached_retriever.py   CachedChromaRetriever · semantic cache · parent-child retrieval
  ingest_corpus.py      ingest_directory() · parallel file extraction · per-task model routing
  extractors/
    __init__.py         FormatRouter · file-type → task name mapping
    txt.py              TxtExtractor · RtfExtractor
    markdown.py         MarkdownExtractor (strips front-matter, fences, links)
    pdf.py              PdfExtractor (pdfminer.six, text-layer PDFs only)
    docx.py             DocxExtractor (python-docx, paragraphs + tables)
    xlsx.py             XlsxExtractor (openpyxl, serializes rows to prose)
    xml_extractor.py    XmlExtractor (lxml, strips tags)
  providers/
    hf_summarizer.py    HFSummarizerLLM — BART/T5 via transformers 5.x
    azure_foundry.py    AzureFoundryLLM — azure-ai-inference SDK
```

### Key data flows

**Ingestion (write-time, runs once):**
```
corpus file
  → ingest_file_blocks()       # splits into blocks, calls compressor
    → _build_local_llm()       # builds BART / Ollama / Azure client
      → LLM.invoke(block_text) # generates ~80-token summary per block
  → BlockIndex.add()           # stores byte offsets (no duplication)
  → ChromaDB L1                # embeds summaries for semantic search
  → TreeIndex.build_from_chunks()
      → L2 clusters (Pass 2)   # same LLM, cluster_size L1 summaries → 1 super-summary
      → L3, L4... (Pass 3+)    # depth determined from actual L1 count
```

**Query (read-time, every request):**
```
user query
  → TreeIndex.search()         # cosine search from top level down
  → TreeReasoningAgent.reason()
      → LLM decides: search_cluster | fetch_raw_block | answer
      → BlockIndex.get_text()  # on-demand raw text (single fseek)
  → synthesized answer + file citations
```

---

## Environment variables

These are set automatically from `bench_config.yaml` — you do not need to set them manually unless running outside the benchmark script.

| Variable | Set by | What it controls |
|---|---|---|
| `CONTEXT_OPTIMIZER_COMPRESSOR_PROVIDER` | `compressor.provider` | Active compressor backend |
| `CONTEXT_OPTIMIZER_COMPRESSOR_MODEL` | `compressor.<provider>.model` | Default compressor model |
| `CONTEXT_OPTIMIZER_CODE_MODEL` | `compressor.<provider>.code_model` | Code-specific compressor model |
| `OLLAMA_BASE_URL` | `compressor.ollama.base_url` | Ollama server URL |
| `AZURE_AI_FOUNDRY_ENDPOINT` | `compressor.azure_foundry.endpoint` | Azure endpoint |
| `AZURE_AI_FOUNDRY_API_KEY` | **manual** — never put in config | Azure auth key |
| `PYTHONIOENCODING` | set in shell before running | Set to `utf-8` on Windows to prevent encoding errors |
