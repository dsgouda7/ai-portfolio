# Context Optimizer: Enterprise Production Readiness

> **Status as of 2026-07-08**
> Multi-format corpus ingestion (Part A) is implemented.
> Codebase search (Part B) was prototyped and removed -- code chunking does not
> split semantically as well as prose, so retrieval quality without full per-function
> LLM summarization (impractically slow on CPU) was too low to be useful.

---

## What is implemented

### Multi-format corpus ingestion
| File | Purpose |
|---|---|
| src/extractors/__init__.py | FormatRouter: detect format, dispatch, return task |
| src/extractors/txt.py | TxtExtractor, RtfExtractor |
| src/extractors/markdown.py | MarkdownExtractor (strips front-matter, links, HTML) |
| src/extractors/pdf.py | PdfExtractor (pdfminer.six, text-layer PDFs) |
| src/extractors/docx.py | DocxExtractor (python-docx, paragraphs + tables) |
| src/extractors/xlsx.py | XlsxExtractor (openpyxl, serializes rows to prose) |
| src/extractors/xml_extractor.py | XmlExtractor (lxml, strips tags) |
| src/ingest_corpus.py | ingest_directory(): parallel extraction + per-task compression |

### Two-model compressor config (bench_config.yaml)
- model -> all non-code content (text, docs, spreadsheets, markup)
- code_model -> source code only (same size, purpose-trained on CodeSearchNet)
- Supported providers: ollama (default), hf (BART, fastest CPU), azure_foundry

## Enterprise production readiness checklist

The following items are required before enterprise deployment.
Items are grouped by priority.

### P0 -- Blocking (must fix before any production traffic)

| Item | Why it blocks | Effort |
|---|---|---|
| **API service layer** | System is CLI-only; no way to integrate into enterprise apps | 2-3 days (FastAPI + Uvicorn) |
| **Authentication + authorization** | Any caller can ingest/query sensitive data | 2 days (bearer token, RBAC on collections) |
| **Input validation + sanitization** | Malicious files (zip-bomb PDFs, macro DOCX) can crash/exploit the extractor | 1 day (file size limits, magic-byte validation) |
| **Secrets management** | API keys (Groq, Azure) are in env vars; no rotation, no vault | 1 day (HashiCorp Vault / Azure Key Vault integration) |
| **Error handling + graceful degradation** | Unhandled exceptions terminate the process; partial ingestion is lost | 1 day (circuit breaker, dead-letter queue for failed files) |
| **Persistent index required** | Without --index-dir, index is lost on restart (temp dirs) | 0.5 day (make --index-dir mandatory; add init check at startup) |

### P1 -- Required for scale (must fix before >10 concurrent users)

| Item | Why it matters | Effort |
|---|---|---|
| **Replace embedded ChromaDB** | Single-process, no replication, no distributed search | 3 days (Qdrant Cloud or ChromaDB distributed) |
| **Async ingestion pipeline** | Ingestion blocks the API; large corpora lock the service | 3 days (Celery + Redis task queue) |
| **Multi-tenant index isolation** | All users share one ChromaDB namespace | 2 days (per-tenant collection prefix or separate databases) |
| **Observability** | No metrics, no traces, no alerts; production incidents are invisible | 2 days (OpenTelemetry -> Prometheus + Grafana) |
| **Structured logging** | Print statements; no correlation IDs, no log levels, not searchable | 1 day (structlog or python-json-logger) |
| **Health / readiness probes** | No way for Kubernetes / load balancer to detect service health | 0.5 day (/healthz, /readyz endpoints) |
| **OCR for scanned PDFs** | Text-layer PDFs only; enterprise document archives are often scanned | 3 days (Tesseract via pytesseract; GPU recommended) |

### P2 -- Quality of life (required before enterprise SLA)

| Item | Why it matters | Effort |
|---|---|---|
| **GPU inference for BART** | 54 min ingestion at 400 MB on CPU; GPU reduces this to ~5 min | 0.5 day (already supported via HF device=-1 -> 0) |
| **Groq/Azure provider for reasoning** | Ollama requires local GPU; cloud providers give <1s latency | 0.5 day (provider already wired; just needs API key config) |
| **Index versioning** | No way to roll back a bad ingestion run | 2 days (named index versions; soft-delete + restore) |
| **Incremental ingestion** | Re-ingesting a changed file re-processes the entire block | 2 days (file hash tracking; delta ingestion) |
| **tree-sitter grammars for all languages** | Regex fallback is imprecise for C/C++ | 1 day (install tree-sitter-languages bundle) |
| **Unit + integration test suite** | No automated tests; regressions are found in production | 3 days (pytest; mock LLM for fast CI) |
| **Docker image + Helm chart** | No deployment artifact; every deploy is manual | 2 days |
| **Rate limiting on API** | Unrestricted callers can exhaust BART/Ollama compute | 0.5 day (fastapi-limiter or nginx) |
| **XLSX numeric data quality** | BART prose summaries of tabular data have low accuracy | 3 days (table-BERT or fine-tuned T5 on tabular data) |

### P3 -- Nice to have (post-launch improvements)

| Item | Notes |
|---|---|
| .pptx support | python-pptx; low effort |
| .html corpus ingestion | lxml extractor already handles HTML; just add .html to FormatRouter |
| CodeBERT embedding (code search) | Better code-semantic vectors than all-MiniLM-L6-v2 |
| Linux full-kernel benchmark | Extend linux_benchmark.py to all of drivers/ (~15k files) |
| Query result caching | Redis cache for repeated queries |
| Streaming answers | SSE endpoint for long Mistral responses |
| Fine-tune BART on domain corpus | Improve summary quality for legal/medical/financial text |

---

## Effort summary

| Priority | Items | Total effort |
|---|---|---|
| P0 (blocking) | 6 items | ~8 days |
| P1 (scale) | 8 items | ~15 days |
| P2 (SLA) | 9 items | ~12 days |
| **Total to enterprise prod** | **23 items** | **~35 developer-days** |

GPU hardware and cloud API keys (Groq/Azure) reduce the biggest latency issues
immediately with minimal code changes -- those should be provisioned first.
