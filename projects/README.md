# Projects

End-to-end projects covering the full lifecycle: problem framing, data, modelling or system design, evaluation, and a working deliverable.

Each project has a concrete problem statement with measurable success criteria, documented constraints, and honest limitations.

---

## [FPL Squad Optimizer](fpl-squad-optimizer/)

**Can a CPU-only ML model trained on community data assemble an FPL squad that performs in the range of an experienced human manager?**

XGBoost team picker for Fantasy Premier League. Four position-specific regressors trained on rolling form, Transfermarkt market value, and FPL API data. Walk-forward simulation across 23 held-out game weeks achieves 51% oracle-capture rate (human manager range: 45–60%). Containerised pipeline with Docker Compose; ready to hook up to Azure ML.

`XGBoost` `SQLite` `Flask` `Docker` `Python`

---

## [Context Optimizer](context-optimizer/)

**Can a two-stage context architecture (compression + targeted retrieval) reduce LLM token consumption from O(corpus size) to O(1) while improving failure observability?**

Multi-stage pipeline that decomposes the context problem: compress rambling incident reports into structured schemas (99.8% reduction), then query logs using extracted keywords to retrieve only relevant evidence (93–99.9% reduction). Result: constant token cost (~1.7K) regardless of corpus size (tested 1K–100K logs). Features schema validation, multi-provider LLM support (Ollama, Groq, mock), deterministic benchmarks, and complete architectural documentation. Demonstrates the design inversion: use a cheap operation (compression) to optimize an expensive operation (reasoning).

Includes scalability tests, failure mode analysis, and diagrams explaining the architectural sophistication of decomposed-stage systems design.

`LangChain` `Pydantic` `LLM` `Python` `Docker` `Architecture`

---

## [RAG Knowledge Pipeline](rag-knowledge-pipeline/)

**Can a fully local, containerised pipeline ingest a text corpus, build a vector index, and serve retrieval-augmented answers with each stage independently deployable?**

Three-phase pipeline: Wikipedia corpus → Delta Lake → ChromaDB → FastAPI RAG server. Each phase has its own Dockerfile, isolated dependencies, and communicates only through durable storage — not in-memory hand-offs.

`PySpark` `Delta Lake` `ChromaDB` `FastAPI` `sentence-transformers` `Docker`

---

## [Video Quality Enhancer](video-quality-enhancer/)

**Can open-source super-resolution and audio-denoising models running entirely locally upscale consumer video to 4K with improved audio — no paid API, no cloud GPU required?**

Local REST service that applies Swin2SR (4× super-resolution) and MetricGAN+ (spectral audio denoising) to arbitrary video files. GPU-accelerated where available; CPU fallback included. First run downloads ~2–3 GB of models; subsequent runs start in ~60 seconds.

`PyTorch` `HuggingFace` `Swin2SR` `MetricGAN+` `Flask` `Docker`


