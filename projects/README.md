# Projects

End-to-end projects covering the full lifecycle: problem framing, data, modelling or system design, evaluation, and a working deliverable.

Each project has a concrete problem statement with measurable success criteria, documented constraints, and honest limitations.

## Evidence Status

- **Implemented source:** inspectable project assets exist; this does not imply they were run.
- **Locally validated:** named non-cloud checks passed in a local environment; this does not imply
	cloud behavior or production readiness.
- **Planned:** the named artifact is not present and cannot count as completion evidence.
- **Live-unvalidated:** cloud, customer, identity, networking, quota, cost, or service behavior still
	needs retained evidence from an authorized target environment.

---

## [FPL Squad Optimizer](fpl-squad-optimizer/)

**Can a CPU-only ML model trained on community data assemble an FPL squad that performs in the range of an experienced human manager?**

XGBoost team picker for Fantasy Premier League. Four position-specific regressors trained on rolling form, Transfermarkt market value, and FPL API data. Walk-forward simulation across 23 held-out game weeks achieves 51% oracle-capture rate (human manager range: 45–60%). Containerised pipeline with Docker Compose; ready to hook up to Azure ML.

`XGBoost` `SQLite` `Flask` `Docker` `Python`

---

## [Context Optimizer](context-optimizer/)

**Can a two-stage context architecture (rolling compression + Tree-of-Thought retrieval) reduce LLM token consumption from O(corpus size) to O(1) while improving failure observability?**

Library that decomposes the context problem into two stages: rolling LLM compression reduces a corpus to structured summaries (91.3% token reduction), then a Tree-of-Thought reasoner branches over compressed evidence to answer targeted queries. Result: constant token cost regardless of corpus size, with full failure observability. Fully local — runs on Ollama with no cloud dependencies. Includes ChromaDB-backed semantic retrieval, a two-tier in-memory + persistent cache, deterministic benchmarks across text/image/incident domains, and architectural documentation.

`LangChain` `ChromaDB` `Pydantic` `Ollama` `Python` `Architecture`

---

## [RAG Knowledge Pipeline](rag-knowledge-pipeline/)

**Can a fully local, containerised pipeline ingest a text corpus, build a vector index, and serve retrieval-augmented answers with each stage independently deployable?**

Three-phase pipeline: Wikipedia corpus → Delta Lake → ChromaDB → FastAPI RAG server. Each phase has its own Dockerfile, isolated dependencies, and communicates only through durable storage — not in-memory hand-offs.

The original local path remains the project baseline. Remote Azure Databricks ingestion, governed
record contracts, job bundles, quality reports, and Direct Vector Access indexing source now also
exist. The local suite passed 55 tests with 2 expected missing-Delta skips. Workspace RBAC, managed identity, Unity Catalog, Delta merge, vector filtering,
deletion, performance, and cost behavior are **live-unvalidated**. Learn the concepts in
[RAG](../learning/genai/03-rag/README.md) and [FDE Data Onboarding](../learning/role-based-tracks/fde/03-data-onboarding-and-contracts/README.md),
then use the [Databricks operations guide](rag-knowledge-pipeline/databricks/indexing/OPERATIONS.md).

`PySpark` `Delta Lake` `ChromaDB` `FastAPI` `sentence-transformers` `Docker`

---

## [Riverside AI Platform](riverside-ai-platform/docs/README.md)

**Can one contract-driven production profile connect fine-tuned model artifacts and a governed
Databricks data plane to an evidence-gated Azure serving system without treating source presence as
production proof?**

Production-oriented source assets for versioned contracts, artifact verification, RAG orchestration,
release gates, telemetry, Azure ML blue/green serving, APIM policies, staged load tests, Bicep/`azd`,
and operations documentation. The non-cloud suite passed 142 tests with 5 cloud tests deselected,
and the offline preflight passed 9 tests. No cloud test or live Azure/Databricks validation was run;
deployment, RBAC, networking, quota, service behavior, SLOs, rollback, cost, and production readiness
are **live-unvalidated**.

Conceptual prerequisites: [AI Engineer route](../learning/role-based-tracks/ai-engineer/README.md),
[FDE route](../learning/role-based-tracks/fde/README.md), and
[Azure Operational LLM Serving](../learning/ai-infrastructure/09-azure-operational-llm-serving/README.md).
The [Riverside project README](riverside-ai-platform/README.md) is the project entry point, with the
[documentation index](riverside-ai-platform/docs/README.md) as the operational reference.

`Azure ML` `API Management` `Azure Databricks` `OpenTelemetry` `Bicep` `azd` `Python`

---

## [WildScope](wildscope/)

**What animals were recently observed across tropical protected areas, and how does a static
wildlife model compare with one adapted to each feed's newest labeled data?**

Two connected learning portals over ten official iNaturalist tropical protected-area feeds. The
observation portal scores SpeciesNet and the deployed label corrector on newly arrived community
identifications. The training portal exposes the exact post-watermark batch, evaluates the deployed
version before training, then trains the next version on all eligible data and records model lineage.

`SpeciesNet` `MegaDetector` `PyTorch` `iNaturalist API` `SQLite` `Flask` `Python`

---

## [Video Quality Enhancer](video-quality-enhancer/)

**Can open-source super-resolution and audio-denoising models running entirely locally upscale consumer video to 4K with improved audio — no paid API, no cloud GPU required?**

Local REST service that applies Swin2SR (4× super-resolution) and MetricGAN+ (spectral audio denoising) to arbitrary video files. GPU-accelerated where available; CPU fallback included. First run downloads ~2–3 GB of models; subsequent runs start in ~60 seconds.

`PyTorch` `HuggingFace` `Swin2SR` `MetricGAN+` `Flask` `Docker`

---

## [Conversation Analyzer](conversation-analyzer/)

**Can a local, CPU-friendly model turn raw conversation transcripts (text or audio) into structured key points without any cloud API?**

Gradio app that transcribes audio with Whisper and extracts key points with FLAN-T5, chunking long transcripts to stay within context limits.

`Gradio` `Whisper` `FLAN-T5` `Transformers` `Python`

---

## [Conversational AI](conversational-ai/)

**Can a small, local causal LM hold a coherent multi-turn conversation with bounded memory on CPU-only hardware?**

Flask chatbot backed by Qwen2.5-1.5B-Instruct with a trimmed rolling chat history to keep responses fast and memory bounded.

`Flask` `Qwen2.5` `Transformers` `PyTorch`

---

## [Image Captioning](image-captioning/)

**Can a single pretrained vision-language model generate accurate image captions through a simple web UI with no fine-tuning?**

Gradio app wrapping BLIP-2 for zero-shot image captioning.

`Gradio` `BLIP-2` `Transformers` `PyTorch`

---

## [Text Translation](text-translation/)

**Can a fully local pipeline chain speech-to-text, translation, and text-to-speech into one working service?**

Flask service chaining Whisper (speech-to-text), Helsinki-NLP MarianMT (translation), and MMS-TTS (text-to-speech) into an end-to-end audio translation flow.

`Flask` `Whisper` `MarianMT` `MMS-TTS` `Transformers`

---

## [Voice Assistant](voice-assistant/)

**Can a fully local, containerised voice assistant handle speech in, LLM reasoning, and speech out with CPU-only models?**

Dockerised Flask app chaining Whisper (STT), DialoGPT (response generation), and SpeechT5 (TTS) behind a web interface.

`Flask` `Whisper` `DialoGPT` `SpeechT5` `Docker`


