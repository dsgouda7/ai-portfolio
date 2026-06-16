# Projects

End-to-end projects covering the full lifecycle: problem framing, data, modelling or system design, evaluation, and a working deliverable.

Each project has a concrete problem statement with measurable success criteria, documented constraints, and honest limitations.

---

## [FPL Squad Optimizer](fpl-squad-optimizer/)

**Can a CPU-only ML model trained on community data assemble an FPL squad that performs in the range of an experienced human manager?**

XGBoost team picker for Fantasy Premier League. Four position-specific regressors trained on rolling form, Transfermarkt market value, and FPL API data. Walk-forward simulation across 23 held-out game weeks achieves 51% oracle-capture rate (human manager range: 45–60%). Containerised pipeline with Docker Compose; ready to hook up to Azure ML.

`XGBoost` `SQLite` `Flask` `Docker` `Python`

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

---

## [Housing Price Predictor](housing-price-predictor/)

**Can a regression model trained on public property records predict sale prices accurately enough to be useful for estimating fair market value?**

Regression workflow on King County, WA housing data: linear baseline → Ridge → Ridge with polynomial features. R² ≈ 0.88 on held-out test split; median absolute error under $35k across properties spanning $75k–$7.7M. Documents where simple regression holds up and where non-linearity makes a meaningful difference.

`scikit-learn` `pandas` `numpy` `Jupyter`
