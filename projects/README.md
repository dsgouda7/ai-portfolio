# Projects

Production-ready applications built from scratch.

## By Domain

### Data Engineering

#### [Databricks RAG Pipeline](data-engineering/databricks_rag/)
RAG pipeline split into 3 independent microservices (ingest, vectorize, serve).

Tech: Delta Lake, ChromaDB, FastAPI, Docker
Features:
- ACID storage with Delta Lake
- Vector search via ChromaDB
- Each phase has its own Dockerfile and dependencies
- REST API for queries
- Runs locally or containerized

---

### Computer Vision

#### [AI Video Enhancer](computer-vision/video_enhancer_ai/)
4K upscaling API using HuggingFace models.

Tech: PyTorch, HuggingFace Transformers, FastAPI, Docker
Models: Swin2SR (video), MetricGAN+ (audio)
Features:
- Detects GPU/CPU and picks appropriate models
- Processes video and audio in parallel
- REST API
- 100% local (no external APIs)
- Dockerized

---

### Machine Learning

#### [King County House Price Modeling](ml/king-county-house-pricing/)
Portfolio notebook focused on practical price prediction workflow and model iteration.

Tech: Python, Pandas, Seaborn, scikit-learn
Features:
- End-to-end regression flow from raw data to refined model
- Data quality checks and baseline imputation
- Linear and regularized model comparison
- Narrative written as engineering progression

## Setup

Each project has its own README with setup instructions. Most include:
- Setup scripts: `setup.ps1` (Windows), `setup.sh` (Linux/macOS)
- Docker support
- Config templates
- Requirements files
