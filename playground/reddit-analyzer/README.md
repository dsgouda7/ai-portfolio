# Reddit Safety Audit Pipeline

## Problem statement

**Can a CPU-only, fully local Python pipeline discover emergent toxic language and negative stereotype terms on Reddit that are not yet captured by open-source moderation lexicons or community safety lists?**

This project is an automated social media safety audit workflow focused on research-relevant subreddits (for example, r/politics and r/news). It combines rule-based lexicon matching, sentiment filtering, unsupervised clustering, and optional local LLM-assisted cluster labeling.

**Constraints we set for ourselves:**
- CPU-only execution for embeddings, clustering, and orchestration
- Local-first inference and storage (JSON/Parquet outputs)
- No mandatory paid APIs
- Baseline lexicon can be seeded from open datasets (default: `SEACrowd/tgl_profanity`)

## Quick start

```powershell
# 1. create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. install dependencies
pip install -r requirements.txt

# 3. run end-to-end pipeline from credential-free Reddit dumps (default mode)
python -m src.main --source hf --hf-max-datasets 10 --hf-max-rows 15000

# optional: use specific Hugging Face datasets
python -m src.main --source hf --hf-datasets \
  Polar/reddit-dataset \
  NUCESI/reddit_comments --hf-max-rows 10000

# optional: ingest local community dumps (Kaggle/HF exports)
python -m src.main --source local --local-dump-paths data/dumps/reddit1.jsonl data/dumps/reddit2.parquet

# optional: combine both HF and local sources
python -m src.main --source mixed --hf-max-datasets 8 --local-dump-paths data/dumps/reddit_seed.parquet

# optional: reuse latest saved raw snapshot
python -m src.main --source latest

# optional: disable LLM labeling stage
python -m src.main --disable-llm-labeling

# 4. launch interactive cluster visualization
python app.py
# → http://localhost:5000
```

Outputs:
- `data/raw/reddit_comments_*.jsonl`
- `data/raw/reddit_comments_*.parquet`
- `data/processed/comments_enriched.parquet`
- `data/processed/comments_negative.parquet`
- `data/processed/comments_clustered.parquet`
- `outputs/cluster_keywords.json`
- `outputs/cluster_labels.json`
- `outputs/detection_drift_metrics.json`
- `reports/safety_audit_report.md`

## Interactive Visualization (`app.py`)

After running the pipeline, launch the Flask web app to visualize clusters and keywords:

```powershell
python app.py
# Opens at http://localhost:5000
```

**Features:**
- **Cluster bubbles**: Main view shows toxic language clusters as bubbles, sized by comment count.
- **Cluster labels**: Each bubble displays the semantically assigned cluster label (from LLM or heuristic).
- **Click to explore**: Click any cluster bubble to drill down and see its top 50 representative keywords.
- **Keyword bubbles**: Keywords appear as smaller bubbles, sized proportionally to their frequency.
- **Reset button**: Return to cluster view at any time.
- **Responsive**: Automatically adapts to window resizing.

Built with D3.js for dynamic bubble pack layouts and Flask for lightweight data serving.

## Project structure

```text
reddit-analyzer/
  src/
    config.py
    ingestion.py
    preprocessing.py
    analysis.py
    moderation.py
    lexicon.py
    metrics.py
    reporting.py
    main.py
  templates/
    index.html
  static/
    css/
      style.css
    js/
      viz.js
  data/
    raw/
    processed/
  outputs/
  reports/
  app.py
  requirements.txt
  .env.example
  README.md
```

## Pipeline stages

1. **Ingestion (`src/ingestion.py`)**
- Supports multiple modes:
  - `praw`: direct Reddit API (credentials required)
  - `hf`: credential-free Reddit dump ingestion from Hugging Face
  - `local`: local JSONL/JSON/CSV/TSV/Parquet dumps (for Kaggle/community exports)
  - `mixed`: combines HF and local
- Normalizes heterogeneous schemas into one unified comment format.
- Saves raw snapshots to JSONL and Parquet.

2. **Lexicon integration (`src/lexicon.py`)**
- Includes a small built-in baseline of toxic/negative terms.
- Optionally extends from Hugging Face dataset seeds (default `SEACrowd/tgl_profanity`).
- Keeps schema-flexible loading for common text/term columns.

3. **Sentiment filtering (`src/preprocessing.py`)**
- Applies VADER sentiment scoring to comments.
- Filters negative comments by configurable compound threshold (default `<= -0.2`).
- Adds tokenized lexicon hits per comment.

4. **Unsupervised clustering (`src/analysis.py`)**
- Generates CPU embeddings with `all-MiniLM-L6-v2`.
- Runs HDBSCAN to find latent toxicity themes without fixed `k`.
- Extracts top representative n-gram keywords per cluster (up to 200).

5. **LLM labeling (`src/moderation.py`)**
- Optional local LLM stage via Ollama HTTP API.
- Produces semantic label, risk summary, and stereotype terms per cluster.
- Falls back to deterministic heuristic labels when LLM is unavailable.

6. **Evaluation/reporting (`src/reporting.py`, `src/metrics.py`)**
- Builds a Safety Audit Report with cluster-level metadata summaries.
- Computes Spearman correlations across cluster/depth/sentiment/time-of-day.
- Runs nonparametric variance tests (Kruskal-Wallis) across clusters.
- Computes Detection Drift and Coverage Gap against baseline lexicon.

## Metrics

`src/metrics.py` outputs:
- **Coverage**: fraction of baseline lexicon rediscovered in cluster keywords
- **Coverage Gap**: `1 - coverage`
- **Detection Drift**: fraction of discovered terms not present in baseline
- Missing baseline terms and novel terms for analyst review

## CPU and infrastructure notes

- `sentence-transformers` is forced to CPU by default.
- HDBSCAN and TF-IDF are CPU-native.
- LLM labeling is optional and defaults to local Ollama endpoint.
- For constrained hardware, lower:
  - `--hf-max-datasets`
  - `--hf-max-rows`
  - `--limit-per-subreddit`
  - `--min-cluster-size`
  - number of keywords/subreddits

## Recommended lexicon seeds

Default is `SEACrowd/tgl_profanity` for seeding. Other practical options to test:
- `cardiffnlp/tweet_eval` (hate/offensive tasks for evaluation framing)
- Hate speech lexicons from Davidson et al. style repositories
- Domain-specific policy lexicons maintained by trust & safety communities

## Limitations

- Reddit API query bias: keyword-driven ingestion can miss euphemisms.
- Cluster IDs are not stable across reruns without fixed data snapshots.
- LLM labels are assistive and should be reviewed by a human moderator.
- Coverage/drift metrics depend heavily on baseline lexicon quality.
