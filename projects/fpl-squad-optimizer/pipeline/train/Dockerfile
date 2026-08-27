# ── Pipeline Step 2: Train ─────────────────────────────────────────────────
#
# Reads fantasy_football.db, builds rolling features, trains XGBoost and GRU
# position models, and writes separate checkpoint artifacts.
#
# Azure ML usage:
#   Mount the ingest step's output (DB directory) as FPL_DB_FILE input.
#   Mount a pipeline Output directory and set FPL_MODELS_FILE to point to it.
#
# Local usage (via docker-compose):
#   docker compose run train
# ──────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# ── Default I/O paths ──────────────────────────────────────────────────────
ENV FPL_DB_FILE=/data/fantasy_football.db \
    FPL_MODELS_FILE=/data/models/models.joblib \
    FPL_RNN_MODELS_FILE=/data/models/models_rnn.joblib \
    FPL_MODEL_TYPE=all

ENTRYPOINT ["python", "pipeline/steps/train.py"]
