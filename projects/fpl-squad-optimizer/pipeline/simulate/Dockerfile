# ── Pipeline Step 3: Simulate ──────────────────────────────────────────────
#
# Runs the walk-forward GW simulation using the trained models and the DB,
# and writes simulation_results.csv + player_rows.csv to FPL_RESULTS_DIR.
#
# Azure ML usage:
#   Mount the DB (ingest output) and models.joblib (train output) as inputs.
#   Mount a pipeline Output directory for FPL_RESULTS_DIR.
#   Pass FPL_SIM_FROM / FPL_SIM_TO as pipeline parameters via env vars.
#
# Local usage (via docker-compose):
#   docker compose run simulate
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
    FPL_RESULTS_DIR=/data/results \
    FPL_SIM_FROM=15 \
    FPL_SIM_TO=37

ENTRYPOINT ["python", "pipeline/steps/simulate.py"]
