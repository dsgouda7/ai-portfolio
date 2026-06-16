# ── Web App: FPL Generator ─────────────────────────────────────────────────
#
# Flask web server serving /generate-team and /validation-report.
# NOT a pipeline step — this is a long-running service that reads the
# artifacts produced by the pipeline (DB, models, simulation results).
#
# Usage:
#   docker build -f fpl-generator/Dockerfile -t fpl-serve .
#   docker run --rm -p 5000:5000 \
#     -e FPL_DB_FILE=/data/fantasy_football.db \
#     -e FPL_MODELS_FILE=/data/models/models.joblib \
#     -e FPL_RESULTS_DIR=/data/results \
#     -v pipeline_data:/data \
#     fpl-serve
#
# Or via docker-compose:
#   docker compose up serve
# ──────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# ── Default artifact paths ─────────────────────────────────────────────────
# Mount the volume that was written by the pipeline steps and set these
# env vars to match the mount paths.
ENV FPL_DB_FILE=/data/fantasy_football.db \
    FPL_MODELS_FILE=/data/models/models.joblib \
    FPL_RESULTS_DIR=/data/results

EXPOSE 5000

ENTRYPOINT ["python", "fpl-generator/web.py"]
