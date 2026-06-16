# ── Pipeline Step 1: Ingest ────────────────────────────────────────────────
#
# Reads the vaastav CSV archive (mounted at FPL_VAASTAV_DIR), calls the live
# FPL Bootstrap API, enriches with SOFIFA attributes + Transfermarkt values,
# and writes fantasy_football.db to FPL_DATA_DIR.
#
# Azure ML usage:
#   Register the vaastav clone as an AML Dataset (FileDataset).
#   Mount it as an input; mount a pipeline Output for the DB directory.
#   Inject FPL_VAASTAV_DIR and FPL_DB_FILE as environment variables.
#
# Local usage (via docker-compose):
#   docker compose run ingest
# ──────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

WORKDIR /app

# Install system-level build deps required by some wheels (lxml, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the full source tree (all steps share the same utils, eligibility, etc.)
COPY . .

# ── Default I/O paths ──────────────────────────────────────────────────────
# Override these at runtime (--env / docker-compose environment:) to point
# the container at mounted volumes or AML-mounted datasets/outputs.
ENV FPL_VAASTAV_DIR=/data/vaastav \
    FPL_DB_FILE=/data/output/fantasy_football.db

ENTRYPOINT ["python", "pipeline/steps/ingest.py"]
