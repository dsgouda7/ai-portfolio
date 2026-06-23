#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Set up the voice-assistant for local (Ollama) operation.
#
# Usage:
#   ./setup.sh                         # defaults: mistral LLM
#   LLM_MODEL=llama3.2:3b ./setup.sh  # lightweight model
# ---------------------------------------------------------------------------
set -euo pipefail

LLM_MODEL="${LLM_MODEL:-mistral}"
PYTHON="${PYTHON:-python3}"

step() { printf "\n\033[0;36m==> %s\033[0m\n" "$*"; }
ok()   { printf "\033[0;32m✓  %s\033[0m\n"   "$*"; }
warn() { printf "\033[0;33m!  %s\033[0m\n"   "$*"; }
die()  { printf "\033[0;31mERROR: %s\033[0m\n" "$*" >&2; exit 1; }

# ── 1. Check prerequisites ────────────────────────────────────────────────────
step "Checking prerequisites"
command -v ollama  >/dev/null 2>&1 || die "'ollama' not found. Install from https://ollama.com"
command -v "$PYTHON" >/dev/null 2>&1 || die "'$PYTHON' not found."

# ── 2. Start Ollama if not running ────────────────────────────────────────────
step "Ensuring Ollama server is running"
if curl -sf http://localhost:11434/api/tags >/dev/null 2>&1; then
    ok "Ollama already running"
else
    warn "Starting Ollama in the background ..."
    ollama serve &
    sleep 3
fi

# ── 3. Pull LLM model(s) ──────────────────────────────────────────────────────
step "Pulling LLM model: $LLM_MODEL"
echo "This may take a few minutes on first run (model ~4 GB for mistral)."
ollama pull "$LLM_MODEL"

step "Pulling lightweight fallback model: llama3.2:3b"
ollama pull "llama3.2:3b"

echo ""
ok "Pulled models:"
ollama list

# ── 4. Python virtual environment ─────────────────────────────────────────────
step "Creating Python virtual environment (.venv)"
if [ ! -d ".venv" ]; then
    "$PYTHON" -m venv .venv
fi

PIP=".venv/bin/pip"
step "Installing Python dependencies"
"$PIP" install --upgrade pip --quiet
"$PIP" install -r requirements.txt

# ── 5. Environment file ───────────────────────────────────────────────────────
step "Setting up .env"
if [ ! -f ".env" ]; then
    cp .env.example .env
    warn ".env created from .env.example — edit it to configure providers."
else
    ok ".env already exists, skipping."
fi

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
ok "Setup complete."
echo "  Start the server:      .venv/bin/python server.py"
echo "  Change the LLM model:  set OLLAMA_LLM_MODEL=<model> in .env"
echo "  Switch to Azure:       set LLM_PROVIDER=azure in .env"
