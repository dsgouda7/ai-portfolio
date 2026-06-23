#!/usr/bin/env bash
# setup.sh - Install dependencies and pull Ollama models for LLM_Guardrails
# Usage: bash setup.sh
#
# Models pulled:
#   mistral      -- main agent model          (~4 GB)
#   llama3.2:1b  -- lightweight safety judge  (~1.3 GB)

set -euo pipefail

FULL_MODEL="${FULL_MODEL:-mistral}"
LIGHT_MODEL="${LIGHT_MODEL:-llama3.2:1b}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

step() { printf "\n\033[0;36m==> %s\033[0m\n" "$*"; }
ok()   { printf "\033[0;32m✓  %s\033[0m\n"   "$*"; }
die()  { printf "\033[0;31mERROR: %s\033[0m\n" "$*" >&2; exit 1; }

# ── 1. Python dependencies ───────────────────────────────────────────────────
step "Installing Python dependencies"
pip install -r "$SCRIPT_DIR/requirements.txt"

# ── 2. Check Ollama ──────────────────────────────────────────────────────────
step "Checking Ollama installation"
command -v ollama >/dev/null 2>&1 || die "'ollama' not found. Install from https://ollama.com/download"
ok "ollama found"

# Ensure Ollama server is running
if curl -sf http://localhost:11434/api/tags >/dev/null 2>&1; then
    ok "Ollama already running"
else
    echo "Starting Ollama in the background ..."
    ollama serve &
    sleep 3
fi

# ── 3. Pull models ────────────────────────────────────────────────────────────
step "Pulling main agent model: $FULL_MODEL  (~4 GB on first pull)"
ollama pull "$FULL_MODEL"

step "Pulling lightweight safety-judge model: $LIGHT_MODEL  (~1.3 GB on first pull)"
ollama pull "$LIGHT_MODEL"

echo ""
ok "Setup complete."
echo "  Open LLM_Guardrails.ipynb in VS Code and run all cells."
echo "  To use Azure AI Foundry instead: export LLM_PROVIDER=azure"
