#!/usr/bin/env bash
set -euo pipefail

ENABLE_MODEL_PULL=false
# Default models match deployments/local/.env.example
MODELS=("llama3.2:3b")

while [[ $# -gt 0 ]]; do
  case "$1" in
    --enable-model-pull)
      ENABLE_MODEL_PULL=true
      shift
      ;;
    --models)
      IFS=',' read -r -a MODELS <<< "$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      echo "Usage: ./setup.sh [--enable-model-pull] [--models model1,model2]"
      exit 1
      ;;
  esac
done

cd "$(dirname "$0")"

# ── Python virtual environment ────────────────────────────────────────────────
if [[ ! -d .venv ]]; then
  python -m venv .venv
fi

VENV_PY=".venv/bin/python"
"$VENV_PY" -m pip install --upgrade pip

# Core library + optional evaluation extras (matplotlib, pillow)
"$VENV_PY" -m pip install -e ".[evaluation]"

# Full runtime/benchmark deps not declared in pyproject.toml
# (chromadb, sentence-transformers, litellm, httpx, fastapi, uvicorn, etc.)
if [[ -f requirements.txt ]]; then
  "$VENV_PY" -m pip install -r requirements.txt
fi

# ── Docker check (needed for local deployment) ────────────────────────────────
if docker info &>/dev/null; then
  echo "Docker: OK"
else
  echo "Warning: Docker not running or not installed."
  echo "  Install Docker Desktop (Mac/Windows) or Docker Engine (Linux)"
  echo "  to use: bash deployments/deploy.sh local up"
fi

# ── Ollama model pulls ────────────────────────────────────────────────────────
if [[ "$ENABLE_MODEL_PULL" == true ]]; then
  if ! command -v ollama >/dev/null 2>&1; then
    echo "Warning: Ollama CLI not found. Install from https://ollama.com and re-run with --enable-model-pull."
  else
    for model in "${MODELS[@]}"; do
      echo "Pulling Ollama model: $model"
      ollama pull "$model"
    done
  fi
fi

# ── Local deployment .env init ────────────────────────────────────────────────
LOCAL_ENV="deployments/local/.env"
LOCAL_ENV_EX="deployments/local/.env.example"
if [[ ! -f "$LOCAL_ENV" && -f "$LOCAL_ENV_EX" ]]; then
  cp "$LOCAL_ENV_EX" "$LOCAL_ENV"
  echo "Created deployments/local/.env from .env.example — review model names if needed."
fi

echo ""
echo "Setup complete."
echo "  Activate venv : source .venv/bin/activate"
echo "  Local deploy  : bash deployments/deploy.sh local up"
if [[ "$ENABLE_MODEL_PULL" == false ]]; then
  echo "  Pull models   : bash setup.sh --enable-model-pull  (default: ${MODELS[*]})"
fi
