#!/usr/bin/env bash
set -euo pipefail

ENABLE_MODEL_PULL=false
MODELS=("phi4:mini" "qwen3")

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

if [[ ! -d .venv ]]; then
  python -m venv .venv
fi

VENV_PY=".venv/bin/python"
"$VENV_PY" -m pip install --upgrade pip
"$VENV_PY" -m pip install -e ".[evaluation]"

if [[ "$ENABLE_MODEL_PULL" == true ]]; then
  if ! command -v ollama >/dev/null 2>&1; then
    echo "Warning: Ollama CLI not found. Install Ollama or run setup without --enable-model-pull."
    exit 0
  fi

  for model in "${MODELS[@]}"; do
    echo "Pulling Ollama model: $model"
    ollama pull "$model"
  done
fi

echo "Setup complete."
echo "Activate with: source .venv/bin/activate"
if [[ "$ENABLE_MODEL_PULL" == false ]]; then
  echo "Model downloads skipped (opt-in). Re-run with --enable-model-pull to pull Ollama models."
fi
