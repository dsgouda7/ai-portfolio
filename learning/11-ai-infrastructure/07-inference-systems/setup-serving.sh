#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ "$(uname -s)" != "Linux" ]; then
    echo "Use Linux or WSL with an NVIDIA GPU. Native Windows/CPU is not a substitute." >&2
    exit 1
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi is unavailable; configure GPU access before installing this lab." >&2
    exit 1
fi
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
PYTHON="${PYTHON:-python3}"
"$PYTHON" -c 'import sys; assert (3, 10) <= sys.version_info[:2] <= (3, 12), "Use Python 3.10-3.12 for the pinned runtime."'
"$PYTHON" -m venv "$SCRIPT_DIR/.venv-serving"
VENV_PYTHON="$SCRIPT_DIR/.venv-serving/bin/python"
"$VENV_PYTHON" -m pip install --upgrade pip
"$VENV_PYTHON" -m pip install -r "$SCRIPT_DIR/requirements-serving.txt"
"$VENV_PYTHON" "$SCRIPT_DIR/serving_experiments.py" --preflight
"$VENV_PYTHON" -m ipykernel install --user --name real-llm-serving --display-name "Python (real LLM serving GPU)"
echo "Open real-llm-serving-lab.ipynb and choose Python (real LLM serving GPU)."
