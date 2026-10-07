#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$SCRIPT_DIR/.venv"
PYTHON="$VENV/bin/python"
KERNEL="pytorch-llms-03"
DISPLAY="Python (PyTorch for LLMs 03)"
if [ ! -x "$PYTHON" ]; then python3 -m venv "$VENV"; fi
"$PYTHON" -m pip install --upgrade pip
"$PYTHON" -m pip install -r "$SCRIPT_DIR/requirements.txt"
if [ "${1:-}" != "--skip-kernel" ]; then
  "$PYTHON" -m ipykernel install --user --name "$KERNEL" --display-name "$DISPLAY"
  "$PYTHON" "$SCRIPT_DIR/../../../scripts/set-notebook-kernel.py" \
    --directory "$SCRIPT_DIR" --name "$KERNEL" --display-name "$DISPLAY"
fi
echo "Ready: $DISPLAY"
