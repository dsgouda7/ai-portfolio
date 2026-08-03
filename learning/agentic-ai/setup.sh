#!/usr/bin/env bash
set -euo pipefail

TRACK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$TRACK_DIR/.venv"

if [[ ! -d "$VENV_DIR" ]]; then
  python3 -m venv "$VENV_DIR"
fi

PYTHON="$VENV_DIR/bin/python"
"$PYTHON" -m pip install --upgrade pip
"$PYTHON" -m pip install -r "$TRACK_DIR/requirements.txt"
"$PYTHON" -m ipykernel install --user --name agentic-ai --display-name "Python (agentic-ai)"

echo "Agentic AI environment ready: $PYTHON"
echo "Validate with: $PYTHON $TRACK_DIR/scripts/validate_track.py"
