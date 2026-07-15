#!/usr/bin/env bash
# Creates a local .venv and installs the dependencies for ibm_genai_conversation_analyzer.py
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"
VENV_PYTHON="$VENV_DIR/bin/python"

if command -v python3 >/dev/null 2>&1; then PYTHON=python3
elif command -v python >/dev/null 2>&1; then PYTHON=python
else echo "Error: Python was not found on PATH. Install Python 3.9+ and re-run." >&2; exit 1
fi

if [ -x "$VENV_PYTHON" ]; then
    echo "Reusing existing virtual environment at $VENV_DIR"
else
    echo "Creating virtual environment at $VENV_DIR ..."
    "$PYTHON" -m venv "$VENV_DIR"
fi

echo "Upgrading pip..."
"$VENV_PYTHON" -m pip install --upgrade pip -q

echo "Installing dependencies from requirements.txt..."
"$VENV_PYTHON" -m pip install -r "$SCRIPT_DIR/requirements.txt"

echo ""
echo "Setup complete."
echo "Run the app with: $VENV_PYTHON ibm_genai_conversation_analyzer.py"
