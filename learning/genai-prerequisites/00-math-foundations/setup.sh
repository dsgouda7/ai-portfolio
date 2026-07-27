#!/usr/bin/env bash
#
# Creates a local virtual environment and installs everything needed to run
# math-foundations-for-ml.ipynb.
#
# Creates a `.venv` next to this script (if it does not already exist),
# installs the dependencies from requirements.txt into it, and registers a
# Jupyter kernel named "math-foundations" pointing at that venv. Select the
# "Python (math-foundations .venv)" kernel in VS Code to use it.
#
# Usage:
#   ./setup.sh
#   ./setup.sh --skip-kernel

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REQUIREMENTS="$SCRIPT_DIR/requirements.txt"
VENV_DIR="$SCRIPT_DIR/.venv"
VENV_PYTHON="$VENV_DIR/bin/python"
SKIP_KERNEL=0

for arg in "$@"; do
    case "$arg" in
        --skip-kernel) SKIP_KERNEL=1 ;;
        *) echo "Unknown option: $arg" >&2; exit 1 ;;
    esac
done

# Pick a base Python to create the venv with
if command -v python3 >/dev/null 2>&1; then
    PYTHON=python3
elif command -v python >/dev/null 2>&1; then
    PYTHON=python
else
    echo "Error: Python was not found on PATH. Install Python 3.9+ and re-run." >&2
    exit 1
fi

if [ ! -f "$REQUIREMENTS" ]; then
    echo "Error: requirements.txt not found next to this script." >&2
    exit 1
fi

# Create the virtual environment if it doesn't already exist
if [ -x "$VENV_PYTHON" ]; then
    echo "Reusing existing virtual environment at $VENV_DIR"
else
    echo "Creating virtual environment with $($PYTHON --version 2>&1) at $VENV_DIR"
    "$PYTHON" -m venv "$VENV_DIR"
fi

echo "Upgrading pip..."
"$VENV_PYTHON" -m pip install --upgrade pip -q

echo "Installing notebook dependencies from requirements.txt..."
"$VENV_PYTHON" -m pip install -r "$REQUIREMENTS" -q

echo "Installing Jupyter kernel support (ipykernel)..."
"$VENV_PYTHON" -m pip install ipykernel -q

echo "Installing nbconvert for headless execution..."
"$VENV_PYTHON" -m pip install nbconvert -q

if [ "$SKIP_KERNEL" -eq 0 ]; then
    echo "Registering Jupyter kernel 'math-foundations'..."
    "$VENV_PYTHON" -m ipykernel install --user --name math-foundations --display-name "Python (math-foundations .venv)"
fi

echo ""
echo "Setup complete."
echo "Open math-foundations-for-ml.ipynb and pick the 'Python (math-foundations .venv)' kernel."
echo "The venv lives at: $VENV_DIR"
