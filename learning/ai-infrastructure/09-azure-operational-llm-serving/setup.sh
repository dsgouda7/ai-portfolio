#!/usr/bin/env bash
# Create the local notebook environment. This script does not install Azure
# tooling, start Docker, or contact Azure.

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

if command -v python3 >/dev/null 2>&1; then
    PYTHON=python3
elif command -v python >/dev/null 2>&1; then
    PYTHON=python
else
    echo "Error: Python 3.10+ was not found on PATH." >&2
    exit 1
fi

if [ ! -f "$REQUIREMENTS" ]; then
    echo "Error: requirements.txt was not found next to this script." >&2
    exit 1
fi

if [ -x "$VENV_PYTHON" ]; then
    echo "Reusing virtual environment at $VENV_DIR"
else
    echo "Creating virtual environment at $VENV_DIR"
    "$PYTHON" -m venv "$VENV_DIR"
fi

"$VENV_PYTHON" -m pip install --upgrade pip -q
"$VENV_PYTHON" -m pip install -r "$REQUIREMENTS" -q
"$VENV_PYTHON" -m pip install ipykernel -q

if [ "$SKIP_KERNEL" -eq 0 ]; then
    "$VENV_PYTHON" -m ipykernel install --user \
        --name azure-operational-serving \
        --display-name "Python (azure-operational-serving .venv)"
fi

echo ""
echo "Setup complete. No Azure or container service was contacted."
echo "Open azure-operational-llm-serving.ipynb and select the chapter kernel."
