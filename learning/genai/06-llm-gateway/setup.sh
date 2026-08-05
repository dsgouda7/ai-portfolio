#!/usr/bin/env bash
# Creates the chapter-local environment for GenAI 06 LLM Gateway.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REQUIREMENTS="$SCRIPT_DIR/requirements.txt"
VENV_DIR="$SCRIPT_DIR/.venv"
VENV_PYTHON="$VENV_DIR/bin/python"
KERNEL_NAME="genai-06-llm-gateway"
KERNEL_DISPLAY_NAME="Python (GenAI 06 LLM Gateway .venv)"
KERNEL_SETTER="$SCRIPT_DIR/../../../scripts/set-notebook-kernel.py"
SKIP_KERNEL=0

for argument in "$@"; do
    case "$argument" in
        --skip-kernel) SKIP_KERNEL=1 ;;
        *) echo "Unknown option: $argument" >&2; exit 1 ;;
    esac
done

if command -v python3 >/dev/null 2>&1; then
    PYTHON=python3
elif command -v python >/dev/null 2>&1; then
    PYTHON=python
else
    echo "Python was not found on PATH. Install Python 3.10+ and rerun this script." >&2
    exit 1
fi

[ -f "$REQUIREMENTS" ] || { echo "requirements.txt was not found at $REQUIREMENTS" >&2; exit 1; }
[ -f "$KERNEL_SETTER" ] || { echo "Kernel metadata helper was not found at $KERNEL_SETTER" >&2; exit 1; }

if [ -x "$VENV_PYTHON" ]; then
    echo "Reusing virtual environment at $VENV_DIR"
else
    echo "Creating virtual environment with $($PYTHON --version 2>&1) at $VENV_DIR"
    "$PYTHON" -m venv "$VENV_DIR"
fi

echo "Upgrading pip, setuptools, and wheel..."
"$VENV_PYTHON" -m pip install --upgrade pip setuptools wheel --quiet

echo "Installing dependencies from $REQUIREMENTS..."
"$VENV_PYTHON" -m pip install -r "$REQUIREMENTS"

if [ "$SKIP_KERNEL" -eq 0 ]; then
    echo "Registering Jupyter kernel '$KERNEL_NAME'..."
    "$VENV_PYTHON" -m ipykernel install --user --name "$KERNEL_NAME" --display-name "$KERNEL_DISPLAY_NAME"

    echo "Assigning '$KERNEL_NAME' to chapter notebooks..."
    "$VENV_PYTHON" "$KERNEL_SETTER" \
        --directory "$SCRIPT_DIR" \
        --name "$KERNEL_NAME" \
        --display-name "$KERNEL_DISPLAY_NAME"
fi

echo ""
echo "Setup complete for GenAI 06 LLM Gateway."
echo "Virtual environment: $VENV_DIR"
echo "Jupyter kernel: $KERNEL_DISPLAY_NAME"
