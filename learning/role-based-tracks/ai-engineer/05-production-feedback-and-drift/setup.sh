#!/usr/bin/env bash
# Create the chapter-local notebook environment without executing notebooks or
# contacting a model, provider, or cloud service.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REQUIREMENTS="$SCRIPT_DIR/requirements.txt"
VENV_DIR="$SCRIPT_DIR/.venv"
VENV_PYTHON="$VENV_DIR/bin/python"
KERNEL_NAME="ai-engineer-05-feedback-drift"
KERNEL_DISPLAY_NAME="Python (AI Engineer 05 Feedback and Drift .venv)"
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
    echo "Error: Python 3.10+ was not found on PATH." >&2
    exit 1
fi

if [ ! -f "$REQUIREMENTS" ] || [ ! -f "$KERNEL_SETTER" ]; then
    echo "Error: requirements or kernel helper is missing." >&2
    exit 1
fi

if [ -x "$VENV_PYTHON" ]; then
    echo "Reusing virtual environment at $VENV_DIR"
else
    echo "Creating virtual environment at $VENV_DIR"
    "$PYTHON" -m venv "$VENV_DIR"
fi

"$VENV_PYTHON" -m pip install --upgrade pip setuptools wheel --quiet
"$VENV_PYTHON" -m pip install -r "$REQUIREMENTS"

if [ "$SKIP_KERNEL" -eq 0 ]; then
    "$VENV_PYTHON" -m ipykernel install --user \
        --name "$KERNEL_NAME" \
        --display-name "$KERNEL_DISPLAY_NAME"
    "$VENV_PYTHON" "$KERNEL_SETTER" \
        --directory "$SCRIPT_DIR" \
        --name "$KERNEL_NAME" \
        --display-name "$KERNEL_DISPLAY_NAME"
fi

echo ""
echo "Setup complete for AI Engineer 05 Feedback and Drift."
echo "No notebook was executed and no external service was contacted."
