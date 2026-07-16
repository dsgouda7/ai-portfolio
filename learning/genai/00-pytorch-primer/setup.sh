#!/usr/bin/env bash
# Creates a local .venv and installs notebook dependencies for mnist-cnn-keras-vs-pytorch.ipynb

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "Creating virtual environment at $SCRIPT_DIR/.venv ..."
python3 -m venv "$SCRIPT_DIR/.venv"

echo "Installing dependencies from requirements.txt ..."
"$SCRIPT_DIR/.venv/bin/pip" install --upgrade pip --quiet
"$SCRIPT_DIR/.venv/bin/pip" install -r "$SCRIPT_DIR/requirements.txt"

echo "Registering Jupyter kernel as 'pytorch-primer' ..."
"$SCRIPT_DIR/.venv/bin/python" -m ipykernel install --user \
    --name "pytorch-primer" \
    --display-name "Python (pytorch-primer)"

echo ""
echo "Done. In VS Code, select the 'Python (pytorch-primer)' kernel (or the .venv interpreter) for the notebook."
