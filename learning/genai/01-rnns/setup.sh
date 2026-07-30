#!/usr/bin/env bash
# Creates a local .venv and installs dependencies for the RNN notebooks.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "Creating virtual environment at $SCRIPT_DIR/.venv ..."
python3 -m venv "$SCRIPT_DIR/.venv"

echo "Installing build tooling ..."
"$SCRIPT_DIR/.venv/bin/pip" install --upgrade pip setuptools wheel --quiet

echo "Installing dependencies from requirements.txt ..."
"$SCRIPT_DIR/.venv/bin/pip" install -r "$SCRIPT_DIR/requirements.txt"

echo "Installing Jupyter kernel support (ipykernel, nbconvert) ..."
"$SCRIPT_DIR/.venv/bin/pip" install --quiet ipykernel nbconvert

echo "Registering Jupyter kernel as 'genai-rnns' ..."
"$SCRIPT_DIR/.venv/bin/python" -m ipykernel install --user \
    --name "genai-rnns" \
    --display-name "Python (genai-rnns)"

echo ""
echo "Done. In VS Code, select the 'Python (genai-rnns)' kernel (or the .venv interpreter) for the notebook."
