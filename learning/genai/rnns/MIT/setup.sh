#!/usr/bin/env bash
# Creates a local .venv and installs notebook dependencies for PT_Part1_Intro.ipynb

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "Creating virtual environment at $SCRIPT_DIR/.venv ..."
python3 -m venv "$SCRIPT_DIR/.venv"

echo "Installing dependencies from requirements.txt ..."
"$SCRIPT_DIR/.venv/bin/pip" install --upgrade pip --quiet
"$SCRIPT_DIR/.venv/bin/pip" install -r "$SCRIPT_DIR/requirements.txt"

echo "Registering Jupyter kernel as 'rnns-mit' ..."
"$SCRIPT_DIR/.venv/bin/python" -m ipykernel install --user \
    --name "rnns-mit" \
    --display-name "Python (rnns-mit)"

echo ""
echo "Done. In VS Code, select the 'Python (rnns-mit)' kernel (or the .venv interpreter) for the notebook."
