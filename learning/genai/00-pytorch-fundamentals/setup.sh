#!/usr/bin/env bash
# Creates the local environment for 01-keras-to-pytorch-antarctic-field-guide.ipynb.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH="$SCRIPT_DIR/.venv"

if [[ ! -d "$VENV_PATH" ]]; then
  echo "Creating virtual environment at $VENV_PATH ..."
  python3 -m venv "$VENV_PATH"
fi

echo "Installing PyTorch foundations dependencies ..."
"$VENV_PATH/bin/python" -m pip install --upgrade pip setuptools wheel --quiet
"$VENV_PATH/bin/python" -m pip install -r "$SCRIPT_DIR/requirements.txt"

echo "Registering Jupyter kernel as 'pytorch-foundations' ..."
"$VENV_PATH/bin/python" -m ipykernel install --user \
  --name "pytorch-foundations" \
  --display-name "Python (pytorch-foundations)"

echo "Done. Select the 'Python (pytorch-foundations)' kernel in VS Code."
