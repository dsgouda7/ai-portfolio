#!/usr/bin/env bash
# setup.sh - Install dependencies and pull the Ollama model for LLM_Guardrails
# Usage: bash setup.sh

set -euo pipefail

MODEL="llama3.2:1b"

echo "Installing Python dependencies..."
pip install -r requirements.txt

echo ""
echo "Checking Ollama..."
if ! command -v ollama &>/dev/null; then
    echo "Error: Ollama is not installed or not in PATH." >&2
    echo "Download it from: https://ollama.com/download" >&2
    exit 1
fi

echo "Pulling model: $MODEL (~1.3 GB)..."
ollama pull "$MODEL"

echo ""
echo "Done! Model '$MODEL' is ready."
echo "Ensure Ollama is running ('ollama serve') before executing the notebook."
