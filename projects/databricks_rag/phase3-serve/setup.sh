#!/bin/bash
set -e

echo "Setting up Phase 3: RAG Server"
echo "==============================="

# Create virtual environment
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "Virtual environment created"
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

# Install shared utilities
pip install -e ../shared

echo ""
echo "Phase 3 setup complete"
echo "To activate: source venv/bin/activate"
echo "To run: uvicorn src.server:app --reload"
