#!/bin/bash
set -e

echo "Setting up Phase 1: Ingestion"
echo "=============================="

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
echo "Phase 1 setup complete"
echo "To activate: source venv/bin/activate"
echo "To run: python src/ingest.py"
