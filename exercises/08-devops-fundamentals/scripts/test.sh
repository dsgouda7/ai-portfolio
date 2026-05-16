#!/bin/bash
# Test Runner Script

set -e

echo "🧪 Running tests..."

# Activate virtual environment
source venv/bin/activate

# Run linting
echo "🔍 Running flake8..."
flake8 src tests --max-line-length=100 --exclude=venv || true

# Check formatting
echo "🎨 Checking code formatting..."
black --check src tests || true

# Run tests with coverage
echo "🧪 Running pytest..."
pytest tests/ -v --cov=src --cov-report=term --cov-report=html

echo "✅ Tests complete! Coverage report: htmlcov/index.html"
