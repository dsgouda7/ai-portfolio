#!/bin/bash

# Setup script for AI Exploration Notebooks
# This script creates a virtual environment and installs dependencies

echo "🚀 Setting up AI Exploration Notebooks environment..."

# Check Python installation
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version)
    echo "✓ Found Python: $PYTHON_VERSION"
    PYTHON_CMD=python3
elif command -v python &> /dev/null; then
    PYTHON_VERSION=$(python --version)
    echo "✓ Found Python: $PYTHON_VERSION"
    PYTHON_CMD=python
else
    echo "✗ Python not found. Please install Python 3.8+ first."
    exit 1
fi

# Create virtual environment
echo ""
echo "📦 Creating virtual environment..."
if [ -d ".venv" ]; then
    echo "Virtual environment already exists."
else
    $PYTHON_CMD -m venv .venv
    echo "✓ Virtual environment created"
fi

# Activate virtual environment
echo ""
echo "🔌 Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
echo ""
echo "⬆️  Upgrading pip..."
$PYTHON_CMD -m pip install --upgrade pip

# Install dependencies
echo ""
echo "📚 Installing dependencies from requirements.txt..."
pip install -r requirements.txt

# Create API keys file if it doesn't exist
echo ""
echo "🔑 Setting up API key management..."
if [ ! -f "api_keys.py" ]; then
    cp api_keys_template.py api_keys.py
    echo "✓ Created api_keys.py from template"
    echo "⚠️  Please edit api_keys.py and add your actual API keys"
else
    echo "api_keys.py already exists"
fi

# Create data directory
echo ""
echo "📁 Creating data directory..."
if [ ! -d ".data" ]; then
    mkdir -p .data
    echo "✓ Created .data directory"
else
    echo ".data directory already exists"
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Edit api_keys.py with your actual API keys"
echo "2. Activate the environment: source .venv/bin/activate"
echo "3. Open any notebook and start exploring!"
