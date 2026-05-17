#!/bin/bash

# Setup script for Azure AI Exploration
# This script creates a virtual environment and installs dependencies

echo "🚀 Setting up Azure AI Exploration environment..."

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

# Create config file if it doesn't exist
echo ""
echo "🔑 Setting up Azure configuration..."
if [ ! -f "config.txt" ]; then
    cp config.txt.template config.txt
    echo "✓ Created config.txt from template"
    echo "⚠️  Please edit config.txt and add your Azure credentials"
else
    echo "config.txt already exists"
fi

# Success message
echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Edit config.txt with your Azure credentials"
echo "2. Activate the virtual environment: source .venv/bin/activate"
echo "3. Launch Jupyter: jupyter notebook"
echo "4. Open 01-azure-ai-integration.ipynb"
