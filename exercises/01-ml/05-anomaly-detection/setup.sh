#!/bin/bash
# Generic Python ML Project Setup Script (Unix/macOS/WSL)
# Copy this file to your project directory and run: ./setup.sh

set -e  # Exit on error

echo "========================================="
echo "  FraudShield Setup"
echo "========================================="
echo ""

# Check Python version
echo "→ Checking Python version..."
if ! command -v python3 &> /dev/null; then
    echo "❌ ERROR: Python 3 is not installed"
    echo "   Install from: https://www.python.org/downloads/"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || { [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 8 ]; }; then
    echo "❌ ERROR: Python 3.8+ required (found $PYTHON_VERSION)"
    exit 1
fi

echo "✓ Python $PYTHON_VERSION detected"
echo ""

# Create virtual environment
echo "→ Creating virtual environment..."
if [ -d "venv" ]; then
    echo "⚠️  Virtual environment already exists"
    read -p "   Delete and recreate? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf venv
        python3 -m venv venv
        echo "✓ Virtual environment recreated"
    else
        echo "✓ Using existing virtual environment"
    fi
else
    python3 -m venv venv
    echo "✓ Virtual environment created"
fi
echo ""

# Activate virtual environment
echo "→ Activating virtual environment..."
source venv/bin/activate
echo "✓ Virtual environment activated"
echo ""

# Upgrade pip
echo "→ Upgrading pip..."
pip install --quiet --upgrade pip
echo "✓ pip upgraded to $(pip --version | cut -d' ' -f2)"
echo ""

# Install dependencies
if [ -f "requirements.txt" ]; then
    echo "→ Installing dependencies from requirements.txt..."
    pip install --quiet -r requirements.txt
    echo "✓ Dependencies installed successfully"
else
    echo "⚠️  No requirements.txt found - skipping dependency installation"
fi
echo ""

# Verify installation
echo "→ Verifying installation..."
python -c "
import sys
try:
    import numpy
    import pandas
    import sklearn
    print('✓ Core libraries imported successfully')
    print('  - NumPy:', numpy.__version__)
    print('  - pandas:', pandas.__version__)
    print('  - scikit-learn:', sklearn.__version__)
except ImportError as e:
    print('⚠️  Some libraries could not be imported:', e)
    sys.exit(1)
"
echo ""

# Final instructions
echo "========================================="
echo "  ✅ Setup Complete!"
echo "========================================="
echo ""
echo "Next steps:"
echo "  1. Activate venv:  source venv/bin/activate"
echo "  2. Run tests:      make test"
echo "  3. Train model:    make train"
echo ""
echo "To deactivate venv later: deactivate"
echo ""
