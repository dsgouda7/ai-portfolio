# Generic Python ML Project Setup Script (Windows PowerShell)
# Copy this file to your project directory and run: .\setup.ps1

$ErrorActionPreference = "Stop"

Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "  Python ML Project Setup" -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host ""

# Check Python version
Write-Host "→ Checking Python version..." -ForegroundColor Yellow
try {
    $pythonVersion = & python --version 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw "Python not found"
    }
} catch {
    Write-Host "❌ ERROR: Python 3 is not installed" -ForegroundColor Red
    Write-Host "   Install from: https://www.python.org/downloads/" -ForegroundColor Red
    exit 1
}

$version = $pythonVersion -replace "Python ", ""
$major, $minor = $version.Split(".")[0..1]

if ([int]$major -lt 3 -or ([int]$major -eq 3 -and [int]$minor -lt 8)) {
    Write-Host "❌ ERROR: Python 3.8+ required (found $version)" -ForegroundColor Red
    exit 1
}

Write-Host "✓ Python $version detected" -ForegroundColor Green
Write-Host ""

# Create virtual environment
Write-Host "→ Creating virtual environment..." -ForegroundColor Yellow
if (Test-Path "venv") {
    Write-Host "⚠️  Virtual environment already exists" -ForegroundColor Yellow
    $response = Read-Host "   Delete and recreate? (y/N)"
    if ($response -eq "y" -or $response -eq "Y") {
        Remove-Item -Recurse -Force venv
        & python -m venv venv
        Write-Host "✓ Virtual environment recreated" -ForegroundColor Green
    } else {
        Write-Host "✓ Using existing virtual environment" -ForegroundColor Green
    }
} else {
    & python -m venv venv
    Write-Host "✓ Virtual environment created" -ForegroundColor Green
}
Write-Host ""

# Activate virtual environment
Write-Host "→ Activating virtual environment..." -ForegroundColor Yellow
try {
    & .\venv\Scripts\Activate.ps1
    Write-Host "✓ Virtual environment activated" -ForegroundColor Green
} catch {
    Write-Host "⚠️  Activation failed - you may need to run:" -ForegroundColor Yellow
    Write-Host "   Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned" -ForegroundColor Yellow
    Write-Host "   Then re-run this script" -ForegroundColor Yellow
    exit 1
}
Write-Host ""

# Upgrade pip
Write-Host "→ Upgrading pip..." -ForegroundColor Yellow
& python -m pip install --quiet --upgrade pip
$pipVersion = & pip --version
Write-Host "✓ pip upgraded to $($pipVersion.Split()[1])" -ForegroundColor Green
Write-Host ""

# Install dependencies
if (Test-Path "requirements.txt") {
    Write-Host "→ Installing dependencies from requirements.txt..." -ForegroundColor Yellow
    & pip install --quiet -r requirements.txt
    Write-Host "✓ Dependencies installed successfully" -ForegroundColor Green
} else {
    Write-Host "⚠️  No requirements.txt found - skipping dependency installation" -ForegroundColor Yellow
}
Write-Host ""

# Verify installation
Write-Host "→ Verifying installation..." -ForegroundColor Yellow
& python -c @"
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
"@
Write-Host ""

# Final instructions
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "  ✅ Setup Complete!" -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:"
Write-Host "  1. Activate venv:  .\venv\Scripts\Activate.ps1"
Write-Host "  2. Run tests:      make test  (or pytest)"
Write-Host "  3. Train model:    make train  (or python main.py)"
Write-Host ""
Write-Host "To deactivate venv later: deactivate"
Write-Host ""
