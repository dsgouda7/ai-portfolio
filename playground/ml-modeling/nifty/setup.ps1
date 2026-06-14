#!/usr/bin/env pwsh

# Check Python
try {
    $pythonVersion = python --version 2>&1
    Write-Host "Found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "Python not found. Install Python 3.8+ first." -ForegroundColor Red
    exit 1
}

# Create venv
if (Test-Path ".venv") {
    Write-Host ".venv already exists, skipping creation." -ForegroundColor Yellow
} else {
    python -m venv .venv
    Write-Host "Virtual environment created." -ForegroundColor Green
}

# Activate and install
& .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip --quiet

pip install -r requirements.txt --quiet
# kaggle credentials
if (-not (Test-Path "kaggle.json")) {
    Write-Host ""
    Write-Host "kaggle.json not found — needed to download the dataset." -ForegroundColor Yellow
    Write-Host "Get your API token from https://www.kaggle.com/settings (Account -> API -> Create New Token)"
    Write-Host "Then either:"
    Write-Host "  1. Copy ~/.kaggle/kaggle.json here, or"
    Write-Host "  2. Create kaggle.json manually with: { \"username\": \"you\", \"key\": \"your_key\" }"
} else {
    Write-Host "kaggle.json found" -ForegroundColor Green
}
Write-Host "Done. Activate with: .\.venv\Scripts\Activate.ps1" -ForegroundColor Cyan
