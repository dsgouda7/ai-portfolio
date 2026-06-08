# Phase 3: RAG Server - Setup Script
# PowerShell version

Write-Host "Setting up Phase 3: RAG Server" -ForegroundColor Cyan
Write-Host "===============================" -ForegroundColor Cyan
Write-Host ""

# Create virtual environment
if (-not (Test-Path "venv")) {
    python -m venv venv
    Write-Host "Virtual environment created" -ForegroundColor Green
}

# Activate virtual environment
& "venv\Scripts\Activate.ps1"

# Upgrade pip
python -m pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

# Install shared utilities
pip install -e ..\shared

Write-Host ""
Write-Host "Phase 3 setup complete" -ForegroundColor Green
Write-Host "To activate: venv\Scripts\Activate.ps1" -ForegroundColor Yellow
Write-Host "To run: uvicorn src.server:app --reload" -ForegroundColor Yellow
