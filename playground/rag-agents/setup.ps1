#!/usr/bin/env pwsh

# Setup script for AI Exploration Notebooks
# This script creates a virtual environment and installs dependencies

Write-Host "🚀 Setting up AI Exploration Notebooks environment..." -ForegroundColor Cyan

# Check Python installation
try {
    $pythonVersion = python --version 2>&1
    Write-Host "✓ Found Python: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "✗ Python not found. Please install Python 3.8+ first." -ForegroundColor Red
    exit 1
}

# Create virtual environment
Write-Host "`n📦 Creating virtual environment..." -ForegroundColor Cyan
if (Test-Path ".venv") {
    Write-Host "Virtual environment already exists." -ForegroundColor Yellow
} else {
    python -m venv .venv
    Write-Host "✓ Virtual environment created" -ForegroundColor Green
}

# Activate virtual environment
Write-Host "`n🔌 Activating virtual environment..." -ForegroundColor Cyan
& .\.venv\Scripts\Activate.ps1

# Upgrade pip
Write-Host "`n⬆️  Upgrading pip..." -ForegroundColor Cyan
python -m pip install --upgrade pip

# Install dependencies
Write-Host "`n📚 Installing dependencies from requirements.txt..." -ForegroundColor Cyan
pip install -r requirements.txt

# Create API keys file if it doesn't exist
Write-Host "`n🔑 Setting up API key management..." -ForegroundColor Cyan
if (-not (Test-Path "api_keys.py")) {
    Copy-Item "api_keys_template.py" "api_keys.py"
    Write-Host "✓ Created api_keys.py from template" -ForegroundColor Green
    Write-Host "⚠️  Please edit api_keys.py and add your actual API keys" -ForegroundColor Yellow
} else {
    Write-Host "api_keys.py already exists" -ForegroundColor Yellow
}

# Create data directory
Write-Host "`n📁 Creating data directory..." -ForegroundColor Cyan
if (-not (Test-Path ".data")) {
    New-Item -ItemType Directory -Path ".data" | Out-Null
    Write-Host "✓ Created .data directory" -ForegroundColor Green
} else {
    Write-Host ".data directory already exists" -ForegroundColor Yellow
}

Write-Host "`n✅ Setup complete!" -ForegroundColor Green
Write-Host "`nNext steps:" -ForegroundColor Cyan
Write-Host "1. Edit api_keys.py with your actual API keys" -ForegroundColor White
Write-Host "2. Activate the environment: .\.venv\Scripts\Activate.ps1" -ForegroundColor White
Write-Host "3. Open any notebook and start exploring!" -ForegroundColor White
