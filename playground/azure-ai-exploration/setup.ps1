#!/usr/bin/env pwsh

# Setup script for Azure AI Exploration
# This script creates a virtual environment and installs dependencies

Write-Host "🚀 Setting up Azure AI Exploration environment..." -ForegroundColor Cyan

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

# Create config file if it doesn't exist
Write-Host "`n🔑 Setting up Azure configuration..." -ForegroundColor Cyan
if (-not (Test-Path "config.txt")) {
    Copy-Item "config.txt.template" "config.txt"
    Write-Host "✓ Created config.txt from template" -ForegroundColor Green
    Write-Host "⚠️  Please edit config.txt and add your Azure credentials" -ForegroundColor Yellow
} else {
    Write-Host "config.txt already exists" -ForegroundColor Yellow
}

# Success message
Write-Host "`n✅ Setup complete!" -ForegroundColor Green
Write-Host "`nNext steps:" -ForegroundColor Cyan
Write-Host "1. Edit config.txt with your Azure credentials" -ForegroundColor White
Write-Host "2. Activate the virtual environment: .\.venv\Scripts\Activate.ps1" -ForegroundColor White
Write-Host "3. Launch Jupyter: jupyter notebook" -ForegroundColor White
Write-Host "4. Open 01-azure-ai-integration.ipynb" -ForegroundColor White
