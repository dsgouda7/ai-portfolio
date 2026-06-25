# IBM Applied Data Science Professional Certificate Capstone Project - Space Launch Environment Setup Script (Windows)
# This script creates a virtual environment and installs all required dependencies

$ErrorActionPreference = "Stop"

Write-Host "Setting up IBM Applied Data Science Professional Certificate Capstone Project - Space Launch environment..." -ForegroundColor Cyan

# Get the script directory
$SCRIPT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $SCRIPT_DIR

# Check if Python is installed
try {
    $pythonVersion = python --version
    Write-Host "Found Python: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "Error: Python is not installed or not in PATH" -ForegroundColor Red
    exit 1
}

# Create virtual environment if it doesn't exist
if (-not (Test-Path ".venv")) {
    Write-Host "Creating virtual environment..." -ForegroundColor Yellow
    python -m venv .venv
} else {
    Write-Host "Virtual environment already exists" -ForegroundColor Green
}

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
& ".\.venv\Scripts\Activate.ps1"

# Upgrade pip
Write-Host "Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip

# Install requirements
Write-Host "Installing dependencies from requirements.txt..." -ForegroundColor Yellow
pip install -r requirements.txt

Write-Host "`nSetup complete!" -ForegroundColor Green
Write-Host "To activate the environment, run: .\.venv\Scripts\Activate.ps1" -ForegroundColor Cyan
Write-Host "To deactivate, run: deactivate" -ForegroundColor Cyan
