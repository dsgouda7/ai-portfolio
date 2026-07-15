#!/usr/bin/env pwsh
# Creates a local .venv and installs the dependencies for ibm_genai_translator.py

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvDir = Join-Path $ScriptDir ".venv"
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"

$pythonCmd = if (Get-Command python -ErrorAction SilentlyContinue) { "python" }
             elseif (Get-Command python3 -ErrorAction SilentlyContinue) { "python3" }
             else { $null }

if (-not $pythonCmd) {
    Write-Host "Error: Python was not found on PATH. Install Python 3.9+ and re-run." -ForegroundColor Red
    exit 1
}

if (Test-Path $VenvPython) {
    Write-Host "Reusing existing virtual environment at $VenvDir"
} else {
    Write-Host "Creating virtual environment at $VenvDir ..."
    & $pythonCmd -m venv $VenvDir
}

Write-Host "Upgrading pip..."
& $VenvPython -m pip install --upgrade pip -q

Write-Host "Installing dependencies from requirements.txt..."
& $VenvPython -m pip install -r (Join-Path $ScriptDir "requirements.txt")

Write-Host "`nSetup complete." -ForegroundColor Green
Write-Host "Run the app with: $VenvPython ibm_genai_translator.py"
