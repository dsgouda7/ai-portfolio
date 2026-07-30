#!/usr/bin/env pwsh
# Creates a local .venv and installs dependencies for the RNN notebooks.

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

Write-Host "Creating virtual environment at $ScriptDir\.venv ..."
python -m venv "$ScriptDir\.venv"

if (-not $?) {
    Write-Error "Failed to create virtual environment. Ensure Python 3.8+ is on PATH."
    exit 1
}

Write-Host "Installing build tooling ..."
& "$ScriptDir\.venv\Scripts\pip.exe" install --upgrade pip setuptools wheel --quiet

Write-Host "Installing dependencies from requirements.txt ..."
& "$ScriptDir\.venv\Scripts\pip.exe" install -r "$ScriptDir\requirements.txt"

Write-Host "Installing Jupyter kernel support (ipykernel, nbconvert) ..."
& "$ScriptDir\.venv\Scripts\pip.exe" install --quiet ipykernel nbconvert

Write-Host "Registering Jupyter kernel as 'genai-rnns' ..."
& "$ScriptDir\.venv\Scripts\python.exe" -m ipykernel install --user `
    --name "genai-rnns" `
    --display-name "Python (genai-rnns)"

Write-Host ""
Write-Host "Done. In VS Code, select the 'Python (genai-rnns)' kernel (or the .venv interpreter) for the notebook."
