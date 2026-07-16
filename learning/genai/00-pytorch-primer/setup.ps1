#!/usr/bin/env pwsh
# Creates a local .venv and installs notebook dependencies for mnist-cnn-keras-vs-pytorch.ipynb

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

Write-Host "Creating virtual environment at $ScriptDir\.venv ..."
python -m venv "$ScriptDir\.venv"

if (-not $?) {
    Write-Error "Failed to create virtual environment. Ensure Python 3.8+ is on PATH."
    exit 1
}

Write-Host "Installing dependencies from requirements.txt ..."
& "$ScriptDir\.venv\Scripts\pip.exe" install --upgrade pip --quiet
& "$ScriptDir\.venv\Scripts\pip.exe" install -r "$ScriptDir\requirements.txt"

Write-Host "Registering Jupyter kernel as 'pytorch-primer' ..."
& "$ScriptDir\.venv\Scripts\python.exe" -m ipykernel install --user `
    --name "pytorch-primer" `
    --display-name "Python (pytorch-primer)"

Write-Host ""
Write-Host "Done. In VS Code, select the 'Python (pytorch-primer)' kernel (or the .venv interpreter) for the notebook."
