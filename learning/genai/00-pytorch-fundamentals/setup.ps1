#!/usr/bin/env pwsh
# Creates the local environment for 01-keras-to-pytorch-antarctic-field-guide.ipynb.

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvPath = Join-Path $ScriptDir ".venv"

if (-not (Test-Path $VenvPath)) {
    Write-Host "Creating virtual environment at $VenvPath ..."
    python -m venv $VenvPath
    if (-not $?) {
        Write-Error "Failed to create the virtual environment. Ensure Python 3.10+ is on PATH."
        exit 1
    }
}

Write-Host "Installing PyTorch foundations dependencies ..."
& "$VenvPath\Scripts\pip.exe" install --upgrade pip setuptools wheel --quiet
& "$VenvPath\Scripts\pip.exe" install -r "$ScriptDir\requirements.txt"

Write-Host "Registering Jupyter kernel as 'pytorch-foundations' ..."
& "$VenvPath\Scripts\python.exe" -m ipykernel install --user `
    --name "pytorch-foundations" `
    --display-name "Python (pytorch-foundations)"

Write-Host "Done. Select the 'Python (pytorch-foundations)' kernel in VS Code."
