$ErrorActionPreference = 'Stop'
$venvPath = Join-Path $PSScriptRoot '.venv'

if (-not (Test-Path $venvPath)) {
    Write-Host "Creating virtual environment..."
    python -m venv $venvPath
}

& "$venvPath\Scripts\Activate.ps1"

Write-Host "Installing dependencies..."
pip install -r (Join-Path $PSScriptRoot 'requirements.txt')

$datasetPath = Join-Path $PSScriptRoot 'Fantasy-Premier-League'
if (-not (Test-Path $datasetPath)) {
    Write-Host "Cloning FPL dataset (vaastav archive)..."
    git clone --depth=1 https://github.com/vaastav/Fantasy-Premier-League.git $datasetPath
} else {
    Write-Host "FPL dataset found. Pulling latest data..."
    git -C $datasetPath pull --ff-only
}

Write-Host ""
Write-Host "Setup complete. Activate with: .venv\Scripts\Activate.ps1"
Write-Host "Train models  : python train\train.py"
Write-Host "Pitch UI      : python fpl-generator\web.py"
Write-Host "CLI team pick : python fpl-generator\team_generator.py"
Write-Host "Backtest      : python train\backtest.py"
Write-Host "TM data status: python transfer_values.py --status"
Write-Host "Delete fantasy_football.db to force a full re-ingest after pulling new data."
