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
Write-Host "Run: python train.py  (then python web.py for the pitch UI)"
Write-Host "Delete fantasy_football.db to force a full re-ingest after pulling new data."
