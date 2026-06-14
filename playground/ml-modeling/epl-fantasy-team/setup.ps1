$ErrorActionPreference = 'Stop'
$venvPath = Join-Path $PSScriptRoot '.venv'

if (-not (Test-Path $venvPath)) {
    Write-Host "Creating virtual environment..."
    python -m venv $venvPath
}

& "$venvPath\Scripts\Activate.ps1"

Write-Host "Installing dependencies..."
pip install -r (Join-Path $PSScriptRoot 'requirements.txt')

Write-Host ""
Write-Host "Setup complete. Activate with: .venv\Scripts\Activate.ps1"
Write-Host ""
Write-Host "NOTE: this script expects the Fantasy Premier League dataset at:"
Write-Host "  ./Fantasy-Premier-League/data/2023-24/"
Write-Host "Clone it from: https://github.com/vaastav/Fantasy-Premier-League"
Write-Host ""
Write-Host "Delete fantasy_football.db to force a full re-ingest after updating the dataset."
