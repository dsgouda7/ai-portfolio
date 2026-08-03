$ErrorActionPreference = "Stop"
$TrackDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvDir = Join-Path $TrackDir ".venv"

if (-not (Test-Path $VenvDir)) {
    python -m venv $VenvDir
}

$Python = Join-Path $VenvDir "Scripts\python.exe"
& $Python -m pip install --upgrade pip
& $Python -m pip install -r (Join-Path $TrackDir "requirements.txt")
& $Python -m ipykernel install --user --name agentic-ai --display-name "Python (agentic-ai)"

Write-Output "Agentic AI environment ready: $Python"
Write-Output "Validate with: $Python $TrackDir\scripts\validate_track.py"
