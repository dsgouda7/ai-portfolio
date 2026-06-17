param(
    [switch]$EnableModelPull,
    [string[]]$Models = @("phi4:mini", "qwen3")
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Test-Path ".venv")) {
    python -m venv .venv
}

$venvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"

& $venvPython -m pip install --upgrade pip
& $venvPython -m pip install -e ".[evaluation]"

if ($EnableModelPull) {
    $ollama = Get-Command ollama -ErrorAction SilentlyContinue
    if ($null -eq $ollama) {
        Write-Warning "Ollama CLI was not found. Install Ollama or run setup without -EnableModelPull."
        exit 0
    }

    foreach ($model in $Models) {
        Write-Host "Pulling Ollama model: $model"
        ollama pull $model
    }
}

Write-Host "Setup complete."
Write-Host "Use .venv\Scripts\Activate.ps1 to activate the environment."
if (-not $EnableModelPull) {
    Write-Host "Model downloads skipped (opt-in). Re-run with -EnableModelPull to pull Ollama models."
}
