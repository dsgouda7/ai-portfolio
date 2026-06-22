param(
    [switch]$EnableModelPull,
    # Models to pull — default matches deployments/local/.env.example
    [string[]]$Models = @("llama3.2:3b")
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

# ── Python virtual environment ────────────────────────────────────────────────
if (-not (Test-Path ".venv")) {
    python -m venv .venv
}

$venvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"

& $venvPython -m pip install --upgrade pip

# Core library + optional evaluation extras (matplotlib, pillow)
& $venvPython -m pip install -e ".[evaluation]"

# Full runtime/benchmark deps not declared in pyproject.toml
# (chromadb, sentence-transformers, litellm, httpx, fastapi, uvicorn, etc.)
if (Test-Path "requirements.txt") {
    & $venvPython -m pip install -r requirements.txt
}

# ── Docker check (needed for local deployment) ────────────────────────────────
try {
    $null = docker info 2>&1
    Write-Host "Docker: OK"
} catch {
    Write-Warning "Docker is not running or not installed. Install Docker Desktop to use 'deployments\deploy.ps1 local'."
}

# ── Ollama model pulls ────────────────────────────────────────────────────────
if ($EnableModelPull) {
    $ollama = Get-Command ollama -ErrorAction SilentlyContinue
    if ($null -eq $ollama) {
        Write-Warning "Ollama CLI not found. Install from https://ollama.com and re-run with -EnableModelPull."
    } else {
        foreach ($model in $Models) {
            Write-Host "Pulling Ollama model: $model"
            ollama pull $model
        }
    }
}

# ── Local deployment .env init ────────────────────────────────────────────────
$localEnv     = Join-Path $PSScriptRoot "deployments\local\.env"
$localEnvEx   = Join-Path $PSScriptRoot "deployments\local\.env.example"
if (-not (Test-Path $localEnv) -and (Test-Path $localEnvEx)) {
    Copy-Item $localEnvEx $localEnv
    Write-Host "Created deployments\local\.env from .env.example — review model names if needed."
}

Write-Host ""
Write-Host "Setup complete."
Write-Host "  Activate venv : .venv\Scripts\Activate.ps1"
Write-Host "  Local deploy  : .\deployments\deploy.ps1 local up"
if (-not $EnableModelPull) {
    Write-Host "  Pull models   : .\setup.ps1 -EnableModelPull  (default: $($Models -join ', '))"
}
