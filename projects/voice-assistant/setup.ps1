#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Set up the voice-assistant for local (Ollama) operation.

.DESCRIPTION
    1. Checks that Ollama is installed and running.
    2. Pulls the required LLM model(s).
    3. Creates a Python virtual environment and installs dependencies.
    4. Copies .env.example to .env if it doesn't already exist.

.NOTES
    Run from the voice-assistant directory:
        .\setup.ps1
    Or with a specific model:
        .\setup.ps1 -LlmModel llama3.2:3b
#>

param(
    [string]$LlmModel = "mistral",
    [string]$PythonExe = "python"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Write-Step([string]$msg) {
    Write-Host "`n==> $msg" -ForegroundColor Cyan
}

function Assert-Command([string]$cmd) {
    if (-not (Get-Command $cmd -ErrorAction SilentlyContinue)) {
        Write-Error "'$cmd' not found. Please install it and retry."
        exit 1
    }
}

# ── 1. Check prerequisites ────────────────────────────────────────────────────
Write-Step "Checking prerequisites"
Assert-Command "ollama"
Assert-Command $PythonExe

# ── 2. Start Ollama if not already running ────────────────────────────────────
Write-Step "Ensuring Ollama server is running"
$ollamaRunning = $false
try {
    $null = Invoke-RestMethod -Uri "http://localhost:11434/api/tags" -TimeoutSec 3
    $ollamaRunning = $true
    Write-Host "Ollama is already running." -ForegroundColor Green
} catch {
    Write-Host "Starting Ollama in the background ..." -ForegroundColor Yellow
    Start-Process "ollama" -ArgumentList "serve" -WindowStyle Hidden
    Start-Sleep -Seconds 3
}

# ── 3. Pull LLM model(s) ──────────────────────────────────────────────────────
Write-Step "Pulling LLM model: $LlmModel"
Write-Host "This may take a few minutes on first run (model ~4 GB for mistral)."
ollama pull $LlmModel

# Optional: pull a lighter fallback model
Write-Step "Pulling lightweight fallback model: llama3.2:3b"
ollama pull "llama3.2:3b"

Write-Host "`nPulled models:" -ForegroundColor Green
ollama list

# ── 4. Python virtual environment ─────────────────────────────────────────────
Write-Step "Creating Python virtual environment (.venv)"
if (-not (Test-Path ".venv")) {
    & $PythonExe -m venv .venv
}

$pip = ".\.venv\Scripts\pip.exe"
Write-Step "Installing Python dependencies"
& $pip install --upgrade pip --quiet
& $pip install -r requirements.txt

# ── 5. Environment file ───────────────────────────────────────────────────────
Write-Step "Setting up .env"
if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Host ".env created from .env.example — edit it to configure providers." -ForegroundColor Yellow
} else {
    Write-Host ".env already exists, skipping." -ForegroundColor Green
}

# ── Done ──────────────────────────────────────────────────────────────────────
Write-Host "`n✓ Setup complete." -ForegroundColor Green
Write-Host "  Start the server with:  .\.venv\Scripts\python.exe server.py" -ForegroundColor White
Write-Host "  Change the LLM model:   set OLLAMA_LLM_MODEL=<model> in .env" -ForegroundColor White
Write-Host "  Switch to Azure:        set LLM_PROVIDER=azure in .env" -ForegroundColor White
