# setup.ps1 - Install dependencies and pull Ollama models for LLM_Guardrails
# Usage: pwsh -File setup.ps1
#
# Models pulled:
#   mistral      -- main agent model          (~4 GB)
#   llama3.2:1b  -- lightweight safety judge  (~1.3 GB)

param(
    [string]$FullModel  = "mistral",
    [string]$LightModel = "llama3.2:1b"
)

$ErrorActionPreference = "Stop"

function Write-Step([string]$msg) { Write-Host "`n==> $msg" -ForegroundColor Cyan }

# ── 1. Python dependencies ───────────────────────────────────────────────────
Write-Step "Installing Python dependencies"
pip install -r "$PSScriptRoot\requirements.txt"

# ── 2. Check Ollama ──────────────────────────────────────────────────────────
Write-Step "Checking Ollama installation"
if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
    Write-Warning "Ollama is not installed or not in PATH."
    Write-Host "Download it from: https://ollama.com/download" -ForegroundColor Yellow
    exit 1
}

# Ensure Ollama server is running
try {
    $null = Invoke-RestMethod -Uri "http://localhost:11434/api/tags" -TimeoutSec 3
    Write-Host "Ollama is already running." -ForegroundColor Green
} catch {
    Write-Host "Starting Ollama in the background ..." -ForegroundColor Yellow
    Start-Process "ollama" -ArgumentList "serve" -WindowStyle Hidden
    Start-Sleep -Seconds 3
}

# ── 3. Pull models ────────────────────────────────────────────────────────────
Write-Step "Pulling main agent model: $FullModel  (~4 GB on first pull)"
ollama pull $FullModel

Write-Step "Pulling lightweight safety-judge model: $LightModel  (~1.3 GB on first pull)"
ollama pull $LightModel

Write-Host ""
Write-Host "Pulled models:" -ForegroundColor Green
ollama list

Write-Host ""
Write-Host "Setup complete." -ForegroundColor Green
Write-Host "  Open LLM_Guardrails.ipynb in VS Code and run all cells."
Write-Host "  To use Azure AI Foundry instead: set LLM_PROVIDER=azure in your environment."
