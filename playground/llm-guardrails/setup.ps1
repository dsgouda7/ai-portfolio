# setup.ps1 - Install dependencies and pull the Ollama model for LLM_Guardrails
# Usage: pwsh -File setup.ps1

$Model = "llama3.2:1b"

Write-Host "Installing Python dependencies..." -ForegroundColor Cyan
pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) {
    Write-Error "pip install failed."
    exit 1
}

Write-Host ""
Write-Host "Checking Ollama..." -ForegroundColor Cyan
if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
    Write-Warning "Ollama is not installed or not in PATH."
    Write-Host "Download it from: https://ollama.com/download" -ForegroundColor Yellow
    exit 1
}

Write-Host "Pulling model: $Model (~1.3 GB)..." -ForegroundColor Cyan
ollama pull $Model
if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to pull model '$Model'. Is Ollama running?"
    exit 1
}

Write-Host ""
Write-Host "Done! Model '$Model' is ready." -ForegroundColor Green
Write-Host "Ensure Ollama is running ('ollama serve') before executing the notebook."
