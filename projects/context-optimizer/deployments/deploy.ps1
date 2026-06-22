# deploy.ps1 — unified entry point for all deployment targets (Windows).
#
# Usage:
#   .\deployments\deploy.ps1 [target] [action]
#
# Targets:
#   local    Docker containers + host-installed Ollama (default)
#   azure    Docker containers + Azure Key Vault secrets
#
# Actions:
#   up       Build images and start all services (default)
#   down     Stop and remove containers
#   build    Build images without starting
#   logs     Follow service logs
#   ps       Show running containers
#
# Examples:
#   .\deployments\deploy.ps1 local
#   .\deployments\deploy.ps1 local down
#   .\deployments\deploy.ps1 azure up

param(
  [string]$Target = "local",
  [string]$Action = "up"
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$ComposeFile = Join-Path $ScriptDir "$Target\docker-compose.yml"

if (-not (Test-Path $ComposeFile)) {
  $available = (Get-ChildItem $ScriptDir -Directory).Name -join ", "
  Write-Error "Unknown target '$Target' (no $ComposeFile found). Available: $available"
  exit 1
}

# ── Pre-flight checks ──────────────────────────────────────────────────────

function Test-Docker {
  docker info 2>$null | Out-Null
  return $LASTEXITCODE -eq 0
}

function Start-DockerIfNeeded {
  if (Test-Docker) { return }

  # Locate Docker Desktop executable
  $candidates = @(
    "$env:ProgramFiles\Docker\Docker\Docker Desktop.exe",
    "${env:ProgramFiles(x86)}\Docker\Docker\Docker Desktop.exe",
    "$env:LOCALAPPDATA\Docker\Docker Desktop.exe"
  )
  $dockerExe = $candidates | Where-Object { Test-Path $_ } | Select-Object -First 1

  if (-not $dockerExe) {
    Write-Error "Docker Desktop is not running and could not be found. Install it from https://www.docker.com/products/docker-desktop/"
    exit 1
  }

  Write-Host "  Starting Docker Desktop ($dockerExe) …"
  Start-Process $dockerExe

  $timeout  = 120  # seconds
  $interval = 5
  $elapsed  = 0
  Write-Host "  Waiting for Docker daemon (up to ${timeout}s) …" -NoNewline
  while (-not (Test-Docker)) {
    if ($elapsed -ge $timeout) {
      Write-Host ""
      Write-Error "Docker daemon did not become ready within ${timeout}s. Try starting Docker Desktop manually."
      exit 1
    }
    Start-Sleep $interval
    $elapsed += $interval
    Write-Host "." -NoNewline
  }
  Write-Host " ready (${elapsed}s)"
}

function Test-LocalPrerequisites {
  Write-Host "Target: local"

  $ollamaUrl = if ($env:OLLAMA_BASE_URL) { $env:OLLAMA_BASE_URL } else { "http://localhost:11434" }

  try {
    Invoke-RestMethod "$ollamaUrl/api/tags" -TimeoutSec 3 | Out-Null
    Write-Host "  Ollama: OK ($ollamaUrl)"

    # Check required models
    $tags = (Invoke-RestMethod "$ollamaUrl/api/tags").models.name
    foreach ($model in @($env:COMPRESSOR_MODEL ?? "llama3.2:3b", $env:REASONING_MODEL ?? "llama3.2:3b") | Select-Object -Unique) {
      if ($tags -contains $model) {
        Write-Host "  Model $model`: OK"
      } else {
        Write-Warning "  Model '$model' not found — run: ollama pull $model"
      }
    }
  } catch {
    Write-Warning "  Ollama not reachable at $ollamaUrl — start with: ollama serve"
  }

  # Create .env from example if missing
  $envFile = Join-Path $ScriptDir "local\.env"
  if (-not (Test-Path $envFile)) {
    Copy-Item (Join-Path $ScriptDir "local\.env.example") $envFile
    Write-Host "  Created $envFile from .env.example"
  }
}

function Test-AzurePrerequisites {
  Write-Host "Target: azure"
  $envFile = Join-Path $ScriptDir "azure\.env.azure"
  if (-not (Test-Path $envFile)) {
    Write-Error @"
$envFile not found.
  Run: Copy-Item deployments\azure\.env.example deployments\azure\.env.azure
  Then fill in AZURE_TENANT_ID, AZURE_CLIENT_ID, AZURE_CLIENT_SECRET, AZURE_KEYVAULT_URI
"@
    exit 1
  }
  Write-Host "  .env.azure: found"
}

# ── Main ───────────────────────────────────────────────────────────────────

Start-DockerIfNeeded

switch ($Target) {
  "local" { Test-LocalPrerequisites }
  "azure" { Test-AzurePrerequisites }
}

$compose = "docker compose -f `"$ComposeFile`""

switch ($Action) {
  "up" {
    Write-Host "Starting $Target deployment..."
    Invoke-Expression "$compose up --build -d"
    Write-Host ""
    Write-Host "Services running. Endpoints:"
    Write-Host "  Ingestion:         http://localhost:8001"
    Write-Host "  Embedding:         http://localhost:8002"
    Write-Host "  Vector Store:      http://localhost:8003"
    Write-Host "  Reasoning Gateway: http://localhost:8080"
  }
  "down"  { Invoke-Expression "$compose down" }
  "build" { Invoke-Expression "$compose build" }
  "logs"  { Invoke-Expression "$compose logs -f" }
  "ps"    { Invoke-Expression "$compose ps" }
  default {
    Write-Error "Unknown action '$Action'. Use: up | down | build | logs | ps"
    exit 1
  }
}
