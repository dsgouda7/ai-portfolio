# Voice Assistant - Run Script (PowerShell)
# Builds Docker image, runs container, and opens browser

param(
    [switch]$NoBuild,
    [switch]$NoBrowser,
    [string]$Port = "5000"
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ImageName = "voice-assistant"
$ContainerName = "voice-assistant-app"

# Color output functions
function Write-Step { param($msg) Write-Host "`n▶ $msg" -ForegroundColor Cyan }
function Write-Ok   { param($msg) Write-Host "  ✓ $msg" -ForegroundColor Green }
function Write-Warn { param($msg) Write-Host "  ⚠ $msg" -ForegroundColor Yellow }
function Write-Fail { param($msg) Write-Host "  ✗ $msg" -ForegroundColor Red; exit 1 }

Write-Host "`n╔════════════════════════════════════════╗" -ForegroundColor Magenta
Write-Host "║   Voice Assistant - Starting App      ║" -ForegroundColor Magenta
Write-Host "╚════════════════════════════════════════╝`n" -ForegroundColor Magenta

# Check if Docker is running
Write-Step "Checking Docker..."
try {
    $null = docker info 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Fail "Docker daemon is not running. Please start Docker Desktop."
    }
    Write-Ok "Docker is running"
} catch {
    Write-Fail "Cannot connect to Docker. Please ensure Docker Desktop is running."
}

# Stop and remove existing container
Write-Step "Cleaning up existing containers..."
try {
    $existingContainer = docker ps -a -q -f name=$ContainerName 2>$null
    if ($existingContainer) {
        Write-Host "  Stopping existing container..."
        docker stop $ContainerName 2>&1 | Out-Null
        Write-Host "  Removing existing container..."
        docker rm $ContainerName 2>&1 | Out-Null
        Write-Ok "Cleaned up existing container"
    } else {
        Write-Ok "No existing container found"
    }
} catch {
    Write-Warn "Could not clean up existing container"
}

# Build Docker image
if (-not $NoBuild) {
    Write-Step "Building Docker image..."
    Push-Location $ScriptDir
    try {
        Write-Host "  This may take a few minutes on first build..."
        docker build -t $ImageName . 2>&1 | ForEach-Object {
            if ($_ -match "^Step \d+/\d+|^Successfully") {
                Write-Host "  $_" -ForegroundColor DarkGray
            }
        }
        if ($LASTEXITCODE -ne 0) {
            Write-Fail "Docker build failed"
        }
        Write-Ok "Docker image built successfully"
    } finally {
        Pop-Location
    }
} else {
    Write-Warn "Skipping build (--NoBuild specified)"
}

# Run container
Write-Step "Starting container..."
try {
    $dockerRunCmd = @(
        "run",
        "-d",
        "--name", $ContainerName,
        "-p", "${Port}:5000",
        "-v", "${ScriptDir}/models:/app/models",
        "-v", "${ScriptDir}/cache:/app/cache",
        $ImageName
    )

    $containerId = docker @dockerRunCmd
    if ($LASTEXITCODE -ne 0) {
        Write-Fail "Failed to start container"
    }
    Write-Ok "Container started: $($containerId.Substring(0, 12))"
} catch {
    Write-Fail "Error starting container: $_"
}

# Wait for container to be healthy
Write-Step "Waiting for application to start..."
$maxAttempts = 60
$attempt = 0
$healthy = $false

Write-Host "  Downloading models on first run (this may take 5-10 minutes)..." -ForegroundColor Yellow

while ($attempt -lt $maxAttempts) {
    Start-Sleep -Seconds 2
    $attempt++

    try {
        $response = Invoke-WebRequest -Uri "http://localhost:${Port}/health" -UseBasicParsing -TimeoutSec 2 -ErrorAction Stop
        if ($response.StatusCode -eq 200) {
            $healthy = $true
            break
        }
    } catch {
        # Show progress
        if ($attempt % 5 -eq 0) {
            Write-Host "  Still waiting... ($attempt/$maxAttempts)" -ForegroundColor DarkGray

            # Show last few lines of logs
            $logs = docker logs --tail 3 $ContainerName 2>&1
            if ($logs) {
                $logs | ForEach-Object {
                    if ($_ -match "Downloading|Loading|model") {
                        Write-Host "    $_" -ForegroundColor DarkCyan
                    }
                }
            }
        }
    }
}

if (-not $healthy) {
    Write-Fail "Application failed to start within timeout"
    Write-Host "`nContainer logs:" -ForegroundColor Yellow
    docker logs --tail 50 $ContainerName
    Write-Host "`nStopping container..." -ForegroundColor Yellow
    docker stop $ContainerName 2>&1 | Out-Null
    exit 1
}

Write-Ok "Application is running and healthy"

# Open browser
if (-not $NoBrowser) {
    Write-Step "Opening browser..."
    Start-Sleep -Seconds 1
    Start-Process "http://localhost:${Port}"
    Write-Ok "Browser opened"
}

# Display success message
Write-Host ""
Write-Host "╔════════════════════════════════════════╗" -ForegroundColor Green
Write-Host "║   Voice Assistant is Running! 🎙️      ║" -ForegroundColor Green
Write-Host "╚════════════════════════════════════════╝" -ForegroundColor Green
Write-Host ""
Write-Host "  🌐 Web Interface:  " -NoNewline -ForegroundColor White
Write-Host "http://localhost:${Port}" -ForegroundColor Yellow
Write-Host "  📦 Container Name: " -NoNewline -ForegroundColor White
Write-Host "$ContainerName" -ForegroundColor Yellow
Write-Host ""
Write-Host "Useful commands:" -ForegroundColor Cyan
Write-Host "  View logs:    " -NoNewline -ForegroundColor White
Write-Host "docker logs -f $ContainerName" -ForegroundColor Yellow
Write-Host "  Stop app:     " -NoNewline -ForegroundColor White
Write-Host "docker stop $ContainerName" -ForegroundColor Yellow
Write-Host "  Restart app:  " -NoNewline -ForegroundColor White
Write-Host "docker restart $ContainerName" -ForegroundColor Yellow
Write-Host "  Remove app:   " -NoNewline -ForegroundColor White
Write-Host "docker rm -f $ContainerName" -ForegroundColor Yellow
Write-Host ""
Write-Host "Press Ctrl+C to view logs (container will keep running)" -ForegroundColor DarkGray
Write-Host ""

# Trap Ctrl+C to show logs instead of exiting
try {
    Write-Host "Streaming logs (Press Ctrl+C to exit)..." -ForegroundColor Cyan
    Write-Host "─────────────────────────────────────────" -ForegroundColor DarkGray
    docker logs -f $ContainerName
} catch {
    Write-Host ""
    Write-Ok "Container is still running in background"
    Write-Host "  Access it at: http://localhost:${Port}" -ForegroundColor Yellow
}
