param(
    [switch]$SkipDocker
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

Write-Host "Checking Docker installation..."
$DockerInstalled = $false

if (Get-Command docker -ErrorAction SilentlyContinue) {
    $DockerInstalled = $true
    $dockerVersion = (docker --version).Trim()
    Write-Host "Found Docker: $dockerVersion"
} else {
    Write-Host "Docker is not currently installed."
}

if (-not $DockerInstalled -and -not $SkipDocker) {
    Write-Host "Attempting to install Docker Desktop..."

    if (Get-Command winget -ErrorAction SilentlyContinue) {
        Write-Host "Installing Docker Desktop via winget..."
        winget install --id Docker.DockerDesktop --source winget --silent --accept-package-agreements --accept-source-agreements

        Write-Host "Docker Desktop installed."
        Write-Host "Please launch Docker Desktop, wait for it to initialize, then run this script again."
        exit 0
    } else {
        Write-Host "Error: winget is not available on this system. Please install Docker manually:" -ForegroundColor Red
        Write-Host "https://www.docker.com/products/docker-desktop/"
        exit 1
    }
}

if ($DockerInstalled) {
    Write-Host "Checking Docker daemon status..."
    if (-not (docker info 2>$null)) {
        Write-Host "Docker daemon is not running. Attempting to start Docker Desktop..."

        $dockerPath = "C:\Program Files\Docker\Docker\Docker Desktop.exe"
        if (Test-Path $dockerPath) {
            Start-Process $dockerPath
            Write-Host "Waiting for Docker to start..."

            $maxAttempts = 30
            $attempt = 0
            $started = $false

            while ($attempt -lt $maxAttempts) {
                Start-Sleep -Seconds 2
                if (docker info 2>$null) {
                    $started = $true
                    break
                }
                $attempt++
                Write-Host "."
            }

            if ($started) {
                Write-Host "Docker daemon started successfully."
            } else {
                Write-Host "Error: Docker daemon failed to start within 60 seconds. Please start Docker Desktop manually." -ForegroundColor Red
                exit 1
            }
        } else {
            Write-Host "Error: Docker Desktop executable not found at expected location. Please start Docker Desktop manually." -ForegroundColor Red
            exit 1
        }
    } else {
        Write-Host "Docker daemon is active."
    }
}

Write-Host "Verifying required project files..."
$requiredFiles = @("Dockerfile", "requirements.txt", "video_processor.py", "audio_processor.py", "app.py")
$missingFiles = 0

foreach ($file in $requiredFiles) {
    if (Test-Path (Join-Path $ScriptDir $file)) {
        Write-Host "  OK: $file"
    } else {
        Write-Host "  Missing: $file" -ForegroundColor Red
        $missingFiles++
    }
}

if ($missingFiles -gt 0) {
    Write-Host "Error: Workspace is incomplete. Resolve the missing files before proceeding." -ForegroundColor Red
    exit 1
}

Write-Host "Checking for local Python installation..."
$pythonCmd = if (Get-Command python -ErrorAction SilentlyContinue) { "python" } else { "python3" }

if (Get-Command $pythonCmd -ErrorAction SilentlyContinue) {
    $pyVersion = & $pythonCmd --version 2>&1
    Write-Host "Found Python: $pyVersion (optional for local testing)"
} else {
    Write-Host "Python not found (only needed for local testing without Docker)"
}

Write-Host "`nSetup complete. Run './run.ps1' to start the application."
