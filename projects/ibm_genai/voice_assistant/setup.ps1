param(
    [switch]$SkipDocker
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

# Check Docker Installation
Write-Host "Checking Docker installation..."
$DockerInstalled = $false

if (Get-Command docker -ErrorAction SilentlyContinue) {
    $DockerInstalled = $true
    $dockerVersion = (docker --version).Trim()
    Write-Host "Found Docker: $dockerVersion"
} else {
    Write-Host "Docker is not currently installed."
}

# Install Docker via winget if missing
if (-not $DockerInstalled -and -not $SkipDocker) {
    Write-Host "Attempting to install Docker Desktop..."

    if (Get-Command winget -ErrorAction SilentlyContinue) {
        Write-Host "Installing Docker Desktop via winget..."
        winget install --id Docker.DockerDesktop --source winget --silent --accept-package-agreements --accept-source-agreements

        Write-Host "Docker Desktop installed successfully."
        Write-Host "Please launch Docker Desktop, wait for it to initialize, then run this script again."
        exit 0
    } else {
        Write-Host "Error: winget is not available on this system. Please install Docker manually:" -ForegroundColor Red
        Write-Host "https://www.docker.com/products/docker-desktop/"
        exit 1
    }
}

# Verify Docker Daemon is running
if ($DockerInstalled) {
    Write-Host "Checking Docker daemon status..."
    if (-not (docker info 2>$null)) {
        Write-Host "Error: Docker daemon is not running. Please start Docker Desktop." -ForegroundColor Red
        exit 1
    }
    Write-Host "Docker daemon is active."
}

# Verify Project Integrity
Write-Host "Verifying required project files..."
$requiredFiles = @("Dockerfile", "requirements.txt", "model_manager.py", "controllers.py")
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

# Check Optional Python Environment (Local Dev)
Write-Host "Checking for local Python installation..."
$pythonCmd = if (Get-Command python -ErrorAction SilentlyContinue) { "python" } else { "python3" }

if (Get-Command $pythonCmd -ErrorAction SilentlyContinue) {
    $pyVersion = & $pythonCmd --version 2>&1
    Write-Host "Found local Python: $pyVersion"
} else {
    Write-Host "Python not found. (This is fine if you only plan to run within Docker)"
}

# Success Output
Write-Host "`nEnvironment verification complete."
Write-Host "To start the application, execute: .\run.ps1"
