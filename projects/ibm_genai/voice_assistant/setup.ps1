# Voice Assistant - Setup Script (PowerShell)
# Installs Docker if necessary and sets up the environment

param(
    [switch]$SkipDocker
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent (Split-Path -Parent $ScriptDir)

# Color output functions
function Write-Step { param($msg) Write-Host "`n▶ $msg" -ForegroundColor Cyan }
function Write-Ok   { param($msg) Write-Host "  ✓ $msg" -ForegroundColor Green }
function Write-Warn { param($msg) Write-Host "  ⚠ $msg" -ForegroundColor Yellow }
function Write-Fail { param($msg) Write-Host "  ✗ $msg" -ForegroundColor Red }

Write-Host "`n╔════════════════════════════════════════╗" -ForegroundColor Magenta
Write-Host "║   Voice Assistant - Setup (Windows)    ║" -ForegroundColor Magenta
Write-Host "╚════════════════════════════════════════╝`n" -ForegroundColor Magenta

# Check if Docker is installed
Write-Step "Checking Docker installation..."
$DockerInstalled = $false

try {
    $dockerVersion = docker --version 2>&1
    if ($LASTEXITCODE -eq 0) {
        $DockerInstalled = $true
        Write-Ok "Docker found: $dockerVersion"
    }
} catch {
    Write-Warn "Docker not found"
}

# Install Docker if not found and not skipped
if (-not $DockerInstalled -and -not $SkipDocker) {
    Write-Step "Installing Docker Desktop..."

    # Check for winget
    $wingetFound = $false
    try {
        $null = Get-Command winget -ErrorAction Stop
        $wingetFound = $true
    } catch {
        Write-Warn "winget not found"
    }

    if ($wingetFound) {
        Write-Host "  Installing Docker Desktop via winget..."
        try {
            winget install --id Docker.DockerDesktop --source winget --silent --accept-package-agreements --accept-source-agreements
            Write-Ok "Docker Desktop installed successfully"
            Write-Warn "Please start Docker Desktop and wait for it to fully initialize before running the app"
            Write-Host "  Then run this setup script again or proceed to run.ps1"
            exit 0
        } catch {
            Write-Fail "Failed to install Docker via winget"
        }
    }

    # Fallback: Manual installation instructions
    Write-Warn "Automatic installation not available"
    Write-Host ""
    Write-Host "  Please install Docker Desktop manually:" -ForegroundColor Yellow
    Write-Host "  1. Visit: https://www.docker.com/products/docker-desktop/" -ForegroundColor Yellow
    Write-Host "  2. Download Docker Desktop for Windows" -ForegroundColor Yellow
    Write-Host "  3. Run the installer and follow the prompts" -ForegroundColor Yellow
    Write-Host "  4. Start Docker Desktop and wait for it to initialize" -ForegroundColor Yellow
    Write-Host "  5. Run this setup script again" -ForegroundColor Yellow
    Write-Host ""
    exit 1
}

# Check if Docker daemon is running
if ($DockerInstalled) {
    Write-Step "Checking Docker daemon..."
    try {
        $dockerInfo = docker info 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Ok "Docker daemon is running"
        } else {
            Write-Fail "Docker daemon is not running"
            Write-Warn "Please start Docker Desktop and wait for it to initialize"
            exit 1
        }
    } catch {
        Write-Fail "Cannot connect to Docker daemon"
        Write-Warn "Please start Docker Desktop and wait for it to initialize"
        exit 1
    }
}

# Verify required files exist
Write-Step "Verifying project files..."
$requiredFiles = @(
    "Dockerfile",
    "requirements.txt",
    "model_manager.py",
    "controllers.py"
)

$allFilesExist = $true
foreach ($file in $requiredFiles) {
    $filePath = Join-Path $ScriptDir $file
    if (Test-Path $filePath) {
        Write-Ok "$file exists"
    } else {
        Write-Fail "$file not found"
        $allFilesExist = $false
    }
}

if (-not $allFilesExist) {
    Write-Fail "Missing required files"
    exit 1
}

# Check for Python (optional - for local testing without Docker)
Write-Step "Checking Python installation (optional)..."
$PythonFound = $false
foreach ($pythonCmd in @("python", "python3")) {
    try {
        $pyVersion = & $pythonCmd --version 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Ok "Python found: $pyVersion"
            $PythonFound = $true
            break
        }
    } catch {
        continue
    }
}

if (-not $PythonFound) {
    Write-Warn "Python not found (only needed for local testing without Docker)"
}

# Summary
Write-Host ""
Write-Host "╔════════════════════════════════════════╗" -ForegroundColor Green
Write-Host "║      Setup Complete! ✓                 ║" -ForegroundColor Green
Write-Host "╚════════════════════════════════════════╝" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  1. Run: " -NoNewline -ForegroundColor White
Write-Host ".\run.ps1" -ForegroundColor Yellow
Write-Host "  2. Wait for models to download (first run only)" -ForegroundColor White
Write-Host "  3. Access the app at: " -NoNewline -ForegroundColor White
Write-Host "http://localhost:5000" -ForegroundColor Yellow
Write-Host ""
