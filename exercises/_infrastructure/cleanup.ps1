#!/usr/bin/env pwsh
# Cleanup script: Stop containers, remove images, and free disk space
#
# Usage:
#   .\cleanup.ps1 -Exercise "01-ml/01-regression" -All
#   .\cleanup.ps1 -Everything  # Nuclear option: clean ALL Docker resources

param(
    [string]$Exercise = "",
    [switch]$All,           # Remove containers + images for this exercise
    [switch]$Everything,    # Remove ALL Docker resources (use with caution)
    [switch]$DryRun         # Show what would be deleted without deleting
)

$InfraDir = $PSScriptRoot

Write-Host "
🧹 Docker Cleanup Script" -ForegroundColor Cyan
Write-Host "═══════════════════════════════════════" -ForegroundColor Cyan

if ($Everything) {
    Write-Host "
⚠️  WARNING: Nuclear cleanup - will remove ALL Docker resources!" -ForegroundColor Red
    Write-Host "   This affects ALL projects on your machine, not just this repo." -ForegroundColor Yellow
    $confirm = Read-Host "
Type 'yes' to confirm"
    
    if ($confirm -ne "yes") {
        Write-Host "
❌ Cancelled" -ForegroundColor Red
        exit 0
    }
    
    if ($DryRun) {
        Write-Host "
📋 DRY RUN - Would execute:" -ForegroundColor Yellow
        Write-Host "  docker-compose -f "$InfraDir/docker-compose-generated.yml" down"
        Write-Host "  docker container prune -f"
        Write-Host "  docker image prune -a -f"
        Write-Host "  docker volume prune -f"
        Write-Host "  docker network prune -f"
        exit 0
    }
    
    Write-Host "
🛑 Stopping all containers from generated compose..." -ForegroundColor Yellow
    if (Test-Path "$InfraDir/docker-compose-generated.yml") {
        docker-compose -f "$InfraDir/docker-compose-generated.yml" down
    }
    
    Write-Host "
🗑️  Removing all stopped containers..." -ForegroundColor Yellow
    docker container prune -f
    
    Write-Host "
🗑️  Removing all unused images..." -ForegroundColor Yellow
    docker image prune -a -f
    
    Write-Host "
🗑️  Removing all unused volumes..." -ForegroundColor Yellow
    docker volume prune -f
    
    Write-Host "
🗑️  Removing all unused networks..." -ForegroundColor Yellow
    docker network prune -f
    
    Write-Host "
✅ Nuclear cleanup complete!" -ForegroundColor Green
    docker system df  # Show disk usage
    exit 0
}

if ($Exercise -eq "" -and -not $Everything) {
    Write-Error "Must specify -Exercise or use -Everything"
    Write-Host "
Usage:" -ForegroundColor Yellow
    Write-Host "  .\cleanup.ps1 -Exercise '01-ml/01-regression' -All"
    Write-Host "  .\cleanup.ps1 -Everything  # Clean ALL Docker resources"
    exit 1
}

# Derive project name from exercise
$ProjectName = ($Exercise -replace '/', '-') + "-api"

Write-Host "
📦 Exercise: $Exercise" -ForegroundColor Cyan
Write-Host "   Project: $ProjectName" -ForegroundColor Gray

if ($DryRun) {
    Write-Host "
📋 DRY RUN - Would execute:" -ForegroundColor Yellow
    Write-Host "  docker-compose -f "$InfraDir/docker-compose-generated.yml" down"
    if ($All) {
        Write-Host "  docker rmi $ProjectName -f"
        Write-Host "  docker rmi prom/prometheus:latest -f"
        Write-Host "  docker rmi grafana/grafana:latest -f"
    }
    exit 0
}

# Stop containers
Write-Host "
🛑 Stopping containers..." -ForegroundColor Yellow
if (Test-Path "$InfraDir/docker-compose-generated.yml") {
    docker-compose -f "$InfraDir/docker-compose-generated.yml" down
    Write-Host "   ✓ Containers stopped" -ForegroundColor Green
} else {
    Write-Host "   ⚠️  No docker-compose-generated.yml found (nothing to stop)" -ForegroundColor Yellow
}

# Remove images if -All flag
if ($All) {
    Write-Host "
🗑️  Removing Docker images..." -ForegroundColor Yellow
    
    # Remove exercise image
    $imageExists = docker images -q $ProjectName
    if ($imageExists) {
        docker rmi $ProjectName -f
        Write-Host "   ✓ Removed $ProjectName" -ForegroundColor Green
    } else {
        Write-Host "   ⚠️  Image $ProjectName not found" -ForegroundColor Yellow
    }
    
    # Ask before removing shared images
    Write-Host "
⚠️  Also remove shared monitoring images (prometheus, grafana)?" -ForegroundColor Yellow
    Write-Host "   (This will affect other exercises using the same stack)" -ForegroundColor Gray
    $removeShared = Read-Host "Remove shared images? (y/N)"
    
    if ($removeShared -eq "y" -or $removeShared -eq "Y") {
        docker rmi prom/prometheus:latest -f 2>$null
        docker rmi grafana/grafana:latest -f 2>$null
        Write-Host "   ✓ Removed shared images" -ForegroundColor Green
    }
}

Write-Host "
✅ Cleanup complete!" -ForegroundColor Green
Write-Host "
📊 Current Docker disk usage:" -ForegroundColor Cyan
docker system df
