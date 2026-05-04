#!/usr/bin/env pwsh
# Deploy any exercise to Docker with plug-and-play infrastructure
#
# Usage:
#   .\deploy.ps1 -Exercise "01-ml/01-regression" -Dockerfile "Dockerfile.ml" -Port 5001
#   .\deploy.ps1 -Exercise "03-ai" -Dockerfile "Dockerfile.api" -Port 5103

param(
    [Parameter(Mandatory=$true)]
    [string]$Exercise,
    
    [Parameter(Mandatory=$true)]
    [string]$Dockerfile,
    
    [Parameter(Mandatory=$true)]
    [int]$Port,
    
    [string]$ProjectName = "",
    [switch]$Build,
    [switch]$Stop,
    [switch]$Logs
)

# Resolve paths
$InfraDir = $PSScriptRoot
$ExerciseDir = Join-Path (Split-Path $InfraDir) $Exercise

if (-not (Test-Path $ExerciseDir)) {
    Write-Error "Exercise directory not found: $ExerciseDir"
    exit 1
}

# Derive project name from exercise path if not provided
if ($ProjectName -eq "") {
    $ProjectName = ($Exercise -replace '/', '-') + "-api"
}

Write-Host "`n📦 Deploying Exercise: $Exercise" -ForegroundColor Cyan
Write-Host "   Dockerfile: $Dockerfile" -ForegroundColor Gray
Write-Host "   Port: $Port" -ForegroundColor Gray
Write-Host "   Project: $ProjectName" -ForegroundColor Gray

# Stop containers
if ($Stop) {
    Write-Host "`n🛑 Stopping containers..." -ForegroundColor Yellow
    docker-compose -f "$InfraDir/docker-compose-generated.yml" down
    exit 0
}

# Show logs
if ($Logs) {
    Write-Host "`n📋 Showing logs..." -ForegroundColor Yellow
    docker-compose -f "$InfraDir/docker-compose-generated.yml" logs -f $ProjectName
    exit 0
}

# Generate docker-compose.yml on-the-fly
$ComposeContent = @"
version: '3.8'

services:
  $($ProjectName):
    build:
      context: $ExerciseDir
      dockerfile: $InfraDir/docker/$Dockerfile
    container_name: $ProjectName
    ports:
      - "$($Port):8000"
    volumes:
      - $($ExerciseDir)/src:/app/src
      - $($ExerciseDir)/data:/app/data
      - $($ExerciseDir)/models:/app/models
    environment:
      - PYTHONUNBUFFERED=1
    networks:
      - ml-network

  prometheus:
    image: prom/prometheus:latest
    container_name: prometheus
    ports:
      - "9090:9090"
    volumes:
      - $InfraDir/monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
    networks:
      - ml-network

  grafana:
    image: grafana/grafana:latest
    container_name: grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    networks:
      - ml-network

networks:
  ml-network:
    driver: bridge
"@

# Write generated compose file
$ComposeFile = Join-Path $InfraDir "docker-compose-generated.yml"
$ComposeContent | Out-File -FilePath $ComposeFile -Encoding UTF8

Write-Host "`n✅ Generated: docker-compose-generated.yml" -ForegroundColor Green

# Create network if doesn't exist
docker network create ml-network 2>$null

# Build if requested
if ($Build) {
    Write-Host "`n🔨 Building Docker image..." -ForegroundColor Yellow
    docker-compose -f $ComposeFile build $ProjectName
}

# Start containers
Write-Host "`n🚀 Starting containers..." -ForegroundColor Yellow
docker-compose -f $ComposeFile up -d

Write-Host "`n✅ Deployment complete!" -ForegroundColor Green
Write-Host "`n📊 Services running:" -ForegroundColor Cyan
Write-Host "   • API: http://localhost:$Port" -ForegroundColor Gray
Write-Host "   • Prometheus: http://localhost:9090" -ForegroundColor Gray
Write-Host "   • Grafana: http://localhost:3000 (admin/admin)" -ForegroundColor Gray
Write-Host "`n💡 View logs: .\deploy.ps1 -Exercise '$Exercise' -Dockerfile '$Dockerfile' -Port $Port -Logs" -ForegroundColor Yellow
Write-Host "💡 Stop all: .\deploy.ps1 -Exercise '$Exercise' -Dockerfile '$Dockerfile' -Port $Port -Stop" -ForegroundColor Yellow
