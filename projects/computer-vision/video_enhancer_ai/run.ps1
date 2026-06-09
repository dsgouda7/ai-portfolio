param(
    [switch]$NoBuild,
    [switch]$NoBrowser,
    [string]$Port = "5001"
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ImageName = "video-enhancer-ai"
$ContainerName = "video-enhancer-app"

if (-not (Get-Service -Name "docker" -ErrorAction SilentlyContinue) -and -not (docker info 2>$null)) {
    Write-Error "Docker daemon is not running. Please start Docker Desktop."
    exit 1
}

if (docker ps -a -q -f name=$ContainerName) {
    Write-Host "Cleaning up old container..."
    docker rm -f $ContainerName > $null
}

if (-not $NoBuild) {
    Write-Host "Building Docker image..."
    Push-Location $ScriptDir
    try {
        docker build -t $ImageName .
    } finally {
        Pop-Location
    }
}

Write-Host "Starting container..."
docker run -d `
    --name $ContainerName `
    -p "${Port}:5001" `
    -v "${ScriptDir}/models:/app/models" `
    -v "${ScriptDir}/cache:/app/cache" `
    -v "${ScriptDir}/uploads:/app/uploads" `
    -v "${ScriptDir}/outputs:/app/outputs" `
    $ImageName

Write-Host "Waiting for app to initialize (this may take a few minutes on first run)..."
$healthy = $false
for ($i = 0; $i -lt 40; $i++) {
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:${Port}/health" -UseBasicParsing -TimeoutSec 2
        if ($response.StatusCode -eq 200) {
            $healthy = $true
            break
        }
    } catch {
        if (-not (docker ps -q -f name=$ContainerName)) {
            Write-Error "Container stopped unexpectedly."
            docker logs --tail 20 $ContainerName
            exit 1
        }
    }
    Start-Sleep -Seconds 5
}

if (-not $healthy) {
    Write-Error "Application failed to become healthy within the timeout period."
    docker logs --tail 50 $ContainerName
    exit 1
}

if (-not $NoBrowser) {
    Start-Process "http://localhost:${Port}"
}

Write-Host "`nVideo Enhancer AI is up at http://localhost:${Port}"
Write-Host "Showing logs. Press Ctrl+C to stop viewing logs (container will keep running).`n"

docker logs -f $ContainerName
