param(
    [switch]$NoBuild,
    [switch]$NoBrowser,
    [string]$Port = "5000"
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ImageName = "voice-assistant"
$ContainerName = "voice-assistant-app"

# 1. Verify Docker is alive
if (-not (Get-Service -Name "docker" -ErrorAction SilentlyContinue) -and -not (docker info 2>$null)) {
    Write-Error "Docker daemon is not running. Please start Docker Desktop."
    exit 1
}

# 2. Tear down existing container if it exists
if (docker ps -a -q -f name=$ContainerName) {
    Write-Host "Cleaning up old container..."
    docker rm -f $ContainerName > $null
}

# 3. Build the image
if (-not $NoBuild) {
    Write-Host "Building Docker image..."
    Push-Location $ScriptDir
    try {
        docker build -t $ImageName .
    } finally {
        Pop-Location
    }
}

# 4. Run the container
Write-Host "Starting container..."
docker run -d `
    --name $ContainerName `
    -p "${Port}:5000" `
    -v "${ScriptDir}/models:/app/models" `
    -v "${ScriptDir}/cache:/app/cache" `
    $ImageName

# 5. Poll health endpoint
Write-Host "Waiting for app to initialize (this may take a bit on the first run)..."
$healthy = $false
for ($i = 0; $i -lt 30; $i++) {
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:${Port}/health" -UseBasicParsing -TimeoutSec 2
        if ($response.StatusCode -eq 200) {
            $healthy = $true
            break
        }
    } catch {
        # App isn't up yet, check if container crashed early
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

# 6. Launch and handoff
if (-not $NoBrowser) {
    Start-Process "http://localhost:${Port}"
}

Write-Host "`nVoice Assistant is up at http://localhost:${Port}"
Write-Host "Showing logs. Press Ctrl+C to stop viewing logs (container will keep running).`n"

# Follow logs. Pressing Ctrl+C here naturally exits the script but leaves the background container running.
docker logs -f $ContainerName
