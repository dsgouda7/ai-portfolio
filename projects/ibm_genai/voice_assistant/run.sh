#!/usr/bin/env bash
# Voice Assistant - Run Script (Bash)
# Builds Docker image, runs container, and opens browser

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_NAME="voice-assistant"
CONTAINER_NAME="voice-assistant-app"
PORT="5000"
NO_BUILD=false
NO_BROWSER=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --no-build)
            NO_BUILD=true
            shift
            ;;
        --no-browser)
            NO_BROWSER=true
            shift
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--no-build] [--no-browser] [--port PORT]"
            exit 1
            ;;
    esac
done

# Color output functions
step() { echo; echo "▶ $*"; }
ok()   { echo "  ✓ $*"; }
warn() { echo "  ⚠ $*"; }
fail() { echo "  ✗ $*"; exit 1; }

echo
echo "╔════════════════════════════════════════╗"
echo "║   Voice Assistant - Starting App      ║"
echo "╚════════════════════════════════════════╝"
echo

# Check if Docker is running
step "Checking Docker..."
if ! docker info &>/dev/null; then
    fail "Docker daemon is not running. Please start Docker."
fi
ok "Docker is running"

# Stop and remove existing container
step "Cleaning up existing containers..."
if docker ps -a -q -f name="$CONTAINER_NAME" &>/dev/null; then
    existing=$(docker ps -a -q -f name="$CONTAINER_NAME")
    if [ -n "$existing" ]; then
        echo "  Stopping existing container..."
        docker stop "$CONTAINER_NAME" &>/dev/null || true
        echo "  Removing existing container..."
        docker rm "$CONTAINER_NAME" &>/dev/null || true
        ok "Cleaned up existing container"
    else
        ok "No existing container found"
    fi
else
    ok "No existing container found"
fi

# Build Docker image
if [ "$NO_BUILD" = false ]; then
    step "Building Docker image..."
    cd "$SCRIPT_DIR"
    echo "  This may take a few minutes on first build..."
    if docker build -t "$IMAGE_NAME" . 2>&1 | grep -E "^Step |^Successfully" | sed 's/^/  /'; then
        ok "Docker image built successfully"
    else
        fail "Docker build failed"
    fi
else
    warn "Skipping build (--no-build specified)"
fi

# Run container
step "Starting container..."
container_id=$(docker run -d \
    --name "$CONTAINER_NAME" \
    -p "${PORT}:5000" \
    -v "${SCRIPT_DIR}/models:/app/models" \
    -v "${SCRIPT_DIR}/cache:/app/cache" \
    "$IMAGE_NAME") || fail "Failed to start container"

ok "Container started: ${container_id:0:12}"

# Wait for container to be healthy
step "Waiting for application to start..."
max_attempts=60
attempt=0
healthy=false

echo "  Downloading models on first run (this may take 5-10 minutes)..."

while [ $attempt -lt $max_attempts ]; do
    sleep 2
    ((attempt++))

    if curl -sf --max-time 2 "http://localhost:${PORT}/health" &>/dev/null; then
        healthy=true
        break
    fi

    # Show progress
    if [ $((attempt % 5)) -eq 0 ]; then
        echo "  Still waiting... ($attempt/$max_attempts)"

        # Show last few lines of logs
        docker logs --tail 3 "$CONTAINER_NAME" 2>&1 | \
            grep -iE "Downloading|Loading|model" | \
            sed 's/^/    /' || true
    fi
done

if [ "$healthy" = false ]; then
    fail "Application failed to start within timeout"
    echo
    echo "Container logs:"
    docker logs --tail 50 "$CONTAINER_NAME"
    echo
    echo "Stopping container..."
    docker stop "$CONTAINER_NAME" &>/dev/null || true
    exit 1
fi

ok "Application is running and healthy"

# Open browser
if [ "$NO_BROWSER" = false ]; then
    step "Opening browser..."
    sleep 1

    # Detect OS and open browser accordingly
    if [[ "$OSTYPE" == "darwin"* ]]; then
        open "http://localhost:${PORT}"
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        if command -v xdg-open &>/dev/null; then
            xdg-open "http://localhost:${PORT}" &>/dev/null &
        elif command -v gnome-open &>/dev/null; then
            gnome-open "http://localhost:${PORT}" &>/dev/null &
        fi
    fi

    ok "Browser opened"
fi

# Display success message
echo
echo "╔════════════════════════════════════════╗"
echo "║   Voice Assistant is Running! 🎙️      ║"
echo "╚════════════════════════════════════════╝"
echo
echo "  🌐 Web Interface:  http://localhost:${PORT}"
echo "  📦 Container Name: $CONTAINER_NAME"
echo
echo "Useful commands:"
echo "  View logs:    docker logs -f $CONTAINER_NAME"
echo "  Stop app:     docker stop $CONTAINER_NAME"
echo "  Restart app:  docker restart $CONTAINER_NAME"
echo "  Remove app:   docker rm -f $CONTAINER_NAME"
echo
echo "Press Ctrl+C to view logs (container will keep running)"
echo

# Handle Ctrl+C gracefully
trap 'echo; ok "Container is still running in background"; echo "  Access it at: http://localhost:${PORT}"; exit 0' INT

# Stream logs
echo "Streaming logs (Press Ctrl+C to exit)..."
echo "─────────────────────────────────────────"
docker logs -f "$CONTAINER_NAME"
