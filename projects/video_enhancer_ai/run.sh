#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_NAME="video-enhancer-ai"
CONTAINER_NAME="video-enhancer-app"
PORT="5000"
NO_BUILD=false
NO_BROWSER=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --no-build)   NO_BUILD=true; shift ;;
        --no-browser) NO_BROWSER=true; shift ;;
        --port)       PORT="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if ! docker info &>/dev/null; then
    echo "Error: Docker daemon is not running. Please start Docker." >&2
    exit 1
fi

if [ -n "$(docker ps -aq -f name="^/${CONTAINER_NAME}$")" ]; then
    echo "Cleaning up old container..."
    docker rm -f "$CONTAINER_NAME" >/dev/null
fi

if [ "$NO_BUILD" = false ]; then
    echo "Building Docker image..."
    cd "$SCRIPT_DIR"
    docker build -t "$IMAGE_NAME" .
fi

echo "Starting container..."
docker run -d \
    --name "$CONTAINER_NAME" \
    -p "${PORT}:5000" \
    -v "${SCRIPT_DIR}/models:/app/models" \
    -v "${SCRIPT_DIR}/cache:/app/cache" \
    -v "${SCRIPT_DIR}/uploads:/app/uploads" \
    -v "${SCRIPT_DIR}/outputs:/app/outputs" \
    "$IMAGE_NAME"

echo "Waiting for app to initialize (this may take a few minutes on first run)..."
healthy=false

for ((i=1; i<=40; i++)); do
    if curl -sf --max-time 2 "http://localhost:${PORT}/health" >/dev/null; then
        healthy=true
        break
    fi

    if [ -z "$(docker ps -q -f name="^/${CONTAINER_NAME}$")" ]; then
        echo "Error: Container stopped unexpectedly." >&2
        docker logs --tail 20 "$CONTAINER_NAME"
        exit 1
    fi

    sleep 5
done

if [ "$healthy" = false ]; then
    echo "Error: Application failed to become healthy within the timeout period." >&2
    docker logs --tail 50 "$CONTAINER_NAME"
    exit 1
fi

if [ "$NO_BROWSER" = false ]; then
    if [[ "$OSTYPE" == "darwin"* ]]; then
        open "http://localhost:${PORT}"
    elif [[ "$OSTYPE" == "linux-gnu"* ]] && command -v xdg-open &>/dev/null; then
        xdg-open "http://localhost:${PORT}" &>/dev/null &
    fi
fi

echo -e "\nVideo Enhancer AI is up at http://localhost:${PORT}"
echo -e "Showing logs. Press Ctrl+C to stop viewing logs (container stays running).\n"

trap 'echo -e "\nExiting log stream. Container is still running in background."; exit 0' INT

docker logs -f "$CONTAINER_NAME"
