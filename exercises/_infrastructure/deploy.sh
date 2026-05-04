#!/usr/bin/env bash
# Deploy any exercise to Docker with plug-and-play infrastructure
#
# Usage:
#   ./deploy.sh --exercise "01-ml/01-regression" --dockerfile "Dockerfile.ml" --port 5001
#   ./deploy.sh --exercise "03-ai" --dockerfile "Dockerfile.api" --port 5103

set -e

# Parse arguments
EXERCISE=""
DOCKERFILE=""
PORT=""
PROJECT_NAME=""
BUILD=false
STOP=false
LOGS=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --exercise)
            EXERCISE="$2"
            shift 2
            ;;
        --dockerfile)
            DOCKERFILE="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --project)
            PROJECT_NAME="$2"
            shift 2
            ;;
        --build)
            BUILD=true
            shift
            ;;
        --stop)
            STOP=true
            shift
            ;;
        --logs)
            LOGS=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [[ -z "$EXERCISE" || -z "$DOCKERFILE" || -z "$PORT" ]]; then
    echo "Usage: $0 --exercise EXERCISE --dockerfile DOCKERFILE --port PORT [--project NAME] [--build] [--stop] [--logs]"
    exit 1
fi

# Resolve paths
INFRA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXERCISE_DIR="$(dirname "$INFRA_DIR")/$EXERCISE"

if [[ ! -d "$EXERCISE_DIR" ]]; then
    echo "❌ Exercise directory not found: $EXERCISE_DIR"
    exit 1
fi

# Derive project name
if [[ -z "$PROJECT_NAME" ]]; then
    PROJECT_NAME="${EXERCISE//\//-}-api"
fi

echo -e "\n📦 Deploying Exercise: $EXERCISE"
echo "   Dockerfile: $DOCKERFILE"
echo "   Port: $PORT"
echo "   Project: $PROJECT_NAME"

# Stop containers
if [[ "$STOP" == true ]]; then
    echo -e "\n🛑 Stopping containers..."
    docker-compose -f "$INFRA_DIR/docker-compose-generated.yml" down
    exit 0
fi

# Show logs
if [[ "$LOGS" == true ]]; then
    echo -e "\n📋 Showing logs..."
    docker-compose -f "$INFRA_DIR/docker-compose-generated.yml" logs -f "$PROJECT_NAME"
    exit 0
fi

# Generate docker-compose.yml
COMPOSE_FILE="$INFRA_DIR/docker-compose-generated.yml"
cat > "$COMPOSE_FILE" <<EOF
version: '3.8'

services:
  $PROJECT_NAME:
    build:
      context: $EXERCISE_DIR
      dockerfile: $INFRA_DIR/docker/$DOCKERFILE
    container_name: $PROJECT_NAME
    ports:
      - "$PORT:8000"
    volumes:
      - $EXERCISE_DIR/src:/app/src
      - $EXERCISE_DIR/data:/app/data
      - $EXERCISE_DIR/models:/app/models
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
      - $INFRA_DIR/monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
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
EOF

echo -e "\n✅ Generated: docker-compose-generated.yml"

# Create network
docker network create ml-network 2>/dev/null || true

# Build if requested
if [[ "$BUILD" == true ]]; then
    echo -e "\n🔨 Building Docker image..."
    docker-compose -f "$COMPOSE_FILE" build "$PROJECT_NAME"
fi

# Start containers
echo -e "\n🚀 Starting containers..."
docker-compose -f "$COMPOSE_FILE" up -d

echo -e "\n✅ Deployment complete!"
echo -e "\n📊 Services running:"
echo "   • API: http://localhost:$PORT"
echo "   • Prometheus: http://localhost:9090"
echo "   • Grafana: http://localhost:3000 (admin/admin)"
echo -e "\n💡 View logs: ./deploy.sh --exercise '$EXERCISE' --dockerfile '$DOCKERFILE' --port $PORT --logs"
echo "💡 Stop all: ./deploy.sh --exercise '$EXERCISE' --dockerfile '$DOCKERFILE' --port $PORT --stop"
