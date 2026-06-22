#!/usr/bin/env bash
# deploy.sh — unified entry point for all deployment targets.
#
# Usage:
#   bash deployments/deploy.sh <target> [action]
#
# Targets:
#   local    Docker containers + host-installed Ollama (default)
#   azure    Docker containers + Azure Key Vault secrets
#
# Actions:
#   up       Build images and start all services (default)
#   down     Stop and remove containers
#   build    Build images without starting
#   logs     Follow service logs
#   ps       Show running containers
#
# Examples:
#   bash deployments/deploy.sh local
#   bash deployments/deploy.sh local down
#   bash deployments/deploy.sh azure up
#   bash deployments/deploy.sh azure logs

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

TARGET="${1:-local}"
ACTION="${2:-up}"

COMPOSE_FILE="$SCRIPT_DIR/$TARGET/docker-compose.yml"

if [[ ! -f "$COMPOSE_FILE" ]]; then
  echo "Error: unknown target '$TARGET' (no $COMPOSE_FILE found)"
  echo "Available targets: $(ls "$SCRIPT_DIR" | grep -v '\.sh\|\.ps1\|docker\|README' | tr '\n' ' ')"
  exit 1
fi

# ── Pre-flight checks ──────────────────────────────────────────────────────

check_docker() {
  if ! docker info &>/dev/null; then
    echo "Error: Docker is not running. Start Docker Desktop (or dockerd) and retry."
    exit 1
  fi
}

check_local_prerequisites() {
  echo "Target: local"

  # Verify Ollama is reachable on the host
  OLLAMA_URL="${OLLAMA_BASE_URL:-http://localhost:11434}"
  if ! curl -sf "$OLLAMA_URL/api/tags" &>/dev/null; then
    echo "Warning: Ollama not reachable at $OLLAMA_URL"
    echo "  Start Ollama:   ollama serve"
    echo "  (continuing — containers will retry on startup)"
  else
    echo "  Ollama: OK ($OLLAMA_URL)"

    # Check required models exist
    for model in "${COMPRESSOR_MODEL:-llama3.2:3b}" "${REASONING_MODEL:-llama3.2:3b}"; do
      if ! curl -sf "$OLLAMA_URL/api/tags" | grep -q "\"$model\"" 2>/dev/null; then
        echo "  Warning: model '$model' not found — pull it with: ollama pull $model"
      else
        echo "  Model $model: OK"
      fi
    done
  fi

  # Create .env from example if missing
  ENV_FILE="$SCRIPT_DIR/local/.env"
  if [[ ! -f "$ENV_FILE" ]]; then
    cp "$SCRIPT_DIR/local/.env.example" "$ENV_FILE"
    echo "  Created $ENV_FILE from .env.example — review and adjust model names if needed"
  fi
}

check_azure_prerequisites() {
  echo "Target: azure"
  ENV_FILE="$SCRIPT_DIR/azure/.env.azure"
  if [[ ! -f "$ENV_FILE" ]]; then
    echo "Error: $ENV_FILE not found."
    echo "  Run: cp deployments/azure/.env.example deployments/azure/.env.azure"
    echo "  Then fill in AZURE_TENANT_ID, AZURE_CLIENT_ID, AZURE_CLIENT_SECRET, AZURE_KEYVAULT_URI"
    exit 1
  fi
  echo "  .env.azure: found"
}

# ── Main ───────────────────────────────────────────────────────────────────

check_docker

case "$TARGET" in
  local) check_local_prerequisites ;;
  azure) check_azure_prerequisites ;;
esac

COMPOSE="docker compose -f $COMPOSE_FILE"

case "$ACTION" in
  up)
    echo "Starting $TARGET deployment..."
    $COMPOSE up --build -d
    echo ""
    echo "Services running. Endpoints:"
    echo "  Ingestion:         http://localhost:8001"
    echo "  Embedding:         http://localhost:8002"
    echo "  Vector Store:      http://localhost:8003"
    echo "  Reasoning Gateway: http://localhost:8080"
    ;;
  down)    $COMPOSE down ;;
  build)   $COMPOSE build ;;
  logs)    $COMPOSE logs -f ;;
  ps)      $COMPOSE ps ;;
  *)
    echo "Unknown action '$ACTION'. Use: up | down | build | logs | ps"
    exit 1
    ;;
esac
