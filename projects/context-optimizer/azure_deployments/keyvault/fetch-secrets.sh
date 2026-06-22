#!/usr/bin/env bash
# fetch-secrets.sh — pull secrets from Azure Key Vault and write per-service .env files.
#
# Called automatically by the secrets-init container at compose startup.
# Can also be run manually before 'docker compose up' if the init container
# approach is not desired.
#
# Inputs  (from .env.azure / env vars set in docker-compose.yml):
#   AZURE_KEYVAULT_URI          vault URI
#   AZURE_TENANT_ID             service principal tenant
#   AZURE_CLIENT_ID             service principal app ID
#   AZURE_CLIENT_SECRET         service principal secret
#   KV_SECRET_*                 optional overrides for secret names (see .env.example)
#
# Outputs (written to azure_deployments/services/):
#   reasoning-gateway.env
#   ingestion.env
#   embedding.env
#   vector-store.env

set -euo pipefail

: "${AZURE_KEYVAULT_URI:?AZURE_KEYVAULT_URI must be set}"
KV_NAME=$(echo "$AZURE_KEYVAULT_URI" | sed 's|https://||; s|\.vault\.azure\.net/.*||')
SERVICES_DIR="$(dirname "$0")/../services"
mkdir -p "$SERVICES_DIR"

# Authenticate via service principal if credentials are present
if [[ -n "${AZURE_CLIENT_ID:-}" && -n "${AZURE_CLIENT_SECRET:-}" && -n "${AZURE_TENANT_ID:-}" ]]; then
  az login --service-principal \
    --username  "$AZURE_CLIENT_ID" \
    --password  "$AZURE_CLIENT_SECRET" \
    --tenant    "$AZURE_TENANT_ID" \
    --output none
fi

fetch() {
  local secret_name="${1}"
  az keyvault secret show \
    --vault-name "$KV_NAME" \
    --name       "$secret_name" \
    --query      value \
    --output     tsv
}

echo "Fetching secrets from vault '$KV_NAME'..."

# Secret name variables with defaults matching provision.sh
S_REASONING_API_KEY="${KV_SECRET_REASONING_API_KEY:-co-reasoning-api-key}"
S_REASONING_ENDPOINT="${KV_SECRET_REASONING_ENDPOINT:-co-reasoning-endpoint}"
S_REASONING_DEPLOYMENT="${KV_SECRET_REASONING_DEPLOYMENT:-co-reasoning-deployment}"
S_SUMMARIZER_API_KEY="${KV_SECRET_SUMMARIZER_API_KEY:-co-summarizer-api-key}"
S_SUMMARIZER_ENDPOINT="${KV_SECRET_SUMMARIZER_ENDPOINT:-co-summarizer-endpoint}"
S_SUMMARIZER_DEPLOYMENT="${KV_SECRET_SUMMARIZER_DEPLOYMENT:-co-summarizer-deployment}"
S_EMBEDDING_API_KEY="${KV_SECRET_EMBEDDING_API_KEY:-co-embedding-api-key}"
S_EMBEDDING_ENDPOINT="${KV_SECRET_EMBEDDING_ENDPOINT:-co-embedding-endpoint}"
S_EMBEDDING_MODEL="${KV_SECRET_EMBEDDING_MODEL:-co-embedding-model}"

# ── reasoning-gateway.env ─────────────────────────────────────────────────
# Main app container: connects to the reasoning LLM via LiteLLM (azure provider).
cat > "$SERVICES_DIR/reasoning-gateway.env" <<EOF
AZURE_API_KEY=$(fetch "$S_REASONING_API_KEY")
AZURE_API_BASE=$(fetch "$S_REASONING_ENDPOINT")
AZURE_API_VERSION=2024-02-01
AZURE_DEPLOYMENT_NAME=$(fetch "$S_REASONING_DEPLOYMENT")
CONTEXT_OPTIMIZER_COMPRESSOR_PROVIDER=azure
VECTOR_STORE_URL=http://vector-store:8003
DEFAULT_COLLECTION=default
EOF

# ── ingestion.env ─────────────────────────────────────────────────────────
# Ingestion service: uses a lighter summarizer model for corpus compression.
cat > "$SERVICES_DIR/ingestion.env" <<EOF
AZURE_API_KEY=$(fetch "$S_SUMMARIZER_API_KEY")
AZURE_API_BASE=$(fetch "$S_SUMMARIZER_ENDPOINT")
AZURE_API_VERSION=2024-02-01
AZURE_DEPLOYMENT_NAME=$(fetch "$S_SUMMARIZER_DEPLOYMENT")
CONTEXT_OPTIMIZER_COMPRESSOR_PROVIDER=azure
EMBEDDING_SERVICE_URL=http://embedding:8002
VECTOR_STORE_URL=http://vector-store:8003
EOF

# ── embedding.env ─────────────────────────────────────────────────────────
# Embedding service: uses Azure AI text-embedding-3-small (~10 ms, cloud PoP).
cat > "$SERVICES_DIR/embedding.env" <<EOF
EMBEDDING_BACKEND=azure
AZURE_EMBEDDING_API_KEY=$(fetch "$S_EMBEDDING_API_KEY")
AZURE_EMBEDDING_ENDPOINT=$(fetch "$S_EMBEDDING_ENDPOINT")
AZURE_EMBEDDING_MODEL=$(fetch "$S_EMBEDDING_MODEL")
AZURE_EMBEDDING_API_VERSION=2024-02-01
EOF

# ── vector-store.env ──────────────────────────────────────────────────────
# Vector store: ChromaDB + semantic LRU cache. No external credentials needed.
cat > "$SERVICES_DIR/vector-store.env" <<EOF
EMBEDDING_SERVICE_URL=http://embedding:8002
CHROMA_PERSIST_DIR=/data/chroma_db
CACHE_SIZE=1000
CACHE_THRESHOLD=0.85
EOF

echo "Service .env files written to $SERVICES_DIR/"
