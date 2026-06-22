#!/usr/bin/env bash
# provision.sh — create all required secrets in Azure Key Vault.
#
# Run once to initialise the vault before the first deployment.
# Requires: az CLI authenticated (az login, or AZURE_CLIENT_ID/SECRET/TENANT in env).
#
# Usage:
#   export AZURE_KEYVAULT_URI=https://<your-vault>.vault.azure.net/
#   bash keyvault/provision.sh
#
# Secret naming convention: co-<service>-<attribute>
#   co-reasoning-*    → reasoning LLM called by reasoning-gateway
#   co-summarizer-*   → lighter model used by ingestion for compression
#   co-embedding-*    → embedding model used by the embedding service

set -euo pipefail

: "${AZURE_KEYVAULT_URI:?Set AZURE_KEYVAULT_URI to your vault URI}"
KV_NAME=$(echo "$AZURE_KEYVAULT_URI" | sed 's|https://||; s|\.vault\.azure\.net/||')

secret() {
  local name="$1" prompt="$2"
  read -r -s -p "  $prompt: " value
  echo
  az keyvault secret set --vault-name "$KV_NAME" --name "$name" --value "$value" --output none
  echo "  ✓ $name"
}

echo "=== Provisioning secrets in vault: $KV_NAME ==="
echo

echo "── Reasoning LLM (e.g. Azure OpenAI GPT-4.1-mini, used by reasoning-gateway) ──"
secret co-reasoning-api-key      "Azure OpenAI API key"
secret co-reasoning-endpoint     "Azure OpenAI endpoint (https://....openai.azure.com/)"
secret co-reasoning-deployment   "Deployment name (e.g. gpt-4.1-mini)"
echo

echo "── Summarizer / Compressor LLM (lighter model used by ingestion service) ──"
secret co-summarizer-api-key     "Azure OpenAI API key (can reuse reasoning key)"
secret co-summarizer-endpoint    "Azure OpenAI endpoint (can reuse reasoning endpoint)"
secret co-summarizer-deployment  "Deployment name (e.g. gpt-4o-mini)"
echo

echo "── Embedding model (Azure AI text-embedding-3-small) ──"
secret co-embedding-api-key      "Azure AI / OpenAI API key"
secret co-embedding-endpoint     "Endpoint (https://....openai.azure.com/)"
secret co-embedding-model        "Model name (e.g. text-embedding-3-small)"
echo

echo "All secrets provisioned in vault '$KV_NAME'."
echo "Next: copy .env.example to .env.azure, set AZURE_KEYVAULT_URI, then:"
echo "  docker compose -f azure_deployments/docker-compose.yml up --build"
