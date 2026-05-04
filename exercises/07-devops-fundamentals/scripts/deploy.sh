#!/bin/bash
# Deployment Script

set -e

ENVIRONMENT=${1:-dev}
NAMESPACE="ml-$ENVIRONMENT"

echo "🚀 Deploying to $ENVIRONMENT environment..."

# Validate environment
if [[ ! "$ENVIRONMENT" =~ ^(dev|staging|prod)$ ]]; then
    echo "❌ Invalid environment. Use: dev, staging, or prod"
    exit 1
fi

# Check kubectl
if ! command -v kubectl &> /dev/null; then
    echo "❌ kubectl is not installed"
    exit 1
fi

# Apply Kubernetes manifests
echo "📦 Applying Kubernetes manifests..."
kubectl apply -k kubernetes/overlays/$ENVIRONMENT/

# Wait for deployment
echo "⏳ Waiting for deployment to complete..."
kubectl rollout status deployment/ml-api -n $NAMESPACE --timeout=5m

# Get service endpoint
echo "🔍 Getting service endpoint..."
kubectl get svc ml-api -n $NAMESPACE

echo "✅ Deployment complete!"
