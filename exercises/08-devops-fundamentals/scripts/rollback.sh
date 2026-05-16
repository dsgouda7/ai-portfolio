#!/bin/bash
# Rollback Script

set -e

ENVIRONMENT=${1:-dev}
NAMESPACE="ml-$ENVIRONMENT"
REVISION=${2:-0}  # 0 means previous revision

echo "⚠️  Rolling back deployment in $ENVIRONMENT..."

# Validate environment
if [[ ! "$ENVIRONMENT" =~ ^(dev|staging|prod)$ ]]; then
    echo "❌ Invalid environment. Use: dev, staging, or prod"
    exit 1
fi

# Show rollout history
echo "📜 Deployment history:"
kubectl rollout history deployment/ml-api -n $NAMESPACE

# Confirm rollback
read -p "Are you sure you want to rollback? (yes/no): " CONFIRM
if [ "$CONFIRM" != "yes" ]; then
    echo "❌ Rollback cancelled"
    exit 0
fi

# Perform rollback
echo "🔄 Rolling back..."
if [ "$REVISION" -eq 0 ]; then
    kubectl rollout undo deployment/ml-api -n $NAMESPACE
else
    kubectl rollout undo deployment/ml-api -n $NAMESPACE --to-revision=$REVISION
fi

# Wait for rollback to complete
echo "⏳ Waiting for rollback to complete..."
kubectl rollout status deployment/ml-api -n $NAMESPACE --timeout=5m

echo "✅ Rollback complete!"
