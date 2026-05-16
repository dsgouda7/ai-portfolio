#!/bin/bash
# Docker Build Script

set -e

ENVIRONMENT=${1:-dev}
IMAGE_TAG=${2:-latest}

echo "🐳 Building Docker image for $ENVIRONMENT..."

if [ "$ENVIRONMENT" == "prod" ]; then
    DOCKERFILE="docker/Dockerfile.prod"
else
    DOCKERFILE="docker/Dockerfile.dev"
fi

echo "📦 Building image: ml-devops-demo:$IMAGE_TAG"
docker build -f $DOCKERFILE -t ml-devops-demo:$IMAGE_TAG .

echo "✅ Build complete!"
echo ""
echo "To run the container:"
echo "  docker run -p 5000:5000 ml-devops-demo:$IMAGE_TAG"
