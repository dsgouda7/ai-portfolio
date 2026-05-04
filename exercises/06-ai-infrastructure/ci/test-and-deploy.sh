#!/bin/bash
# Test and Deploy Script

set -e  # Exit on error

echo "🧪 Starting test and deploy pipeline..."

# Step 1: Run tests
echo "📋 Running unit tests..."
pytest tests/ --cov=src --cov-report=term-missing --cov-report=html

# Step 2: Check coverage threshold
COVERAGE=$(pytest --cov=src --cov-report=term-missing | grep "TOTAL" | awk '{print $NF}' | sed 's/%//')
THRESHOLD=80

if (( $(echo "$COVERAGE < $THRESHOLD" | bc -l) )); then
    echo "❌ Coverage $COVERAGE% is below threshold $THRESHOLD%"
    exit 1
fi

echo "✅ Coverage $COVERAGE% meets threshold"

# Step 3: Lint code
echo "🔍 Linting code..."
black --check src/ tests/ || {
    echo "❌ Black formatting check failed"
    exit 1
}

flake8 src/ tests/ --max-line-length=120 || {
    echo "❌ Flake8 check failed"
    exit 1
}

echo "✅ Linting passed"

# Step 4: Build Docker image
echo "🐳 Building Docker image..."
docker build -t ml-model-serving:test .

echo "✅ Docker image built"

# Step 5: Run container health check
echo "🏥 Testing container health..."
docker run -d --name ml-test -p 8000:8000 ml-model-serving:test
sleep 10

# Health check
curl -f http://localhost:8000/health || {
    echo "❌ Health check failed"
    docker logs ml-test
    docker stop ml-test
    docker rm ml-test
    exit 1
}

echo "✅ Container health check passed"

# Cleanup
docker stop ml-test
docker rm ml-test

# Step 6: Deploy (if specified)
if [ "$1" == "deploy" ]; then
    echo "🚀 Deploying to $2..."
    
    if [ "$2" == "staging" ]; then
        kubectl apply -f k8s/ --namespace=ml-staging
        kubectl rollout status deployment/ml-model-serving --namespace=ml-staging
    elif [ "$2" == "production" ]; then
        kubectl apply -f k8s/ --namespace=ml-production
        kubectl rollout status deployment/ml-model-serving --namespace=ml-production
    else
        echo "❌ Unknown environment: $2"
        exit 1
    fi
    
    echo "✅ Deployment complete"
fi

echo "🎉 Pipeline completed successfully!"
