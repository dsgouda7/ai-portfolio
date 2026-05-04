#!/bin/bash
# Start MLflow tracking server on localhost:5000

echo "🚀 Starting MLflow tracking server..."
echo "   UI will be available at http://localhost:5000"

# Create mlruns directory if it doesn't exist
mkdir -p mlruns

# Start MLflow server
mlflow server \
    --backend-store-uri sqlite:///mlruns/mlflow.db \
    --default-artifact-root ./mlruns \
    --host 0.0.0.0 \
    --port 5000

# Note: Use Ctrl+C to stop the server
