#!/bin/bash
# Start Evidently AI monitoring dashboard

echo "🚀 Starting Evidently AI dashboard..."
echo "   UI will be available at http://localhost:8000"

# Create reports directory if it doesn't exist
mkdir -p monitoring_reports

# Start Evidently UI server
evidently ui --workspace ./monitoring_reports --port 8000

# Note: Use Ctrl+C to stop the server
