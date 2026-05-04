#!/bin/bash
# Health Check Script

set -e

URL=${1:-http://localhost:5000}

echo "🏥 Running health checks on $URL..."

# Health check
echo "1️⃣  Health check..."
HEALTH_STATUS=$(curl -s -o /dev/null -w "%{http_code}" $URL/health)
if [ "$HEALTH_STATUS" -eq 200 ]; then
    echo "   ✅ Health: OK"
else
    echo "   ❌ Health: FAILED (Status: $HEALTH_STATUS)"
    exit 1
fi

# Readiness check
echo "2️⃣  Readiness check..."
READY_STATUS=$(curl -s -o /dev/null -w "%{http_code}" $URL/ready)
if [ "$READY_STATUS" -eq 200 ]; then
    echo "   ✅ Ready: OK"
else
    echo "   ⚠️  Ready: NOT READY (Status: $READY_STATUS)"
fi

# Model info
echo "3️⃣  Model info..."
MODEL_INFO=$(curl -s $URL/model/info)
echo "   $MODEL_INFO"

# Metrics check
echo "4️⃣  Metrics check..."
METRICS_STATUS=$(curl -s -o /dev/null -w "%{http_code}" $URL/metrics)
if [ "$METRICS_STATUS" -eq 200 ]; then
    echo "   ✅ Metrics: OK"
else
    echo "   ❌ Metrics: FAILED (Status: $METRICS_STATUS)"
fi

echo ""
echo "✅ All health checks passed!"
