# Context Optimizer AI Gateway - Production Image
FROM python:3.11-slim as builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy compression engine
COPY src/ /build/src/

# Copy gateway code
COPY ai-gateway/wrapper/ /build/ai-gateway/wrapper/
COPY ai-gateway/service/ /build/ai-gateway/service/

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    fastapi>=0.104.0 \
    uvicorn[standard]>=0.24.0 \
    litellm>=1.0.0 \
    pydantic>=2.5.0 \
    redis>=4.0.0

# ============================================================================
# Production image
# ============================================================================
FROM python:3.11-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY --from=builder /build/src/ /app/src/
COPY --from=builder /build/ai-gateway/ /app/ai-gateway/

# Set Python path
ENV PYTHONPATH=/app:/app/src:/app/ai-gateway/wrapper

# Create non-root user
RUN useradd -m -u 1000 gateway && \
    chown -R gateway:gateway /app

USER gateway

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')"

# Run service
CMD ["python", "ai-gateway/service/gateway_service.py"]
