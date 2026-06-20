# Multi-stage build for context-optimizer service
# Production-ready compression service with minimal footprint

# Stage 1: Builder
FROM python:3.11-slim as builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --user --no-cache-dir -r requirements.txt

# Stage 2: Runtime
FROM python:3.11-slim

LABEL maintainer="deepaksha"
LABEL description="Context Optimizer - LLM Context Compression Service"

WORKDIR /app

# Copy Python dependencies from builder
COPY --from=builder /root/.local /root/.local

# Copy application code
COPY src/context_optimizer/ ./context_optimizer/
COPY pipeline/ ./pipeline/
COPY pyproject.toml .
COPY README.md .

# Make sure scripts are executable
ENV PATH=/root/.local/bin:$PATH
ENV PYTHONPATH=/app:$PYTHONPATH
ENV PYTHONUNBUFFERED=1

# Create data directory for corpus storage
RUN mkdir -p /app/data /app/results

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; from context_optimizer import __version__; sys.exit(0)"

# Expose API port (if running as service)
EXPOSE 8000

# Default command - can be overridden
CMD ["python", "-m", "context_optimizer.cli", "--help"]
