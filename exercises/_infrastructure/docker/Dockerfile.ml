# Generic ML Service Dockerfile
# Usage: docker build -f _infrastructure/docker/Dockerfile.ml -t <project-name> .
# Supports: scikit-learn, pandas, numpy, Flask APIs

FROM python:3.10-slim

WORKDIR /app

# Install common ML dependencies
RUN pip install --no-cache-dir \
    scikit-learn>=1.3.0 \
    pandas>=2.0.0 \
    numpy>=1.24.0 \
    flask>=3.0.0 \
    pydantic>=2.4.0 \
    prometheus-client>=0.18.0 \
    joblib>=1.3.0 \
    pyyaml>=6.0

# Copy project code (mounted via docker-compose)
COPY src/ ./src/
COPY config.yaml ./config.yaml

# Create non-root user
RUN useradd -m -u 1000 mluser && \
    chown -R mluser:mluser /app

# Create directories for models and logs
RUN mkdir -p models logs && \
    chown -R mluser:mluser models logs

USER mluser

# Expose API port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/health')"

CMD ["python", "-m", "src.api"]
