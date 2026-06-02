FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    portaudio19-dev \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY model_manager.py .
COPY controllers.py .

# Create directories for models and cache
RUN mkdir -p /app/models /app/cache
ENV TRANSFORMERS_CACHE=/app/cache
ENV HF_HOME=/app/cache

# Expose Flask port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=60s \
    CMD python -c "import requests; requests.get('http://localhost:5000/health', timeout=2)"

# Run the Flask application
CMD ["python", "controllers.py"]
