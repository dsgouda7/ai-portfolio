FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    gcc \
    g++ \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY video_processor.py audio_processor.py app.py config.yaml .
COPY templates/ ./templates/

# Create directories
RUN mkdir -p /app/models /app/cache /app/uploads /app/outputs /app/pretrained_models
ENV TRANSFORMERS_CACHE=/app/cache
ENV HF_HOME=/app/cache

# Expose Flask port
EXPOSE 5001

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=90s \
    CMD python -c "import requests; requests.get('http://localhost:5001/health', timeout=2)"

# Run the Flask application
CMD ["python", "app.py"]
