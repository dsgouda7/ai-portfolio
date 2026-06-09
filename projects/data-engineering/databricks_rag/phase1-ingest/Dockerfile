FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for Delta Lake (requires JVM)
RUN apt-get update && apt-get install -y \
    openjdk-17-jre-headless \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set Java environment
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64

# Copy requirements
COPY phase1-ingest/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy shared utilities
COPY shared/ ./shared/
RUN pip install -e ./shared

# Copy phase1 source
COPY phase1-ingest/src/ ./src/

# Entry point
CMD ["python", "src/ingest.py"]
