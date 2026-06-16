FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY phase2-vectorize/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy shared utilities
COPY shared/ ./shared/
RUN pip install -e ./shared

# Copy phase2 source
COPY phase2-vectorize/src/ ./src/

# Entry point
CMD ["python", "src/vectorize.py"]
