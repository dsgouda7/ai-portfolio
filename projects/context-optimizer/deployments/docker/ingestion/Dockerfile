FROM python:3.11-slim

WORKDIR /app

# Install only what ingestion needs
RUN pip install --no-cache-dir fastapi uvicorn httpx langchain-core langchain-ollama langchain-groq

# Copy the core library
COPY src/ src/

COPY deployments/docker/ingestion/service.py service.py

ENV PYTHONPATH=/app/src

EXPOSE 8001
CMD ["uvicorn", "service:app", "--host", "0.0.0.0", "--port", "8001"]
