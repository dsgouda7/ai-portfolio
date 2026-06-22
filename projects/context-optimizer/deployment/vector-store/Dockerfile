FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir fastapi uvicorn chromadb sentence-transformers httpx

COPY src/ src/
COPY deployment/vector-store/service.py service.py

ENV PYTHONPATH=/app/src

EXPOSE 8003
CMD ["uvicorn", "service:app", "--host", "0.0.0.0", "--port", "8003"]
