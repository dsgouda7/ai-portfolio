FROM python:3.11-slim

WORKDIR /app

# sentence-transformers model download happens at first request (cached to volume)
RUN pip install --no-cache-dir fastapi uvicorn sentence-transformers

COPY deployment/embedding/service.py service.py

EXPOSE 8002
CMD ["uvicorn", "service:app", "--host", "0.0.0.0", "--port", "8002"]
