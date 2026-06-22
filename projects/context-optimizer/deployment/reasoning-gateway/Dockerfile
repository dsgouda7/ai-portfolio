FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir fastapi uvicorn litellm httpx langchain-core langchain-ollama langchain-groq

COPY src/ src/
COPY deployment/reasoning-gateway/service.py service.py

ENV PYTHONPATH=/app/src

EXPOSE 8080
CMD ["uvicorn", "service:app", "--host", "0.0.0.0", "--port", "8080"]
