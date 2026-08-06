ARG PYTHON_BASE_IMAGE
FROM ${PYTHON_BASE_IMAGE}

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

RUN addgroup --system riverside && adduser --system --ingroup riverside riverside

COPY pyproject.toml README.md ./
COPY src ./src
COPY config ./config

RUN python -m pip install --no-cache-dir .

USER riverside
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2", "--timeout-keep-alive", "5"]
