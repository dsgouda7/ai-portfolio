# syntax=docker/dockerfile:1
# ─────────────────────────────────────────────────────────────────────────────
# InferenceBase — Production vLLM Container
# ─────────────────────────────────────────────────────────────────────────────
# Builds the production image for serving Llama-3-8B via vLLM.
#
# Usage (built by CI — see DevOps Ch.4):
#   docker build -t inferencebase/llama3-vllm:latest .
#   docker run --gpus 1 --rm -p 8000:8000 \
#     -e MODEL_ID="meta-llama/Meta-Llama-3-8B-Instruct" \
#     -e HUGGING_FACE_HUB_TOKEN="$(cat ~/.hf_token)" \
#     inferencebase/llama3-vllm:latest
#
# In production, HUGGING_FACE_HUB_TOKEN is injected by K8s Secret (never
# hardcoded here). See k8s/deployment.yaml for the secretKeyRef binding.
# ─────────────────────────────────────────────────────────────────────────────

FROM vllm/vllm-openai:v0.4.3

# ── Security: non-root user (OWASP A05 — Security Misconfiguration) ──────────
# Running as root enables container-escape attacks. Create a dedicated vllm
# system user (uid 1001) and own all writable paths before switching.
RUN groupadd --system --gid 1001 vllm \
    && useradd --system --uid 1001 --gid 1001 \
               --shell /bin/bash \
               --create-home vllm \
    && mkdir -p /home/vllm/.cache/huggingface \
    && chown -R vllm:vllm /home/vllm

USER vllm
WORKDIR /home/vllm

# ── Runtime environment ───────────────────────────────────────────────────────
# Model ID and credentials are injected at runtime by K8s (deployment.yaml).
# Set only non-sensitive defaults here — no tokens, no API keys.
ENV HF_HOME="/home/vllm/.cache/huggingface" \
    VLLM_WORKER_MULTIPROC_METHOD="spawn" \
    TOKENIZERS_PARALLELISM="false" \
    # Default model: override via K8s env; never commit tokens to this file
    MODEL_ID="meta-llama/Meta-Llama-3-8B-Instruct" \
    QUANTIZATION="awq" \
    MAX_MODEL_LEN="4096" \
    GPU_MEMORY_UTILIZATION="0.90"

EXPOSE 8000

# ── Entry point ───────────────────────────────────────────────────────────────
# Shell form so ${VAR} expands from runtime env (K8s injects MODEL_ID, etc.).
# --enable-metrics exposes /metrics for Prometheus scraping (see ch10).
CMD python -m vllm.entrypoints.openai.api_server \
        --host 0.0.0.0 \
        --port 8000 \
        --model ${MODEL_ID} \
        --quantization ${QUANTIZATION} \
        --max-model-len ${MAX_MODEL_LEN} \
        --gpu-memory-utilization ${GPU_MEMORY_UTILIZATION} \
        --served-model-name inferencebase \
        --enable-metrics
