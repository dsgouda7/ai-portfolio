# LLM Inference Mechanics

How LLMs generate text at inference time — sampling, the autoregressive loop, KV caching, prefill vs decode, and production serving tradeoffs.

## Contents

- [inference-mechanics.md](inference-mechanics.md) — Sampling parameters (temperature, top-p), KV cache optimization, prefill/decode phases, PagedAttention, batching strategies, and throughput/latency tradeoffs

## Visual Assets

- **img/** — Diagrams (autoregressive generation with KV cache, prefill vs decode phases, memory cost breakdown)
- **gen-scripts/** — Python scripts to generate inference mechanics visualizations

---

**Related chapters:**
- [Ch.1 — LLM Fundamentals](../ch01-transformer-architecture/) — Transformer architecture, attention mechanism, tokenization
- [Ch.3 — LLM Training Pipeline](../ch03-llm-training-pipeline/) — Pretraining, supervised fine-tuning, RLHF
- [Ch.4 — RAG and Embeddings](../ch07-rag-and-embeddings/) — Why RAG reduces inference cost by lowering token count
