# AI Infrastructure Learning Track

**What this covers:** The engineering that makes large language models practical at scale — GPU hardware architecture, training infrastructure, and inference systems.

**Target learner:** Someone who has finished `learning/genai/` and understands Transformers, fine-tuning, RAG, and LLM gateways at a mechanistic level, and now wants to understand *why* these systems behave the way they do at the hardware level and how to optimize them in production.

**Prerequisites:** `learning/genai/` (all chapters).

---

## Chapters

| # | Chapter | Narrative | Key content |
|---|---|---|---|
| 01 | GPU Hardware Foundations | InferenceBase $80k/month OpenAI bill → which GPU? | CPU vs GPU, memory hierarchy, roofline model, warp occupancy, coalescing |
| 02 | Mixed Precision and Memory Math | Riverside A10G sprint → which models fit at what precision? | Memory footprint, fp16/bf16, GradScaler, gradient checkpointing |
| 03 | PyTorch Profiling | 45-second training step mystery → where does the time go? | torch.profiler, compute vs memory bound, torch.compile |
| 04 | FlashAttention Internals | 60% of attention time at S=512 → why and how to fix it | IO complexity, tiling algorithm, online softmax, SDPA dispatch |
| 05 | Distributed Training | Riverside 70B grant + 4 A100s → which parallelism strategy? | DDP, FSDP, tensor parallelism, pipeline parallelism |
| 06 | Quantization in Depth | Riverside MacBook 16GB → compress the model without breaking it | int8 PTQ, GPTQ, AWQ, GGUF formats, NF4 |
| 07 | Inference Systems | Riverside 100× traffic spike → make the gateway 10× faster | KV cache, continuous batching, speculative decoding |
| 08 | Custom Kernels with Triton | 87% of peak HBM bandwidth → write it in Python | Triton programming model, tiled matmul, fused ops, autotuning |

---

## Entry Points

- **Start at Ch1** if you want to understand GPU hardware from scratch
- **Start at Ch2** if you already understand GPU architecture and want memory optimization
- **Start at Ch6** if you just want to deploy a model on consumer hardware

---

## Setup

```bash
cd learning/ai-infrastructure
pip install torch>=2.0 numpy matplotlib transformers
# Chapter-specific: see each chapter's requirements.txt
```

GPU is recommended but not required. All notebooks run on CPU with graceful fallbacks that show the same learning content with reference numbers where live timing is not possible.
