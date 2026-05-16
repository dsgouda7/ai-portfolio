# LLM Inference Mechanics

**Status:** ✅ Complete

How LLMs generate text at inference time — sampling, the autoregressive loop, KV caching, prefill vs decode, and production serving tradeoffs.

## Contents

- [inference-mechanics.md](inference-mechanics.md) — Core chapter content
  - Sampling parameters (temperature, top-p, top-k)
  - KV cache optimization (the breakthrough that made chatbots viable)
  - Prefill vs decode phases (compute-bound vs memory-bound)
  - Production tradeoffs (throughput vs latency)

- [notebook-exercise.ipynb](notebook-exercise.ipynb) — Hands-on exercises with TODOs
  - Measure KV cache speedup (2-5x improvement)
  - Compare prefill vs decode latency
  - Test batching throughput
  - Experiment with sampling parameters (temperature, top-p, top-k)

- [notebook-solution.ipynb](notebook-solution.ipynb) — Complete implementations

## Learning Objectives

After completing this chapter, you should be able to:

1. **Understand sampling mechanics**
   - Explain how temperature, top-p, and top-k affect generation
   - Choose appropriate sampling parameters for different use cases
   - Implement sampling strategies using HuggingFace Transformers

2. **Optimize inference performance**
   - Measure KV cache impact on generation speed
   - Distinguish between prefill and decode phases
   - Evaluate batching strategies for throughput

3. **Make production tradeoffs**
   - Balance throughput vs latency requirements
   - Calculate KV cache memory requirements
   - Understand compute-bound vs memory-bound bottlenecks

## Prerequisites

- **Ch01 Transformer Architecture** — understand attention mechanism and next-token prediction
- **Python knowledge** — basic PyTorch and HuggingFace Transformers
- **Hardware** — GPU recommended but not required (CPU works, just slower)

## Key Concepts

| Concept | Analogy | Production Impact |
|---|---|---|
| **Temperature** | Confidence dial (0 = boring expert, 1+ = creative brainstormer) | Controls output quality/diversity tradeoff |
| **Top-p (nucleus)** | "Reasonable options" filter (keep top 90% probability mass) | Default choice for quality (0.9) |
| **KV Cache** | Grocery list (avoid re-reading recipe every step) | 2-5x speedup, essential for chatbots |
| **Prefill** | Speed-reading entire book at once | Compute-bound, dominates latency for long prompts |
| **Decode** | Writing one word at a time after glancing at notes | Memory-bound, benefits from quantization |

## Quick Start

```bash
# Open the exercise notebook
code notes/03-ai/ch02-llm-inference-mechanics/notebook-exercise.ipynb

# Install dependencies (if not already installed)
pip install transformers torch matplotlib numpy

# Run all cells to experiment with inference mechanics
```

**First-time model download:** LLaMA 3.2 3B (~6GB) will download automatically on first run.

## Common Questions

**Q: Why do my generations differ from the solution?**
A: Sampling with temperature > 0 is stochastic. Use `torch.manual_seed(42)` for reproducibility or `temperature=0.0` for deterministic output.

**Q: Can I use a smaller model?**
A: Yes! Replace `"meta-llama/Llama-3.2-3B"` with `"gpt2"` (~500MB) for faster downloads. Results will be qualitatively similar.

**Q: Why is my CPU inference so slow?**
A: CPU lacks parallel processing power. KV cache helps but you'll still see ~10x slower than GPU. This is expected and still useful for learning.

**Q: What's the difference between top-p and top-k?**
A: Top-p is adaptive (keeps tokens until cumulative probability reaches p), top-k is fixed (always keeps k tokens). Top-p is generally preferred.

## Next Chapter

**Ch03: LLM Training Pipeline** — Learn how to fine-tune LLMs with LoRA, implement RLHF, and build production training pipelines.
