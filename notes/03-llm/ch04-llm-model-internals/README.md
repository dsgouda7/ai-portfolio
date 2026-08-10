# LLM Model Internals — Parameters, Memory, and Deployment

**Status:** Complete

The engineering vocabulary for deployment decisions: parameter counting, VRAM budgeting, quantization trade-offs (fp16/int8/int4), Mixture of Experts routing, and Grouped Query Attention. Bridges from "I understand how LLMs work" to "I can make informed deployment choices."

## Contents

- [model-internals.md](model-internals.md) — Core chapter content
  - Parameter counting rules of thumb (7B, 13B, 70B, MoE)
  - VRAM budgeting: the 2× and 4× rules
  - Quantization: fp16 → int8 → int4 trade-offs
  - Mixture of Experts: routing and active-parameter economics
  - Grouped Query Attention (GQA): memory bandwidth optimization

> **From-scratch companion (Part 14):** [`learning/genai/08-transformers/02-decoder-only-language-model.ipynb`](../../../learning/genai/08-transformers/02-decoder-only-language-model.ipynb) — Part 14 loads `distilgpt2` through PyTorch `GPT2LMHeadModel`, inspects its transformer blocks and attention heads, and plots next-token probabilities. Use it before this chapter to connect parameter-counting theory to a real model.

## Learning Objectives

After completing this chapter, you should be able to:

1. **Count parameters and budget VRAM**
   - Estimate VRAM requirements from parameter count and precision
   - Calculate the memory footprint of KV cache at different batch sizes
   - Explain why MoE models (e.g., Mixtral-8×7B) activate only 12.9B of 93B parameters per token

2. **Choose quantization correctly**
   - Describe the quality/memory trade-off for fp16, int8, and int4
   - Explain when int4 is acceptable vs when fp16 is required
   - Understand GPTQ and AWQ as calibration-based quantization methods

3. **Evaluate deployment constraints**
   - Given a model size and hardware spec, determine feasibility
   - Explain why GQA reduces memory bandwidth without changing model quality
   - Select the right model size for a given latency/cost/quality target

## Prerequisites

- **Ch.01** — Transformer architecture (you need to know what layers exist to count their parameters)
- **Ch.02** — KV cache mechanics (KV cache memory is a major component of VRAM budgeting)
- **Ch.03** — Training pipeline (quantization is often applied post-training; understanding training precision matters)

## Key Concepts

| Concept | Analogy | Why It Matters |
|---|---|---|
| **Parameter count** | Library capacity | Measures model capacity, not content quality |
| **VRAM budget** | Workshop bench space | Hard constraint: model + KV cache + activations must fit |
| **fp16 / int8 / int4** | 16-bit / 8-bit / 4-bit number precision | Each halving roughly halves memory, with quality costs |
| **MoE (Mixture of Experts)** | Specialist teams with a dispatcher | 93GB total model, only 12.9B active per token → cost efficiency |
| **GQA (Grouped Query Attention)** | Shared key-value heads across query heads | Cuts KV cache memory 4–8× with minimal quality drop |
| **Flash Attention** | Cache-aware computation reordering | Reduces memory bandwidth from O(n²) reads to O(n) |

## ML → LLM Bridge

If you completed **notes/02**, these connections are direct:

- **Ch.10 (Pruning & Mixed Precision)**: The AMP training pattern (fp16 forward pass, fp32 gradient accumulation, GradScaler) from notes/02 is identical here — extended to 175B+ parameter models. GPTQ post-training quantization is the inference-time analog of training-time AMP.
- **Ch.09 (Knowledge Distillation)**: 7B models distilled from 70B teachers (e.g., Alpaca, Vicuna, TinyLLaMA) use the same temperature-scaled KL loss. The deployment decision between a 70B teacher and 7B student maps directly to the latency/quality budget discussed in this chapter.
- **Ch.01 (Residual Networks)**: Every transformer layer applies skip connections ($x + \text{Attention}(x)$, $x + \text{FFN}(x)$). Parameter counting requires knowing that these residual blocks each contribute $4d^2$ parameters to the FFN and $4d^2$ to the attention projection matrices.

## Quick Start

```bash
code notes/03-llm/ch04-llm-model-internals/model-internals.md
```

## Common Questions

**Q: Does bigger always mean better?**
A: Diminishing returns hit hard above 70B. A 7B model with high-quality domain-specific fine-tuning typically outperforms a vanilla 70B model on that domain. Scale buys generality; specialization buys performance.

**Q: When should I use int4 quantization?**
A: int4 is appropriate for inference on consumer hardware (e.g., running LLaMA locally on a MacBook Pro with 16GB RAM). For production serving with quality SLAs, use int8 or fp16. int4 typically loses 1–3% on benchmarks but saves 4× memory vs fp16.

**Q: What is MoE and why does Mixtral-8×7B advertise 46.7B total parameters?**
A: Mixtral has 8 expert FFN blocks per layer but a learned router activates only 2 per token. Total parameters: ~46.7B (all experts). Active parameters per token: ~12.9B (2 experts). You pay 12.9B in compute but have 46.7B in knowledge capacity.

## Next Chapter

[Ch.05 — Prompt Engineering](../ch05-prompt-engineering/README.md): With the model internals understood, Ch.05 covers behavioral control — how system prompts, few-shot examples, temperature, and structured output format translate your understanding of model mechanics into reliable production behavior.
