# Transformer Architecture — From Tokens to Attention

**Status:** ✅ Complete

The foundation for all modern LLMs — from RNNs to transformers, attention mechanics (Q/K/V), positional encoding, and the three architectural families (encoder-only, decoder-only, encoder-decoder).

## Contents

- [transformer-architecture.md](transformer-architecture.md) — Core chapter content
  - Historical evolution: RNNs → Transformers (2017) → GPT-3 (2020)
  - Tokenization and BPE (byte-pair encoding)
  - Attention mechanism: Q/K/V matrices, multi-head attention
  - Positional encoding: sinusoidal vs learned vs RoPE
  - Three architectural families: BERT, GPT, T5

- [notebook-exercise.ipynb](notebook-exercise.ipynb) — Hands-on exercises with TODOs
  - Visualize attention patterns from pretrained models
  - Compare sinusoidal vs learned positional encodings
  - Test encoder vs decoder embeddings (BERT vs GPT-2)

- [notebook-solution.ipynb](notebook-solution.ipynb) — Complete implementations

## Learning Objectives

After completing this chapter, you should be able to:

1. **Understand transformer fundamentals**
   - Explain why transformers replaced RNNs (parallelization + long-range dependencies)
   - Describe the Q/K/V attention mechanism using analogies
   - Identify when to use encoder-only vs decoder-only vs encoder-decoder architectures

2. **Master attention mechanics**
   - Compute attention scores and interpret attention weights
   - Explain multi-head attention specialization (syntax, semantics, position)
   - Understand causal masking for autoregressive generation

3. **Apply architectural knowledge**
   - Choose appropriate models for retrieval (BERT) vs generation (GPT)
   - Explain positional encoding tradeoffs (RoPE for long context)
   - Calculate parameter counts and inference costs

## Prerequisites

- **Basic ML knowledge** — see Appendix A in the chapter for core terminology
- **Python proficiency** — NumPy, PyTorch basics
- **Linear algebra** — matrix multiplication, dot products
- **Hardware** — GPU recommended but not required (CPU works, just slower)

## Key Concepts

| Concept | Analogy | Why It Matters |
|---|---|---|
| **Attention (Q/K/V)** | Library catalog: Q=search query, K=spine labels, V=book contents | Core mechanism that replaced recurrence |
| **Multi-head attention** | Multiple specialist readers analyzing same text | Different heads learn syntax, semantics, position |
| **Causal masking** | Reading mystery novel (can't peek ahead) | Enables autoregressive generation (GPT, Claude) |
| **Positional encoding** | Barcode on tokens showing position | Transformers have no inherent order awareness |
| **Encoder-only (BERT)** | Bidirectional reading | Best for retrieval/embeddings (RAG pipelines) |
| **Decoder-only (GPT)** | Left-to-right generation | Modern LLMs (ChatGPT, Claude, LLaMA) |

## Quick Start

```bash
# Open the exercise notebook
code notes/03-ai/ch01-transformer-architecture/notebook-exercise.ipynb

# Install dependencies (if not already installed)
pip install transformers torch matplotlib seaborn scikit-learn

# Run all cells to explore transformer mechanics
```

**First-time model download:** DistilBERT (~250MB), GPT-2 (~500MB), BERT (~420MB) will download automatically on first run.

## Common Questions

**Q: Do I need to understand all the math formulas?**
A: No. Start with the plain English explanations and concrete examples. The formulas are provided for completeness but intuition comes first.

**Q: Why do attention weights differ each time I run?**
A: The notebooks use pretrained models with frozen weights, so attention patterns are deterministic. If you see variation, ensure you've set `torch.manual_seed(42)`.

**Q: Which architecture should I use for my project?**
A: **Decoder-only (GPT/LLaMA)** for generation, chatbots, reasoning. **Encoder-only (BERT)** for embeddings, retrieval, classification. **Encoder-decoder (T5)** for translation, summarization (but decoder-only now competitive via prompting).

**Q: What's the difference between BERT and GPT?**
A: **Attention pattern.** BERT (encoder) uses bidirectional attention (sees full context) — best for understanding. GPT (decoder) uses causal attention (sees only past tokens) — required for generation.

**Q: Why are transformers O(n²) in sequence length?**
A: Each token computes attention scores with every other token (n × n score matrix per head). This is why 100k+ token contexts are expensive.

**Q: Can I skip this chapter and jump to practical LLM usage?**
A: Not recommended. Every later chapter assumes you understand attention, tokenization, and encoder vs decoder architectures. This is foundational.

## Notebook Exercises Explained

### Exercise 1: Attention Pattern Visualization
- **Goal:** See how different attention heads specialize
- **Model:** DistilBERT (12 heads, 6 layers)
- **Example:** "The river bank was flooded" — watch how "bank" attends to "river"
- **Insight:** Some heads focus on adjacent tokens (local syntax), others on distant tokens (semantic relationships)

### Exercise 2: Positional Encoding Comparison
- **Goal:** Understand why RoPE (LLaMA, Mistral) beats learned embeddings
- **Test:** Generate encodings for positions 0-1023
- **Key finding:** Sinusoidal/RoPE extrapolate beyond training length; learned embeddings don't
- **Production impact:** RoPE models extend from 2k to 8k+ context with minimal fine-tuning

### Exercise 3: Encoder vs Decoder Embeddings
- **Goal:** See why BERT dominates retrieval while GPT dominates generation
- **Models:** BERT (encoder), GPT-2 (decoder)
- **Task:** Semantic similarity on 10 sentence pairs
- **Expected:** BERT produces higher similarity for related pairs (bidirectional context captures richer semantics)

## Troubleshooting

**Model download fails:**
```bash
# Set Hugging Face cache directory
export HF_HOME=/path/to/cache
# Or use smaller models
# Replace "distilbert-base-uncased" with "distilbert-base-uncased" (already smallest)
```

**Out of memory (OOM) errors:**
- Use CPU instead of GPU (set `device='cpu'`)
- Reduce sequence length in examples
- Close other applications

**Slow execution on CPU:**
- Expected! Transformer inference is GPU-optimized
- Exercise 1: ~10-15 seconds per cell
- Exercise 2: ~5 seconds
- Exercise 3: ~30 seconds (loads 2 models)

## Key Takeaways

1. **Transformers = attention + parallelization** — replaced RNNs by processing all tokens simultaneously
2. **Attention = weighted retrieval** — Q/K/V is a learned lookup mechanism
3. **Multi-head = specialization** — different heads track syntax, semantics, distance
4. **Architecture determines use case:**
   - Encoder (BERT): understanding, retrieval, embeddings
   - Decoder (GPT): generation, reasoning, chatbots
   - Encoder-decoder (T5): translation, summarization (less common now)
5. **Production pattern (2025):** Decoder-only LLM (GPT-4, Claude) + encoder embeddings (E5, BGE) for RAG

## Interview Prep

These are the questions every ML engineer should answer cold:

| Question | One-sentence answer |
|---|---|
| What problem did transformers solve? | RNNs couldn't parallelize training (sequential bottleneck) and couldn't learn dependencies beyond 100-200 tokens (vanishing gradients). |
| Explain attention in simple terms | Each token searches all other tokens for relevant context, computes similarity scores (Q·K), and retrieves a weighted blend of their information (weighted sum of V). |
| Why multi-head attention? | No single attention pattern captures all relationships — syntax, semantics, and position require different specialists working in parallel. |
| Encoder vs decoder difference? | Encoder sees full sequence simultaneously (bidirectional) → best for understanding. Decoder sees only past tokens (causal) → required for generation. |
| What's the scaling bottleneck? | Attention is O(n²) in sequence length — a 512-token sequence computes 262k scores per head. This is why long contexts are expensive. |

## Next Chapter

**[Ch02: LLM Inference Mechanics](../ch02-llm-inference-mechanics)** — Now that you understand *what* transformers are, learn *how* they generate text: sampling, KV caching, prefill vs decode, and production tradeoffs.
