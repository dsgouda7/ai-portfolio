# Chapter 1: Transformer Architecture

**Status:** ✅ Production-ready

**What you learn here:**
- The historical path from RNNs → Attention → Transformers
- Core concept: LLMs as next-token predictors
- Tokenization (BPE, context windows, cost estimation)
- Transformer mechanics: Q/K/V attention, multi-head attention, positional encoding
- Three architectural families: encoder-only (BERT), decoder-only (GPT), encoder-decoder (T5)

**Before reading this:**
- Optional: Basic linear algebra (matrix multiplication, dot products)
- Optional: High-level understanding of neural networks

**After reading this:**
- Proceed to [Ch.2 — LLM Inference Mechanics](../ch02-llm-inference-mechanics) for sampling, KV caching, and the pretraining pipeline

---

## Files

| File | Purpose |
|------|---------|
| [transformer-architecture.md](transformer-architecture.md) | Main chapter content |
| `gen-scripts/` | Python scripts to generate diagrams and visualizations |
| `img/` | Generated PNG diagrams referenced in the chapter |

---

## Key Concepts

### §0 · The Historical Thread
- RNN/LSTM limitations (sequential processing, vanishing gradients)
- Attention mechanism (Bahdanau 2014)
- "Attention Is All You Need" (Vaswani et al., 2017)
- Decoder fork (GPT lineage) vs Encoder fork (BERT lineage)
- Scaling laws and emergent capabilities (GPT-3)
- Alignment era (InstructGPT, RLHF)
- Test-time compute scaling (o1, DeepSeek-R1)

### §1 · Core Idea
- **Training objective:** Maximize $P(\text{token}_t \mid \text{token}_{<t})$
- **Three stages:** Pretraining → SFT → RLHF
- All capabilities emerge from next-token prediction at scale

### §2 · Tokenization
- **BPE (Byte-Pair Encoding):** Subword tokenization algorithm
- **Cost estimation:** ~1 token ≈ 0.75 English words, 1 token ≈ 4 bytes
- **Context window:** Maximum tokens per forward pass (4k–1M depending on model)
- **Lost-in-the-middle:** Performance degrades for info buried in long contexts

### §2A · Transformer Architecture
- **Multi-head self-attention:**
  - Project tokens to Query, Key, Value spaces
  - Compute attention scores: $\text{softmax}(QK^T / \sqrt{d_k})$
  - Weighted sum: $\text{attention\_weights} \cdot V$
  - Multiple heads learn different patterns (syntax, semantics, position)
- **Causal masking:** Decoder models mask future tokens ($j > i$)
- **Positional encoding:** Inject position info (sinusoidal, learned, or RoPE)
- **Feed-forward network:** Two-layer MLP applied per token independently
- **Residual connections + LayerNorm:** Stabilize training

### §2B · Encoder vs Decoder
| Architecture | Attention | Use Case | Examples |
|--------------|-----------|----------|----------|
| **Encoder-only** | Bidirectional (no mask) | Classification, embeddings, retrieval | BERT, RoBERTa, E5 |
| **Decoder-only** | Causal (lower-triangular mask) | Text generation, chat, reasoning | GPT, Claude, LLaMA |
| **Encoder-decoder** | Bidirectional encoder + causal decoder + cross-attention | Translation, summarization | T5, BART, Whisper |

**The 2025 pattern:** Use decoder-only LLMs for generation + encoder-only models for retrieval (RAG pipelines).

---

## Prerequisites for Later Chapters

After completing this chapter, you have the foundation for:
- **Ch.2 — LLM Inference Mechanics:** Sampling parameters, KV caching, the pretraining → SFT → RLHF pipeline
- **Ch.3 — CoT Reasoning:** Chain-of-thought prompting builds on understanding how decoders generate sequences
- **Ch.4 — RAG and Embeddings:** Encoder models for retrieval + decoder models for generation
- **Ch.5 — Advanced Prompting:** Prompt engineering techniques assume you understand tokenization and context windows

---

## Visualizations

This chapter includes three key diagrams (generated from `gen-scripts/`):

1. **multihead-attention-flow.png** — Data flow through Q/K/V projection, attention computation, and output projection
2. **encoder-decoder-comparison.png** — Attention patterns for encoder-only, decoder-only, and encoder-decoder architectures
3. **kv-cache-inference.png** — Referenced in Ch.2 for inference optimization (KV caching)

To regenerate diagrams:
```bash
cd gen-scripts
python generate-all.py
```

---

## Next Steps

✅ **You've completed:** Transformer fundamentals
➡️ **Read next:** [Ch.2 — LLM Inference Mechanics](../ch02-llm-inference-mechanics)

