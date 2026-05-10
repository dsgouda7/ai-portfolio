# ⚠️ DEPRECATED — Chapter Split into 4 Focused Chapters

> **This directory is no longer maintained.** The content has been split into 4 separate chapters for better pedagogical structure.

## Old Structure (Single Chapter)

```
ch01-llm-fundamentals/
└── llm-fundamentals.md  (1,300+ lines covering everything)
```

## New Structure (4 Focused Chapters)

The content has been split into:

### **[Ch.1 — Transformer Architecture](../ch01-transformer-architecture/)**
**What it covers:** §0-§2B from original chapter
- Historical thread (RNN → Transformer evolution)
- Tokenization and BPE
- Q/K/V attention mechanics
- Multi-head attention
- Encoder vs Decoder architectures (BERT vs GPT vs T5)
- Positional encoding (sinusoidal, learned, RoPE)

**Path:** `notes/03a-ai/ch01-transformer-architecture/transformer-architecture.md`

---

### **[Ch.2 — LLM Inference Mechanics](../ch02-llm-inference-mechanics/)**
**What it covers:** §3-§3A from original chapter
- Sampling parameters (temperature, top-p, top-k)
- Autoregressive generation loop
- KV caching (10-20× speedup)
- Prefill vs decode phases
- PagedAttention and continuous batching
- Production serving considerations

**Path:** `notes/03a-ai/ch02-llm-inference-mechanics/inference-mechanics.md`

---

### **[Ch.3 — LLM Training Pipeline](../ch03-llm-training-pipeline/)**
**What it covers:** §4-§5 from original chapter
- Pretraining (next-token prediction)
- Supervised Fine-Tuning (SFT)
- RLHF / DPO / RLVR alignment
- PEFT methods (LoRA, prefix tuning, prompt tuning)
- Emergent capabilities (in-context learning, CoT, scaling thresholds)

**Path:** `notes/03a-ai/ch03-llm-training-pipeline/training-pipeline.md`

---

### **[Ch.4 — LLM Model Internals](../ch04-llm-model-internals/)**
**What it covers:** §6-§7 from original chapter
- Parameter breakdown (where the 7B lives)
- Mixture of Experts (MoE)
- VRAM usage (weights, activations, KV cache)
- Quantization (fp16, int8, int4)
- Flash Attention
- Key technical distinctions (18 pairs)

**Path:** `notes/03a-ai/ch04-llm-model-internals/model-internals.md`

---

## Why the Split?

**Before:** 1,300+ line monolithic chapter covering everything from attention mechanics to production serving
**After:** 4 focused chapters, each ~300-400 lines, with clear learning objectives

**Benefits:**
- ✅ Better pedagogical flow (foundational → applied)
- ✅ Easier to navigate and reference
- ✅ Each chapter has focused notebook exercises
- ✅ Clearer dependencies between concepts

## Migration Guide

If you have bookmarks or references to the old `ch01-llm-fundamentals`, use this mapping:

| Old Section | New Location |
|-------------|--------------|
| §0 · Historical Thread | [Ch.1 §0](../ch01-transformer-architecture/transformer-architecture.md) |
| §1 · Core Idea | [Ch.1 §1](../ch01-transformer-architecture/transformer-architecture.md) |
| §2 · Tokenization | [Ch.1 §2](../ch01-transformer-architecture/transformer-architecture.md) |
| §2A · Transformer Architecture | [Ch.1 §2A](../ch01-transformer-architecture/transformer-architecture.md) |
| §2B · Encoder vs Decoder | [Ch.1 §2B](../ch01-transformer-architecture/transformer-architecture.md) |
| §3 · Sampling | [Ch.2 §3](../ch02-llm-inference-mechanics/inference-mechanics.md) |
| §3A · Inference Mechanics | [Ch.2 §3A](../ch02-llm-inference-mechanics/inference-mechanics.md) |
| §4 · Training Stages | [Ch.3 §4](../ch03-llm-training-pipeline/training-pipeline.md) |
| §5 · Emergent Capabilities | [Ch.3 §5](../ch03-llm-training-pipeline/training-pipeline.md) |
| §6 · Model Size & MoE | [Ch.4 §6](../ch04-llm-model-internals/model-internals.md) |
| §6A · Model Internals | [Ch.4 §6A](../ch04-llm-model-internals/model-internals.md) |
| §7 · Key Distinctions | [Ch.4 §7](../ch04-llm-model-internals/model-internals.md) |

---

## Next Steps

Start with [Ch.1 — Transformer Architecture](../ch01-transformer-architecture/) for the foundational understanding, then proceed through Ch.2-4 before moving to applications (Ch.5-8).
