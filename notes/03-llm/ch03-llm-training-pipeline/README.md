# LLM Training Pipeline — Pretraining, SFT, and Alignment

**Status:** Complete

How raw internet text becomes GPT-4 or Claude: the three-stage pipeline (pretraining → supervised fine-tuning → RLHF/DPO) that every frontier model uses, plus parameter-efficient fine-tuning (LoRA) for domain adaptation.

## Contents

- [training-pipeline.md](training-pipeline.md) — Core chapter content
  - The three training stages: pretraining, SFT, RLHF/DPO
  - Why models from different companies have different personalities
  - Pretraining corpus composition (LLaMA 2 breakdown)
  - RLHF vs DPO: when to use each
  - LoRA and prefix tuning: adapting without full retraining

> **From-scratch companion (Part 8):** [`learning/genai/08-transformers/02-decoder-only-language-model.ipynb`](../../../learning/genai/08-transformers/02-decoder-only-language-model.ipynb) — Part 8 trains a MiniLM decoder-only transformer end to end: vocabulary → embeddings → transformer blocks → cross-entropy training loop → accuracy curve. Read it before this chapter to make the training-pipeline concepts concrete.

## Learning Objectives

After completing this chapter, you should be able to:

1. **Understand the training pipeline**
   - Distinguish pretraining (language knowledge), SFT (instruction format), and RLHF (alignment) as separate regimes with different data, cost, and outcomes
   - Explain why RLHF optimizes for human approval rather than truth
   - Trace how the same base architecture produces GPT-4 vs Claude's distinct behaviors

2. **Reason about scale and cost**
   - Estimate pretraining cost from parameter count and token count
   - Explain why emergent capabilities appear at specific scale thresholds
   - Understand why more pretraining data doesn't fix alignment problems

3. **Apply parameter-efficient fine-tuning**
   - Explain how LoRA reduces trainable parameters by 99% vs full fine-tuning
   - Choose between LoRA, prefix tuning, and full fine-tuning for a given scenario
   - Understand why fine-tuning a pretrained model outperforms training from scratch

## Prerequisites

- **Ch.01** — Transformer architecture (attention, positional encoding, architecture families)
- **Ch.02** — LLM inference mechanics (autoregressive generation, sampling)
- **notes/02 Ch.7–8** *(optional)* — Self-supervised pretraining context (SimCLR, MAE) deepens the pretraining intuition

## Key Concepts

| Concept | Analogy | Why It Matters |
|---|---|---|
| **Pretraining** | Building a brain from reading the entire internet | Creates the language knowledge and world model; 95% of capability |
| **SFT (Supervised Fine-Tuning)** | Installing a steering wheel | Teaches instruction-following format from human demonstrations |
| **RLHF** | Teaching which roads humans prefer | Aligns behavior to human preferences via pairwise comparisons |
| **DPO (Direct Preference Optimization)** | Offline RLHF without a separate reward model | Simpler and often comparable to RLHF with less training instability |
| **LoRA** | Adding adjustable knobs to a frozen machine | Adapts any fine-tuning task by training only 0.1–1% of parameters |
| **Emergence** | Latent capacity unlocked by scale | Capabilities in training data, only accessible above a parameter threshold |

## ML → LLM Bridge

If you completed **notes/02**, these connections anchor the new material:

- **Ch.09 (Knowledge Distillation)**: DistilBERT and TinyLLaMA use the *identical* temperature-scaled KL loss from distillation. The "teacher" model at inference provides soft targets over the 50K-token vocabulary — this is "dark knowledge" at LLM scale.
- **Ch.10 (Pruning & Mixed Precision)**: LoRA is a form of structured compression applied *before* training rather than after. The rank decomposition ($\Delta W = BA$, $r \ll d$) exploits the same low-rank hypothesis as structured pruning.
- **Ch.07 (Contrastive Learning)**: The reward model in RLHF is trained with a contrastive-style objective — rank the preferred response higher than the rejected one. The InfoNCE-style margin loss from SimCLR is the conceptual ancestor.

## Quick Start

```bash
# Open the training pipeline notes
code notes/03-llm/ch03-llm-training-pipeline/training-pipeline.md
```

## Common Questions

**Q: Does more pretraining data always help?**
A: No. After a certain quality threshold, data diversity matters more than quantity. Models trained on carefully filtered 1T tokens routinely outperform those trained on noisily scraped 10T tokens (see Mistral 7B vs early LLaMA variants).

**Q: Why can't we just skip pretraining and RLHF a randomly initialized model?**
A: RLHF refines existing capabilities; it doesn't create them. A randomly initialized model has no language knowledge to align. Pretraining provides 95% of capability; RLHF shapes the final 5%.

**Q: What's the practical difference between RLHF and DPO?**
A: RLHF trains a separate reward model then uses PPO to optimize against it — two training steps, higher instability. DPO directly optimizes the policy on preference pairs without a separate reward model — simpler, often comparable quality, and now the default at most labs.

## Next Chapter

[Ch.04 — LLM Model Internals](../ch04-llm-model-internals/README.md): With the training pipeline understood, Ch.04 goes inside the deployed model — parameter counts, VRAM budgeting, quantization trade-offs, and Mixture of Experts architecture.
