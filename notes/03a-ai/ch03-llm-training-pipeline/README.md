# LLM Training Pipeline

How raw text predictors become instruction-following assistants through pretraining, supervised fine-tuning, and alignment (RLHF/DPO/RLVR).

## Contents

- [training-pipeline.md](training-pipeline.md) — The three training stages, PEFT preview (LoRA, prefix tuning), and emergent capabilities at scale

## Visual Assets

- **img/** — Diagrams (training pipeline flowchart, RLHF loop, DPO comparison, PEFT adapter architectures)
- **gen-scripts/** — Python scripts to generate training stage visualizations

---

**Related chapters:**
- [Ch.1 — LLM Fundamentals](../ch01-transformer-architecture/) — Architecture, tokenization, inference mechanics
- [Ch.2 — Prompt Engineering](../ch05-prompt-engineering/) — Instruction design, few-shot learning, prompt patterns
- [Ch.4 — Model Internals](../ch04-llm-model-internals/) — Attention patterns, layer specialization, interpretability
- [03b-agentic-ai Ch.5 — Fine-Tuning](../../03b-agentic-ai/ch05-fine-tuning/) — Full PEFT implementation (LoRA, QLoRA, prefix tuning)
