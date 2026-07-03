# Chain-of-Thought Reasoning — From Prompting to Trained Reasoning Models

**Status:** Complete

From Wei et al. (2022) showing that "Let's think step by step" jumps GSM8K from 17% to 60%, to o1 and DeepSeek-R1 training reasoning as a first-class capability. Covers the full arc: prompted CoT, self-consistency (majority vote), Tree of Thoughts, and the architectural shift where reasoning is no longer a prompting trick but a trained behavior.

## Contents

- [cot-reasoning.md](cot-reasoning.md) — Core chapter content
- [cot-reasoning-supplement.md](cot-reasoning-supplement.md) — Extended worked examples and implementation patterns
  - CoT prompting: zero-shot and few-shot variants
  - Self-consistency: majority vote across multiple reasoning paths
  - Tree of Thoughts: breadth-first search over reasoning branches
  - Process Reward Models (PRMs) vs. Outcome Reward Models (ORMs)
  - o1 / DeepSeek-R1: reasoning as trained behavior
  - Test-time compute scaling

## Learning Objectives

After completing this chapter, you should be able to:

1. **Apply CoT prompting correctly**
   - Distinguish when CoT helps (multi-step reasoning) vs. when it doesn't (lookup tasks)
   - Write effective zero-shot CoT prompts and construct few-shot CoT examples
   - Explain why CoT works: reasoning traces surface intermediate quantities the model can condition on

2. **Use self-consistency and Tree of Thoughts**
   - Apply majority voting across CoT samples to improve reliability
   - Understand when ToT's structured branching justifies its cost vs. simple self-consistency
   - Estimate the cost of N-sample self-consistency vs. a single trained reasoning model call

3. **Understand the trained reasoning paradigm**
   - Explain the architectural and training difference between a prompted CoT model and o1/R1
   - Describe what "thinking tokens" are and why they're invisible in most APIs
   - Reason about when test-time compute scaling makes sense vs. when a direct answer is better

## Prerequisites

- **Ch.05** — Prompt engineering (CoT prompting is a prompting technique; understanding few-shot and system prompts from Ch.05 is assumed)
- **Ch.03** *(helpful)* — RLHF/DPO context helps explain why trained reasoning models (o1) differ from RLHF-aligned chat models

## Key Concepts

| Concept | Analogy | Why It Matters |
|---|---|---|
| **Chain-of-Thought (CoT)** | Showing your working | 17% → 60% on GSM8K benchmark (Wei et al. 2022) |
| **Self-consistency** | Majority vote across independent attempts | Reduces variance; works because reasoning paths are diverse |
| **Tree of Thoughts (ToT)** | Structured search with backtracking | Best for combinatorial problems where early errors are catastrophic |
| **Process Reward Model (PRM)** | Grading each reasoning step, not just the answer | Enables training reasoning quality, not just outcome correctness |
| **Test-time compute scaling** | Thinking longer → better answers | GPT-o1 / DeepSeek-R1 principle: compute budget at inference trades off against model size |
| **Trained reasoning (o1/R1)** | Reasoning baked in at training time | Consistent CoT without prompting; thinking tokens invisible to user |

## Key Stat

**GSM8K arithmetic benchmark (Wei et al., 2022):**
- Standard prompting: 17% accuracy (GPT-3 175B)
- Chain-of-Thought prompting: **60% accuracy** (same model, same weights)

Zero prompt change to the model — only the *input format* changed.

## Quick Start

```bash
code notes/03-llm/ch06-cot-reasoning/cot-reasoning.md
```

## Common Questions

**Q: Does CoT always help?**
A: No. CoT helps when the task requires multi-step reasoning where intermediate results matter (math, logic, code analysis). It can *hurt* for tasks that benefit from fast, direct retrieval (e.g., capital city lookups, simple factual queries) because forcing step-by-step reasoning introduces more opportunities for the model to confabulate.

**Q: How does self-consistency compare to simply using a larger model?**
A: At equal budget, self-consistency (e.g., 8 samples from a 7B model with majority vote) often matches or exceeds a single call to a model 5–10× larger. The tradeoff: latency (you wait for 8 calls) vs. throughput (they can run in parallel).

**Q: What is DeepSeek-R1 and how does it differ from o1?**
A: Both train reasoning as an explicit capability. o1's internals are proprietary. DeepSeek-R1 is open-source (Apache 2.0) and uses GRPO (Group Relative Policy Optimization) — reinforcement learning with process-level rewards — to train chains of reasoning directly. The model generates visible `<think>...</think>` blocks before its final answer.

## Next Chapter

[Ch.07 — RAG and Embeddings](../ch07-rag-and-embeddings/README.md): CoT helps models reason better over what they know. Ch.07 addresses a different problem: what they don't know — or what's outdated in their weights. RAG (Retrieval-Augmented Generation) grounds model responses in external knowledge, reducing hallucinations by 38%–90% on knowledge-intensive tasks.
