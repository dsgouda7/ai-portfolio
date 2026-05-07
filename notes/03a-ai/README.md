# LLM Fundamentals Track

> **The Mission**: Conduct the **AI Adoption Review** — reverse-engineer how large language models actually work, chapter by chapter, using **GPT-4** and **Claude 3.5 Sonnet** as the two models under investigation. Your deliverable is a documented **AI Literacy Kit** proving deep model understanding.

> **Track Status:** ✅ **COMPLETE** — All 5 chapters finished

This is not a tutorial. Every chapter follows an investigative arc: you are a Staff Engineer leading your company's AI Adoption Review, and the board wants proof you understand these models before committing engineering resources. You run controlled experiments, document findings, and build up from tokenization to full retrieval pipelines — using GPT-4 and Claude as *subjects of study*, not black-box API endpoints.

---

## The Investigation: What the AI Literacy Kit Proves

| Investigation Area | Question | Evidence Produced |
|---|---|---|
| **Model Architecture** | How does a transformer produce text? | Token probability traces, sampling parameter experiments |
| **Training Pipeline** | Why do GPT-4 and Claude behave differently? | Side-by-side output comparisons on identical prompts |
| **Grounding** | What happens when the model doesn't know your data? | Hallucination rate before vs. after RAG |
| **Retrieval** | How does semantic search scale? | HNSW vs IVF benchmark on a 50k-doc corpus |
| **Reasoning** | When does step-by-step thinking help vs. hurt? | CoT accuracy on logic puzzles; confident-but-wrong traces |

---

## Chapter Overview

| Ch | Title | Investigation Beat | Key Experiment | Status |
|----|-------|--------------------|----------------|--------|
| **1** | [LLM Fundamentals](ch01-llm-fundamentals/llm-fundamentals.md) | **"The Black Box"** — what actually happens inside a transformer | Run the same prompt on GPT-4 vs Claude; trace output divergence to tokenization and sampling | ✅ Complete |
| **2** | [Prompt Engineering](ch02-prompt-engineering/prompt-engineering.md) | **"The Control Interface"** — system prompts and few-shot as behavioral levers | Compare GPT-4 vs Claude on identical instructions; map where they diverge | ✅ Complete |
| **3** | [Chain-of-Thought Reasoning](ch03-cot-reasoning/cot-reasoning.md) | **"The Reasoning Engine"** — when step-by-step thinking helps vs. when it hallucinates confidently | GPT-4o1 vs Claude extended thinking on logic puzzles and ambiguous queries | ✅ Complete |
| **4** | [RAG & Embeddings](ch04-rag-and-embeddings/rag-and-embeddings.md) | **"The Memory Problem"** — LLMs don't know your data; grounding changes everything | Hallucination rate on internal docs: 38% before RAG → 4% after | ✅ Complete |
| **5** | [Vector Databases](ch05-vector-dbs/vector-dbs.md) | **"The Index"** — scaling retrieval from 200 docs to 50,000 | HNSW vs IVF benchmark: recall at 1ms, 10ms, 100ms latency budgets | ✅ Complete |

---

## Narrative Arc: The Intelligence Audit

### 🔬 Phase 1: Opening the Black Box (Ch.1-3)
**Understand what LLMs are, how to steer them, and when they reason**

- **Ch.1**: Transformer architecture, tokenization, training pipeline → trace GPT-4 vs Claude output divergence to first principles
- **Ch.2**: System prompts, few-shot, structured output → document how the same instruction produces different behavior across models
- **Ch.3**: Chain-of-thought, extended thinking, Tree-of-Thoughts → identify where reasoning helps and where it produces confident hallucinations

**Finding**: Model behavior is deterministic and explainable — once you understand the architecture and sampling parameters.

---

### 📚 Phase 2: Grounding & Memory (Ch.4-5)
**Solve the knowledge problem: LLMs don't know your data**

- **Ch.4**: RAG & Embeddings → hallucination rate on internal docs drops from 38% → 4% after retrieval grounding
- **Ch.5**: Vector DBs → scale the retrieval layer to 50,000 documents with HNSW; benchmark recall vs. latency

**Finding**: Retrieval-augmented generation is not optional for domain-specific applications — it is the difference between a demo and a deployable system.

---

## The Investigation Progression

**Your experiment log:**

```
After Ch.1: Can explain why GPT-4 and Claude give different answers to the same prompt
After Ch.2: Can predict how each model responds to a given system prompt before running it
After Ch.3: Know exactly when chain-of-thought reasoning helps vs. when to skip it
After Ch.4: Have a working RAG pipeline; hallucination rate on test corpus: 4%
After Ch.5: Can choose the right ANN index for a given latency/recall trade-off
```

---

## What You'll Build

By the end of this track, you'll have:

1. ✅ **Experiment log** comparing GPT-4 vs Claude 3.5 Sonnet behavior across tokenization, sampling, and instruction-following
2. ✅ **Prompt experiment library** documenting system prompt divergence across model families
3. ✅ **Reasoning audit** — which tasks benefit from CoT; which tasks it hurts
4. ✅ **Working RAG pipeline** grounding a 50k-document corpus with <5% hallucination rate
5. ✅ **Retrieval benchmark** — HNSW vs IVF recall/latency at three document scales
6. 🎓 **AI Literacy Kit** — a deliverable proving deep understanding of LLM mechanics to any technical audience

---

## How to Use This Track

### Sequential (Recommended)
Work through Ch.1 → Ch.5 in order. Each chapter builds on the previous and explicitly states what the experiment adds to your AI Literacy Kit.

### By Goal

**"I need to understand why GPT-4 and Claude behave differently"**
→ [Ch.1](ch01-llm-fundamentals/llm-fundamentals.md), [Ch.2](ch02-prompt-engineering/prompt-engineering.md)

**"I need to build a RAG pipeline"**
→ [Ch.1](ch01-llm-fundamentals/llm-fundamentals.md), [Ch.4](ch04-rag-and-embeddings/rag-and-embeddings.md), [Ch.5](ch05-vector-dbs/vector-dbs.md)

**"I need to know when chain-of-thought reasoning helps"**
→ [Ch.3](ch03-cot-reasoning/cot-reasoning.md)

**"I need to scale retrieval to large document corpora"**
→ [Ch.5](ch05-vector-dbs/vector-dbs.md)

**"I'm ready to build agentic systems"**
→ Complete this track, then see [Agentic AI Track](../03b-agentic-ai/README.md)

---

## What Comes Next

**To build production agentic systems on top of this foundation:**
→ [**Agentic AI Track** (03b-agentic-ai)](../03b-agentic-ai/README.md) — PizzaBot Grand Challenge: tool orchestration, safety hardening, cost/latency optimization, fine-tuning, advanced agentic patterns

**To work with multi-agent coordination:**
→ [**Multi-Agent AI Track** (04-multi-agent-ai)](../04-multi-agent-ai/README.md) — message formats, MCP, A2A delegation, event-driven architectures, shared memory

---

## Prerequisites

**Recommended before starting:**
- Python programming (all code examples use Python)
- Basic ML concepts (Ch.1 assumes you know what "training" and "parameters" mean)
- HTTP APIs (you'll call OpenAI/Anthropic/Cohere APIs)

**Not required but helpful:**
- [ML Track](../01-ml/README.md) — especially Ch.18 (Transformers)
- [Math Under The Hood](../00-math_under_the_hood) — for deeper transformer understanding

---

## Chapter Structure

Every chapter follows the same template:

1. **§0 The Challenge** — Current constraint status, what's blocked, what this unlocks
2. **§1 Core Idea** — Plain-English intuition
3. **§2-N** — Detailed sections on concepts, techniques, code
4. **Progress Check** — ✅ Unlocked capabilities, ❌ Still blocked, next preview

Plus a **Jupyter notebook** with runnable code and diagrams.

📚 **Start here**: [AIPrimer.md](ai-primer.md) — running example, conceptual overview, and complete reading guide in one place.

---

## Let's Build

🔬 **Your mission**: Transform from "I can call GPT" (Ch.1) to "I can explain exactly why any LLM behaves the way it does — and ground it in real data" (Ch.5).

Start here: **[Ch.1 — LLM Fundamentals](ch01_llm_fundamentals/llm-fundamentals.md)**
