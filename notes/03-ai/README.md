# Agentic AI Track

> **The Mission**: Build **Mamma Rosa's PizzaBot** — a production AI ordering system that beats human phone staff on business metrics while maintaining safety and reliability.

> **Track Status:** ✅ **COMPLETE** — All 11 chapters finished including advanced agentic patterns

This is not a demo chatbot. Every chapter threads through a single production challenge: you're the Lead AI Engineer at a pizza restaurant, and the CEO is skeptical that AI can outperform the existing phone ordering team. You must prove AI can deliver superior business value while meeting strict accuracy, latency, cost, safety, and reliability requirements.

---

## The Grand Challenge: 6 Core Constraints

| # | Constraint | Target | Why It Matters |
|---|------------|--------|----------------|
| **#1** | **BUSINESS VALUE** | >25% conversion + +$2.50 AOV + 70% labor savings | Must beat phone baseline (22% conv, $38.50 AOV, $157k/yr labor). CEO kills project if AI doesn't deliver ROI |
| **#2** | **ACCURACY** | <5% error rate on menu facts | Wrong prices/calories/ingredients destroy customer trust. Phone staff: <2% error |
| **#3** | **LATENCY** | <3s p95 response time | Customers hang up after 5s. Phone staff: ~2s to respond |
| **#4** | **COST** | <$0.08 per conversation | ~$0.08/conv = break-even vs phone labor. Need headroom for profit |
| **#5** | **SAFETY** | Zero successful attacks | Prompt injection, data leakage, jailbreaks = PR disaster + regulatory risk |
| **#6** | **RELIABILITY** | >99% uptime | Downtime = lost orders. Phone system: 99.2% uptime |

---

## Progressive Capability Unlock

| Ch | Title | What Unlocks | Constraints | Status |
|----|-------|--------------|-------------|--------|
| **1** | [LLM Fundamentals](ch01_llm_fundamentals/llm-fundamentals.md) | Understand tokenization, sampling, context windows | Foundation knowledge | 8% conv, 40% errors |
| **2** | [Prompt Engineering](ch02_prompt_engineering/prompt-engineering.md) | System prompts, structured output, few-shot | #2 Partial, #4 tracking | 12% conv, 15% errors |
| **3** | [Chain-of-Thought Reasoning](ch03_cot_reasoning/cot-reasoning.md) | Multi-step query logic, reasoning traces | Logic (not facts) | 15% conv, 10% errors |
| **4** | [RAG & Embeddings](ch04_rag_and_embeddings/rag-and-embeddings.md) | **Ground answers in real menu data** | **#2 ✅ <5% errors!** | 18% conv, 5% errors |
| **5** | [Vector Databases](ch05_vector_dbs/vector-dbs.md) | HNSW, IVF, scaling retrieval | #3 Partial | Fast retrieval |
| **6** | [ReAct & Semantic Kernel](ch06_react_and_semantic_kernel/react-and-semantic-kernel.md) | **Tool orchestration, agent loop** | **#1 ✅ >25% conv!** | 27% conv, +$2.80 AOV |
| **7** | [Safety & Hallucination](ch07_safety_and_hallucination/safety-and-hallucination.md) | **Prompt injection defense, guardrails** | **#5 ✅ Zero attacks!** | Safety hardened |
| **8** | [Evaluating AI Systems](ch08_evaluating_ai_systems/evaluating-ai-systems.md) | Metrics, evals, A/B testing | #6 Partial | Monitoring |
| **9** | [Cost & Latency Optimization](ch09_cost_and_latency/cost-and-latency.md) | **Caching, batching, model selection** | **#3 ✅ #4 ✅** | <2s p95, $0.06/conv |
| **10** | [Fine-Tuning](ch10_fine_tuning/fine-tuning.md) | **Domain adaptation, production polish** | **#6 ✅ >99% uptime** | Production ready |
| **11** | [Advanced Agentic Patterns](ch11_advanced_agentic_patterns/README.md) | **Reflection, debate, memory, Tree-of-Thoughts** | Edge cases: 8% → 0.7% errors | 🎉 **MASTERY!** |

---

## Narrative Arc: From Raw LLM to Production System

### 🎬 Act 1: Foundation (Ch.1-3)
**Understand LLMs, control output format, add reasoning**

- **Ch.1**: What is an LLM? → Tokenization, sampling, training stages
- **Ch.2**: Prompt engineering → System prompts, structured output, few-shot examples
- **Ch.3**: Chain-of-thought → Multi-step reasoning, but still hallucinating facts

**Status**: ❌ All constraints unmet. 15% conversion (worse than 22% phone baseline).

---

### ⚡ Act 2: Grounding & Retrieval (Ch.4-5)
**Eliminate hallucinations with real data**

- **Ch.4**: RAG & Embeddings → **#2 ACHIEVED!** <5% error rate 🎉
- **Ch.5**: Vector DBs → Scale retrieval to millions of documents

**Status**: ✅❌❌❌❌❌ (Accuracy achieved! Conversion improving: 18% → approaching baseline)

---

### 🚀 Act 3: Orchestration & Safety (Ch.6-7)
**Build the agent loop, defend against attacks**

- **Ch.6**: ReAct & SK → **#1 ACHIEVED!** >25% conversion + +$2.80 AOV 🎉
- **Ch.7**: Safety → **#5 ACHIEVED!** Zero successful prompt injections 🎉

**Status**: ✅✅❌❌✅❌ (Business value + Safety unlocked!)

---

### 📊 Act 4: Production Readiness (Ch.8-10)
**Optimize cost/latency, monitor reliability, fine-tune**

- **Ch.8**: Evaluation → Metrics, evals, regression testing
- **Ch.9**: Cost & Latency → **#3 ✅ #4 ✅** <2s p95, $0.06/conv 🎉
- **Ch.10**: Fine-tuning → **#6 ✅** >99% uptime, production polish 🎉

**Status**: ✅✅✅✅✅✅ **ALL CONSTRAINTS SATISFIED!**

---

### 🧠 Act 5: Advanced Patterns & Edge Cases (Ch.11)
**Beyond single-pass reasoning: Handle complex edge cases**

- **Ch.11**: Advanced Agentic Patterns → Reflection, debate, hierarchical orchestration, Tree-of-Thoughts, memory
  - **Edge case error rate**: 8% → **0.7%** (11× improvement)
  - **Escalations to human**: 12% → **2%** (6× reduction)
  - **Complex order handling**: Contradictions, pricing conflicts, multi-step catering
  - **10 patterns**: Reflection, Debate, Hierarchical, Tool Selection, Tree-of-Thoughts, Chain-of-Verification, Constitutional AI, Ensemble, Plan-and-Execute, Memory-Augmented

**Status**: ✅✅✅✅✅✅ **PRODUCTION MASTERY + EDGE CASE HANDLING**

---

## The Business Context: Mamma Rosa's Pizza

**Current baseline (phone staff):**
- 22% order conversion rate (of callers)
- $38.50 average order value
- $157,680/year labor costs (3 staff × $18/hr × 8hr × 365 days)
- ~45 simultaneous call capacity during peak
- <2% error rate (wrong prices, items)
- ~2s average response time

**CEO's challenge:** "Prove AI can beat that — or I'm not investing $300k in a chatbot."

**Your progression:**

```
After Ch.1:  8% conv, 40% errors → "This is embarrassing. Phone staff never give wrong info."
After Ch.2: 12% conv, 15% errors → "Better, but still unreliable. Can't deploy this."
After Ch.3: 15% conv, 10% errors → "Logic is good, facts are wrong. Unusable."
After Ch.4: 18% conv,  5% errors → "Okay, we're approaching parity. What about upselling?"
After Ch.6: 27% conv, +$2.80 AOV → "Wait... this is BETTER than phone staff?!"
After Ch.9: <2s p95, $0.06/conv → "And it's faster AND cheaper?! Let's deploy."
After Ch.10: >99% uptime       → 🎉 "Best decision we made. Scaling to all locations."
After Ch.11: 0.7% edge cases   → 🧠 "Handles complex scenarios that stump phone staff!"
```

---

## What You'll Build

By the end of this track, you'll have:

1. ✅ **RAG pipeline** with vector retrieval eliminating menu hallucinations (Constraint #2)
2. ✅ **Agent orchestration** with tool-calling for order processing, upselling (Constraint #1)
3. ✅ **Safety hardening** against prompt injection, jailbreaks (Constraint #5)
4. ✅ **Production optimization** hitting cost and latency targets (Constraints #3 & #4)
5. ✅ **Monitoring & reliability** infrastructure for >99% uptime (Constraint #6)
6. 🎓 **Deep understanding** of when to use RAG vs fine-tuning, how to eval AI systems, and how to ship agentic products

---

## How to Use This Track

### Sequential (Recommended)
Work through Ch.1 → Ch.10 in order. Each chapter builds on previous concepts and explicitly states what was unlocked vs what's still blocked.

### By Goal

**"I need to build a RAG chatbot"**  
→ [Ch.1](ch01_llm_fundamentals/llm-fundamentals.md), [Ch.2](ch02_prompt_engineering/prompt-engineering.md), [Ch.4](ch04_rag_and_embeddings/rag-and-embeddings.md), [Ch.5](ch05_vector_dbs/vector-dbs.md)

**"I need an agent that uses tools"**  
→ [Ch.1-2](ch01_llm_fundamentals/llm-fundamentals.md), [Ch.6](ch06_react_and_semantic_kernel/react-and-semantic-kernel.md)

**"I need to defend against prompt injection"**  
→ [Ch.2](ch02_prompt_engineering/prompt-engineering.md), [Ch.7](ch07_safety_and_hallucination/safety-and-hallucination.md)

**"I need to optimize cost and latency"**  
→ [Ch.9](ch09_cost_and_latency/cost-and-latency.md)

**"I need production monitoring and evals"**  
→ [Ch.8](ch08_evaluating_ai_systems/evaluating-ai-systems.md)

**"I need to handle edge cases and complex scenarios"**  
→ [Ch.11](ch11_advanced_agentic_patterns/README.md)

### By Constraint

- **#1 Business Value**: [Ch.6](ch06_react_and_semantic_kernel/react-and-semantic-kernel.md), [Ch.10](ch10_fine_tuning/fine-tuning.md)
- **#2 Accuracy**: [Ch.2](ch02_prompt_engineering/prompt-engineering.md), [Ch.4](ch04_rag_and_embeddings/rag-and-embeddings.md)
- **#3 Latency**: [Ch.9](ch09_cost_and_latency/cost-and-latency.md)
- **#4 Cost**: [Ch.9](ch09_cost_and_latency/cost-and-latency.md)
- **#5 Safety**: [Ch.2](ch02_prompt_engineering/prompt-engineering.md), [Ch.7](ch07_safety_and_hallucination/safety-and-hallucination.md)
- **#6 Reliability**: [Ch.8](ch08_evaluating_ai_systems/evaluating-ai-systems.md), [Ch.10](ch10_fine_tuning/fine-tuning.md)

---

## Multi-Agent AI Extension

For multi-agent systems (agent-to-agent delegation, shared memory, event-driven coordination):  
→ See [**Multi-Agent AI Track**](../multi_agent_ai/README.md)

The Multi-Agent track extends the single-agent foundation from this track with:
- Message formats and agent communication protocols
- MCP (Model Context Protocol) for tool standardization
- A2A (Agent-to-Agent) delegation
- Event-driven architectures
- Shared memory patterns (blackboard, pub/sub)
- Trust and sandboxing for multi-agent safety

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

🎯 **Your mission**: Transform from "I can call GPT" (Ch.1) to "I can ship a production agentic system that beats human baselines" (Ch.10).

Start here: **[Ch.1 — LLM Fundamentals](ch01_llm_fundamentals/llm-fundamentals.md)**
