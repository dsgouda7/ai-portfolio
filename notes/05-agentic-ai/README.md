# Agentic AI Track — 05

> **Prerequisites:** Complete [LLM Fundamentals (03-ai)](../03-ai) before starting this track. You need a solid understanding of prompting, chain-of-thought reasoning, and RAG before building agents.

This track is the **PizzaBot Grand Challenge** — 6 chapters taking Mamma Rosa's pizza ordering bot from a broken prototype (8% conversion) to a production-grade agent (32% conversion) satisfying all 6 business constraints.

---

## The Grand Challenge — 6 Constraints

The CEO has set 6 non-negotiable requirements:

| # | Constraint | Target | Status |
|---|---|---|---|
| 1 | BUSINESS VALUE | >25% conversion + +$2.50 AOV + 70% labor savings | Ch.1 |
| 2 | ACCURACY | <5% error rate | Ch.3 |
| 3 | LATENCY | <3s p95 | Ch.4 |
| 4 | COST | <$0.08/conversation | Ch.4 |
| 5 | SAFETY | Zero successful attacks | Ch.2 |
| 6 | RELIABILITY | >99% uptime | Ch.5 |

---

## Track Overview

| Chapter | Capability Unlocked | Constraint Met | Conversion |
|---|---|---|---|
| [Ch.1 — ReAct & Semantic Kernel](ch01-react-and-semantic-kernel) | Tool orchestration + proactive upselling | #1 Business Value | 8% → 27% |
| [Ch.2 — Safety & Hallucination](ch02-safety-and-hallucination) | Prompt injection defense, guardrails | #5 Safety | 27% → 27% (hardened) |
| [Ch.3 — Evaluating AI Systems](ch03-evaluating-ai-systems) | RAGAS metrics, LLM-as-judge, hallucination rate | #2 Accuracy | 27% → 30% |
| [Ch.4 — Cost & Latency](ch04-cost-and-latency) | KV caching, model tiers, streaming | #3 Latency + #4 Cost | 30% → 32% |
| [Ch.5 — Fine-Tuning](ch05-fine-tuning) | LoRA adapter for brand voice | #6 Reliability | 32% → 32% (optimized) |
| [Ch.6 — Advanced Agentic Patterns](ch06-advanced-agentic-patterns) | Multi-agent, LangGraph, HITL | System maturity | Full production |

---

## Narrative Arc — Phase Breakdown

### Phase 1: Making It Work (Ch.1)
The bot can finally take orders. ReAct orchestration gives it tool-calling capability — it calls `find_nearest_location()`, `check_item_availability()`, and `calculate_order_total()`. Proactive upselling ("add garlic bread?") pushes conversion past the phone baseline.

### Phase 2: Making It Safe (Ch.2–3)
The CEO read about prompt injection attacks. Ch.2 hardens the system: input validation, output filtering, indirect injection defense. Ch.3 adds measurement — RAGAS metrics quantify exactly where errors come from.

### Phase 3: Making It Fast and Cheap (Ch.4)
Latency at 27% conversion is 4s p95. Cost is $0.11/conv — 37% over budget. Ch.4 fixes both: prefix caching for the menu corpus, model tier routing, streaming responses.

### Phase 4: Making It Excellent (Ch.5–6)
Fine-tuning bakes Mamma Rosa's brand voice into the model — reduces output variance and cost. Ch.6 adds multi-agent architecture for franchise-level coordination.

---

## Progression Metrics

```
Ch.1: ReAct orchestration → 27% conv, +$2.80 AOV, 4s p95 (broke through baseline)
Ch.2: Safety hardening → 27% conv, 0 attacks [#5 SAFETY]
Ch.3: Evaluation framework → 30% conv (quality assurance, faster iteration)
Ch.4: Cost/latency optimization → 32% conv, 1.8s p95, $0.06/conv [#3 #4]
Ch.5: Fine-tuning → 32% conv, $41.00 AOV [#6 RELIABILITY]
Ch.6: Advanced patterns → Production-grade multi-agent system
```

**Final metrics:** 32% conversion (>25% target), $41.80 AOV (+$3.30), 1.8s p95 (<3s), $0.06/conv (<$0.08), 0 attacks, >99% uptime

---

## Navigation

- **Previous track:** [LLM Fundamentals (03-ai)](../03-ai) — complete before starting here
- **Next track:** [Multi-Agent AI (06-multi-agent-ai)](../06-multi-agent-ai)
- **Grand Solution:** [grand-solution.md](grand-solution.md) — full synthesis
- **Authoring Guide:** [authoring-guide.md](authoring-guide.md)
