# Agentic AI Track — Authoring Guide

> **This document tracks the chapter-by-chapter structure of the Agentic AI notes library (03b-agentic-ai).**
> Each chapter lives under `notes/03b-agentic-ai/` in its own folder.
> Read this before editing any chapter to keep tone, structure, and the running example consistent.

<!-- LLM-STYLE-FINGERPRINT-V1
canonical_chapters: ["notes/03b-agentic-ai/ch01-react-and-semantic-kernel/react-and-semantic-kernel.md", "notes/03b-agentic-ai/ch02-safety-and-hallucination/safety-and-hallucination.md"]
voice: second_person_practitioner
register: technical_but_conversational_business_focused
formula_motivation: required_before_each_formula
numerical_walkthroughs: judicious_pizzabot_traces_when_clarifying
dataset: mamma_rosa_pizzabot_only_no_generic_chatbot_examples
failure_first_pedagogy: true
callout_system: {insight:"Insight", warning:"Warning", constraint:"Constraint", optional_depth:"Optional", forward_pointer:"Forward pointer"}
mermaid_color_palette: {primary:"#1e3a8a", success:"#15803d", caution:"#b45309", danger:"#b91c1c", info:"#1d4ed8"}
image_background: dark_facecolor_1a1a2e_for_generated_plots
section_template: [story_header, challenge_0, animation, core_idea_1, running_example_2, technical_content, progress_check_N, bridge_N1]
conversation_trace_style: step_by_step_with_token_counts_and_costs
security_pattern: environment_variables_only_no_hardcoded_keys
forward_backward_links: every_concept_links_to_where_it_was_introduced_and_where_it_reappears
conformance_check: compare_new_chapter_against_ch01_react_before_publishing
red_lines: [no_formula_without_business_metric_consequence, no_concept_without_pizzabot_grounding, no_generic_chatbot_examples, no_section_without_forward_backward_context, no_code_with_security_antipatterns, no_callout_box_without_actionable_content]
-->

---

## The Plan

The Agentic AI track covers 6 core chapters: everything from tool-calling orchestration to multi-agent production architecture. This track builds directly on [LLM Fundamentals (03a-ai)](../03a-ai).

```
notes/03b-agentic-ai/
├── ch01-react-and-semantic-kernel/
│ ├── react-and-semantic-kernel.md
│ └── notebook.ipynb
├── ch02-safety-and-hallucination/
│ ├── safety-and-hallucination.md
│ └── notebook.ipynb
├── ch03-evaluating-ai-systems/
│ ├── evaluating-ai-systems.md
│ └── notebook.ipynb
├── ch04-cost-and-latency/
│ ├── cost-and-latency.md
│ └── notebook.ipynb
├── ch05-fine-tuning/
│ ├── fine-tuning.md
│ └── notebook.ipynb
├── ch06-advanced-agentic-patterns/
│ ├── advanced-agentic-patterns.md
│ └── notebook.ipynb
```

---

## The Running Example — Mamma Rosa's PizzaBot Grand Challenge

Every chapter uses **one consistent system**: **Mamma Rosa's Pizza** — a regional pizza chain where the AI agent must satisfy 6 hard constraints to stay in production.

**The scenario**: *You're the Lead AI Engineer. The CEO has given you 6 weeks to take the prototype PizzaBot (8% conversion, broken tool-calling) to a production-grade agent that beats phone ordering on all 6 metrics.*

| Chapter | What We Build / Constraint Met |
|---|
| Ch.1 — ReAct & Semantic Kernel | Tool orchestration: `find_nearest_location()`, `check_item_availability()`, `calculate_order_total()`. Proactive upselling. **#1 BUSINESS VALUE: 27% conversion** |
| Ch.2 — Safety & Hallucination | Prompt injection defense, input/output guardrails. **#5 SAFETY: 0 attacks** |
| Ch.3 — Evaluating AI Systems | RAGAS metrics, LLM-as-judge, conversion tracking. **#2 ACCURACY: <5% error** |
| Ch.4 — Cost & Latency | KV caching, model tiers, streaming. **#3 LATENCY: 1.8s p95 | #4 COST: $0.06/conv** |
| Ch.5 — Fine-Tuning | LoRA adapter for brand voice + cost reduction. **#6 RELIABILITY** |
| Ch.6 — Advanced Agentic Patterns | Multi-agent coordination, LangGraph, HITL for franchise operations |

**The 6 constraints**: threshold values are: conversion >25%, AOV +$2.50, labor savings >70%, error <5%, latency <3s p95, cost <$0.08/conv, zero attacks, uptime >99%.

---

## Pedagogical Patterns

### The § 0 "Challenge" Pattern
Every chapter opens with `## 0 · The Challenge — Where We Are` showing:
1. Current metric state (what's failing)
2. A concrete failing scenario (CEO confrontation / customer complaint)
3. Why the previous chapter's solution is insufficient
4. What this chapter unlocks (metrics we expect to move)

### Callout Convention
- `**Insight → PizzaBot:**` — connects concept to PizzaBot system
- `**Warning:**` — common pitfall
- `**Constraint #N:**` — ties progress to a specific business constraint
- `**Optional depth:**` — deeper theory for those who want it
- `**Forward pointer:**` — links to where this concept reappears

### Progress Check Pattern
Near chapter end: `## N · Progress Check — Constraint Status` with a metrics table showing before/after for each of the 6 constraints.

### Bridge Pattern
Chapter ends with: `## N+1 · Bridge — What's Next` explaining what gap remains and which chapter addresses it next.

---

## Style Guidelines

### No Emojis

**Do not use emojis in technical content.** All emoji-based callouts have been systematically removed from the repository (27,921 emojis removed across 168 files as of May 2026).

Use text-only formatting:
- **Checkpoint:** (not 💡 **Checkpoint:**)
- **Warning:** (not ⚠️ **Warning:**)
- **Rule of Thumb:** (not 🎯 **Rule of Thumb:**)
- [Complete] or Complete (not ✅)
- [WRONG] or [Failed] (not ❌)

**Rationale:** Emojis create visual clutter, reduce professionalism, and can render inconsistently across platforms. Technical documentation should rely on clear text formatting.

---
