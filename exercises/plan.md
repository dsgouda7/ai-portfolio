# Exercises Track Split Plan: 03-ai → 03-ai + 05-agentic-ai

> **Status:** exercises/03-ai renamed from exercises/03-ai ✅ | exercises/05-agentic-ai NOT YET CREATED ❌

---

## Current State

`exercises/03-ai/` was renamed from `exercises/03-ai`. Its current content is **PizzaBot-focused** (LLM fine-tuning + RAG pipeline for Mamma Rosa's menu). The notes split means:

- **notes/03-ai** → LLM Fundamentals, **Intelligence Audit / Investigation arc** (wiki RAG, prompt engineering, CoT)
- **notes/05-agentic-ai** → Agentic AI, **PizzaBot Grand Challenge arc** (tool orchestration, multi-agent, safety)

The exercises must follow the same split.

---

## Target State

### exercises/03-ai — LLM Fundamentals (Investigation Arc)

Aligns with notes/03-ai chapters (ch01–ch05):
- **Theme:** Staff Engineer AI Literacy Kit investigation — comparing GPT-4 and Claude on wiki RAG
- **Grand Challenge:** Build a wiki Q&A system; measure hallucination rate before/after RAG; benchmark ANN indexes

**Files to keep (generic infrastructure):**
- `src/api.py` — keep (generic LLM API wrapper)
- `src/rag.py` — keep (generic RAG pipeline)
- `src/evaluate.py` — keep (generic evaluation metrics)
- `src/utils.py` — keep (utility functions)
- `src/monitoring.py` — keep (monitoring patterns)
- `requirements.txt` — keep (dependencies)
- `setup.ps1` / `setup.sh` — keep (setup scripts)
- `tests/` — keep (test suite)

**Files to update:**
- `README.md` → Replace PizzaBot framing with Investigation arc (wiki Q&A, hallucination measurement)
- `SOLUTION.md` → Update to reflect Investigation arc solutions
- `config.yaml` → Update dataset references (menu.json → wiki docs)
- `main.py` → Update entry point to use wiki knowledge base
- `src/data.py` → Update data loading to use wiki docs instead of menu JSON
- `src/models.py` → Keep LLM model wrappers, remove pizza-specific fine-tuning
- `src/features.py` → Update feature extraction for wiki/text domain

**Files to replace:**
- `knowledge-base/menu.json` → `knowledge-base/wiki-docs/` (markdown engineering docs)
- `knowledge-base/faqs.txt` → `knowledge-base/wiki-faqs.txt` (generic engineering FAQs)
- `knowledge-base/policies.txt` → `knowledge-base/wiki-policies.txt` (SLA/policy docs)
- `_REFERENCE/api-complete.py` → Update to show completed Investigation arc solution
- `_REFERENCE/models-complete.py` → Update to show completed wiki RAG solution

**Chapter exercise alignment:**
- **ch01 (LLM Fundamentals):** Tokenization, sampling, GPT-4 vs Claude baseline comparison
- **ch02 (Prompt Engineering):** System prompts, few-shot, structured output, injection defense
- **ch03 (CoT Reasoning):** Logic puzzle experiments, self-consistency, visible vs hidden reasoning
- **ch04 (RAG & Embeddings):** Wiki ingestion pipeline, hallucination rate measurement (38% → 4%)
- **ch05 (Vector DBs):** HNSW vs IVF benchmark at 50k docs, ANN index comparison

---

### exercises/05-agentic-ai — Agentic AI (PizzaBot Grand Challenge)

**Does NOT exist yet — must be created.**

Aligns with notes/05-agentic-ai chapters (ch01–ch06):
- **Theme:** PizzaBot Grand Challenge — build a production-grade agentic ordering system
- **Grand Challenge:** Achieve >25% conversion, <5% error, <3s p95 latency, <$0.08/conv

**Directory structure to create:**
```
exercises/05-agentic-ai/
├── README.md                    ← PizzaBot Grand Challenge description
├── SOLUTION.md                  ← Full solution overview
├── config.yaml                  ← PizzaBot config (model, tools, constraints)
├── main.py                      ← Main entry point (agent runner)
├── requirements.txt             ← Dependencies (langchain, langgraph, etc.)
├── setup.ps1                    ← Windows setup
├── setup.sh                     ← Unix setup
├── .gitignore                   ← Python gitignore
├── knowledge-base/
│   ├── menu.json                ← MOVED from 03-ai (Mamma Rosa's full menu)
│   ├── faqs.txt                 ← MOVED from 03-ai (pizza FAQs)
│   └── policies.txt             ← MOVED from 03-ai (delivery policies)
├── src/
│   ├── __init__.py
│   ├── agent.py                 ← ReAct agent loop (Thought→Action→Observation)
│   ├── tools.py                 ← Tool definitions (retrieve_from_rag, check_availability, etc.)
│   ├── rag.py                   ← COPIED from 03-ai (RAG pipeline for menu knowledge)
│   ├── safety.py                ← Guardrails (injection defense, output validation)
│   ├── evaluate.py              ← Agentic eval (task completion, constraint satisfaction)
│   ├── monitoring.py            ← Cost/latency tracking per conversation
│   └── utils.py                 ← COPIED from 03-ai (utility functions)
├── tests/
│   ├── test_agent.py            ← Agent loop tests
│   ├── test_tools.py            ← Tool execution tests
│   └── test_safety.py           ← Injection defense tests
└── _REFERENCE/
    ├── agent-complete.py        ← Full ReAct agent implementation
    └── safety-complete.py       ← Complete guardrails implementation
```

**Chapter exercise alignment:**
- **ch01 (ReAct & Semantic Kernel):** Implement ReAct loop with menu tools
- **ch02 (Agentic RAG):** Wire RAG pipeline to agent tool schema
- **ch03 (Evaluating AI Systems):** Measure task completion rate, constraint satisfaction
- **ch04 (Safety & Guardrails):** Add injection defense, output validation
- **ch05 (Multi-Agent Systems):** Coordinate planner + executor agents
- **ch06 (Cost & Latency):** Optimize per-conversation cost to <$0.08

---

## Implementation Plan

### Phase 1 — exercises/05-agentic-ai Creation (Parallel Agents)

Run these 4 agents in parallel:

**Agent A — Directory scaffold:**
- Create `exercises/05-agentic-ai/` directory structure (all folders)
- Copy `knowledge-base/menu.json`, `knowledge-base/faqs.txt`, `knowledge-base/policies.txt` from `exercises/03-ai/knowledge-base/` to `exercises/05-agentic-ai/knowledge-base/`
- Copy `exercises/03-ai/src/rag.py` → `exercises/05-agentic-ai/src/rag.py`
- Copy `exercises/03-ai/src/utils.py` → `exercises/05-agentic-ai/src/utils.py`
- Copy `exercises/03-ai/requirements.txt` → `exercises/05-agentic-ai/requirements.txt` (add langgraph, langchain-community)
- Copy `exercises/03-ai/.gitignore` → `exercises/05-agentic-ai/.gitignore`
- Create empty `src/__init__.py`, `tests/` directory

**Agent B — Core files (README, SOLUTION, config):**
- Create `exercises/05-agentic-ai/README.md` with PizzaBot Grand Challenge description
  - Mirror format of `exercises/03-ai/README.md` but with PizzaBot arc content
  - Include 6-constraint table, phone baseline, scaffolding level
- Create `exercises/05-agentic-ai/SOLUTION.md` with chapter-by-chapter solution overview
- Create `exercises/05-agentic-ai/config.yaml` with model names, tool schemas, constraint targets

**Agent C — Agent and Tools implementation scaffold:**
- Create `exercises/05-agentic-ai/src/agent.py` with ReAct loop skeleton (TODOs for student to fill)
- Create `exercises/05-agentic-ai/src/tools.py` with tool definitions (retrieve_from_rag, check_availability, calculate_order_total)
- Create `exercises/05-agentic-ai/src/safety.py` with guardrails skeleton
- Create `exercises/05-agentic-ai/main.py` entry point

**Agent D — Reference implementations:**
- Create `exercises/05-agentic-ai/_REFERENCE/agent-complete.py` (full ReAct agent)
- Create `exercises/05-agentic-ai/_REFERENCE/safety-complete.py` (full guardrails)
- Create `exercises/05-agentic-ai/src/evaluate.py` (task completion metrics)
- Create `exercises/05-agentic-ai/src/monitoring.py` (cost/latency per conversation)

### Phase 2 — exercises/03-ai Update (Sequential, lower priority)

After Phase 1 is complete, update exercises/03-ai to align with Investigation arc:
1. Update `README.md` to replace PizzaBot framing with Investigation arc
2. Update `knowledge-base/` to include wiki-style documents instead of pizza menu
3. Update `main.py` entry point to use wiki knowledge base
4. Update `config.yaml` dataset references

> **Note:** The src/ Python files in 03-ai are mostly generic LLM/RAG patterns — they don't need major changes. The key updates are in README, knowledge-base content, and config.

---

## Key Files Reference

| Source | Destination | Action |
|--------|-------------|--------|
| `exercises/03-ai/knowledge-base/menu.json` | `exercises/05-agentic-ai/knowledge-base/menu.json` | COPY |
| `exercises/03-ai/knowledge-base/faqs.txt` | `exercises/05-agentic-ai/knowledge-base/faqs.txt` | COPY |
| `exercises/03-ai/knowledge-base/policies.txt` | `exercises/05-agentic-ai/knowledge-base/policies.txt` | COPY |
| `exercises/03-ai/src/rag.py` | `exercises/05-agentic-ai/src/rag.py` | COPY |
| `exercises/03-ai/src/utils.py` | `exercises/05-agentic-ai/src/utils.py` | COPY |
| `exercises/03-ai/requirements.txt` | `exercises/05-agentic-ai/requirements.txt` | COPY + add langgraph |

> **Do NOT move files** — copy so 03-ai retains its existing exercise infrastructure.

---

## Constraints for Implementation Agents

- Follow notes/05-agentic-ai/authoring-guide.md for PizzaBot arc conventions
- Follow notes/03-ai/authoring-guide.md for Investigation arc conventions
- Scaffold level: 🔴 Minimal (student fills TODOs, reference shows complete solution)
- Security: No hardcoded API keys — use environment variables only
- All Python code must pass basic linting (no obvious syntax errors)
- Minimize LLM calls: agents should read one file, implement it fully, not iterate back-and-forth
- Maximize parallel execution: Agents A-D in Phase 1 are fully independent, run them simultaneously
