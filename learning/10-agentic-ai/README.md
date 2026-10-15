# Stage 10 — Agentic AI

**Contents:** [one-line map of this directory](CONTENTS.md)

**Previous:** [Stage 09 — LLM Gateway](../09-llm-gateway/) · **Current:** Stage 10 — Agentic AI · **Next:** [Stage 11 — AI Infrastructure](../11-ai-infrastructure/)

Build **OrderFlow**, an AI-native B2B purchase-order system, from a fragile tool-calling loop into an auditable multi-agent workflow.

The default path is deterministic, CPU-only, and offline. It uses committed fixtures and provider/service doubles so control flow, safety, recovery, and evaluation claims do not depend on an API key or model availability.

## Prerequisites

1. GenAI foundations: [Transformer architecture](../04-transformers/), [LLM fine-tuning](../06-llm-finetuning/), [RAG](../07-rag/), and [LLM evaluation](../08-llm-evaluation/).
2. Python 3.12.
3. Basic familiarity with typed Python functions and JSON.

The [Agentic AI System Design track](system-design/system-design.md) is a complementary architecture reference, not a prerequisite.

## OrderFlow Targets

| Constraint | Baseline | Track target |
|---|---:|---:|
| Throughput | 50 POs/day manual | Modeled capacity at or above 1,000/day |
| End-to-end latency | 18 hours average | Modeled 95% within 4 hours |
| Decision errors | 12% after naive overflow | Below 2% on fixtures |
| Context occupancy | Generalist exceeds budget | Every specialist at or below 80% |
| Auditability | Missing attribution | 100% financial transitions attributed |
| Safety | Supplier text can affect approval | Zero successful fixture attacks |
| Reliability | Retry can duplicate PO | Zero duplicate commitments under injected failures |

Production-scale figures are modeled targets unless a notebook explicitly runs a load test. Local fixture metrics are reported separately.

## Chapters

Each local contract states what to predict, implement, inspect, and prove before moving on. The
system-design links provide **reference-only depth**; they do not replace the notebook's executable
evidence.

| # | Notebook | Local contract | Failure exposed | Measured unlock | Reference-only system-design depth |
|---:|---|---|---|---|---|
| 00 | [Agent Foundations and Tool Contracts](00-agent-foundations-and-tool-contracts/00-agent-foundations-and-tool-contracts.ipynb) | [Contract](00-agent-foundations-and-tool-contracts/README.md) | Plausible text is not an executable call | 10/10 valid calls; unknown SKU fails closed | [Foundations](system-design/01-foundations-of-agentic-systems.md), [tool/MCP/skill registry](system-design/03-tool-mcp-and-skill-registry.md) |
| 01 | [Reasoning, Planning, and Bounded Control](01-reasoning-planning-and-control/01-reasoning-planning-and-control.ipynb) | [Contract](01-reasoning-planning-and-control/README.md) | Repeated action loop | Every run bounded to eight steps; at least 90% solvable completion | [Lifecycle/runtime](system-design/02-agent-lifecycle-and-runtime.md), [loops/termination](system-design/06-non-determinism-loops-and-termination.md) |
| 02 | [State, Context, and Memory](02-state-context-and-memory/02-state-context-and-memory.ipynb) | [Contract](02-state-context-and-memory/README.md) | Context overflow and cross-thread leakage | 100% fixture recall, 19% occupancy, zero leakage, restart recovery | [State/memory](system-design/05-state-management-and-memory.md) |
| 02a | [Skills: Discovery, Composition, and Versioning](02a-skills-discovery-composition-and-versioning/02a-skills-discovery-composition-and-versioning.ipynb) · [Theory notes](02a-skills-discovery-composition-and-versioning/02a-skills-discovery-composition-and-versioning-theory.md) | [Contract](02a-skills-discovery-composition-and-versioning/README.md) | Copied know-how floats across tool versions and silently changes decisions | Gated discovery, pinned dependencies, evaluated promotion, deterministic conflicts, and named rollback | [Tool/MCP/skill registry](system-design/03-tool-mcp-and-skill-registry.md), [evaluation](system-design/07-agent-evaluation-frameworks.md) |
| 03 | [Durable Workflows with LangGraph](03-durable-workflows-with-langgraph/03-durable-workflows-with-langgraph.ipynb) | [Contract](03-durable-workflows-with-langgraph/README.md) | Hidden branching and crash restart | Correct Finance route; no repeated gather after resume | [Lifecycle/runtime](system-design/02-agent-lifecycle-and-runtime.md), [state/memory](system-design/05-state-management-and-memory.md), [framework comparison](system-design/13-semantic-kernel-vs-langgraph.md) |
| 04 | [Agentic RAG and Self-Correction](04-agentic-rag-and-self-correction/04-agentic-rag-and-self-correction.ipynb) | [Contract](04-agentic-rag-and-self-correction/README.md) | Superseded policy retrieval | 10/10 grounded citations; at most two attempts | [Loops/termination](system-design/06-non-determinism-loops-and-termination.md), [governance/security](system-design/11-governance-guardrails-and-security.md) |
| 05 | [Agent Evaluation and Observability](05-agent-evaluation-and-observability/05-agent-evaluation-and-observability.ipynb) | [Contract](05-agent-evaluation-and-observability/README.md) | Final status hides route and argument defects | 2/2 seeded regressions detected and localized | [Evaluation](system-design/07-agent-evaluation-frameworks.md), [observability](system-design/08-observability-tracing-and-health.md) |
| 06 | [Safety, Human Control, and Governance](06-safety-human-control-and-governance/06-safety-human-control-and-governance.ipynb) | [Contract](06-safety-human-control-and-governance/README.md) | Supplier prompt injection crosses authority | 0/5 attacks succeed; complete audit record | [Governance/security](system-design/11-governance-guardrails-and-security.md) |
| 07 | [MCP and Agent Interoperability](07-mcp-and-agent-interoperability/07-mcp-and-agent-interoperability.ipynb) | [Contract](07-mcp-and-agent-interoperability/README.md) | N-by-M integration glue | Discovery, schema rejection, configuration-only provider switch | [Tool/MCP/skill registry](system-design/03-tool-mcp-and-skill-registry.md) |
| 08 | [Multi-Agent Communication and Coordination](08-multi-agent-communication-and-coordination/08-multi-agent-communication-and-coordination.ipynb) | [Contract](08-multi-agent-communication-and-coordination/README.md) | Generalist overflows and blocks | Specialist context below 80%; modeled latency improves | [Communication patterns](system-design/09-multi-agent-communication-patterns.md) |
| 09 | [Reliability, Recovery, and Production Decisions](09-reliability-recovery-and-production-decisions/09-reliability-recovery-and-production-decisions.ipynb) | [Contract](09-reliability-recovery-and-production-decisions/README.md) | Retry duplicates financial side effects | Zero duplicates; failures resume or compensate | [Recovery/sagas](system-design/10-recoverability-rollbacks-and-saga.md), [scale/capacity](system-design/12-production-scale-and-capacity.md) |

## Canonical Ownership Matrix

The notebook in the **canonical owner** column owns the runnable teaching contract. Supporting
chapters compose that mechanism in a new failure context; the architecture documents remain
reference-only depth.

| Concern | Canonical hands-on owner | Supporting composition | Reference-only depth | Evidence required before moving on |
|---|---|---|---|---|
| Tool safety | **00** — parse, register, validate, time out, and fail closed | **06** binds authority and approval; **07** rejects protocol/schema drift | [03 — Tool, MCP & Skill Registry](system-design/03-tool-mcp-and-skill-registry.md), [11 — Governance, Guardrails & Security](system-design/11-governance-guardrails-and-security.md) | Invalid JSON, unknown tools, invalid arguments, and unknown SKUs never reach an unsafe action |
| State and restart | **02** owns state/memory and namespace isolation; **03** owns workflow checkpoint, restart, and resume | **09** recovers durable side effects after ambiguous outcomes | [05 — State Management & Memory](system-design/05-state-management-and-memory.md), [02 — Agent Lifecycle & Runtime](system-design/02-agent-lifecycle-and-runtime.md), [10 — Recoverability](system-design/10-recoverability-rollbacks-and-saga.md) | Zero cross-thread leakage and a restart that restores state without redoing the completed gather step |
| Termination | **01** — progress checks, run budgets, retry policy, and terminal states | **04** applies the same boundary to self-correction | [06 — Non-Determinism, Loops & Termination](system-design/06-non-determinism-loops-and-termination.md) | Every run reaches an explicit terminal state within its step/time budget; correction stops after its declared attempt bound |
| Evaluation | **05** — trajectory regression, replay, trace attribution, cost, and latency | Every chapter keeps a local mechanism proof; **02a** gates a skill release | [07 — Agent Evaluation Frameworks](system-design/07-agent-evaluation-frameworks.md), [08 — Observability, Tracing & Health](system-design/08-observability-tracing-and-health.md) | Seeded route and argument regressions are both detected and localized; a final-status-only check is insufficient |

## Inbound Composition from Stage 09

Stage 10 consumes Stage 09 gateway controls rather than silently rebuilding them. This mapping lets
you carry the provider boundary forward without editing or rereading Stage 09 as an agent chapter.

| Stage 09 artifact | Stage 10 composition point | Boundary to preserve |
|---|---|---|
| Normalized provider request/result and typed provider errors | **00** treats the gateway as a typed tool surface; **07** exposes it through a discoverable protocol boundary | Provider normalization does not grant tool authority |
| Eligibility, routing, retry, fallback, and request deadlines | **01** turns outcomes into bounded retry/terminal decisions; **09** handles side-effect recovery | Gateway fallback is not agent-loop termination or transaction compensation |
| Correlation IDs, token/cost fields, and latency telemetry | **05** nests gateway evidence inside the agent trajectory | A successful final response does not prove a correct route or tool argument |
| Exact/semantic cache behavior | **02** records only durable agent state and memory with explicit namespaces | A gateway cache is not agent memory and must not become the restart source of truth |
| Rate, concurrency, and budget signals | **01** enforces per-run limits; **05** attributes consumption; **09** makes the production decision | Modeled gateway capacity is not measured agent throughput |

For the gateway-side mechanisms, return to [Stage 09](../09-llm-gateway/README.md). For the platform
view that connects gateways to agent runtimes, use
[04 — Model Gateway & LLM Providers](system-design/04-model-gateway-and-llm-providers.md) as
**reference-only depth**.

## Setup

PowerShell:

```powershell
cd learning/10-agentic-ai
./setup.ps1
```

Bash:

```bash
cd learning/10-agentic-ai
./setup.sh
```

Both scripts create `.venv`, install [requirements.txt](requirements.txt), and register the `10-agentic-ai` Jupyter kernel.

## Validation

Run shared fixture tests:

```powershell
learning/10-agentic-ai/.venv/Scripts/python.exe -m unittest discover -s learning/10-agentic-ai/tests -v
```

Execute every code cell in every notebook and check notebook metadata:

```powershell
learning/10-agentic-ai/.venv/Scripts/python.exe learning/10-agentic-ai/scripts/validate_track.py
```

Use `--static-only` to validate JSON, metadata, and Python syntax without execution.

## Shared Code Boundary

[shared/](shared/) contains fixtures, deterministic doubles, trace/token accounting, and common assertions only. Agent loops, planners, memory strategies, graphs, policy engines, protocols, coordination, and recovery mechanisms remain inline in notebooks.

## Authoring Standard

All chapters follow the root [Authoring Guide](../../AUTHORING_GUIDE.md): failure first, one OrderFlow incident per chapter, adjacent proof cells, predictable exercises, mandatory Mermaid diagrams, explicit coverage ledgers, and cleared outputs.
