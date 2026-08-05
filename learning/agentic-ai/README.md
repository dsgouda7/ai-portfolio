# Agentic AI Learning Track

Build **OrderFlow**, an AI-native B2B purchase-order system, from a fragile tool-calling loop into an auditable multi-agent workflow.

The default path is deterministic, CPU-only, and offline. It uses committed fixtures and provider/service doubles so control flow, safety, recovery, and evaluation claims do not depend on an API key or model availability.

## Prerequisites

1. GenAI foundations: [Transformer architecture](../genai/02-transformers/), [LLM fine-tuning](../genai/03-llm-finetuning/), [RAG](../genai/04-rag/), and [LLM evaluation](../genai/05-llm-evaluation/).
2. Python 3.12.
3. Basic familiarity with typed Python functions and JSON.

The [Agentic AI System Design track](../agentic-ai-system-design/system-design.md) is a complementary architecture reference, not a prerequisite.

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

| # | Notebook | Failure exposed | Measured unlock |
|---:|---|---|---|
| 00 | [Agent Foundations and Tool Contracts](00-agent-foundations-and-tool-contracts/00-agent-foundations-and-tool-contracts.ipynb) | Plausible text is not an executable call | 10/10 valid calls; unknown SKU fails closed |
| 01 | [Reasoning, Planning, and Bounded Control](01-reasoning-planning-and-control/01-reasoning-planning-and-control.ipynb) | Repeated action loop | Every run bounded to eight steps; at least 90% solvable completion |
| 02 | [State, Context, and Memory](02-state-context-and-memory/02-state-context-and-memory.ipynb) | Context overflow and cross-thread leakage | 100% fixture recall, 19% occupancy, zero leakage, restart recovery |
| 03 | [Durable Workflows with LangGraph](03-durable-workflows-with-langgraph/03-durable-workflows-with-langgraph.ipynb) | Hidden branching and crash restart | Correct Finance route; no repeated gather after resume |
| 04 | [Agentic RAG and Self-Correction](04-agentic-rag-and-self-correction/04-agentic-rag-and-self-correction.ipynb) | Superseded policy retrieval | 10/10 grounded citations; at most two attempts |
| 05 | [Agent Evaluation and Observability](05-agent-evaluation-and-observability/05-agent-evaluation-and-observability.ipynb) | Final status hides route and argument defects | 2/2 seeded regressions detected and localized |
| 06 | [Safety, Human Control, and Governance](06-safety-human-control-and-governance/06-safety-human-control-and-governance.ipynb) | Supplier prompt injection crosses authority | 0/5 attacks succeed; complete audit record |
| 07 | [MCP and Agent Interoperability](07-mcp-and-agent-interoperability/07-mcp-and-agent-interoperability.ipynb) | N-by-M integration glue | Discovery, schema rejection, configuration-only provider switch |
| 08 | [Multi-Agent Communication and Coordination](08-multi-agent-communication-and-coordination/08-multi-agent-communication-and-coordination.ipynb) | Generalist overflows and blocks | Specialist context below 80%; modeled latency improves |
| 09 | [Reliability, Recovery, and Production Decisions](09-reliability-recovery-and-production-decisions/09-reliability-recovery-and-production-decisions.ipynb) | Retry duplicates financial side effects | Zero duplicates; failures resume or compensate |

## Setup

PowerShell:

```powershell
cd learning/agentic-ai
./setup.ps1
```

Bash:

```bash
cd learning/agentic-ai
./setup.sh
```

Both scripts create `.venv`, install [requirements.txt](requirements.txt), and register the `agentic-ai` Jupyter kernel.

## Validation

Run shared fixture tests:

```powershell
learning/agentic-ai/.venv/Scripts/python.exe -m unittest discover -s learning/agentic-ai/tests -v
```

Execute every code cell in every notebook and check notebook metadata:

```powershell
learning/agentic-ai/.venv/Scripts/python.exe learning/agentic-ai/scripts/validate_track.py
```

Use `--static-only` to validate JSON, metadata, and Python syntax without execution.

## Shared Code Boundary

[shared/](shared/) contains fixtures, deterministic doubles, trace/token accounting, and common assertions only. Agent loops, planners, memory strategies, graphs, policy engines, protocols, coordination, and recovery mechanisms remain inline in notebooks.

## Authoring Standard

All chapters follow the root [Authoring Guide](../../AUTHORING_GUIDE.md): failure first, one OrderFlow incident per chapter, adjacent proof cells, predictable exercises, mandatory Mermaid diagrams, explicit coverage ledgers, and cleared outputs.
