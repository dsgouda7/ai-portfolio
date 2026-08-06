# Learning

Concept-building notebooks and projects for the full GenAI stack.

For original engineering work see [projects/](../projects/).
For exploratory exercises and third-party course material see [playground/](../playground/).

---

## GenAI — `genai/`

The primary learning track.  Covers sequence models, the Transformer architecture,
applied LLM patterns, fine-tuning, and applied mini-projects.

See [genai/README.md](genai/README.md) for the full directory listing, learning arc,
and prerequisites per chapter.

Authoring standard: [genai/authoring-guide.md](genai/authoring-guide.md)

---

## Role-Based Tracks

These tracks live under [role-based-tracks/](role-based-tracks/README.md) because they combine
subject chapters into job-shaped practice rather than introduce another top-level subject. Learn
them alongside the relevant GenAI, agentic AI, and infrastructure chapters.

| Route | Use it when | Current status |
|---|---|---|
| [AI Engineer](role-based-tracks/ai-engineer/README.md) | You need to connect model mechanics, adaptation, retrieval, evaluation, gateways, infrastructure, release lineage, and production feedback. | Route, five operational chapters, Azure-shaped serving tutorial, capstone scaffold, and Riverside platform source are present. The route notebooks were executed successfully locally and then cleared; live Azure evidence and production readiness are not claimed. |
| [Forward Deployed Engineer](role-based-tracks/fde/README.md) | You need to structure discovery, architecture, scope, and evidence across data, identity, rollout, incident, and handoff decisions. | Route, eight engagement chapters, and capstone are present. The notebooks were executed successfully locally and then cleared. Completion supports supervised practicum entry, not independent production execution. |

The FDE route depends on the AI Engineer technical core. It builds discovery, architecture translation,
and evidence discipline. Completing its local or synthetic work does not establish customer acceptance,
production readiness, cloud behavior, commercial authority, or independent execution competence;
production practice requires a supervised authorized engagement.

### Operational Bridges

| Surface | Relationship | Evidence status |
|---|---|---|
| [Azure Operational LLM Serving](ai-infrastructure/09-azure-operational-llm-serving/README.md) | Connects fine-tuning, gateway, quantization, and inference concepts to a local Azure-shaped serving lab. | All 13 code cells executed successfully locally and were then cleared. The p95 gates returned `HOLD_LOCAL_RELEASE` and `HOLD_AZURE_PROMOTION`; Azure behavior is **live-unvalidated**. |
| [Databricks-backed RAG data plane](../projects/rag-knowledge-pipeline/databricks/indexing/OPERATIONS.md) | Extends the local RAG pipeline with remote ingestion, governed records, and Direct Vector Access indexing assets. | Source and bundle assets exist; Databricks behavior is **live-unvalidated**. |
| [Riverside AI Platform](../projects/riverside-ai-platform/docs/README.md) | Composes the shared contracts into an Azure production profile with Azure ML, APIM, evaluation, telemetry, load-test, and IaC assets. | Implemented source assets exist. The non-cloud suite passed 142 tests with 5 cloud tests deselected, and the offline preflight passed 9 tests; deployment and Azure behavior are **live-unvalidated**. |

`Planned` means the target artifact is not present. `Executed and cleared` means a notebook completed
locally before its outputs and execution counts were removed. `Live-unvalidated` means source or design
assets exist but no retained result from the target cloud or customer environment is claimed.

---

## Agentic AI — `agentic-ai/`

A hands-on, ten-notebook track that builds **OrderFlow**, an AI-native purchase-order system, from
typed tool calls through bounded planning, memory, durable LangGraph workflows, agentic RAG,
evaluation, governance, MCP interoperability, multi-agent coordination, and saga recovery.

The default path is deterministic, CPU-only, and offline. Start with
[agentic-ai/README.md](agentic-ai/README.md) for setup, fixtures, chapter order, measured unlocks,
and the rich visual-asset plan.

---

## Agentic AI System Design — `agentic-ai-system-design/`

A system-design reference track (plain Markdown, no notebooks) answering the Staff/Principal-level
interview question *"Design an Agentic AI Platform."* Start at
[agentic-ai-system-design/system-design.md](agentic-ai-system-design/system-design.md) for the
master architecture, then drill into the numbered component docs (agent lifecycle, tool/MCP/skill
registry, model gateway, state & memory, loop/termination control, evaluation, observability,
multi-agent patterns, recoverability & the Saga pattern, governance, production scale) and the
dedicated Semantic Kernel vs. LangGraph comparison.

---

Note: `data-engineering/` and `ml/` (IBM course artefacts) have been moved to
[playground/data-engineering/](../playground/data-engineering/) and
[playground/ml/](../playground/ml/).
