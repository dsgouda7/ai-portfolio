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

| Route | What is this? | When should I use it? | Where do I start? |
|---|---|---|---|
| AI Engineer | A role-shaped path through model mechanics, adaptation, retrieval, evaluation, gateways, infrastructure, release lineage, and feedback. | You want to connect technical chapters into an observable AI application lifecycle. | [AI Engineer route and baseline](role-based-tracks/ai-engineer/README.md) |
| Forward Deployed Engineer | A customer-engagement path through discovery, architecture, scope, data, identity, rollout, incidents, and handoff. | You want to practice making bounded technical and delivery decisions with customer constraints. | [FDE route and engagement baseline](role-based-tracks/fde/README.md) |

The FDE route builds on the AI Engineer technical core, then adds discovery, architecture translation,
and evidence discipline.

### Operational Bridges

| Surface | What is this? | When should I use it? | Where do I start? |
|---|---|---|---|
| Azure Operational LLM Serving | An Azure-shaped local lab joining fine-tuning, gateways, quantization, and inference. | You want to practice bounded serving and release decisions after learning the component concepts. | [Serving tutorial](ai-infrastructure/09-azure-operational-llm-serving/README.md) |
| Databricks-backed RAG data plane | Remote-ingestion, governed-record, and Direct Vector Access indexing extensions to the local RAG pipeline. | You want to study the data-plane boundary beyond the local RAG chapters. | [RAG data-plane operations](../projects/rag-knowledge-pipeline/databricks/indexing/OPERATIONS.md) |
| Riverside AI Platform | A production-shaped source implementation joining contracts, Azure ML, APIM, evaluation, telemetry, load testing, and IaC. | You want a larger integration surface for route capstones or architecture review. | [Riverside documentation](../projects/riverside-ai-platform/docs/README.md) |

### Validation and Status Transparency

These details describe the evidence behind the routes and bridges; they do not change which route a
learner should choose.

- **Route notebooks:** The AI Engineer notebooks and all nine FDE notebooks were executed successfully
  against local or synthetic fixtures, then outputs and execution counts were cleared so each learner
  begins from a clean state. FDE completion supports entry into supervised practice, not independent
  production execution.
- **Serving safety gates:** All 13 cells in the Azure Operational LLM Serving tutorial executed locally.
  Its p95 checks returned `HOLD_LOCAL_RELEASE` and `HOLD_AZURE_PROMOTION`. A `HOLD` is a deliberate
  safety gate, not an error: release or cloud promotion stays blocked until the required threshold and
  environment evidence pass.
- **Cloud scope:** `Live-unvalidated` means source or design assets are present without a retained result
  from the named cloud or customer environment. This currently applies to the serving tutorial's Azure
  behavior and the Databricks RAG extensions.
- **Riverside scope:** Implemented source assets are present. The non-cloud suite passed 142 tests with
  5 cloud tests deselected, and the offline preflight passed 9 tests. Deployment, live Azure behavior,
  customer acceptance, and production readiness are not claimed.

`Planned` means the target artifact is not present. These labels preserve evidence boundaries; they
are not learner-facing error states.

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
