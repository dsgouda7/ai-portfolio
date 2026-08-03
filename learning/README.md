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
