# Chapter 03 — Durable Workflows with LangGraph

**Contents:** [one-line map of this directory](CONTENTS.md)

**Failure being repaired:** Hidden branches make Finance routing unauditable, and a crash causes completed gathering work to run again.

**Prerequisites:** [Chapter 02a](../02a-skills-discovery-composition-and-versioning/README.md), durable state from [Chapter 02](../02-state-context-and-memory/README.md), and Python graph/state-machine basics.

## Local Learning Contract

- **Predict:** Identify which transition should resume after a crash and which completed node must not repeat.
- **Implement:** Build the state machine before mapping it to LangGraph, then add fan-out/fan-in, checkpoints, and state migration.
- **Inspect:** Read node transitions, persisted state, branch decisions, checkpoint version, and per-node invocation counts.
- **Prove:** Route the Finance case correctly and resume with no repeated gather step.
- **Required evidence:** The notebook's branch, crash/resume, invocation-count, and migration assertions.
- **Bounded transfer:** Move the injected crash one transition later; rerun once and prove every pre-crash node still executes exactly once.
- **Relevant system-design reference:** [02 — Agent Lifecycle & Runtime](../system-design/02-agent-lifecycle-and-runtime.md), [05 — State Management & Memory](../system-design/05-state-management-and-memory.md), and [13 — Semantic Kernel vs. LangGraph](../system-design/13-semantic-kernel-vs-langgraph.md), all **reference-only depth**.

**Previous:** [Chapter 02a — Skills](../02a-skills-discovery-composition-and-versioning/README.md) · **Next:** [Chapter 04 — Agentic RAG](../04-agentic-rag-and-self-correction/README.md)

[Open the notebook](03-durable-workflows-with-langgraph.ipynb)
