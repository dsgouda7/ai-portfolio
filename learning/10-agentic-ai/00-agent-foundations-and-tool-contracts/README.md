# Chapter 00 — Agent Foundations and Tool Contracts

**Contents:** [one-line map of this directory](CONTENTS.md)

**Failure being repaired:** OrderFlow can emit plausible text, but malformed or unauthorized calls can still reach an action boundary.

**Prerequisites:** [Stage 09 gateway contracts](../../09-llm-gateway/README.md), typed Python functions, and JSON objects.

## Local Learning Contract

- **Predict:** Decide which invalid call fails first and which typed failures are actually retryable.
- **Implement:** Build the smallest registered model-tool-observation loop with schema validation, typed errors, timeouts, and idempotent surfaces.
- **Inspect:** Read the call envelope, validation result, retry category, observation, and terminal outcome—not the prose rationale.
- **Prove:** Show 10/10 valid fixture calls succeed while malformed calls and an unknown SKU fail closed.
- **Required evidence:** The notebook's adjacent assertions and printed pass/reject results, including the existing quantity-change exercise.
- **Relevant system-design reference:** [01 — Foundations](../system-design/01-foundations-of-agentic-systems.md) and [03 — Tool, MCP & Skill Registry](../system-design/03-tool-mcp-and-skill-registry.md), both **reference-only depth**.

**Previous:** [Stage 09 — LLM Gateway](../../09-llm-gateway/README.md) · **Next:** [Chapter 01 — Reasoning, Planning, and Bounded Control](../01-reasoning-planning-and-control/README.md)

[Open the notebook](00-agent-foundations-and-tool-contracts.ipynb)
