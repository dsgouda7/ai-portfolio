# Chapter 02a — Skills: Discovery, Composition, and Versioning

**Failure being repaired:** Copied know-how floats across tool versions, loads too much context, and changes decisions without a release boundary.

**Prerequisites:** [Chapter 02](../02-state-context-and-memory/README.md), tool contracts from [Chapter 00](../00-agent-foundations-and-tool-contracts/README.md), and semantic versioning basics.

## Local Learning Contract

- **Predict:** Decide which manifest omission breaks reproducibility and how much context eager disclosure wastes.
- **Implement:** Define manifests, gated discovery, least-capability composition, pinned dependencies, promotion, rollback, and deterministic conflict resolution.
- **Inspect:** Compare catalog versus full-manifest tokens, selected capabilities, pinned versions, evaluation results, and release state.
- **Prove:** Unauthorized discovery fails closed; dependency pins preserve behavior; failed evaluations cannot promote; rollback names a known release.
- **Required evidence:** The notebook scorecard and existing capability-discovery exercise; use the [theory notes](02a-skills-discovery-composition-and-versioning-theory.md) only as a concise recap.
- **Relevant system-design reference:** [03 — Tool, MCP & Skill Registry](../system-design/03-tool-mcp-and-skill-registry.md) and [07 — Agent Evaluation Frameworks](../system-design/07-agent-evaluation-frameworks.md), both **reference-only depth**.

**Previous:** [Chapter 02 — State, Context, and Memory](../02-state-context-and-memory/README.md) · **Next:** [Chapter 03 — Durable Workflows](../03-durable-workflows-with-langgraph/README.md)

[Open the notebook](02a-skills-discovery-composition-and-versioning.ipynb)
