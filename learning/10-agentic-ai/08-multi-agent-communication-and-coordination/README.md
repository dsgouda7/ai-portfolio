# Chapter 08 — Multi-Agent Communication and Coordination

**Failure being repaired:** One generalist owns every fact and action, overflows context, blocks parallel work, and obscures responsibility.

**Prerequisites:** [Chapter 07](../07-mcp-and-agent-interoperability/README.md), typed envelopes, bounded controllers, and namespace isolation.

## Local Learning Contract

- **Predict:** Decide when routing specialists beats a generalist and where supervisor-worker coordination can deadlock or conflict.
- **Implement:** Add typed task envelopes, task lifecycle, router and supervisor-worker patterns, context partitioning, conflict handling, and A2A discovery.
- **Inspect:** Read sender/recipient/role, task status, correlation and causation IDs, specialist context occupancy, conflict resolution, and modeled critical path.
- **Prove:** Keep every specialist below 80% context occupancy and improve modeled latency without losing attribution.
- **Required evidence:** The notebook's envelope, lifecycle, partition, conflict, occupancy, and latency assertions.
- **Bounded transfer:** Reduce one specialist's context budget by 10%; rerun the same fixture and prove the occupancy gate either still passes or names that specialist.
- **Relevant system-design reference:** [09 — Multi-Agent Communication Patterns](../system-design/09-multi-agent-communication-patterns.md), **reference-only depth**.

**Previous:** [Chapter 07 — MCP and Interoperability](../07-mcp-and-agent-interoperability/README.md) · **Next:** [Chapter 09 — Reliability and Recovery](../09-reliability-recovery-and-production-decisions/README.md)

[Open the notebook](08-multi-agent-communication-and-coordination.ipynb)
