# Contents: Agentic AI Platform System Design

Read the whole-platform reference first, then deepen the component you need.

## Required route

This branch adds no required lessons to the main route.

## Optional

- [Designing an Agentic AI Platform — System Design](system-design.md) — (optional) Map the full platform and separate what a model proposes from what the surrounding system may authorize and record.
- [Foundations of Agentic Systems](01-foundations-of-agentic-systems.md) — (optional) Separate a fixed workflow from a model-directed one and locate the system components that retain control.
- [Agent Lifecycle & Runtime](02-agent-lifecycle-and-runtime.md) — (optional) Follow how an agent definition becomes a versioned running execution with explicit lifecycle boundaries.
- [03 — Tool, MCP & Skill Registry](03-tool-mcp-and-skill-registry.md) — (optional) Register reusable tools and instructions so discovery cannot bypass version checks or action permissions.
- [04 — Model Gateway & LLM Providers](04-model-gateway-and-llm-providers.md) — (optional) Change model providers without scattering provider-specific behavior through every agent prompt and tool.
- [State Management & Memory in Agentic AI Systems](05-state-management-and-memory.md) — (optional) Persist plans, observations, approvals, and versions so a run can be explained and replayed after restart.
- [Non-Determinism, Loops & Termination](06-non-determinism-loops-and-termination.md) — (optional) Detect repeated behavior and enforce time, token, and spending limits even when the model does not stop itself.
- [Agent Evaluation Frameworks](07-agent-evaluation-frameworks.md) — (optional) Evaluate a multi-step path when the final answer alone cannot reveal wrong decisions or tool use.
- [Observability, Tracing & Agent Health](08-observability-tracing-and-health.md) — (optional) Track whether an agent is making useful decisions rather than merely whether its service is online.
- [Multi-Agent Communication Patterns](09-multi-agent-communication-patterns.md) — (optional) Choose how specialists exchange messages, converge on a result, and retain attribution for each decision.
- [Recoverability, Rollbacks & the Saga Pattern](10-recoverability-rollbacks-and-saga.md) — (optional) Handle partial tool failures with recorded compensating actions when earlier side effects cannot simply be undone.
- [Governance, Guardrails & Security for Agentic Platforms](11-governance-guardrails-and-security.md) — (optional) Enforce action permissions and audit trails even when untrusted text tries to redirect the agent.
- [Production Scale & Capacity for Agentic Platforms](12-production-scale-and-capacity.md) — (optional) Bound concurrent executions while preserving separate customer data, cost limits, and operational evidence.
- [Semantic Kernel vs. LangGraph](13-semantic-kernel-vs-langgraph.md) — (optional) Compare how two workflow frameworks represent control, state, and execution responsibilities in code.
