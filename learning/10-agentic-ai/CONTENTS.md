# Contents: Agentic AI

Build OrderFlow from a fragile tool-calling loop into a bounded, auditable, recoverable workflow.

## Required route

- [Agent Foundations and Tool Contracts](00-agent-foundations-and-tool-contracts/README.md) — Reject malformed or unauthorized tool requests before plausible model text can trigger an action.
- [Reasoning, Planning, and Bounded Control](01-reasoning-planning-and-control/README.md) — Detect repeated actions, require progress, and make every controller stop within an explicit budget.
- [State, Context, and Memory](02-state-context-and-memory/README.md) — Keep threads separate and restore trustworthy workflow state without treating the whole conversation as memory.
- [Skills: Discovery, Composition, and Versioning](02a-skills-discovery-composition-and-versioning/README.md) — Load reusable instructions only when needed and bind them to the tool versions and release that make them safe to reuse.
- [Durable Workflows with LangGraph](03-durable-workflows-with-langgraph/README.md) — Make workflow branches visible and resume after a crash without repeating completed gathering work.
- [Agentic Retrieval and Bounded Self-Correction](04-agentic-rag-and-self-correction/README.md) — Reject superseded evidence and limit how often a workflow may rewrite a question or retry an answer.
- [Agent Evaluation and Observability](05-agent-evaluation-and-observability/README.md) — Inspect the route, tool arguments, cost, and timing that a successful final status can hide.
- [Safety, Human Control, and Governance](06-safety-human-control-and-governance/README.md) — Enforce identity, authority, policy, and approval outside the prompt so supplier text cannot grant permission.
- [MCP and Agent Interoperability](07-mcp-and-agent-interoperability/README.md) — Discover tool capabilities and reject incompatible message shapes before a client reaches the wrong provider.
- [Multi-Agent Communication and Coordination](08-multi-agent-communication-and-coordination/README.md) — Divide a workflow among specialists while preserving message ownership, context limits, and responsibility.
- [Reliability, Recovery, and Production Decisions](09-reliability-recovery-and-production-decisions/README.md) — Recover from uncertain financial actions without duplicating commitments or losing the evidence needed to resume.

## Optional

- [Agentic AI System Design](system-design/CONTENTS.md) — (optional) Design the surrounding platform that governs model proposals, tool authority, persistent state, and recovery.
