# Chapter 01 — Reasoning, Planning, and Bounded Control

**Contents:** [one-line map of this directory](CONTENTS.md)

**Failure being repaired:** A plausible controller repeats an unchanged action because nothing checks progress or forces a terminal state.

**Prerequisites:** [Chapter 00](../00-agent-foundations-and-tool-contracts/README.md), especially typed tools, retry categories, and fail-closed execution.

## Local Learning Contract

- **Predict:** Decide what an unbounded controller does when observations stop changing and how deadline policy changes useful completion.
- **Implement:** Add an explicit plan, progress check, per-tool timeout, retry policy, run budget, and terminal-state machine.
- **Inspect:** Compare action fingerprints, consumed time, attempts started, remaining steps, and final terminal state.
- **Prove:** Every fixture run stops within eight steps and at least 90% of solvable cases complete.
- **Required evidence:** The notebook's budget assertions plus the existing time-budget and maximum-step exercises.
- **Relevant system-design reference:** [02 — Agent Lifecycle & Runtime](../system-design/02-agent-lifecycle-and-runtime.md) and [06 — Non-Determinism, Loops & Termination](../system-design/06-non-determinism-loops-and-termination.md), both **reference-only depth**.

**Previous:** [Chapter 00 — Agent Foundations and Tool Contracts](../00-agent-foundations-and-tool-contracts/README.md) · **Next:** [Chapter 02 — State, Context, and Memory](../02-state-context-and-memory/README.md)

[Open the notebook](01-reasoning-planning-and-control.ipynb)
