# Chapter 02 — State, Context, and Memory

**Failure being repaired:** Prompt history overflows context, leaks facts across threads, and cannot restore a trustworthy run after restart.

**Prerequisites:** [Chapter 01](../01-reasoning-planning-and-control/README.md), explicit terminal states, and basic serialization.

## Local Learning Contract

- **Predict:** Decide which facts belong in current state, working context, long-term memory, or a durable checkpoint.
- **Implement:** Compare memory strategies, namespace retrieval, reject poisoned entries, and checkpoint replayable state.
- **Inspect:** Measure recall, context occupancy, namespace identity, checkpoint contents, and the post-restart transition.
- **Prove:** Reach 100% fixture recall at 19% occupancy, zero cross-thread leakage, and successful restart recovery.
- **Required evidence:** The notebook's recall, occupancy, isolation, poisoning, and replay assertions.
- **Bounded transfer:** Duplicate one isolation probe under a second tenant namespace; change only the namespace and prove the first tenant's fact is absent.
- **Relevant system-design reference:** [05 — State Management & Memory](../system-design/05-state-management-and-memory.md), **reference-only depth**.

**Previous:** [Chapter 01 — Reasoning, Planning, and Bounded Control](../01-reasoning-planning-and-control/README.md) · **Next:** [Chapter 02a — Skills](../02a-skills-discovery-composition-and-versioning/README.md)

[Open the notebook](02-state-context-and-memory.ipynb)
