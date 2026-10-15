# Chapter 02 — State, Context, and Memory

**Contents:** [one-line map of this directory](CONTENTS.md)

**Failure being repaired:** Prompt history overflows context, leaks facts across threads, and cannot restore a trustworthy run after restart.

**Prerequisites:** [Chapter 01](../01-reasoning-planning-and-control/README.md), explicit terminal states, and basic serialization.

**Before the notebook notation:** keep OrderFlow's current shopping list separate from its archive. **Working memory** is the temporary information for this step. **Episodic memory** records earlier purchase events. **Semantic memory** stores reusable facts. **Procedural memory** stores how to perform a task; a supplier email cannot grant itself that status. State is the authoritative purchase record, while context is the selected material shown on one model call.

**Canonical handoff:** after proving recall, namespace isolation, and restart, continue through [Chapter 02a — Skills](../02a-skills-discovery-composition-and-versioning/README.md). The notebook's retained direct link to Chapter 03 is an older short route; it does not waive the versioned know-how, dependency, and authority checks owned by Chapter 02a.

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
