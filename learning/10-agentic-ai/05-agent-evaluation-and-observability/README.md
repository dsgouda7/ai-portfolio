# Chapter 05 — Agent Evaluation and Observability

**Contents:** [one-line map of this directory](CONTENTS.md)

**Failure being repaired:** A successful final status hides a wrong route, a wrong tool argument, and where cost or latency accumulated.

**Prerequisites:** [Chapter 04](../04-agentic-rag-and-self-correction/README.md), Stage 08 evaluation concepts, and the Stage 10 trace vocabulary.

## Local Learning Contract

- **Predict:** Decide which seeded defect can pass a final-status-only check and which span must reveal it.
- **Implement:** Emit correlated spans, attribute cost/latency, replay deterministic fixtures, and compare trajectories to regression expectations.
- **Inspect:** Read route order, tool arguments, parent/child span IDs, token/cost fields, latency, and localized diff output.
- **Prove:** Detect and localize both seeded regressions (2/2), not merely report a failed final status.
- **Required evidence:** The notebook's span-integrity, attribution, replay, and two-regression assertions.
- **Bounded transfer:** Mutate one non-terminal tool argument in a copied trace fixture; prove the trajectory check names the argument mismatch while final status remains unchanged.
- **Relevant system-design reference:** [07 — Agent Evaluation Frameworks](../system-design/07-agent-evaluation-frameworks.md) and [08 — Observability, Tracing & Health](../system-design/08-observability-tracing-and-health.md), both **reference-only depth**.

**Previous:** [Chapter 04 — Agentic RAG](../04-agentic-rag-and-self-correction/README.md) · **Next:** [Chapter 06 — Safety, Human Control, and Governance](../06-safety-human-control-and-governance/README.md)

[Open the notebook](05-agent-evaluation-and-observability.ipynb)
