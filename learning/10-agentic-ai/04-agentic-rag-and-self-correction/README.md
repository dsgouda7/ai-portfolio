# Chapter 04 — Agentic RAG and Bounded Self-Correction

**Failure being repaired:** Retrieval returns a superseded policy, while an unconstrained rewrite loop can spend forever polishing the wrong evidence.

**Prerequisites:** [Chapter 03](../03-durable-workflows-with-langgraph/README.md) and [Stage 07 RAG](../../07-rag/README.md), especially retrieval metadata and citation grounding.

## Local Learning Contract

- **Predict:** Decide whether a retrieved passage is current, authoritative, and sufficient before allowing an answer or rewrite.
- **Implement:** Wrap retrieval as a metadata-aware tool, permit one bounded rewrite, and add conflict and citation checks.
- **Inspect:** Compare document version/status, query attempt, retrieved evidence, citation support, and fallback reason.
- **Prove:** Produce 10/10 grounded citations while stopping after at most two total attempts.
- **Required evidence:** The notebook's freshness, conflict, citation, attempt-bound, and fallback assertions.
- **Bounded transfer:** Set the rewrite allowance to zero for one seeded miss; prove the run falls back immediately rather than issuing an extra retrieval.
- **Relevant system-design reference:** [06 — Non-Determinism, Loops & Termination](../system-design/06-non-determinism-loops-and-termination.md) and [11 — Governance, Guardrails & Security](../system-design/11-governance-guardrails-and-security.md), both **reference-only depth**.

**Previous:** [Chapter 03 — Durable Workflows](../03-durable-workflows-with-langgraph/README.md) · **Next:** [Chapter 05 — Evaluation and Observability](../05-agent-evaluation-and-observability/README.md)

[Open the notebook](04-agentic-rag-and-self-correction.ipynb)
