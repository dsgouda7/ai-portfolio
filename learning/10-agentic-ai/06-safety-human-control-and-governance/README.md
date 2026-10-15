# Chapter 06 — Safety, Human Control, and Governance

**Failure being repaired:** Supplier-controlled text can influence approval unless identity, authority, policy, and approval state are enforced outside the prompt.

**Prerequisites:** [Chapter 05](../05-agent-evaluation-and-observability/README.md), tool validation from [Chapter 00](../00-agent-foundations-and-tool-contracts/README.md), and trust-boundary basics.

## Local Learning Contract

- **Predict:** Decide which prompt-only guardrail fails when untrusted supplier content asks for financial authority.
- **Implement:** Add trust labels, least-privilege tools, policy-as-code, approval bound to state, and append-only decision records.
- **Inspect:** Read content provenance, principal/scope, policy decision, approval binding, attempted action, and audit attribution.
- **Prove:** Block all five fixture attacks (0/5 succeed) and retain a complete audit record for every financial transition.
- **Required evidence:** The notebook's attack-suite, scope, approval-binding, and audit-record assertions.
- **Bounded transfer:** Add one supplier-text variant to the existing attack fixtures without changing policy; prove it remains data and cannot mint approval scope.
- **Relevant system-design reference:** [11 — Governance, Guardrails & Security](../system-design/11-governance-guardrails-and-security.md), **reference-only depth**.

**Previous:** [Chapter 05 — Evaluation and Observability](../05-agent-evaluation-and-observability/README.md) · **Next:** [Chapter 07 — MCP and Interoperability](../07-mcp-and-agent-interoperability/README.md)

[Open the notebook](06-safety-human-control-and-governance.ipynb)
