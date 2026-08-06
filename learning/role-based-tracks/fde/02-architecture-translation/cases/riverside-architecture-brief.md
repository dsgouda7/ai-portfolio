# Riverside Architecture Translation Brief

## Decision question

What is the smallest intervention that reduces policy lookup and bounded
continuation time while preserving Riverside's confidentiality, rights,
regional, human-accountability, and workflow-write boundaries?

Use only the canonical facts in
`../../shared/fixtures/riverside-engagement-v1.json`. This brief supplies a
reading order and challenge prompts; it does not replace or amend that fixture.

## Facts to carry into the review

| Decision surface | Stable IDs to inspect | Why it matters |
|---|---|---|
| Required outcomes | `BRF-RIV-001`, `UC-RIV-001`, `UC-RIV-002` | Separate cited answers from bounded drafting |
| Optional workflow scope | `UC-RIV-003`, `CON-RIV-008`, `UNK-RIV-005` | A desired integration is not evidence that writes are safe |
| Rights boundary | `UC-RIV-004`, `NG-RIV-002`, `ROLE-RIGHTS-COUNSEL` | Retrieval may support counsel; it cannot make the legal decision |
| Current baseline | `MET-RIV-001`, `MET-RIV-003`, `MET-RIV-005` | Compare against 18-minute search, 42-minute drafting, and 61% current-first results |
| Identity context | `SEC-RIV-002`, required request context | Missing tenant, role, region, purpose, or title must deny access |
| Model boundary | `SEC-RIV-001`, `CON-RIV-005`, `UNK-RIV-008` | No public upload; regional capacity and approved hosting remain unvalidated |
| Side effects | `SEC-RIV-004`, `INC-RIV-005`, `SLA-RIV-006` | Approval and retries must not create duplicate commits |
| Anti-AI option | `NG-RIV-001` through `NG-RIV-004`, `RISK-RIV-001` | Process repair and manual fallback must be considered before inference |

## Challenge prompts

1. Which parts of the current seven-step workflow are lookup, judgment,
   drafting, approval, and system-of-record mutation?
2. What can policy lifecycle metadata and deterministic filtering fix before a
   model is introduced?
3. When does search suffice, and which requirement specifically needs RAG?
4. Which facts belong in retrieval rather than model weights?
5. What stable behavior, if any, would justify fine-tuning after prompt evidence?
6. Can every control-flow branch be enumerated from the four use cases and
   policy constraints? If yes, why is an agent needed?
7. What independently complex subproblem requires another reasoning agent? If
   none exists, reject multi-agent design.
8. Which exact payload does a human approve before any workflow write?
9. What continues to work when all generative inference is disabled?
10. Which claims remain blocked on customer or external validation?

## Review rule

Prefer composition over labels. Riverside may need deterministic policy,
authorized retrieval, a bounded model call, and a fixed workflow in one product.
That does not make the product an agent. Call it agentic only if a model must
choose the next action at runtime and that freedom is supported by evidence.
