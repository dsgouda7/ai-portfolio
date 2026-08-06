# Architecture Translation

> **Evidence banner:** `FROZEN FIXTURE`, `HEURISTIC OPTION SCORES`,
> `EXECUTION VERIFIED THEN CLEARED`, `UNVALIDATED PRODUCTION DESIGN`.

This chapter turns Riverside House's bounded use cases into the smallest
defensible system design. You will compare no AI, deterministic software,
search, RAG, a prompt call, fine-tuning, a deterministic workflow, one agent,
and multiple agents. Each option must fail on a named Riverside requirement
before a more complex mechanism is allowed into the design.

The intended result is not an agent platform. It is a phased deterministic
application that composes the minimum useful capabilities:

1. repair policy lifecycle metadata and retain a manual/search-only path;
2. add authorized current-source search and cited answers for `UC-RIV-001`;
3. add a region-approved bounded prompt path for `UC-RIV-002`, using only the
   explicitly selected manuscript;
4. keep rights lookup read-only and route legal interpretation to Rights counsel;
5. defer PageTurn writes until `UNK-RIV-005` resolves idempotency and
   reconciliation behavior;
6. reconsider fine-tuning only after prompt/RAG evidence isolates a stable
   behavior gap, and reconsider an agent only after a real runtime branch cannot
   be enumerated as a workflow.

This is a proposed teaching decision over synthetic facts. It is not a customer
approval, security review, cloud validation, production benchmark, or SLA.

## Start Here

1. Read the frozen [Riverside architecture brief](cases/riverside-architecture-brief.md).
2. Run `setup.ps1` on Windows or `setup.sh` on macOS/Linux when you are ready to
   create the local environment.
3. Open `architecture-translation.ipynb` and select
   `Python (FDE 02 Architecture Translation .venv)`.
4. Run from the top only when you are ready to create local fixture evidence.
5. Copy the resulting decisions into the templates only after reviewing every
   open assumption and evidence class.

The route setup environment was verified, and the notebook executed successfully
against the committed synthetic fixtures. Its outputs were then cleared, so the
committed notebook contains no outputs and all execution counts remain null. This
validation does not establish customer approval, cloud behavior, or production design.

## Failure-First Route

| Step | What it can establish | Riverside failure that forces the next step |
|---|---|---|
| No AI / process repair | Owners, current-policy lifecycle, manual fallback, explicit authority | Cannot produce a bounded continuation or answer paraphrased questions |
| Deterministic software | Identity checks, ACL filters, version rules, schemas, transition validation | Cannot interpret open-ended editorial language or draft prose |
| Search | Returns authorized current passages and stable source IDs | Does not synthesize a supported answer from several passages |
| RAG | Produces a cited answer from authorized current evidence | Generation still needs an approved regional model boundary and cannot grant authority |
| Prompt call | Drafts against an explicitly selected manuscript | Prompt context alone does not discover current policy or enforce source ACLs |
| Fine-tuning | May improve stable style, format, or instruction behavior | Weights are stale, uninspectable evidence and cannot enforce document authorization |
| Deterministic workflow | Encodes known routes, approval, state, retry, and recovery | Only fails if a required next step truly cannot be enumerated in advance |
| Single agent | Chooses a runtime branch or tool from observations | Riverside's frozen use cases do not require that freedom; added loop and tool risks are unearned |
| Multi-agent | Splits independently complex, dynamic work among reasoners | No frozen need requires distributed reasoning; coordination and trust boundaries multiply |

## Required Artifacts

| Artifact | Purpose | Template |
|---|---|---|
| `ARC-01` | Evidence-labeled option matrix and dispositions | [Option matrix](templates/option-matrix-template.md) |
| `ARC-02` | Context, container, sequence, and boundary design | [Boundary register](templates/boundary-register-template.md) |
| `ADR-001` | Smallest-design decision and revisit triggers | [ADR](templates/adr-template.md) |
| `ARC-03` | Customer-readable scope and limitations | [Customer explanation](templates/customer-explanation-template.md) |
| Run record | Observed checks, limitations, and external gaps after an authorized run | [Notebook output record](templates/notebook-output-record.md) |

## Architecture Gate

The chapter passes only when all of these statements are true:

1. Every selected component maps to a frozen use case or policy constraint.
2. Every rejected option has a named Riverside-specific reason and revisit
   trigger.
3. Tenant, actor, role, region, purpose, title, and trace context fail closed at
   ingress and remain available to retrieval and tools.
4. Current-source lifecycle and ACL filtering occur before retrieval ranking.
5. A model never decides authorization, rights, publication, payment, or whether
   its own workflow write is approved.
6. Human approval binds to the exact title, prior state, next state, actor, and
   business idempotency key.
7. PageTurn writes stay disabled while `UNK-RIV-005` is open.
8. UK/EU and US model routes remain separate until regional service, quota,
   price, and failover claims receive external validation under `UNK-RIV-008`.
9. The anti-AI path still supports manual work and authorized source search when
   generation is disabled.
10. The customer explanation distinguishes known facts, modeled assumptions,
    policy constraints, unknowns, and required external validation.

## Files

| Path | Purpose |
|---|---|
| `architecture-translation.ipynb` | Failure-first decision notebook; synthetic execution verified, then cleared |
| `requirements.txt` | Minimal chapter-local kernel dependency |
| `setup.ps1`, `setup.sh` | Local virtual environment and kernel registration |
| `cases/` | Frozen-case reading instructions and decision prompts |
| `templates/` | Reusable `ARC-01`, `ARC-02`, `ADR-*`, `ARC-03`, and notebook run-record artifacts |
| `../shared/fixtures/riverside-engagement-v1.json` | Canonical case facts, unchanged by this chapter |
| `../shared/fixtures/expected-facts-v1.json` | Cross-notebook fact ledger |

## Conceptual Owners

This chapter composes existing mechanisms rather than reteaching them:

- [FDE lifecycle and architecture gate](../00-role-baseline-and-engagement-lifecycle.md)
- [Agentic system design](../../agentic-ai-system-design/system-design.md)
- [Workflow versus agent](../../agentic-ai-system-design/01-foundations-of-agentic-systems.md)
- [Multi-agent tradeoffs](../../agentic-ai-system-design/09-multi-agent-communication-patterns.md)
- [Governance and authority](../../agentic-ai-system-design/11-governance-guardrails-and-security.md)
- [Recovery and idempotency](../../agentic-ai-system-design/10-recoverability-rollbacks-and-saga.md)
- [Hybrid search and the RAG/fine-tuning boundary](../../genai/04-rag/04-hybrid-search.ipynb)
- [RAG failure localization](../../genai/04-rag/05-rag-evaluation.ipynb)
- [Fine-tuning comparison and release decisions](../../genai/03-llm-finetuning/03-llm-finetuning-comparison-and-decision.ipynb)
- [Repository authoring standard](../../../AUTHORING_GUIDE.md)

## Validation Status

The route setup environment was verified, and every notebook cell executed
successfully against the committed synthetic fixtures before outputs were cleared.
The run validates local option logic, assertions, and computed teaching tables.
It does not validate model quality, customer outcomes, shell portability on every
platform, cloud behavior, production mappings, security approval, or customer
acceptance; those remain scoped external or practicum work.
