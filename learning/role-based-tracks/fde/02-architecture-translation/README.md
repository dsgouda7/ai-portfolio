# Architecture Translation

> **Evidence banner:** `FROZEN FIXTURE`, `HEURISTIC OPTION SCORES`,
> `EXECUTION VERIFIED THEN CLEARED`, `UNVALIDATED PRODUCTION DESIGN`.

This chapter turns Riverside House's bounded use cases into the smallest
defensible system design. You will compare no AI, deterministic software,
search, RAG, a prompt call, fine-tuning, a deterministic workflow, one agent,
and multiple agents. Each option must fail on a named Riverside requirement
before a more complex mechanism is allowed into the design.

## Architecture Translation

Riverside House has four jobs to improve: find current policy, draft a bounded
continuation, propose a workflow change, and look up rights restrictions. The
brief suggests AI, agents, and broad automation, but those labels do not say
which problem each component solves or who keeps authority.

This chapter turns that brief into the smallest design that covers the four
jobs. Every component must solve a named Riverside failure. If a simpler path
works, the more complex option stays out.

> **Evidence boundary:** `FROZEN FIXTURE`, `HEURISTIC OPTION SCORES`,
> `EXECUTION VERIFIED THEN CLEARED`, `UNVALIDATED PRODUCTION DESIGN`.
>
> The notebook uses synthetic Riverside facts. It does not establish customer
> approval, security review, cloud behavior, production performance, or an SLA.

## Situation

Editors spend a median of 18 minutes finding policy and 42 minutes drafting a
bounded continuation. Current guidance appears first in only 61 percent of the
replayed policy searches. Speed matters, but Riverside must still enforce title
access, regional processing, rights authority, and human review.

The design therefore starts with process repair and deterministic controls:

1. clean up policy lifecycle metadata and keep a manual/search-only path;
2. search only current sources the requester may use;
3. add cited answers only when search results need synthesis;
4. draft only from an explicitly selected manuscript in an approved region;
5. keep rights lookup read-only and send interpretation to Rights counsel;
6. keep PageTurn writes disabled until Riverside proves retry and reconciliation
   behavior under `UNK-RIV-005`;
7. reconsider fine-tuning or agents only when measured evidence shows a simpler
   path cannot meet the need.

## Sketch

```mermaid
flowchart LR
    A["Four Riverside jobs"] --> B["Repair process and source lifecycle"]
    B --> C["Add deterministic identity, policy, and workflow controls"]
    C --> D["Add authorized search"]
    D --> E["Add cited generation or bounded drafting only where needed"]
    E --> F["Human decides use or approves exact action"]
    F --> G["Keep writes disabled until recovery evidence exists"]
    style A fill:#1e3a8a,stroke:#e2e8f0,stroke-width:2px,color:#ffffff
    style B fill:#1d4ed8,stroke:#e2e8f0,stroke-width:2px,color:#ffffff
    style C fill:#1d4ed8,stroke:#e2e8f0,stroke-width:2px,color:#ffffff
    style D fill:#1d4ed8,stroke:#e2e8f0,stroke-width:2px,color:#ffffff
    style E fill:#b45309,stroke:#e2e8f0,stroke-width:2px,color:#ffffff
    style F fill:#15803d,stroke:#e2e8f0,stroke-width:2px,color:#ffffff
    style G fill:#b91c1c,stroke:#e2e8f0,stroke-width:2px,color:#ffffff
```

The paths remain separate. Policy and rights lookups share authorized search,
but only policy answers may use cited generation. Continuation drafting receives
one selected title. Workflow changes use fixed steps and exact human approval.
A model never grants access, interprets rights, publishes, pays, or approves its
own write.

## Hands-On Check

1. Read the frozen [Riverside architecture brief](cases/riverside-architecture-brief.md).
2. Run `setup.ps1` on Windows or `setup.sh` on macOS/Linux.
3. Open `architecture-translation.ipynb` and select
   `Python (FDE 02 Architecture Translation .venv)`.
4. Run from the top to create local fixture evidence.
5. For each option, record the Riverside failure it fixes, the authority it must
   not receive, and the evidence that would justify revisiting it.
6. Copy reviewed results into the templates. Do not promote notebook output to
   customer or production evidence.

Use these checks while working:

- Remove one required request field. The request must be denied.
- Substitute a superseded policy. It must be excluded before ranking.
- Change an approved workflow argument. The action must require new approval.
- Simulate a model outage. Manual work and authorized search must remain.
- Leave `UNK-RIV-005` open. PageTurn writes must remain disabled.

The route environment and notebook were previously executed successfully against
the committed synthetic fixtures. Outputs were then cleared; execution counts are
null. That run proves only the local fixture logic.

## Decision

The proposed Riverside design is a phased deterministic application, not a
general agent platform.

| Path | Decision now | Reason |
|---|---|---|
| Process repair and deterministic controls | Select first | Own lifecycle, validation, access checks, and fixed transitions |
| Authorized search | Select | Returns current permitted passages and remains available when generation is off |
| Cited policy answer | Select as a read-only option | Helps synthesize several passages; the editor decides whether to use it |
| Bounded continuation draft | Select with regional and title gates | Uses one selected manuscript; the editor accepts or rejects the draft |
| Rights lookup | Select as read-only | Rights counsel keeps interpretation authority |
| PageTurn workflow write | Design but disable | Retry and reconciliation behavior is unresolved |
| Fine-tuning | Defer | Current facts, access, citations, and deletion cannot live safely in model weights |
| Single agent | Reject for now | All four current routes can be written down in advance |
| Multiple agents | Reject for now | No measured coordination need justifies added trust and operating risk |

Revisit fine-tuning only after evaluation finds a stable behavior gap. Revisit
one agent only after representative cases reveal a valuable next step that
cannot be listed in advance. Revisit multiple agents only after one agent is
justified and measured coordination benefit exists.

## Architecture Gate

The chapter passes only when all of these statements are true:

1. Every selected component maps to a frozen use case or policy constraint.
2. Every rejected option has a Riverside-specific reason and revisit trigger.
3. Tenant, actor, role, region, purpose, title, and trace context fail closed at
   entry and remain available to retrieval and tools.
4. Lifecycle and access filtering happen before retrieval ranking.
5. Models never decide authorization, rights, publication, payment, or approval.
6. Human approval is bound to the exact title, prior state, next state, actor,
   and business request key.
7. PageTurn writes stay disabled while `UNK-RIV-005` is open.
8. UK/EU and US model routes stay separate until regional service, quota, price,
   and failover claims are externally validated under `UNK-RIV-008`.
9. Generation can be disabled without removing manual work or authorized search.
10. Customer-facing explanations separate known facts, assumptions, policy
    constraints, unknowns, and required external validation.

## Takeaway

The smallest useful architecture is a composition, not a product label. Keep
current evidence in authorized retrieval, keep authority in deterministic and
human-owned controls, and add model behavior only where a named Riverside task
needs it. Unknowns block exposure; they do not become assumptions.

## Required Artifacts

| Artifact | Purpose | Template |
|---|---|---|
| `ARC-01` | Option decisions and revisit triggers | [Option matrix](templates/option-matrix-template.md) |
| `ARC-02` | System paths and authority boundaries | [Boundary register](templates/boundary-register-template.md) |
| `ADR-001` | Smallest-design decision | [ADR](templates/adr-template.md) |
| `ARC-03` | Customer-readable scope and limitations | [Customer explanation](templates/customer-explanation-template.md) |
| Run record | Observed checks, environment, and limitations | [Notebook output record](templates/notebook-output-record.md) |

## Files and Follow-On Reading

| Path | Purpose |
|---|---|
| `architecture-translation.ipynb` | Riverside option and boundary checks |
| `requirements.txt` | Chapter-local kernel dependency |
| `setup.ps1`, `setup.sh` | Local environment and kernel registration |
| `cases/` | Frozen case and decision prompts |
| `templates/` | Reusable architecture decision artifacts |
| `../shared/fixtures/riverside-engagement-v1.json` | Canonical case facts |
| `../shared/fixtures/expected-facts-v1.json` | Cross-notebook fact ledger |

This chapter builds on the [FDE lifecycle and architecture gate](../00-role-baseline-and-engagement-lifecycle.md),
[workflow and agent guidance](../../../agentic-ai-system-design/01-foundations-of-agentic-systems.md),
[governance and authority](../../../agentic-ai-system-design/11-governance-guardrails-and-security.md),
[recovery and idempotency](../../../agentic-ai-system-design/10-recoverability-rollbacks-and-saga.md),
[Hybrid Search](../../../genai/03-rag/01-hybrid-search.ipynb), and
[RAG Evaluation](../../../genai/03-rag/02-rag-evaluation.ipynb).
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
- [Agentic system design](../../../agentic-ai-system-design/system-design.md)
- [Workflow versus agent](../../../agentic-ai-system-design/01-foundations-of-agentic-systems.md)
- [Multi-agent tradeoffs](../../../agentic-ai-system-design/09-multi-agent-communication-patterns.md)
- [Governance and authority](../../../agentic-ai-system-design/11-governance-guardrails-and-security.md)
- [Recovery and idempotency](../../../agentic-ai-system-design/10-recoverability-rollbacks-and-saga.md)
- [Hybrid search and the RAG/fine-tuning boundary](../../../genai/03-rag/01-hybrid-search.ipynb)
- [RAG failure localization](../../../genai/03-rag/02-rag-evaluation.ipynb)
- [Fine-tuning comparison and release decisions](../../../genai/02-llm-finetuning/03-llm-finetuning-comparison-and-decision.ipynb)
- [Repository authoring standard](../../../../AUTHORING_GUIDE.md)

## Validation Status

The route setup environment was verified, and every notebook cell executed
successfully against the committed synthetic fixtures before outputs were cleared.
The run validates local option logic, assertions, and computed teaching tables.
It does not validate model quality, customer outcomes, shell portability on every
platform, cloud behavior, production mappings, security approval, or customer
acceptance; those remain scoped external or practicum work.
