# Shared Riverside FDE Case

This directory contains the frozen, synthetic customer engagement used by all
eight Forward Deployed Engineer notebooks. Riverside House is a fictional
publisher. Every person, organization detail, identifier, metric, incident,
document, and system record here is invented for teaching. The fixtures contain
no customer data, credentials, tokens, connection strings, or usable secrets.

## Canonical Files

| File | Purpose |
|---|---|
| `fixtures/riverside-engagement-v1.json` | Canonical customer, workflow, constraints, assumptions, risks, incidents, rollout, and support facts |
| `fixtures/riverside-source-samples-v1.json` | Deterministic PDF, text, ERP, and API-shaped source records for onboarding exercises |
| `fixtures/expected-facts-v1.json` | Machine-readable facts and intentional tensions notebook outputs can check |
| `schemas/riverside-engagement.schema.json` | JSON Schema for the canonical engagement |
| `schemas/riverside-source-samples.schema.json` | JSON Schema for source samples |
| `schemas/expected-facts.schema.json` | JSON Schema for expected facts |
| `SCHEMA.md` | Field semantics, ID conventions, evidence classes, and extension rules |

`riverside-engagement-v1.json` is the narrative source of truth. The source
samples and expected-facts ledger refer back to its stable IDs. Notebook authors
must not copy and silently modify case facts inside a notebook.

## Engagement In One Paragraph

Riverside House asks for help so editors can "find answers and continue
manuscripts faster." Today an editor searches several policy PDFs, manuscript
folders, an ERP catalog, and workflow APIs, then reconciles conflicting guidance
by email. Riverside wants an authorized editorial assistant that retrieves
current evidence with citations, drafts bounded continuations, and proposes
workflow updates without autonomously publishing text or changing rights and
payment records. The brief is deliberately incomplete: stakeholders disagree on
latency, cloud use, automation, support coverage, volume, and launch scope.

## Evidence Classes

Every quantitative or normative statement uses one of these classes:

| Class | Meaning |
|---|---|
| `measured_baseline` | Synthetic observation supplied as if measured from the current workflow |
| `modeled_assumption` | Planning input for capacity, cost, or rollout analysis; not a measured result |
| `customer_claim` | A stakeholder statement that requires corroboration |
| `policy_constraint` | A stated rule the design must enforce until an owner changes it |
| `external_validation_required` | A cloud, legal, security, or operational claim the local notebooks cannot prove |
| `intentional_conflict` | Two or more supplied statements cannot all be accepted without a decision |
| `unknown` | Required information is absent and must become discovery work |

Notebook results should preserve these labels. A local calculation over a
modeled assumption remains a modeled result; it does not become a production
measurement.

## Stable ID Rules

IDs are immutable within fixture version `RIV-FDE-1.0.0`.

- Use the existing ID when referring to an entity or fact.
- Do not encode a mutable display name, date, or status into a new ID.
- Join fixtures by ID, never by array position or human-readable name.
- Keep superseded records and change their lifecycle fields rather than reusing
  their IDs.
- A future incompatible case must use a new fixture version and new file names.

Common prefixes are `ORG`, `ENG`, `PER`, `ROLE`, `TEN`, `REG`, `SYS`, `SRC`,
`DOC`, `WF`, `MET`, `ASM`, `CON`, `RISK`, `INC`, `COH`, `SUP`, `UNK`, and
`FACT`. Full definitions are in `SCHEMA.md`.

## Intentional Discovery Work

The following tensions are seeded, not data errors:

1. The sponsor says "instant" while Operations proposes different p95 targets
   for retrieval and continuation.
2. Finance forecasts 900 sessions per day while the workflow sample supports a
   620-session planning baseline.
3. Editorial wants automatic workflow updates while Legal prohibits autonomous
   publication, rights, and payment changes.
4. The US imprint accepts approved managed services while the UK/EU imprint says
   unpublished manuscript content must remain in its primary region.
5. The launch brief implies continuous availability while the support agreement
   covers only weekday business hours.
6. Two policy sources disagree about the approval threshold because one is
   superseded but still searchable.
7. Deletion timing, contractor access, metadata failover, and quality acceptance
   slices are intentionally unresolved.

The discovery notebook should surface and assign these items. Later notebooks
must consume the recorded decision or keep the item explicitly unresolved; they
must not invent a convenient answer.

## Notebook Responsibilities

| Notebook | Primary shared inputs |
|---|---|
| 01 Discovery and success criteria | brief, stakeholders, workflow, baselines, conflicts, unknowns |
| 02 Architecture translation | use cases, non-goals, decision boundaries, systems, constraints |
| 03 Data onboarding and contracts | source inventory, source samples, ACLs, versions, parser and deletion failures |
| 04 Identity, isolation, and compliance | tenants, roles, regions, purpose, tool scopes, isolation incident |
| 05 SLA, capacity, and commercials | volume, token, latency, availability, cost, budget, and support assumptions |
| 06 Rollout and change management | cohorts, entry and exit gates, communications, rollback ownership |
| 07 Incident response and recovery | seeded incidents, containment, evidence, severity, re-enablement ownership |
| 08 Handoff and customer success | support matrix, runbook ownership, training, acceptance, health-review cadence |

## Change Control

This package is frozen before notebook implementation. A notebook author who
finds a missing case fact should propose a change to the shared-case owner rather
than editing these fixtures. Accepted changes must update the canonical fixture,
relevant schema, source sample if applicable, expected-facts ledger, and fixture
version together.
