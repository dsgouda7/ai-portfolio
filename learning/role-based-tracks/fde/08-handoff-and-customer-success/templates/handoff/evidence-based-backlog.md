# Evidence-Based Backlog Template

> A backlog item must be triggered by evidence, an expired assumption, an unresolved authority, or an accepted product gap. Stakeholder volume alone does not establish priority.

## Backlog control

| Field | Value |
|---|---|
| Artifact ID | `HOF-BACKLOG-01` |
| Review cadence | `<weekly in hypercare; monthly afterward>` |
| Product owner | `<owner>` |
| Operations reviewer | `<owner>` |
| Security/data/finance reviewers | `<roles as needed>` |
| Last prioritization decision | `<reference/date>` |

## Item record

| Field | Required entry |
|---|---|
| Backlog ID/title | Stable ID and bounded outcome |
| Evidence trigger | Incident, drill, health review, user feedback set, failed gate, expired claim, unknown, or product gap |
| Claim class/reference | Preserve `Measured`, `Modeled`, or `Customer-validated`; an unknown stays unknown |
| Affected workflow/criterion/control | Stable IDs |
| Current exposure and workaround | Who is affected and how risk is bounded now |
| Expected decision/outcome | Observable change, not implementation preference |
| Priority dimensions | Safety/authority, customer impact, frequency, confidence, urgency, effort, reversibility |
| Owner/reviewer/due date | Named roles and date |
| Validation plan | Test, slice, threshold/decision rule, environment, evidence location |
| Rollout/rollback | Cohort, abort condition, known-good target, compensation if needed |
| Revalidation trigger | Material change or expiry |

## Priority guardrails

1. Security/authorization, prohibited actions, deletion, and evidence-integrity failures outrank convenience features.
2. A blocker to accepted exposure outranks an improvement outside the accepted scope.
3. Measured repeated impact generally outranks a single unsupported request, but sparse high-severity events remain critical.
4. Modeled benefit stays discounted until assumptions are validated.
5. Customer validation establishes importance for a scope; it does not prove technical effect.
6. Work without an owner, validation method, or rollback path is discovery, not implementation-ready.

## Riverside starting backlog

These are evidence-based starting items, not completed prioritization decisions.

| Backlog ID | Trigger | Bounded outcome | Current state | Owner | Exposure decision |
|---|---|---|---|---|---|
| `BLG-RIV-001` | `UNK-RIV-010`; FDE hypercare ownership | Assign post-hypercare model/retrieval quality accountability and evidence duties | Unknown | `PER-RIV-001` | Broad availability blocked until accepted |
| `BLG-RIV-002` | `UNK-RIV-006`; support-hours conflict | Decide funded out-of-hours coverage or explicit fail-closed/queued posture | Unknown | `PER-RIV-001` | First user cohort blocked by frozen case gate |
| `BLG-RIV-003` | Platform limitations | Reconcile production region, endpoint naming, and timeout configuration | Source inconsistency | `<integration owner>` | Cloud readiness blocked |
| `BLG-RIV-004` | Promise/evidence ledger | Produce timed rollback and re-enablement rehearsal evidence | Procedure only | `<operations/release owner>` | Rollback claim blocked |
| `BLG-RIV-005` | `UNK-RIV-005`, `INC-RIV-005` | Establish idempotent PageTurn write or adapter/reconciliation control | Unknown plus seeded incident | `PER-RIV-005` | Workflow writes blocked |
| `BLG-RIV-006` | `INC-RIV-001` | Prevent superseded policy retrieval with lifecycle-aware indexing and evaluation | Seeded incident | `PER-RIV-002` | Affected guidance disabled until gate passes |
| `BLG-RIV-007` | Platform evidence ledger | Establish scoped SLO/capacity/cost baselines from approved environment | Unset/unproven | Operations/finance | Do not make contractual claims |
| `BLG-RIV-008` | Training gate | Run trace, rollback, deletion/access, isolation, and threshold-change drills | Not run during authoring | Customer operations | Handoff remains open |

## Decision record

| Backlog ID | Priority decision | Evidence considered | Tradeoff/rejected alternative | Owner | Next gate |
|---|---|---|---|---|---|
| `<ID>` | `<now/next/later/reject/discovery>` | `<references>` | `<reason>` | `<owner>` | `<gate>` |

## Backlog health check

- [ ] Every item has an evidence trigger or explicit unknown.
- [ ] Safety and authority blockers cannot be outvoted by feature demand.
- [ ] Implementation tasks state the outcome and validation, not only a chosen technology.
- [ ] Modeled benefit is not presented as measured impact.
- [ ] Accepted limitations have an owner, expiry, and exposure bound.
- [ ] Closed items link retained evidence; closure by comment alone is not allowed.
