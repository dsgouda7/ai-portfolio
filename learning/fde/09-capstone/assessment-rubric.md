# FDE Capstone Assessment Rubric

Score the submitted package, not the learner's intent or presentation fluency.
Reviewers must cite artifact IDs and findings. Do not average away a critical
identity, authorization, safety, data-deletion, recovery, or ownership failure.

## Readiness scale

Use this scale inside each weighted area:

| Score | Evidence standard |
|---:|---|
| 0 | Absent, unsafe, fabricated, or contradicted by the package |
| 1 | Described but untraceable, unowned, or not testable |
| 2 | Constructed consistently from frozen, synthetic, or modeled inputs; key proof remains external |
| 3 | Demonstrated with reproducible, scoped evidence and independent review in a representative non-production setting |
| 4 | Customer-validated for the exact workflow, environment, owners, conditions, and revalidation trigger |

This authored capstone can support scores 1-2 through static construction. A score
3 needs later executed evidence. A score 4 cannot be created by this notebook or
by fictional approval.

## Weighted assessment

| Area | Weight | Full-credit evidence | Automatic weakness signals |
|---|---:|---|---|
| Claim discipline and traceability | 10 | Complete claim register; correct classes; bidirectional acceptance/component trace; limitations and expiry | Unlabeled numbers, silent claim promotion, broken lineage |
| Discovery and success metrics | 10 | Current workflow, bounded baseline, stakeholders/authority, non-goals, conflicts, unknowns, sliced criteria, golden workflows | Solution-first questions, invented targets, aggregate-only success |
| Architecture and ADR quality | 12 | No-AI through multi-agent comparison; smallest design; boundaries; phased scope; rejected options and revisit triggers | Agent/service/fine-tuning selected by preference or novelty |
| Data contract and onboarding | 10 | Owners, purpose, schemas, mapping, quarantine, ACLs, lineage, sync, deletion, drift, per-source readiness | Treating parse success as retrieval readiness; permissive nulls or stale ACLs |
| Identity, threat, residency, and controls | 12 | Trusted context flow, RBAC, threat model, fail-closed negative tests, residency/service review, residual risk and external owners | Client-supplied authority, unqualified secure/compliant/resident claims |
| Cost, capacity, SLA, and support | 8 | Low/expected/high and stress ranges; sensitivity; all cost categories; quota/headroom; proposed service terms and authorities | Average-only sizing, point estimate as quote, target as measured result |
| Evaluation and rollout | 10 | Versioned slices; separate retrieval/generation/policy/ops gates; uncertainty; shadow/canary/ramp; rollback and compensation | Aggregate gate, threshold chosen after scoring, canary without rollback |
| Incident response and communication | 8 | Containment-first simulation; evidence preservation; known/unknown split; redacted updates; regression and re-enablement authority | Diagnosis before containment, speculative root cause, unsupported recovery promise |
| Operations and handoff | 7 | Signal-to-owner/action/runbook map; drills; support boundaries; acceptance, exit, health review, retirement | Document dump, attendance as competence, FDE remains hidden dependency |
| Project mapping, product gaps, and next iteration | 5 | Honest asset/gap map; documented project mismatches; prioritized gaps; smallest evidence-backed next step | Source existence called capability; roadmap ranked by novelty |
| Scope control and final recommendation | 8 | Recommendation follows evidence, preserves anti-AI/manual path, names blockers and conditions, permits hold/no-go | Calendar-driven launch, hidden exclusions, unsupported production language |
| **Total** | **100** |  |  |

For each area, assign a 0-4 score, multiply by `weight / 4`, and cite findings.
Round only the final total.

## Mandatory deductions

Apply deductions after the weighted score. Record every deduction with the exact
claim and artifact location.

| Unsupported statement or design choice | Deduction |
|---|---:|
| Invented `[Customer-validated]` decision or approval | -15 each |
| Unqualified `secure`, `compliant`, `resident`, `production-ready`, `highly available`, `scalable`, or `within SLA` claim | -10 each |
| Source asset, test file, command, or ADR described as executed/passing evidence | -8 each |
| Live Azure/Databricks behavior, quota, region, price, failover, deletion, identity, networking, or rollback asserted without retained evidence | -8 each |
| Unlabeled quantitative target, estimate, saving, quality, latency, availability, or cost value | -3 each |
| Agent, multi-agent design, fine-tuning job, or cloud service added without a named failed criterion for the smaller option | -5 each |
| Critical authorization, policy, safety, deletion, or duplicate-commit failure averaged into an aggregate pass | -15 each |
| Deployment rollback presented as reversal of a committed external action | -10 each |
| Compliance/legal conclusion made without an authorized owner and scoped review | -10 each |
| Customer content, credentials, secrets, or sensitive identifiers placed in the package or telemetry design without an approved need | -20 each and stop review |

Deductions may reduce the score below zero. Report a minimum displayed score of
zero while retaining the full deduction ledger.

## Critical caps

Regardless of weighted score:

- Cap at **49 / 100** if any cross-tenant or over-privileged path is allowed by
  design, consequential action lacks bound human approval, deleted content is
  knowingly retrievable, or a critical safety failure is averaged away.
- Cap at **59 / 100** if no rollback/compensation path, incident communication,
  or customer operations owner exists.
- Cap at **69 / 100** if the package lacks bidirectional traceability, a golden
  workflow set, sensitivity analysis, or an honest project-gap map.
- A fabricated measurement, customer approval, legal conclusion, or cloud result
  is an automatic **Not ready** outcome even if the arithmetic score is higher.

## Outcome bands

| Final score | Outcome |
|---:|---|
| 85-100 | Review-ready for the next named gate; not a blanket production approval |
| 70-84 | Strong constructed package with material evidence or authority gaps to close |
| 55-69 | Partial package; major traceability, control, rollout, or handoff work remains |
| 0-54 | Not ready; unsafe, unsupported, or insufficiently bounded |

To pass the capstone, the learner needs at least **70 / 100**, no fabricated claim,
no critical cap below 70, and at least score 2 in every weighted area. Identity,
data, evaluation, rollout, incident response, and handoff must each score at least
2 independently.

## Reviewer record

| Field | Value |
|---|---|
| Package version | `[TODO]` |
| Reviewer role and authority | `[TODO]` |
| Weighted score | `[TODO]` |
| Deductions | `[TODO]` |
| Critical cap | `[TODO]` |
| Final score and outcome | `[TODO]` |
| Blocking findings | `[TODO]` |
| Conditions for rereview | `[TODO]` |
| Review date and expiry | `[TODO]` |
