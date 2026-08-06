# FDE Capstone Assessment Rubric

Use this rubric after draft one and the worked example, when you already have a
decision thread to inspect. Do not design the first draft around point values or
deductions. The rubric is feedback on whether the package tells the truth and
supports a bounded decision; it is not the lesson itself.

Score the submitted package, not the learner's intent or presentation fluency.
Reviewers must cite artifact IDs and findings. Do not average away a critical
identity, authorization, safety, data-deletion, recovery, or ownership failure.

## First-Thread Feedback

Before using points, review draft one with four questions:

1. Can I see the original conflict and who may resolve it?
2. Can I test the acceptance criterion without inventing missing evidence?
3. Does the architecture decision explain why a smaller option is insufficient?
4. Does the recommendation name the safe default, next gate, and owner?

If any answer is no, give narrative feedback and revise the thread. Use the
weighted assessment only after the learner has expanded into the full package.

## Readiness scale

Use this scale inside each weighted area:

| Score | Evidence standard | Riverside example |
|---:|---|---|
| 0 | Absent, unsafe, fabricated, or contradicted by the package | The package calls regional processing approved even though the `uksouth`/`eastus2` mismatch is unresolved |
| 1 | Described but untraceable, unowned, or not testable | "Monitor stale policy" appears in a runbook, but no signal, owner, threshold, or evidence location is named |
| 2 | Constructed consistently from frozen, synthetic, or modeled inputs; key proof remains external | A complete stale-index alert record uses frozen facts and names the live alert test as an external blocker |
| 3 | Demonstrated with reproducible, scoped evidence and independent review in a representative non-production setting | Riverside operators complete the stale-index drill in staging and retain timestamps, actions, limitations, and reviewer findings |
| 4 | Customer-validated for the exact workflow, environment, owners, conditions, and revalidation trigger | The authorized operations owner accepts the exact alert path, support boundary, expiry, and retraining trigger for the launch scope |

This authored capstone can support scores 1-2 through static construction. A score
3 needs later executed evidence. A score 4 cannot be created by this notebook or
by fictional approval.

## Weighted assessment

| Area | Weight | Full-credit evidence | Automatic weakness signals |
|---|---:|---|---|
| Claim discipline and traceability | 10 | Complete claim register; correct classes; bidirectional acceptance/component trace; limitations and expiry | "500 ms" has no class; a modeled load result becomes measured; a claim cannot trace back to its Riverside source |
| Discovery and success metrics | 10 | Current workflow, bounded baseline, stakeholders/authority, non-goals, conflicts, unknowns, sliced criteria, golden workflows | "Improve productivity" has no timer or slice; eleven conflicts become one average score; a target appears without an authorized owner |
| Architecture and ADR quality | 12 | No-AI through multi-agent comparison; smallest design; boundaries; phased scope; rejected options and revisit triggers | An agent is selected before search fails a named criterion; fine-tuning is added without a stable behavior gap |
| Data contract and onboarding | 10 | Owners, purpose, schemas, mapping, quarantine, ACLs, lineage, sync, deletion, drift, per-source readiness | A PageTurn row parses and is called ready despite stale ACLs; deletion stops at the source and ignores the index |
| Identity, threat, residency, and controls | 12 | Trusted context flow, RBAC, threat model, fail-closed negative tests, residency/service review, residual risk and external owners | The request body supplies tenant authority; a disabled contractor can retrieve a title; the region field is called residency proof |
| Cost, capacity, SLA, and support | 8 | Low/expected/high and stress ranges; sensitivity; all cost categories; quota/headroom; proposed service terms and authorities | 620 daily sessions is treated as measured; token cost is presented as a quote; weekday support is described as 24/7 coverage |
| Evaluation and rollout | 10 | Versioned slices; separate retrieval/generation/policy/ops gates; uncertainty; shadow/canary/ramp; rollback and compensation | One average hides a rights failure; the threshold is chosen after results; PageTurn writes enter canary without reconciliation |
| Incident response and communication | 8 | Containment-first simulation; evidence preservation; known/unknown split; redacted updates; regression and re-enablement authority | The update claims root cause before containment; customer text enters the incident record; the operator self-approves re-enable |
| Operations and handoff | 7 | Signal-to-owner/action/runbook map; drills; support boundaries; acceptance, exit, health review, retirement | Riverside receives files but the stale-index alert has no owner; attendance is counted as a drill; only the FDE can recover |
| Project mapping, product gaps, and next iteration | 5 | Honest asset/gap map; documented project mismatches; prioritized gaps; smallest evidence-backed next step | A source file is called deployed capability; the region/name/timeout mismatches disappear from the recommendation; novelty sets priority |
| Scope control and final recommendation | 8 | Recommendation follows evidence, preserves anti-AI/manual path, names blockers and conditions, permits hold/no-go | The autumn date forces launch; US policy is silently included; "production-ready" appears despite critical open gates |
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
