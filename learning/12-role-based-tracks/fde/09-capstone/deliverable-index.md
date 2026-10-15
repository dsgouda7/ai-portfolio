# Capstone Deliverable Index

Use this file after one minimal decision thread works end to end. It is the
manifest for expanding that thread into a complete engagement review, not the
starting point for learning the capstone.

## Pass 1: Minimal Decision Thread

Before tracking the full package, complete one row of this smaller path:

| Riverside conflict | Acceptance criterion | Smallest-option decision | Evidence or validation gate | Rollout stop | Handoff owner | Recommendation |
|---|---|---|---|---|---|---|
| `[TODO: CON-RIV-*]` | `[TODO: AC-*]` | `[TODO: ADR-*]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO: HOLD / NARROW / PROCEED TO NAMED GATE / DO NOT PROCEED]` |

If you cannot explain that row in plain language, return to the workbook or the
worked example. More templates will not repair a broken decision thread.

## Pass 2: Full Package Manifest

Do not mark an artifact complete because a file exists. `Complete` means its
required evidence, owner, limitations, decision, and downstream links have been
reviewed.

Expand in three waves rather than opening every template at once:

1. **Define the decision:** engagement control, discovery, and the smallest architecture.
2. **Prove the bounded path:** data, identity, service envelope, evaluation, and rollout.
3. **Make it operable:** incident response, handoff, project mapping, and the next iteration.

Finish the current wave's decision links before opening the next one. A later
finding may reopen an earlier wave; that is evidence doing its job, not failure.

## Package metadata

| Field | Value |
|---|---|
| Package ID | `ENG-CAP-01` |
| Frozen case version | `RIV-FDE-1.0.0` |
| Candidate package version | `[TODO]` |
| Learner/author | `[TODO]` |
| Reviewers and authority | `[TODO]` |
| Decision date | `[TODO]` |
| Overall recommendation | `[TODO: HOLD / NARROW SCOPE / PROCEED TO NAMED GATE / DO NOT PROCEED]` |
| Highest supported evidence tier | `[TODO]` |
| Open critical blockers | `[TODO]` |

## Required artifacts

| Order | Artifact IDs | Template | Minimum decision | Status | Reviewer |
|---:|---|---|---|---|---|
| 0 | `ENG-CAP-01`, `CLM-CAP-*`, `DEC-CAP-*` | [Engagement and claims](templates/00-engagement-charter-and-claim-register.md) | Is the engagement bounded and are all claims classified? | `NOT STARTED` | `[TODO]` |
| 1 | `DSC-CAP-01` to `DSC-CAP-03` | [Discovery and success](templates/01-discovery-and-success.md) | What workflow, baseline, outcome, non-goal, and unknown define the work? | `NOT STARTED` | `[TODO]` |
| 2 | `ARC-CAP-01`, `ARC-CAP-02`, `ADR-CAP-*` | [Architecture and ADRs](templates/02-architecture-and-adrs.md) | What is the smallest valid intervention and why? | `NOT STARTED` | `[TODO]` |
| 3 | `DATA-CAP-01` to `DATA-CAP-04` | [Data onboarding](templates/03-data-contract-and-onboarding.md) | Which sources can be governed, mapped, refreshed, deleted, and retrieved? | `NOT STARTED` | `[TODO]` |
| 4 | `SEC-CAP-01` to `SEC-CAP-04` | [Identity and controls](templates/04-identity-threat-residency-controls.md) | Does context fail closed, and which approvals remain external? | `NOT STARTED` | `[TODO]` |
| 5 | `CAP-CAP-01`, `COST-CAP-01`, `SVC-CAP-01` | [Service envelope](templates/05-cost-capacity-sla.md) | What scenario range is supportable, and what cannot yet be committed? | `NOT STARTED` | `[TODO]` |
| 6 | `EVAL-CAP-01`, `ROL-CAP-01`, `ROL-CAP-02` | [Evaluation and rollout](templates/06-evaluation-and-rollout.md) | What evidence permits only the next bounded exposure stage? | `NOT STARTED` | `[TODO]` |
| 7 | `INC-CAP-01` to `INC-CAP-03` | [Incident and communication](templates/07-incident-and-customer-communication.md) | Can the team contain, communicate, revalidate, and re-enable safely? | `NOT STARTED` | `[TODO]` |
| 8 | `OPS-CAP-01`, `HOF-CAP-01`, `HOF-CAP-02` | [Operations and handoff](templates/08-operations-and-handoff.md) | Can named customer owners operate and change the service? | `NOT STARTED` | `[TODO]` |
| 9 | `MAP-CAP-01` | [Project mapping](templates/09-project-mapping.md) | Which assets are reusable, missing, conflicting, or unvalidated? | `NOT STARTED` | `[TODO]` |
| 10 | `GAP-CAP-01`, `NEXT-CAP-01` | [Gaps and next iteration](templates/10-product-gaps-and-next-iteration.md) | What is the smallest next investment justified by evidence? | `NOT STARTED` | `[TODO]` |

## Required traceability chain

Create one row per acceptance criterion and one row per selected component. Every
cell needs an artifact ID or an explicit `NOT APPLICABLE` reason.

| Acceptance criterion | Workflow/user | Claim IDs | ADR/component | Data contract | Control/threat | Evaluation case/gate | Capacity/SLA assumption | Rollout/rollback | Signal/incident path | Handoff owner | Gap/revalidation trigger |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `[TODO: AC-*]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` |

Reverse traceability is also required:

| Selected component/service | Requirement or control served | Evidence class | Evidence or validation owner | Failure/rollback path | Remove if evidence never arrives? |
|---|---|---|---|---|---|
| `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` |

A component with no mapped requirement is a removal candidate. A requirement with
no evaluation, operating signal, incident path, or owner is not ready for exposure.

## Review order and reopening rules

1. Review engagement control and discovery before architecture.
2. Review architecture before accepting data and identity dependencies.
3. Review data and controls before evaluation or service commitments.
4. Review evaluation and service envelope before rollout.
5. Review rollback, incident, and communication paths before any user cohort.
6. Review operations, drills, and ownership before broad availability or FDE exit.
7. Review project mapping and gaps before presenting the next iteration as funded
   scope.

Schema drift reopens data readiness. A role or region change reopens identity and
residency. A model, prompt, index, evaluator, workload, support, or architecture
change reopens evaluation and rollout. A failed drill reopens handoff.

## Ambiguity disposition queue

Do not remove a conflict from the queue merely because the recommendation avoids it. Link every supplied conflict and every material unknown to its owning artifacts.

| Conflict/unknown ID | Competing inputs or missing fact | Resolution owner and authority | Evidence/decision needed | Needed-by gate | Exposure blocked | Safe default/escalation | Status |
|---|---|---|---|---|---|---|---|
| `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO: HOLD / NARROW / MANUAL / DISABLE / ESCALATE]` | `OPEN` |

`OPEN` is an acceptable final status when the blocked exposure, owner, and next decision are explicit. An invented resolution is not.

## Final package check

- [ ] Every required artifact has a stable ID, version, status, owner, reviewer,
      scope, exclusions, and decision.
- [ ] Every quantitative or normative claim is in the claim register.
- [ ] Every claim has a class, evidence reference or validation request,
      limitations, and revalidation trigger.
- [ ] Every conflict and unknown is resolved, conditionally owned, or blocks a
      named exposure stage.
- [ ] Every acceptance criterion has forward and reverse traceability.
- [ ] All 11 supplied conflicts and every material unknown appear in the ambiguity disposition queue.
- [ ] Critical failures cannot be averaged away.
- [ ] Project source assets are not described as executed evidence.
- [ ] The recommendation names what was measured, modeled, customer-validated,
      externally unvalidated, and still unknown.
