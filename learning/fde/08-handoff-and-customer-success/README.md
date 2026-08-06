# Handoff and Customer Success

This module closes the Riverside engagement by transferring operating capability, change authority, evidence, and known limitations to named customer owners. It does not treat a folder of documents, a walkthrough, or a signature as proof that customer operations can run the system.

The running failure is deliberate: the first handoff package is a document dump. It contains plausible files but no reliable map from alert to owner, action, evidence, escalation, or revalidation. The notebook makes that package fail before the learner repairs it.

## Mission

Riverside House is preparing to move from bounded canaries toward broad editorial availability. The frozen case supplies support hours, severity targets, capability owners, seeded incidents, rollout gates, and unresolved questions. The platform repository supplies implemented source assets and operational procedures, but no executed validation or live Azure evidence.

Your job is to produce a handoff package that lets Riverside operations:

1. decide whether the service is ready to accept;
2. interpret service, quality, retrieval, policy, cost, and customer-specific signals;
3. respond to alerts without relying on private FDE context;
4. change policies and releases through evidence gates;
5. escalate within explicit support boundaries;
6. demonstrate high-risk drills before acceptance;
7. review health, limitations, and retirement criteria after hypercare;
8. turn evidence gaps into an owned, prioritized backlog.

## Evidence boundary

This module preserves the route's claim classes:

- `[Measured]` requires an executed test, trace, drill, query, or observation with scope and limitations.
- `[Modeled]` requires explicit assumptions, range, sensitivity, and a replacement-by-measurement owner.
- `[Customer-validated]` requires a named authorized customer role, bounded scope, decision, conditions, and revalidation trigger.

The notebook executed successfully against the committed synthetic fixtures and reproduced the documented `BLOCKED` handoff decision, then its outputs were cleared. Template examples are teaching examples, not completed operator drills or production approvals. The Riverside platform contains source assets and documentation, but no live deployment, SLO, rollback rehearsal, support approval, customer acceptance, or production evidence is established here. Do not promote those assets or the local fixture run into readiness claims.

## Module map

| Artifact | Purpose |
|---|---|
| [SETUP.md](SETUP.md) | Verified local setup and instructions for a retained exercise run |
| [requirements.txt](requirements.txt) | Runtime dependency declaration for the notebook |
| [handoff-and-customer-success.ipynb](handoff-and-customer-success.ipynb) | Failure-first learning path and in-memory checks |
| [templates/operations/operational-readiness-review.md](templates/operations/operational-readiness-review.md) | Gate review across service, data, security, operations, and ownership |
| [templates/operations/dashboard-guide.md](templates/operations/dashboard-guide.md) | Signal-to-decision dashboard contract |
| [templates/operations/alert-ownership-catalog.md](templates/operations/alert-ownership-catalog.md) | Alert severity, owner, first action, escalation, and noise handling |
| [templates/operations/runbook.md](templates/operations/runbook.md) | Reusable runbook structure and minimum Riverside runbook set |
| [templates/operations/recurring-health-review.md](templates/operations/recurring-health-review.md) | Monthly workflow, quality, policy, capacity, cost, incident, and retirement review |
| [templates/handoff/handoff-package-index.md](templates/handoff/handoff-package-index.md) | Traceable package manifest; the antidote to the document dump |
| [templates/handoff/policy-change-process.md](templates/handoff/policy-change-process.md) | Versioned policy-change proposal, gates, rollout, rollback, and retained evidence |
| [templates/handoff/support-escalation-matrix.md](templates/handoff/support-escalation-matrix.md) | Hours, severity, response targets, authorities, vendors, and exclusions |
| [templates/handoff/acceptance-signoff.md](templates/handoff/acceptance-signoff.md) | Conditional acceptance, limitations, authority, and revalidation record |
| [templates/handoff/exit-criteria.md](templates/handoff/exit-criteria.md) | Hypercare and FDE exit gates based on demonstrated ownership |
| [templates/handoff/evidence-based-backlog.md](templates/handoff/evidence-based-backlog.md) | Ranked iterations linked to incidents, reviews, feedback, or expired evidence |
| [templates/training/training-agenda-and-drill-record.md](templates/training/training-agenda-and-drill-record.md) | Role-based agenda, drills, scoring, remediation, and attendance evidence |

## Required references

Use these sources as dependencies rather than copying them:

1. [FDE route contract](../README.md)
2. [Role baseline, lifecycle, and handoff standard](../00-role-baseline-and-engagement-lifecycle.md)
3. [Frozen Riverside case](../shared/README.md)
4. [Observability, tracing, and agent health](../../agentic-ai-system-design/08-observability-tracing-and-health.md)
5. [Recoverability, rollback, and saga](../../agentic-ai-system-design/10-recoverability-rollbacks-and-saga.md)
6. [Production scale and capacity](../../agentic-ai-system-design/12-production-scale-and-capacity.md)
7. [Riverside architecture](../../../projects/riverside-ai-platform/docs/architecture.md)
8. [Riverside operations runbook](../../../projects/riverside-ai-platform/docs/operations-runbook.md)
9. [Riverside incident response](../../../projects/riverside-ai-platform/docs/incident-response.md)
10. [Riverside rollback](../../../projects/riverside-ai-platform/docs/rollback.md)
11. [Promise versus evidence](../../../projects/riverside-ai-platform/docs/promise-vs-evidence.md)
12. [Current limitations](../../../projects/riverside-ai-platform/docs/limitations.md)

## Downstream integration path

Use the templates to assemble a candidate operating package from the Riverside operations, incident, rollback, evaluation, deployment, telemetry, and limitation records. Then, in a supervised practicum, replace every placeholder with the exact deployed resource, dashboard, alert, access process, support route, runbook action, release identifier, evidence location, and owner for the authorized environment.

Handoff validation requires operators to perform the agreed drills while the learner is observed by an experienced FDE or operations owner. Attendance, a document review, source assets, or a signature without demonstrated operation do not establish transfer. A failed drill remains a blocker and becomes evidence for remediation and repeat practice.

## Completion rule

The module is complete only when every recurring task and alert has an accepted owner, operators complete the agreed drills, support and change authorities are explicit, limitations and exclusions remain visible, and FDE-only access or undocumented knowledge is removed. A failed drill keeps handoff open even if the acceptance form is signed.

## Validation and evidence limitation

The route setup environment was verified, and the notebook executed successfully against the committed synthetic fixtures before its outputs were cleared. The run reproduced the documented `BLOCKED` and `REJECT` paths, but it did not perform an operator drill, cloud query, customer handoff, support acceptance, legal/compliance review, or production operation. All committed notebook outputs remain empty and all code-cell execution counts remain null.
