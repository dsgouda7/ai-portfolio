# Handoff and Customer Success

## Transfer Knowledge, Not Just Files

When Riverside launches, the editorial team cannot call the FDE every time an alert fires or a policy needs updating. This module builds the package that makes the team independent and keeps the service operable.

**The problem:** a polished folder of documents is not operating capability. When the index is six hours stale on a Saturday, the on-call editor needs to know who owns the incident, what to check first, what evidence to preserve, and whether they may roll back safely. Those decisions require an ownership map, a runbook linked to each alert, and proof that the team has practiced the response.

**What you will build:** a traceable handoff package where every alert, task, and decision follows a visible path:

`symptom -> owner -> safe action -> evidence -> escalation -> drill -> revalidation`

The frozen Riverside case provides support hours, seeded incidents, capability owners, rollout gates, and unresolved questions. You connect those facts into a package the customer can actually operate.

## What You Will Build: From Document Dump to Ownership Map

### Before

- Riverside receives `dashboard.pdf`, `runbook.docx`, `support.xlsx`, training slides, and an acceptance form.
- At 2 AM, the on-call editor receives an alert and a phone tree, but no clear first action.
- The FDE still holds the missing context, so handoff remains open.

### After

Your package contains:

1. **Ownership map:** every alert, policy change, and recurring task has a named role and escalation path.
2. **Runbook links:** each alert maps to severity, first safe action, evidence checklist, runbook, and escalation decision.
3. **Drill records:** the team has practiced trace isolation, rollback, access containment, deletion propagation, and threshold changes.
4. **Health loop:** a monthly review connects metrics, incidents, ownership changes, support burden, and backlog priorities.
5. **Revalidation triggers:** material changes expire the relevant evidence and reopen the right gate.

### Timeline

| When | Riverside scenario | Required ownership outcome |
|---|---|---|
| Day 1 | An alert fires | The owner takes the first safe action, preserves evidence, and escalates within the agreed boundary |
| Week 1 | Editorial proposes a policy change | Proposer, reviewer, test owner, canary approver, and rollback owner are explicit |
| Weeks 2-4 | Operators train and drill | The team demonstrates isolation, rollback, deletion, and threshold-change paths without private FDE knowledge |
| Month 1 and later | Health review recurs | Owners review value, quality, incidents, support load, backlog priority, and retirement triggers |

## How We Separate Proof from Promise

This module keeps three evidence classes distinct:

- **`[Measured]`** - You ran the drill or check and retained the scoped result.
	Example: "The rotation on-call performed `DRL-RIV-TRACE` and isolated a simulated tool timeout in four minutes."
- **`[Modeled]`** - You built the logic from stated assumptions, but nobody has validated it in the target environment.
	Example: "If index refresh stalls, the alert should fire within six hours under `ASM-RIV-012`; the live alert path remains untested."
- **`[Customer-validated]`** - An authorized Riverside role made a bounded decision with conditions and a revalidation trigger.
	Example: "The Editorial Director approved staged canaries for policy changes. Revalidate if volume exceeds 900 sessions per day or support exceeds four hours per month."

Do not mix the classes. A written runbook is not `[Measured]` until an operator has used it in a scoped drill. An SLO target is not `[Measured]` until the target environment has produced retained results. A responsibility is not `[Customer-validated]` when only the FDE assigned it.

The notebook runs against synthetic fixtures and reproduces the documented `BLOCKED` decision. That checks the local logic, not Riverside operations or production readiness. Templates are teaching examples; source files and local execution do not prove a live deployment, SLO, rollback rehearsal, support approval, customer acceptance, or production capability.

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
4. [Observability, tracing, and agent health](../../../agentic-ai-system-design/08-observability-tracing-and-health.md)
5. [Recoverability, rollback, and saga](../../../agentic-ai-system-design/10-recoverability-rollbacks-and-saga.md)
6. [Production scale and capacity](../../../agentic-ai-system-design/12-production-scale-and-capacity.md)
7. [Riverside architecture](../../../../projects/riverside-ai-platform/docs/architecture.md)
8. [Riverside operations runbook](../../../../projects/riverside-ai-platform/docs/operations-runbook.md)
9. [Riverside incident response](../../../../projects/riverside-ai-platform/docs/incident-response.md)
10. [Riverside rollback](../../../../projects/riverside-ai-platform/docs/rollback.md)
11. [Promise versus evidence](../../../../projects/riverside-ai-platform/docs/promise-vs-evidence.md)
12. [Current limitations](../../../../projects/riverside-ai-platform/docs/limitations.md)

## Downstream integration path

Use the templates to assemble a candidate operating package from the Riverside operations, incident, rollback, evaluation, deployment, telemetry, and limitation records. Then, in a supervised practicum, replace every placeholder with the exact deployed resource, dashboard, alert, access process, support route, runbook action, release identifier, evidence location, and owner for the authorized environment.

Handoff validation requires operators to perform the agreed drills while the learner is observed by an experienced FDE or operations owner. Attendance, a document review, source assets, or a signature without demonstrated operation do not establish transfer. A failed drill remains a blocker and becomes evidence for remediation and repeat practice.

## Completion rule

The module is complete only when every recurring task and alert has an accepted owner, operators complete the agreed drills, support and change authorities are explicit, limitations and exclusions remain visible, and FDE-only access or undocumented knowledge is removed. A failed drill keeps handoff open even if the acceptance form is signed.

## Validation and evidence limitation

The route setup environment was verified, and the notebook executed successfully against the committed synthetic fixtures before its outputs were cleared. The run reproduced the documented `BLOCKED` and `REJECT` paths, but it did not perform an operator drill, cloud query, customer handoff, support acceptance, legal/compliance review, or production operation. All committed notebook outputs remain empty and all code-cell execution counts remain null.
