# ARC-02: Architecture Boundary Register

## Context contract

| Context field | Source | Validated by | Propagated to | Missing-context behavior | Audit evidence |
|---|---|---|---|---|---|
| Tenant | | | | Deny | |
| Actor | | | | Deny | |
| Roles | | | | Deny | |
| Region | | | | Deny | |
| Purpose | | | | Deny | |
| Title assignment | | | | Deny when title-scoped | |
| Trace ID | | | | Reject or mint at trusted ingress | |

## Boundary register

| Boundary ID | Type | Proposal or data entering | Deterministic control | Human authority | Side effect or data leaving | Failure behavior | Evidence owner |
|---|---|---|---|---|---|---|---|
| | Identity | | | | | | |
| | Data | | | | | | |
| | Retrieval | | | | | | |
| | Model | | | | | | |
| | Policy | | | | | | |
| | Human approval | | | | | | |
| | Tool / system of record | | | | | | |
| | State / audit | | | | | | |
| | External validation | | | | | | |

## Side-effect inventory

| Action | Risk class | Model may propose? | Policy decision | Approval payload | Idempotency / reconciliation | Compensation or correction | Owner |
|---|---|---|---|---|---|---|---|
| | | | | | | | |

## Boundary health check

- [ ] Every component has one upstream need and one named owner.
- [ ] All authorization inputs fail closed when absent.
- [ ] Retrieval filters lifecycle and ACLs before ranking.
- [ ] Model text is never authority or a system of record.
- [ ] Human approval binds to exact, current arguments.
- [ ] Every mutation has idempotency, reconciliation, audit, and recovery terms.
- [ ] Telemetry excludes prohibited content and high-cardinality metric labels.
- [ ] Unknown and external-validation boundaries remain visibly blocked.
