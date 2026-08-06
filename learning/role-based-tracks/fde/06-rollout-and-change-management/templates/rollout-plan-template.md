# ROL-01 Rollout Plan

## Control record

| Field | Value |
|---|---|
| Rollout ID | `ROL-*` |
| Change record | `CHG-*` |
| Candidate release / index / policy IDs |  |
| Known-good rollback targets |  |
| Customer workflow and approved use cases |  |
| Excluded workflows, users, data, and actions |  |
| Business owner |  |
| Technical owner |  |
| Release approver |  |
| Incident commander / support owner |  |
| Communications owner |  |
| Evidence retention location and period |  |

## Baseline and claims

Record each baseline with class, population, window, source, limitations, and owner.
Do not convert a modeled target into a measured result when copying it here.

| Claim ID | Statement | Class | Population / window | Evidence | Limitation | Revalidation trigger |
|---|---|---|---|---|---|---|
|  |  | Measured / Modeled / Customer-validated |  |  |  |  |

## Cohort ladder

| Cohort | Mode / exposure | Entry gate | Observation window | Exit gate | Abort condition | Rollback target | Business owner | Technical owner |
|---|---|---|---|---|---|---|---|---|
| Offline | No user exposure |  |  |  |  | No deployment |  |  |
| Shadow | Output not served; writes sandboxed or disabled |  |  |  |  | Stop replay |  |  |
| Champion | Named trained users; bounded use cases |  |  |  |  | Prior release |  |  |
| Imprint / workflow canary | Bounded users, tenants, tools, and quota |  |  |  |  | Prior mode and release |  |  |
| Regional canary | Region-approved path only |  |  |  |  | Disable regional route |  |  |
| Broad | Approved use cases only |  |  |  |  | Per-tenant prior release |  |  |

## Disagreement review

Define categories before reviewing: agreement, candidate preferred, historical
preferred, policy block, unresolved, evaluator defect, and data/slice gap.

| Slice | Cases | Disagreements | Reviewed | Policy blocks | Unresolved | Owner | Disposition artifact |
|---|---:|---:|---:|---:|---:|---|---|
|  |  |  |  |  |  |  |  |

## Health signals

| Domain | Metric and slice | Threshold | Window | Data freshness | Alert owner | First action |
|---|---|---|---|---|---|---|
| Authorization | Forbidden access count | `0` | Continuous |  |  | Disable affected route |
| Consequential action | Duplicate commit count | `0` | Continuous |  |  | Pause writes and reconcile |
| Quality |  |  |  |  |  |  |
| Adoption | Weekly active rate / feedback coverage |  |  |  |  | Interview and retrain cohort |
| Latency / availability |  |  |  |  |  |  |
| Cost / quota |  |  |  |  |  |  |
| Support | Ticket rate and unresolved severity |  |  |  |  |  |

## Customer change plan

| Audience | Change / concern | Preparation | Training | Feedback route | Support route | Owner | Evidence of readiness |
|---|---|---|---|---|---|---|---|
|  |  |  |  |  |  |  |  |

## Ramp decision rules

- Missing or stale evidence produces `HOLD`.
- Any global safety abort produces `ABORT`; an aggregate improvement cannot override it.
- One completed window cannot satisfy a two-window gate.
- `GO` authorizes only the next named cohort and exposure level.
- Every traffic rollback records committed actions that need reconciliation, correction, compensation, or escalation.
