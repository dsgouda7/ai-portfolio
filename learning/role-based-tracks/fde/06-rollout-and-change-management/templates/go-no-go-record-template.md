# ROL-03 Go / No-Go Record

## Decision

| Field | Value |
|---|---|
| Decision ID | `ROL-DEC-*` |
| Decision | `GO` / `HOLD` / `ABORT` / `ROLLBACK` |
| Candidate and baseline release IDs |  |
| Current cohort and requested next cohort |  |
| Exposure authorized | users, tenants, regions, use cases, tools, quota |
| Decision time in UTC |  |
| Effective until / revalidation trigger |  |
| Decision authority |  |
| Technical recommendation owner |  |
| Customer workflow owner |  |
| Security / operations concurrence |  |

## Gate evidence

| Gate | Threshold / rule | Observed result | Class | Status | Evidence | Owner |
|---|---|---|---|---|---|---|
| Historical baseline comparable |  |  |  | Pass / Hold / Fail |  |  |
| Shadow disagreements dispositioned |  |  |  |  |  |  |
| Critical slices |  |  |  |  |  |  |
| Authorization and prohibited actions | `0` failures |  |  |  |  |  |
| Adoption and structured feedback |  |  |  |  |  |  |
| Quality and workflow outcome |  |  |  |  |  |  |
| Latency, availability, cost, support |  |  |  |  |  |  |
| Rollback target and drill | named and verified |  |  |  |  |  |
| Communications and support rota | accepted |  |  |  |  |  |

## Conditions and dissent

Record conditions, owner, due date, exposure limit, and automatic response if missed.
Preserve reviewer dissent rather than converting it into meeting-note consensus.

## Decision rationale

State which evidence changed exposure, which evidence remains incomplete, and why
the selected decision is safer than the alternatives. A positive aggregate does
not override a failed critical slice or global abort condition.

## Retained record

Link the manifest, evaluation report, rollout observations, cohort membership
approval, communications, rollback target, drill, support rota, and change record.
