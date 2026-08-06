# ROL-04 Rollback and Compensation Drill

## Scenario and controls

| Field | Value |
|---|---|
| Drill ID / change ID |  |
| Candidate / known-good release |  |
| Active cohort and exposure |  |
| Trigger injected or observed |  |
| Rollback owner / approver |  |
| Incident commander / scribe |  |
| Start / target completion / actual completion UTC |  |
| Evidence location |  |

## Track A: deployment rollback

Deployment rollback answers: **How do we stop new work from using the candidate?**

| Step | Expected state | Actual evidence | Owner | Result |
|---|---|---|---|---|
| Stop ramp and freeze changes | No exposure increase |  |  |  |
| Route new work to named known-good release | Candidate traffic is zero |  |  |  |
| Disable affected tools / writes if required | Read-only or blocked mode |  |  |  |
| Invalidate or drain candidate sessions safely | No silent continuation on withdrawn config |  |  |  |
| Run readiness, negative authorization, quality, and telemetry checks | Known-good behavior restored |  |  |  |
| Observe stable signals for the approved window | End-to-end rollback complete |  |  |  |

## Track B: committed actions

Committed-action recovery answers: **What already changed in an external system?**

| Action / key | Commit known? | State query | Classification | Correction / compensation | Business owner | Technical owner | Status |
|---|---|---|---|---|---|---|---|
|  | yes / no / unknown | query by business key first | compensable / retryable / irreversible / approval-required |  |  |  |  |

Rules:

1. An unknown timeout outcome is a state-query problem first and an escalation problem second.
2. Reuse the same idempotency key on an approved retry.
3. Never compensate an action that has not been shown to exist.
4. A corrective action is not a true undo; record customer-visible residue.
5. Deployment rollback completion does not close Track B.

## Communications and close

Record impact, known facts, unknowns, containment, next update, customer decision,
residual risk, temporary-control expiry, and follow-up owner. The drill passes only
when operators can execute both tracks without relying on undocumented FDE context.
