# Re-enablement Decision: `<INCIDENT-ID>`

> Decision: `HOLD | APPROVE BOUNDED RE-ENABLEMENT | REVERT TO CONTAINMENT`
> A repaired component is not automatically authorized for production exposure.

## Scope and authority

| Field | Record |
|---|---|
| Decision time (UTC) | `<timestamp>` |
| Release/index/policy/config | `<stable IDs>` |
| Tenant/workflow/cohort | `<bounded scope>` |
| Requested exposure | `<percentage/users/functions/regions>` |
| Incident commander | `<name/role>` |
| Affected service/control owners | `<names/roles>` |
| Customer decision owner | `<name/role>` |
| Security/privacy/legal approval | `<reference or N/A with authorized rationale>` |

## Preconditions

| Precondition | Evidence reference | Owner | Result |
|---|---|---|---|
| Containment remains available and tested | `<reference>` | `<role>` | `PASS | FAIL` |
| Supported cause is addressed or bounded | `<reference>` | `<role>` | `PASS | FAIL` |
| Positive behavior gate passes | `<reference>` | `<role>` | `PASS | FAIL` |
| Negative authorization/policy gate passes | `<reference>` | `<role>` | `PASS | FAIL` |
| Adjacent workflows show no unacceptable regression | `<reference>` | `<role>` | `PASS | FAIL` |
| Telemetry detects recurrence without prohibited content | `<reference>` | `<role>` | `PASS | FAIL` |
| Rollback, pause, revoke, or compensation path is ready | `<reference>` | `<role>` | `PASS | FAIL` |
| Customer communication is approved | `<reference>` | `<role>` | `PASS | FAIL` |

Any failed mandatory precondition yields `HOLD`. Record an explicit exception
authority and exposure limit if organizational policy permits an exception;
silence is not approval.

## Residual risk

| Risk | Evidence/uncertainty | Accepted scope | Owner with authority | Expiry/revalidation trigger |
|---|---|---|---|---|
| `<risk>` | `<reference>` | `<bounded exposure>` | `<role>` | `<time/change/event>` |

## Observation plan

| Field | Record |
|---|---|
| Observation window | `<duration and UTC bounds>` |
| Cohort/traffic limit | `<bounded value>` |
| Signals and slices | `<availability, quality, policy, retrieval, side effect, cost>` |
| Signal owners | `<roles>` |
| Automatic stop conditions | `<specific thresholds/events>` |
| Next decision time | `<UTC timestamp>` |
| Customer update cadence | `<times or trigger>` |

## Decision

Decision rationale: `<evidence-backed statement>`

Approved temporary controls: `<control, owner, expiry, removal gate>`

Rollback/containment target: `<stable target and execution owner>`

Approvals:

| Role | Decision | Conditions | Reference/time |
|---|---|---|---|
| Incident commander | `APPROVE | HOLD` | `<conditions>` | `<reference>` |
| Service/control owner | `APPROVE | HOLD` | `<conditions>` | `<reference>` |
| Security/privacy/legal, when applicable | `APPROVE | HOLD | N/A` | `<conditions/rationale>` | `<reference>` |
| Customer decision owner | `APPROVE | HOLD` | `<conditions>` | `<reference>` |

## Health check

- [ ] Every mandatory precondition has retained evidence.
- [ ] Negative authorization/policy checks are included, not only happy paths.
- [ ] Residual risks have authorized owners and expiry.
- [ ] Exposure is bounded and automatically stoppable.
- [ ] Observation signals have owners and a next decision time.
- [ ] Rollback, containment, and committed-action correction are not conflated.
- [ ] Required service, control, incident, and customer approvals are recorded.
