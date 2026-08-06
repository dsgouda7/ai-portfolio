# Operations and Handoff Package

## Document control

| Field | Value |
|---|---|
| Artifact IDs | `OPS-CAP-01`, `HOF-CAP-01`, `HOF-CAP-02` |
| Version / status | `[TODO] / DRAFT` |
| FDE / customer operations / support / business / security owners | `[TODO]` |
| Target release/index/environment | `[TODO]` |
| Scope / exclusions / revalidation trigger | `[TODO]` |

## Service and dependency inventory

| Capability/dependency | Version/config/region | Owner/support contact | Quota/credential process | Data/control boundary | Known limitation | Rollback/retirement path |
|---|---|---|---|---|---|---|
| `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` |

## Signal, alert, decision, and runbook map

| Signal/alert ID | Service/quality/security/cost question | Threshold/status | Owner | First safe action | Runbook | Escalation/authority | Evidence retained | Revalidate on |
|---|---|---|---|---|---|---|---|---|
| `ALERT-CAP-*` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` |

Cover availability/deadlines, TTFT/TPOT/total latency, retrieval/citation quality,
authorization/policy, deletion freshness, PageTurn reconciliation, dependency and
quota health, cost, telemetry freshness/cardinality, and customer adoption/support.

## Runbook contract

| Runbook ID/scenario | Trigger and safety boundary | Preconditions/authority | Steps and evidence | Stop/rollback/compensation | Validation and re-enablement | Owner |
|---|---|---|---|---|---|---|
| `RB-CAP-*` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` |

Minimum runbooks: degraded provider, stale/unauthorized retrieval, identity
failure, quota/budget exhaustion, workflow ambiguity, data sync/deletion failure,
release/index rollback, compensation/reconciliation, and re-enablement.

## Support and escalation matrix

| Severity/capability | Covered hours | Acknowledgement/update proposal | L1/L2/incident owner | Customer communication owner | Vendor dependency | Exclusions | Commitment authority |
|---|---|---|---|---|---|---|---|
| `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` |

Explicitly disposition `CON-RIV-010` and `UNK-RIV-006`. A proposed response target
is not a contractual SLA.

## Training and drill record

| Drill ID | Operator role | Scenario | Pass criteria | Execution evidence | Result | Remediation/retest | Reviewer |
|---|---|---|---|---|---|---|---|
| `DRILL-CAP-*` | `[TODO]` | `[TODO]` | `[TODO]` | `[NOT RUN until executed]` | `[TODO]` | `[TODO]` | `[TODO]` |

Required drills:

1. Trace a failing request across gateway, retrieval/model, tool, policy, and audit.
2. Roll back a release without confusing rollback with compensation.
3. Process access revocation or deletion and verify downstream propagation.
4. Contain a seeded cross-tenant attempt and preserve evidence.
5. Change an evaluation threshold through the approved process.

Attendance is not competence. A failed drill keeps handoff open.

## Readiness, acceptance, and exit

| Gate | Required evidence | Owner/approver | Result | Limitation/condition | Revalidation trigger |
|---|---|---|---|---|---|
| Operational readiness | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` |
| Security/data readiness | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` |
| Support readiness | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` |
| Training/drills | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` |
| Customer acceptance | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` |
| FDE exit/hypercare close | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` |

## Recurring health and retirement review

| Review area | Evidence/window | Decision owner | Trigger/threshold | Action or no-action decision | Next review |
|---|---|---|---|---|---|
| Workflow/adoption | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` |
| Data/retrieval/quality drift | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` |
| Policy/security/residency | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` |
| Capacity/cost/support | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` |
| Incidents/changes/evidence expiry | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` |
| Retirement/replace criteria | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` |

Handoff decision and remaining FDE-only dependency: `[TODO]`
