# Riverside Isolation Test Report

## Report status

| Field | Value |
|---|---|
| Execution status | `NOT RUN` |
| Evidence class | `[Modeled]` expected outcomes plus `[Local-static]` fixture inspection |
| Fixture | `RIV-FDE-04-LOCAL-1.0.0` referencing frozen `RIV-FDE-1.0.0` |
| Environment | None; committed source only |
| Source commit | Record when executed |
| Executed by / time | Not applicable |
| Release verdict | `NOT EVALUATED` |

No row below is a measured pass. The implementation task intentionally did not
execute the notebook or setup scripts.

## Expected cases

| Scenario | Boundary | Expected | Expected reason | Execution | Actual | Evidence |
|---|---|---:|---|---|---|---|
| `ISO-RIV-001` | Same-tenant policy retrieval | Allow | `authorized` | NOT RUN | Not recorded | Modeled |
| `ISO-RIV-002` | Cross-tenant retrieval | Deny | `tenant_mismatch` | NOT RUN | Not recorded | Modeled |
| `ISO-RIV-003` | Disabled contractor / stale group | Deny | `identity_disabled` | NOT RUN | Not recorded | Modeled |
| `ISO-RIV-004` | Role escalation / prohibited rights tool | Deny | `role_escalation` | NOT RUN | Not recorded | Modeled |
| `ISO-RIV-005` | Region policy | Deny | `region_not_allowed` | NOT RUN | Not recorded | Modeled |
| `ISO-RIV-006` | Purpose required | Deny | `missing_context` | NOT RUN | Not recorded | Modeled |
| `ISO-RIV-007` | Title assignment | Deny | `title_not_assigned` | NOT RUN | Not recorded | Modeled |
| `ISO-RIV-008` | Prohibited rights tool | Deny | `tool_not_authorized` | NOT RUN | Not recorded | Modeled |
| `ISO-RIV-009` | Bounded workflow proposal | Allow | `authorized` | NOT RUN | Not recorded | Modeled |

## Acceptance rule

- All expected allows must allow for the stated reason.
- All expected denies must deny at the earliest applicable boundary.
- Forbidden access count must equal zero.
- Every decision must produce a content-free audit event.
- A mismatch, exception, missing audit event, or unexpected allow is a stop-ship.
- `ISO-RIV-003` failure invokes `INC-RIV-003` containment: disable the EU route,
  revoke the identity, preserve access evidence, and engage Security and Legal.

## Evidence to record after an authorized run

| Item | Required content |
|---|---|
| Run identity | Reviewer, timestamp, environment, source commit, fixture/version |
| Result | Total, matched, mismatched, false allows, false denies |
| Decision trace | Scenario ID, boundary, policy/rule version, reason, audit-event reference |
| Limitations | Local substitutions, untested backends, cache/network/IdP gaps |
| External gaps | Token/RBAC/network/residency/audit/retention/deletion/legal validation owners |
| Decision | Stop-ship, local mechanism accepted, or approved for external staging validation |

## External validation remains open

Even a nine-of-nine local result would not prove Azure RBAC, managed identity,
private networking, index enforcement, cache partitioning, IdP revocation, regional
processing, backup location, telemetry behavior, deletion, legal basis, or regulatory
compliance. Complete the external validation checklist before any customer assurance.
