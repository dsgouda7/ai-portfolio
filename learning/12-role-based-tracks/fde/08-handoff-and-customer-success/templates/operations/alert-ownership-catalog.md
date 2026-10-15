# Alert Ownership Catalog Template

> An alert without an owner and safe first action is a notification, not an operating control.

## Catalog control

| Field | Value |
|---|---|
| Artifact ID | `OPS-01` |
| Catalog version | `<version>` |
| Routing configuration reference | `<immutable reference>` |
| On-call rota owner | `<owner>` |
| Last delivery drill | `<Measured evidence or not run>` |
| Revalidate on | Rule, threshold, routing, support-hours, service, or ownership change |

## Alert record

| Field | Required entry |
|---|---|
| Alert ID and name | Stable ID and operator-readable title |
| Signal/query/window | Exact definition, exclusions, and data freshness |
| Severity and rationale | Impact-based severity; start higher when scope is uncertain |
| Threshold status | `unset / modeled / measured / customer-approved` |
| Primary owner and backup | Named role/rota; never a team alias alone |
| Covered-hours behavior | Page, queue, fail closed, or invoke funded after-hours path |
| First safe action | Containment that does not bypass identity, policy, or evidence preservation |
| Runbook and dashboard | Exact IDs/links |
| Escalation and authority | Who can change severity, communicate externally, roll back, compensate, or re-enable |
| Evidence to preserve | Bounded identifiers, versions, timestamps, and approved records |
| False-positive handling | Suppression owner, expiry, and review; never silent permanent disablement |
| Clear condition | Signal and observation window required to resolve the alert |
| Limitations | Blind spots, sampling, lag, sparse slices, or dependency assumptions |

## Riverside example catalog

These rows are design examples. Thresholds and delivery remain unvalidated.

| Alert ID | Condition | Initial severity | Primary owner | First safe action | Runbook | Escalation/authority | Evidence | Noise treatment |
|---|---|---|---|---|---|---|---|---|
| `ALT-RIV-001` | Any successful forbidden or cross-tenant access | `SEV-1` | `ROLE-SECURITY` | Fail closed for affected route; preserve authorization evidence | `RBK-RIV-IDENTITY` | Security/privacy lead and incident commander; re-enable only with negative tests | Release/index/policy IDs, entitlement snapshot, filter decision, timestamps | Never sample or auto-suppress |
| `ALT-RIV-002` | Current-policy retrieval/citation gate regresses | `SEV-2` | `ROLE-EDITORIAL-DIRECTOR` | Disable affected guidance; freeze index promotion | `RBK-RIV-STALE-RETRIEVAL` | Data/model lead plus workflow owner | Query-set version, index/document versions, ranking, authorization context | Require two-window rule only if no forbidden content is involved |
| `ALT-RIV-003` | Workflow reconciliation finds duplicate commit | `SEV-2` | `ROLE-SUPPORT-L2` | Pause writes; keep read-only assistance; query committed state | `RBK-RIV-TOOL-AMBIGUOUS` | Tool owner and incident commander; compensation authority required | Business key, idempotency key, attempts, checkpoint, target history | Group duplicate symptoms by business key; never dedupe away separate commits |
| `ALT-RIV-004` | Regional route unavailable | `SEV-2` | `ROLE-IT-OWNER` | Fail closed for manuscript requests; use policy-only degraded mode only if authorized | `RBK-RIV-PROVIDER` | Security decides region boundary; communications owner approves updates | Dependency health, routing, failed counts, capacity, communications | Correlate dependency events; do not suppress customer-impact signal |
| `ALT-RIV-005` | Telemetry stale or dropping protected traces | `SEV-2` | `ROLE-SUPPORT-L2` | Stop rollout/change decisions that depend on missing evidence | `RBK-RIV-TELEMETRY` | Security if content/redaction is involved | Exporter state, ingestion delay, sampling config, dropped-span count | Maintenance suppression requires owner, window, and expiry |
| `ALT-RIV-006` | Cost/budget burn exceeds accepted envelope | `<unset>` | `ROLE-FINANCE` | Stop ramp and isolate retry/routing change; do not invent budget headroom | `RBK-RIV-BUDGET` | Finance owns budget exception | Usage, successful-work denominator, allocation, billing reconciliation | Review seasonality and traffic mix before threshold change |

## Ownership gap check

Before acceptance, flag:

- alerts whose primary or backup owner is the FDE after the agreed hypercare period;
- pages that arrive outside covered hours with no funded response path;
- alerts with no runbook, no first safe action, or no re-enablement authority;
- alert thresholds copied from examples without baseline evidence;
- safety/security alerts subject to ordinary sampling or auto-suppression;
- dashboards that can fail silently without a telemetry-health alert.
