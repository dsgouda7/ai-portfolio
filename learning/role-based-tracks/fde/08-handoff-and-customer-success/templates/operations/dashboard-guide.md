# Dashboard Guide Template

> A dashboard is useful only when an operator can turn a signal into a bounded decision. Example thresholds are placeholders until measured and approved.

## Guide control

| Field | Value |
|---|---|
| Artifact ID | `OPS-DASH-01` |
| Dashboard version | `<version>` |
| Environment/release scope | `<scope>` |
| Dashboard owner | `<owner>` |
| Data source owner | `<owner>` |
| Retention/sampling/redaction approval | `<reference>` |
| Last freshness and access check | `<evidence reference>` |
| Revalidate on | Signal schema, release, index, policy, sampling, retention, or access change |

## Operator landing view

| Question | Panels/signals | Required slices | Decision enabled | Limitation to show beside the panel |
|---|---|---|---|---|
| Is the request path available and within deadline? | Availability, deadline success, p50/p95 TTFT, TPOT, total latency, errors/rejections | Service, environment, release, deployment, region, route, tenant tier, bounded error code | Continue, stop ramp, shed load, isolate boundary, or roll back | HTTP success does not prove answer quality or authorization correctness |
| Is retrieval current, authorized, and useful? | Retrieval success, top-k bucket, no-result rate, citation coverage, freshness/deletion checks | Approved low-cardinality tenant tier and workflow slice; index version in trace/log context | Freeze index promotion, enter degraded mode, or roll back index | Metric labels exclude tenant/document IDs; case investigation needs controlled traces |
| Is generation grounded and safe? | Evaluation trend, abstention, citation use, policy denials, unsafe-output escalations | Release, approved use case, canary cohort | Stop candidate, route to human, or reopen evaluation | Automated evaluators have bias and sparse rare-slice coverage |
| Are tools and workflow actions reliable? | Tool error/timeout rate, duplicate/reconciliation events, compensation queue | Tool name in traces, release, workflow operation class | Pause writes, keep read-only mode, reconcile, or escalate | Traffic rollback does not undo a committed workflow action |
| Is identity/isolation behaving correctly? | Forbidden/unauthorized outcomes, negative-test status, entitlement freshness | Region, tenant tier, role/purpose in protected audit context | Fail closed, isolate route, preserve evidence, engage Security | A false allow may be rare and must never be sampled away |
| Is capacity degrading before customers fail? | Queue depth, tokens/sec versus quota, tool latency, checkpoint latency, trace ingestion delay | Region, service, tenant tier | Defer admission, protect known-good traffic, request quota, or isolate dependency | CPU alone is not a reliable load signal for network-bound agent execution |
| Is cost within the accepted envelope? | Cost per successful request/token, fixed cost, retry waste, budget burn | Environment, release, tenant tier | Stop ramp, investigate retries/routing, or request budget decision | Cost is unproven until reconciled with billing and successful work |
| Can we trust the dashboard itself? | Telemetry freshness, exporter errors, sampling mode, dropped spans, alert delivery | Service and region | Lower confidence, switch to approved fallback evidence, or block change | Missing telemetry does not prove health |

## Signal contract

For each panel, complete one row:

| Signal ID | Definition and query | Unit/window | Source and freshness | Baseline/threshold status | Owner | Alert ID | Runbook ID | Decision | Known blind spot |
|---|---|---|---|---|---|---|---|---|---|
| `<OPS-SIG-001>` | `<exact numerator, denominator, exclusions, query>` | `<unit/window>` | `<source/freshness>` | `unset / modeled / measured / approved` | `<owner>` | `<alert>` | `<runbook>` | `<decision>` | `<blind spot>` |

## Riverside telemetry boundary

The platform source allowlists bounded metric dimensions such as service, environment, release, deployment, region, route, outcome, token bucket, tenant tier, error code, and retrieval-depth bucket. Prompts, completions, source text, user/request/tenant/document/chunk identifiers, and source URIs are excluded from metric labels.

Do not add high-cardinality or sensitive values to make a dashboard easier to filter. Use controlled trace/log investigation with approved access and redaction instead.

## Drill prompts

1. `INC-RIV-001`: decide whether the first safe action is model rollback, index freeze/rollback, or policy-only degraded mode.
2. `INC-RIV-003`: locate the evidence for a stale contractor entitlement without exposing manuscript text.
3. `INC-RIV-005`: distinguish timeout rate from duplicate committed workflow actions.
4. `INC-RIV-006`: show why a regional failover chart cannot authorize cross-region manuscript processing.

## Quick health check

- [ ] Every panel states the decision it enables.
- [ ] Every alerting signal links to an owner and runbook.
- [ ] Quality, policy, cost, and customer-specific health appear beside service health.
- [ ] Composite health decomposes into actionable component signals.
- [ ] Telemetry freshness and blind spots are visible.
- [ ] Thresholds are labeled unset, modeled, measured, or approved.
- [ ] Sensitive/high-cardinality values are absent from metric labels.
