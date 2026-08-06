# Operations Runbook

## Operating boundary

This runbook assumes an authorized Azure deployment built from the implemented
Riverside source assets. The repository contains service libraries, Azure ML, APIM,
monitoring/IaC, evaluation, and load assets, but no composed deployment or live
evidence. Every command below is an expected production command and was not executed
while authoring this document.

No SLO, alert threshold, on-call rotation, dashboard URL, resource name, or recovery
objective is committed. Fill those values from approved production evidence before
go-live.

Use the [dependency failure policy](dependency-failure-policy.md) for APIM,
orchestrator, Databricks, Azure ML, and telemetry failures. No alternate model,
index, answer-without-retrieval mode, or cross-region failover is approved.

## Shift start

1. Confirm the active incident/change channel and current on-call owner.
2. Read unresolved incidents, active changes, and planned evaluations.
3. Confirm release ID, active slot, index version, runtime version, region, and
   source commit from deployment metadata.
4. Review availability, deadline success, rejections, timeouts, backend failures,
   p50/p95 TTFT/TPOT/total latency, successful tokens/second, retrieval/citation
   signals, and cost anomaly status.
5. Verify telemetry freshness, alert delivery, sampling/retention state, and that
   content-redaction controls have not changed.
6. Confirm blue and green state and that the known-good rollback target remains
   available.

## Inspect the Azure resources

```powershell
$ResourceGroup = "<approved-resource-group>"
$WorkspaceName = "<azure-ml-workspace-name>"
$EndpointName = "<managed-online-endpoint-name>"
$DeploymentName = "<active-or-suspect-deployment>"
$ApimServiceName = "<api-management-service-name>"
$AppInsightsName = "<application-insights-component-name>"

az ml online-endpoint show `
  --name $EndpointName `
  --resource-group $ResourceGroup `
  --workspace-name $WorkspaceName

az ml online-deployment list `
  --endpoint-name $EndpointName `
  --resource-group $ResourceGroup `
  --workspace-name $WorkspaceName `
  --output table

az ml online-deployment get-logs `
  --endpoint-name $EndpointName `
  --name $DeploymentName `
  --resource-group $ResourceGroup `
  --workspace-name $WorkspaceName `
  --lines 200

az apim show `
  --name $ApimServiceName `
  --resource-group $ResourceGroup

az monitor app-insights component show `
  --app $AppInsightsName `
  --resource-group $ResourceGroup

az monitor activity-log list `
  --resource-group $ResourceGroup `
  --offset 2h
```

Treat command output as sensitive operational data. Redact identifiers and never
attach tokens, request bodies, environment variables, or customer content.

## First triage by contract error

| Error code | First checks | Default action |
|---|---|---|
| `invalid_request` | API version, token/message bounds, schema, caller release | Reject; do not retry. Identify client regression if rate changes. |
| `unauthorized` | Token issuer/audience/expiry, APIM validation policy | Reject; do not bypass auth. Escalate widespread failures. |
| `forbidden` | Trusted tenant context, role/ACL filters, index version | Reject; treat any false allow as a security incident. |
| `policy_violation` | Gateway policy version and safe normalized message | Reject; review only through approved policy change. |
| `overloaded` | Offered/achieved load, queue, instance health, retry amplification | Honor bounded retry guidance; shed load and stop rollout. |
| `timeout` | End-to-end deadline, retrieval/model spans, retries, cold/warm state | Stop retry amplification; isolate the slow dependency. |
| `backend_failure` | Azure ML/index health, identity, network, dependency changes | Stop rollout; fail closed or use approved fallback only. |
| `release_unavailable` | Digest/runtime/profile/precision checks, warm-up/readiness | Remove candidate traffic; inspect release evidence. |
| `internal_error` | Correlated trace and deployment metadata, safe logs | Stop rollout; avoid exposing backend exception text. |

## Symptom playbooks

### Availability or deadline success drops

1. Declare the incident level using [incident response](incident-response.md).
2. Freeze deploys, config changes, index updates, and traffic increases.
3. Separate gateway rejection, orchestrator failure, retrieval failure, and model
   failure by service and deployment metadata.
4. Check for a recent traffic, release, index, policy, RBAC, network, or quota change.
5. If the active release/index is implicated, follow [rollback](rollback.md).
6. Retain metric queries, logs, change IDs, and timestamps without customer content.

### TTFT or total latency regresses

1. Confirm telemetry and test-engine health; compare achieved load and output length.
2. Split gateway, retrieval, queue, model prefill, model decode, and stream-delivery
   time.
3. Check cold start/warm-up, instance count, queue depth, token buckets, top-k bucket,
   retries, and dependency throttling.
4. Stop canary or shadow expansion. Scale only within approved quota and cost limits.
5. Roll back when the candidate is causal or the deadline error budget says to abort.

### Overload or 429 rate rises

1. Verify whether demand, retries, or a capacity loss changed.
2. Ensure APIM and clients honor bounded `retry_after_seconds`; reject retry storms.
3. Protect known-good traffic with admission control and approved tenant-tier policy.
4. Do not increase retry count beyond the config maximum of three.
5. Capture recovery time after offered load falls; it is part of the operational gate.

Until autoscale behavior is proven and approved, treat source scale rules as safety
bounds rather than an operating policy. Use the
[manual capacity change record](checklists/manual-capacity-change.md) for any
temporary replica, instance, APIM unit, or concurrency change. Change one tier at a
time, confirm downstream headroom and cost approval, and retain before/after state.

### Retrieval or citation quality regresses

1. Freeze index promotion and model rollout independently.
2. Confirm query set, index version, embedding model/revision, chunk strategy/version,
   filters, and deletion state.
3. Separate retrieval miss from generation misuse with a gold-context check.
4. Roll back the index when lineage/index evidence points to data; roll back the
   release when candidate generation is causal.

### Authorization or content leakage

1. Treat any cross-tenant allow, forbidden-document retrieval, prompt/completion log,
   or sensitive metric dimension as a security incident.
2. Stop affected traffic and telemetry export as allowed by the approved containment
   plan; preserve evidence without spreading content.
3. Do not delete logs before the incident commander and security owner define legal
   hold and safe handling.
4. Rotate credentials only when a credential exists and compromise is plausible;
   prefer revoking role assignment/session paths for identity incidents.

## Change controls

Changes to release, traffic, index, APIM policy, RBAC, private networking, telemetry,
retention, autoscale, quota, or region require a change record and rollback target.
No direct production hotfix bypasses contracts or evidence gates. Emergency changes
record the same evidence after containment and receive retrospective review.

## Routine evidence

Daily, retain only approved aggregates for availability, deadline success, latency,
throughput, rejection/error categories, release/index versions, and alert health.
Weekly, review capacity headroom, cost attribution, access/policy changes, telemetry
cardinality, deletion propagation, stale slots/artifacts, and unresolved holds.
Per release, retain the complete evidence set in [deployment](deployment.md).
Reconcile each closed billing window and every material release/workload change with
the [cost reconciliation record](checklists/cost-reconciliation.md).

## Shift handoff

Record active release/slot/index, customer impact, open incidents, disabled alerts,
temporary mitigations, capacity/cost deviations, pending changes, evidence links,
and the next decision owner. Do not paste customer content into handoff notes.
