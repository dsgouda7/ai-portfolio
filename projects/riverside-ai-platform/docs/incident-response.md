# Incident Response

## Purpose

Use this process for an authorized Riverside Azure environment. It is a response
framework, not evidence that alerting, paging, containment, or recovery has been
implemented or rehearsed.

Customer safety, authorization, evidence preservation, and clear ownership take
priority over keeping every request flowing. Do not bypass identity, ACL, policy,
or content-redaction controls to restore availability.

## Draft severity model

The business owner must approve impact windows and notification obligations before
go-live.

| Severity | Example impact | Initial posture |
|---|---|---|
| SEV-0 | Confirmed cross-tenant disclosure, restricted-content exposure, credential compromise with active abuse | Stop affected traffic, engage security/legal, preserve evidence |
| SEV-1 | Broad outage, sustained deadline failure, unsafe generation reaching customers, deletion-control failure | Incident command, freeze changes, contain or roll back |
| SEV-2 | Partial degradation, elevated errors/latency, quality regression in a bounded slice, cost runaway without disclosure | Assign owner, stop rollout, mitigate within approved controls |
| SEV-3 | Internal-only defect, alert noise, non-urgent operational gap | Track and correct through normal change control |

If impact is uncertain, start at the higher plausible severity and downgrade with
evidence.

## Roles

- **Incident commander:** owns severity, priorities, decisions, and handoffs.
- **Operations lead:** queries health, applies approved mitigations, and validates
  recovery.
- **Security/privacy lead:** owns disclosure, credential, policy, residency, and
  evidence-handling decisions.
- **Data/model lead:** separates index, retrieval, artifact, runtime, and evaluation
  causes.
- **Communications lead:** provides approved, factual stakeholder updates.
- **Scribe:** maintains a UTC timeline, evidence references, hypotheses, commands,
  outputs, and approvals without copying customer content.

One person may fill multiple roles for a small incident, but incident command and
high-risk production changes require a second reviewer where access allows.

## Declare and stabilize

1. Open the incident record and channel; assign incident commander and scribe.
2. Record detection time, reporter, environment, region, release, slot, index,
   source commit, symptoms, affected tenants/tiers, and current change window.
3. Freeze deploys, promotions, index updates, APIM policy changes, autoscale edits,
   and nonessential data jobs.
4. Identify the last known-good release, index, policy, and infrastructure state.
5. Choose containment: reject affected requests, disable candidate traffic, pause
   ingestion/index promotion, isolate telemetry export, revoke a role/session, or
   execute the approved rollback.
6. State the next update time and decision owner.

## Initial inspection commands

These commands are documentation only and require an authorized deployment of the
implemented Azure source assets:

```powershell
$ResourceGroup = "<approved-resource-group>"
$WorkspaceName = "<azure-ml-workspace-name>"
$EndpointName = "<managed-online-endpoint-name>"
$SuspectDeployment = "<blue-or-green>"
$ApimServiceName = "<api-management-service-name>"

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
  --name $SuspectDeployment `
  --resource-group $ResourceGroup `
  --workspace-name $WorkspaceName `
  --lines 200

az apim show `
  --name $ApimServiceName `
  --resource-group $ResourceGroup

az monitor activity-log list `
  --resource-group $ResourceGroup `
  --offset 4h
```

Do not print environment variables, tokens, request/response bodies, source text,
or raw customer identifiers. If logs already contain content, restrict access and
follow the security/privacy lead's preservation instructions.

## Diagnose by boundary

### Identity and authorization

Check issuer/audience validation, APIM policy revision, managed identity token
acquisition, role assignments, trusted tenant context, ACL filters, and denied-path
tests. A false allow is a security incident; a false deny is an availability issue
until evidence shows otherwise.

### Data and retrieval

Check index version, source/embedding/chunk versions, tenant/ACL/region/classification
filters, deletion state, freshness, and recent pipeline jobs. Stop index promotion
before modifying serving traffic when the failure is data-specific.

### Model release

Check release/report digests, base/adapter/tokenizer digests, model profile,
precision, runtime interface, warm-up/readiness, deployment slot, and traffic.
Artifact mismatch or failed readiness requires zero candidate traffic.

### Gateway and orchestration

Check request bounds, deadlines, retry count, circuit state, overload response,
stream termination, citation normalization, and backend error mapping. Do not
increase retries during an overload incident.

### Capacity and dependencies

Check offered and achieved load, queueing, instance health/count, quota, throttling,
regional service health, network/DNS/private endpoint changes, and retry
amplification. Capacity changes need cost and quota approval even during response.
Apply the [dependency failure policy](dependency-failure-policy.md); do not create
an untested fallback under incident pressure. Use the
[manual capacity change record](checklists/manual-capacity-change.md) until an
autoscale policy has representative approved evidence.

### Telemetry

Check exporter health, sampling, ingestion delay, alert rule changes, dimension
cardinality, and accidental content capture. Missing telemetry lowers confidence;
it does not prove the service is healthy.

## Containment rules

- Prefer reversible traffic and index-version changes over in-place mutation.
- Keep authentication and authorization fail-closed.
- Do not delete a suspect deployment, index, log destination, or artifact before
  evidence and rollback needs are resolved.
- Do not rotate all identities speculatively; revoke the narrow compromised path
  and record blast radius.
- Do not send customer content through unapproved debugging tools or channels.
- Do not use production teardown as incident mitigation.

## Restore and validate

Follow [rollback](rollback.md) when a prior state is safer. Recovery requires:

1. health/readiness and contract smoke;
2. negative authorization and telemetry-redaction checks;
3. representative retrieval/citation and safe-refusal checks;
4. stable latency/error/rejection signals for an approved observation window;
5. confirmation that the active release/index/policy matches the incident record;
6. incident commander and relevant security/release approval.

## Communications

State what is known, unknown, affected, contained, and next. Use UTC timestamps and
evidence-backed scope. Do not speculate about root cause or promise a recovery time
without support. Customer, legal, regulator, and partner notifications follow the
approved contractual and jurisdictional process, which is not yet defined here.

## Close and learn

Close only after customer impact ends, temporary controls have owners/expiry, and
monitoring is stable. Produce a blameless review with timeline, detection gap, root
and contributing causes, evidence, decision quality, customer impact, security and
residency implications, corrective actions, owners, due dates, and tests that will
prove completion. A document-only action is not remediation.

Exercise this process with the
[incident and rollback tabletop template](checklists/incident-rollback-tabletop.md).
Label discussion-only decisions separately from commands actually run in an
authorized staging exercise; neither is production recovery evidence.
