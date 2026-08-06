# Rollback

## Rollback contract

Rollback restores an explicitly named known-good release, index, policy, or
infrastructure state. It is not "redeploy something older" and it does not erase
incident evidence.

Before every rollout, record:

- known-good release ID/version/digests, deployment slot, runtime, and source commit;
- known-good index name/version and compatible data contract;
- current APIM policy artifact/digest and environment config version;
- traffic allocation and mirror allocation;
- backward/forward compatibility window;
- rollback owner, approver, trigger, and observation window.

Riverside runtime, rollout, APIM, load, and IaC source assets are present. No
rollback was executed or rehearsed in this task, and no retained rollback evidence
is linked.

## Automatic abort triggers

Stop advancement and default to rollback review when a release-report gate fails,
an authorization test allows forbidden data, readiness fails, a digest/runtime
mismatch appears, deadline/error/rejection abort criteria are crossed, quality or
safety regresses beyond threshold, telemetry leaks content, or evidence becomes
untrustworthy.

## Model traffic rollback

Assume blue is known-good and green is the candidate. Reverse the values only when
the change record proves green is known-good.

```powershell
$ResourceGroup = "<approved-resource-group>"
$WorkspaceName = "<azure-ml-workspace-name>"
$EndpointName = "<managed-online-endpoint-name>"

az ml online-endpoint update `
  --name $EndpointName `
  --resource-group $ResourceGroup `
  --workspace-name $WorkspaceName `
  --mirror-traffic "green=0"

az ml online-endpoint update `
  --name $EndpointName `
  --resource-group $ResourceGroup `
  --workspace-name $WorkspaceName `
  --traffic "blue=100 green=0"

az ml online-endpoint show `
  --name $EndpointName `
  --resource-group $ResourceGroup `
  --workspace-name $WorkspaceName

az ml online-deployment get-logs `
  --endpoint-name $EndpointName `
  --name blue `
  --resource-group $ResourceGroup `
  --workspace-name $WorkspaceName `
  --lines 200
```

Do not delete green during response. Preserve logs and deployment metadata until
the incident/release owner releases it.

## Model rollback validation

```powershell
az ml online-endpoint invoke `
  --name $EndpointName `
  --resource-group $ResourceGroup `
  --workspace-name $WorkspaceName `
  --request-file azureml/samples/chat-request.json
```

The committed sample is an implemented source asset; a composed cloud smoke suite
is still absent. Validate readiness,
stream/non-stream contracts, safe errors, citations, negative authorization,
deployment metadata, telemetry redaction, and stable operational signals over the
approved window.

## Index rollback

Select a retained immutable `index_version`; do not edit records in place. Confirm
that the active model/orchestrator supports the old vector-record contract,
embedding dimensions/revision, chunk strategy/version, citation lineage, and
deletion state. Re-run retrieval, leakage, citation, freshness, and deletion tests
before restoring traffic.

The implemented backend is Databricks Direct Vector Access. Its source can create,
upsert, query, and delete records, but it does not provide an exact atomic serving
alias/version rollback command. Add and cloud-validate that control-plane interface
before production use; do not simulate rollback by mutating records.

## APIM policy rollback

Restore only the exact pre-change snapshot created by `Publish-APIM.ps1`. The
snapshot records prior resource bodies and explicit absence for the managed API,
named values, blue/green/pool backends, fragments, and API policy. Retain its
SHA-256 independently from the snapshot file.

```powershell
# Default dry run verifies digest and target binding without Azure access.
./scripts/Restore-APIM.ps1 `
  -ConfigPath <approved-apim-input.json> `
  -SnapshotPath <prechange-snapshot.json> `
  -ExpectedSnapshotSha256 <lowercase-sha256>

./scripts/Restore-APIM.ps1 `
  -ConfigPath <approved-apim-input.json> `
  -SnapshotPath <prechange-snapshot.json> `
  -ExpectedSnapshotSha256 <lowercase-sha256> `
  -Apply
```

Restore re-creates prior resources and deletes only Riverside-managed objects
recorded as absent. It refuses secret named values. Do not edit production policy
in the portal as an undocumented fix. Validate Entra token checks,
managed-identity backend auth, limits, deadlines, retries, routing, normalized
errors, and bounded telemetry. The script was not executed or rehearsed while
authored, so rollback remains unproven.

## Infrastructure/config rollback

Revert the IaC/config change in source, preview the exact target state, obtain review,
then provision. Never use `azd down` as rollback.

```powershell
Set-Location projects/riverside-ai-platform
az bicep build --file infra/main.bicep
azd provision --preview
azd provision
```

For resource-group deployments, retain the provider-level what-if output. A source
revert alone does not prove Azure state changed back.

## Data pipeline rollback

Pause new index publication, preserve Bronze/Silver/Gold lineage and quarantine,
and repoint serving only to a known-good immutable index. Do not undelete data that
is subject to a valid deletion request. Reprocessing must retain source version,
content hashes, pipeline/chunk/embedding versions, ACL, classification, region, and
deletion lineage.

Deletion is not reversed by a general data rollback. Before production use, run the
[deletion propagation rehearsal](checklists/deletion-rehearsal.md) in an authorized
non-production environment and prove that restoring a pre-deletion backup reapplies
the deletion ledger before serving or export. No rehearsal has been run here.

## Roll forward instead

Use a roll-forward only when rollback is less safe, such as an irreversible security
or deletion correction, incompatible migration, or known defect in the prior
release. It still needs bounded scope, independent review, evidence gates, and a
new rollback target.

## Completion record

Record trigger, decision owner, commands/outputs, traffic/index/policy before and
after, release/deployment metadata, validation evidence, customer impact, start/end
time, residual risk, temporary controls, and follow-up owner. A successful traffic
command is not a successful rollback until end-to-end behavior is stable.

Use the [incident and rollback tabletop template](checklists/incident-rollback-tabletop.md)
for discussion or authorized staging exercises. A completed template must identify
its evidence class and cannot be represented as a production rollback.
