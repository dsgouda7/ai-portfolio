# Deployment

## Scope

This is the expected production deployment procedure for `dev`, `staging`, and
`production`. There is no local deployment profile. The commands are documentation
only and were not executed while this file was authored.

The repository now contains `azure.yaml`, Bicep modules and parameter examples,
Azure ML templates/scoring source and deterministic registration/deployment
scripts, APIM policies plus publish/restore scripts, service libraries,
evaluations, Azure Load Testing run/export assets, and static tests. It does not
contain `azd` service entries, approved release packages, cloud test results, or
retained command output. The scripts are implemented source interfaces only and
were not executed while authored. Resolve the remaining evidence gaps in
[limitations](limitations.md) before any authorized provision.

## Required inputs

Do not start provisioning until the change record contains:

- target subscription, tenant, environment, region, and resource group;
- approved region/residency and service-availability review;
- current source commit and clean build provenance;
- immutable release manifest and matching evaluation report URIs/digests;
- selected inactive slot and known-good rollback release/index versions;
- reviewed Bicep what-if output, policy/RBAC changes, and deletion list;
- regional quota and SKU availability evidence;
- owner, approver, observation windows, abort criteria, and incident channel;
- backup/restore and teardown decisions for durable data.

Production deploy permission, release approval, and APIM policy-edit permission
should be separate roles. No operator places credentials in environment files,
parameters, command arguments, or retained output.

## Expected tool preflight

Run from the repository root in PowerShell after dependencies are installed through
an approved project environment:

```powershell
azd version
az version
az extension show --name ml
az bicep version
azd auth login
az account show --output table
```

Record tool versions and the active tenant/subscription. `az account show` is an
identity check, not approval to deploy.

## Static and non-cloud gate

These commands are the expected project interface. They must fail closed when a
required asset or invariant is absent:

```powershell
python -m pytest projects/riverside-ai-platform/tests/unit
python -m pytest projects/riverside-ai-platform/tests/contract
python -m pytest tests/unit/release_gates tests/contract/evaluations tests/contract/apim
az bicep build --file projects/riverside-ai-platform/infra/main.bicep
```

Required retained evidence:

- test and validator output tied to the source commit;
- valid/invalid contract fixture results with format assertions enabled;
- release manifest/report digest and decision consistency;
- source secret-scan result;
- Bicep build/lint result;
- proof that no `local` environment profile exists.

Passing this gate proves only local/static behavior. It does not prove Azure RBAC,
quota, networking, service availability, identity, autoscaling, latency, cost, or
residency.

## Select the Azure environment

Use an existing approved `azd` environment or create one with a non-secret name.
Set identifiers through the approved environment mechanism; keep secrets in Azure
Key Vault or identity-based access.

```powershell
$Environment = "staging"
$Location = "<approved-azure-region>"
$SubscriptionId = "<approved-subscription-id>"
$ResourceGroup = "<approved-resource-group>"

azd env select $Environment
azd env set AZURE_LOCATION $Location
azd env set AZURE_SUBSCRIPTION_ID $SubscriptionId
azd env get-values
```

Review `azd env get-values` before retaining it. Do not attach output containing a
secret or sensitive endpoint to the change record.

## Preview infrastructure

The expected `azd` preview is:

```powershell
azd provision --preview
```

The installed `azd` version must advertise `--preview`; otherwise stop and update
the approved toolchain or use the reviewed resource-group what-if path below. Do
not replace preview with a real provision.

Run Azure Resource Manager what-if with the exact target parameters:

```powershell
$TemplateFile = "projects/riverside-ai-platform/infra/main.bicep"
$ParametersFile = "projects/riverside-ai-platform/infra/main.parameters.json"

az deployment group what-if `
  --resource-group $ResourceGroup `
  --template-file $TemplateFile `
  --parameters $ParametersFile `
  --validation-level Provider
```

What-if needs Azure access and is live control-plane validation, but it makes no
resource changes. Review creates, modifications, deletes, RBAC assignments,
diagnostic destinations, public-network settings, and deployment-script effects.
What-if can contain noise and cannot prove runtime behavior.

## Provision infrastructure

Provision only after preview approval:

```powershell
azd provision
azd show
```

The current `azure.yaml` has infrastructure configuration but no `services` entries,
so `azd deploy` does not deploy the Python libraries, scoring source, Azure ML
deployments, or APIM policies. Do not use `azd up` to hide that gap. Capture
provisioning IDs/outputs and reconcile them with the approved what-if.

## Materialize and register Azure ML assets

Prepare a reviewed JSON input using the exact contract in `scripts/README.md`.
Endpoint name, region, environment, deployment names, model/environment versions,
image digest, index version, timestamp, code path, package paths, and expected
digests must all be explicit. The committed YAML files are templates and must not
be passed directly to `az ml`.

```powershell
Set-Location projects/riverside-ai-platform

./scripts/Materialize-AzureML.ps1 `
  -ConfigPath <approved-azureml-input.json> `
  -OutputDirectory <new-generated-directory>

# Default dry run: verify generated/package digests and print registration.
./scripts/Register-AzureMLAssets.ps1 `
  -ConfigPath <approved-azureml-input.json> `
  -MaterializedDirectory <generated-directory>

# Mutation requires independent approval and the explicit switch.
./scripts/Register-AzureMLAssets.ps1 `
  -ConfigPath <approved-azureml-input.json> `
  -MaterializedDirectory <generated-directory> `
  -Apply
```

Materialization verifies the complete package digest plus every adapter,
tokenizer, training-manifest, and evaluation-report digest declared by each
release manifest. Registration refuses an existing model/environment version
whose digest tags differ. The pinned base image must use `@sha256:`. These checks
do not prove the registered bytes were served correctly.

## Publish Azure ML endpoint and deployments

```powershell
# Default dry run: verify materialized digests and print create/update intent.
./scripts/Deploy-AzureML.ps1 `
  -ConfigPath <approved-azureml-input.json> `
  -MaterializedDirectory <generated-directory> `
  -Deployment green

./scripts/Deploy-AzureML.ps1 `
  -ConfigPath <approved-azureml-input.json> `
  -MaterializedDirectory <generated-directory> `
  -Deployment green `
  -Apply
```

The publication script changes no traffic. The 180-second Azure ML timeout is
the outer container boundary; it is not the application deadline. The selected
profile's lower serving and APIM deadlines remain the user-visible contract.

## Inspect the model endpoint

```powershell
$WorkspaceName = "<azure-ml-workspace-name>"
$EndpointName = "<managed-online-endpoint-name>"
$CandidateDeployment = "<inactive-blue-or-green-deployment>"

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
  --name $CandidateDeployment `
  --resource-group $ResourceGroup `
  --workspace-name $WorkspaceName `
  --lines 200

az ml online-endpoint invoke `
  --name $EndpointName `
  --resource-group $ResourceGroup `
  --workspace-name $WorkspaceName `
  --request-file azureml/samples/chat-request.json
```

The committed request is synthetic and conforms by source inspection only. Use the
stream sample separately only after the target inference image and all front ends
are proven not to buffer SSE.

## Publish APIM and run cloud smoke

```powershell
$ApimServiceName = "<api-management-service-name>"

./scripts/Publish-APIM.ps1 `
  -ConfigPath <approved-apim-input.json> `
  -BackupDirectory <new-prechange-snapshot-directory>

# After reviewed dry run and backend provider what-if:
./scripts/Publish-APIM.ps1 `
  -ConfigPath <approved-apim-input.json> `
  -BackupDirectory <new-prechange-snapshot-directory> `
  -Apply

```

The default path verifies explicit named values/backend inputs and approved
source digests without contacting Azure. `-Apply` verifies the selected Azure
CLI account, snapshots the managed API/named values/backends/fragments/policy,
runs backend Bicep what-if, and publishes in dependency order. Snapshot creation
is not rollback rehearsal. Cloud smoke must cover
authorization denial, token/input bounds, readiness, streaming and non-streaming
shapes, citations, normalized errors, deployment metadata, and telemetry redaction.

## Shadow and canary

Assume blue is known-good and green is the candidate; reverse the names when blue
is inactive. Mirrored traffic does not change the response returned to clients.

```powershell
./scripts/Set-AzureMLTraffic.ps1 `
  -ConfigPath <approved-azureml-input.json> `
  -BluePercent 90 `
  -GreenPercent 10

./scripts/Set-AzureMLTraffic.ps1 `
  -ConfigPath <approved-azureml-input.json> `
  -BluePercent 90 `
  -GreenPercent 10 `
  -Apply
```

The committed rollout files cover canary and full blue/green traffic; they do not
configure Azure ML mirrored traffic. Treat shadow as an unimplemented deployment
interface until an approved command/asset is added. Never advance because a command
succeeded. At each stage, retain test-engine health,
request counts, uncertainty, errors, TTFT/TPOT/total latency, quality and safety
results, cost, deployment logs, approver, and observation window. Use the abort
criteria in the approved release report.

## Run and retain Azure Load Testing evidence

Prepare the explicit managed-identity input described in `scripts/README.md`.
The selected system- or user-assigned identity must already be attached to the
Azure Load Testing resource and authorized for the APIM application scope.

```powershell
./scripts/Materialize-AzureLoadTest.ps1 `
  -ConfigPath <approved-load-input.json> `
  -OutputDirectory <new-generated-load-directory>

# Dry run, then independently approved mutation.
./scripts/Start-AzureLoadTest.ps1 `
  -ConfigPath <approved-load-input.json> `
  -MaterializedDirectory <generated-load-directory> `
  -TestRunId <lowercase-immutable-run-id>

./scripts/Start-AzureLoadTest.ps1 `
  -ConfigPath <approved-load-input.json> `
  -MaterializedDirectory <generated-load-directory> `
  -TestRunId <lowercase-immutable-run-id> `
  -Apply

# Use only after Azure reports a terminal state.
./scripts/Export-AzureLoadTestEvidence.ps1 `
  -ConfigPath <approved-load-input.json> `
  -MaterializedDirectory <generated-load-directory> `
  -TestRunId <run-id> `
  -OutputDirectory <new-evidence-directory> `
  -Apply
```

Export downloads result, report, and log archives; captures load-run and
engine-health metric namespaces; and writes a SHA-256 inventory without
overwriting prior evidence. A failed/error/stopped run is retained but fails the
workflow. Converting raw preview metric JSON to the normalizer's engine-health
shape remains unvalidated and must be reviewed against a retained service result.

## Completion record

Deployment is complete only when the change record includes:

- release ID/version/digests and source commit;
- environment, region, endpoint, active slot, runtime, and index version;
- static gate, what-if, cloud smoke, load, shadow, and canary evidence;
- policy/RBAC and private-network negative tests;
- monitor/alert and telemetry-content review;
- final traffic allocation and retained rollback target;
- cost/capacity observations and unresolved deviations;
- operator, reviewer, approver, and timestamps.

## Teardown

Teardown is destructive and requires a separate approved retention/deletion plan.
Do not run it for production incident mitigation.

```powershell
azd down
```

Before teardown, export required evidence, protect retained artifacts and evaluation
reports, confirm legal/retention obligations, and verify whether the command will
remove shared or durable resources.
