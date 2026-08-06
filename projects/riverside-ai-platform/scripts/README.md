# Riverside deployment scripts

These PowerShell scripts are source interfaces for Azure ML, API Management,
and Azure Load Testing. They were authored without execution. No Azure resource,
model, environment, API, policy, backend, named value, load test, or test run was
created, updated, read, or deleted while these files were added.

All mutating scripts default to dry-run behavior and require `-Apply`. Azure
access uses the current `az login`, workload-identity, or managed-identity
session. The scripts accept no password, client secret, API key, bearer token,
connection string, or Key Vault secret URI. They verify that the active Azure
CLI tenant and subscription exactly match explicit JSON inputs before mutation.

## Offline preflight

`Preflight.ps1` is a fail-closed static validator. It has no `-Apply` switch,
does not invoke `az`, does not read cloud state, and never deploys. Run it after
Azure ML materialization and before any registration, publication, traffic, or
infrastructure command:

```powershell
./scripts/Preflight.ps1 `
  -Mode Offline `
  -ConfigPath ./config/staging.resolved.yaml `
  -InfrastructureParametersPath ./config/staging.parameters.json `
  -AzureMLConfigPath ./generated/staging.azureml.json `
  -MaterializedDirectory ./generated/azureml/staging `
  -ApimConfigPath ./generated/staging.apim.json
```

The preflight requires resolved, environment-specific inputs. Checked-in
`*.example.json` files and YAML containing `${ENVIRONMENT_REFERENCES}` are
expected to fail. For production, add `-ProductionOptIn`; this opt-in permits
static validation only and grants no deployment permission.

The validator checks:

- the resolved profile shape and all required schema sections;
- absence of unresolved environment references, template tokens, zero GUIDs,
  `replace-with-*` values, angle-bracket placeholders, and `.example.invalid`
  endpoints;
- exact environment and region agreement across profile, infrastructure,
  Azure ML input, materialization manifest, endpoint YAML, and deployment YAML;
- application, gateway, and Azure ML deadline ordering;
- materialized file digests, endpoint and slot names, index version, and exact
  Azure ML model/environment asset references;
- required subnet, private DNS zone, reused APIM identity, workspace, and load
  testing inputs when the corresponding resource is not provisioned;
- APIM named-value coverage, service name, backend IDs, backend pool binding,
  HTTPS URLs, optional extension backend IDs, and backend timeout;
- Databricks subscription, workspace URL/name, catalog, schema, vector-search,
  embedding endpoint, dimensions, and immutable release values;
- explicit production opt-in and agreement of every production environment
  label.

Passing preflight is `STATIC_VALIDATION` evidence only. It cannot prove that an
existing resource exists, an identity has access, a region has capacity, a
policy is accepted by APIM, or Databricks/Azure ML behaves correctly.

Run the pure helper and orchestration tests without Azure access:

```powershell
Invoke-Pester ./tests/unit/preflight/Preflight.Tests.ps1
```

## Digest contract

File digests are lowercase SHA-256. Directory digests are deterministic:

1. recursively list regular files;
2. sort by `/`-separated path relative to the directory;
3. create one line per file as `<relative-path><TAB><file-sha256>`;
4. join lines with LF and include a final LF;
5. SHA-256 the UTF-8 bytes.

Materialization stops on a missing file, empty directory, malformed digest,
digest mismatch, unresolved placeholder, unsafe scalar, non-digest container
image, invalid Azure ML asset name/version, unsupported environment, or
application deadline outside 1-120 seconds.
The 180-second Azure ML `request_timeout_ms` remains the outer container
boundary. The profile's lower application deadline remains authoritative and is
recorded as deployment metadata; the scripts do not claim that Azure ML enforces
the application deadline.

## Azure ML input

`Materialize-AzureML.ps1` requires a JSON object with:

```json
{
  "subscription_id": "<explicit-guid>",
  "tenant_id": "<explicit-guid>",
  "resource_group": "<explicit-name>",
  "workspace_name": "<explicit-name>",
  "environment": "<dev|staging|production>",
  "region": "<approved-region>",
  "endpoint_name": "<bicep-output-name>",
  "blue_slot_name": "<slot-name>",
  "green_slot_name": "<slot-name>",
  "blue_deployment_name": "<telemetry-name>",
  "green_deployment_name": "<telemetry-name>",
  "blue_model_name": "<registered-asset-name>",
  "blue_model_version": "<immutable-version>",
  "green_model_name": "<registered-asset-name>",
  "green_model_version": "<immutable-version>",
  "environment_asset_name": "<registered-environment-name>",
  "environment_asset_version": "<immutable-version>",
  "base_image_by_digest": "<registry/image@sha256:64-lowercase-hex>",
  "index_version": "<immutable-index-version>",
  "deployed_at": "<yyyy-MM-ddTHH:mm:ssZ>",
  "application_deadline_seconds": 100,
  "paths": {
    "code_package": "<directory-containing-score.py>",
    "blue_model_package": "<complete-model-package-directory>",
    "green_model_package": "<complete-model-package-directory>"
  },
  "expected_sha256": {
    "code_package": "<directory-digest>",
    "blue_model_package": "<directory-digest>",
    "green_model_package": "<directory-digest>",
    "conda_file": "<file-digest>",
    "environment_template": "<file-digest>"
  }
}
```

Each model package must contain `model-release-manifest.json`, `base-model/`,
`adapter_config.json`, and every `repo://` artifact declared by the release
manifest. Adapter, tokenizer, training-manifest, and evaluation-report digests
are verified before YAML is written. Materialized deployment tags bind the
release-manifest, model-package, code-package, environment-image, conda, and
environment-template SHA-256 values to exact model/environment asset versions.
Registration refuses to reuse an existing model/environment version whose
digest tags differ, validates exact returned Azure resource IDs and name/version
fields in apply mode, and emits `registration-manifest.json`.

The orchestrator Dockerfile has no default base image. Build it only with a
digest reference, for example `--build-arg
PYTHON_BASE_IMAGE=python@sha256:<64-lowercase-hex>`. The Container Apps module
also rejects an `orchestratorImage` that is not an `@sha256:` reference.

Expected order:

```powershell
./scripts/Materialize-AzureML.ps1 -ConfigPath <config.json> -OutputDirectory <generated>
./scripts/Register-AzureMLAssets.ps1 -ConfigPath <config.json> -MaterializedDirectory <generated>
./scripts/Register-AzureMLAssets.ps1 -ConfigPath <config.json> -MaterializedDirectory <generated> -Apply
./scripts/Deploy-AzureML.ps1 -ConfigPath <config.json> -MaterializedDirectory <generated>
./scripts/Deploy-AzureML.ps1 -ConfigPath <config.json> -MaterializedDirectory <generated> -Deployment green -Apply
./scripts/Set-AzureMLTraffic.ps1 -ConfigPath <config.json> -BluePercent 90 -GreenPercent 10
./scripts/Set-AzureMLTraffic.ps1 -ConfigPath <config.json> -BluePercent 90 -GreenPercent 10 -Apply
```

## APIM input

`Publish-APIM.ps1` requires explicit target fields, `api_id`, `api_path`, every
non-secret key from `apim/parameters/named-values.json`, backend names/HTTPS
URLs/weights/priorities, and expected source digests:

```json
{
  "subscription_id": "<explicit-guid>",
  "tenant_id": "<explicit-guid>",
  "resource_group": "<explicit-name>",
  "apim_service_name": "<explicit-name>",
  "api_id": "<explicit-api-id>",
  "api_path": "<explicit-path>",
  "named_values": {"<every-contract-name>": "<non-secret-value>"},
  "backends": {
    "blue_name": "<name>", "green_name": "<name>", "pool_name": "<name>",
    "blue_url": "<https-url>", "green_url": "<https-url>",
    "blue_weight": 100, "green_weight": 1,
    "blue_priority": 1, "green_priority": 2
  },
  "expected_sha256": {
    "openapi": "<file-digest>",
    "api_policy": "<file-digest>",
    "fragments": "<directory-digest>",
    "named_values_contract": "<file-digest>",
    "backend_template": "<file-digest>"
  }
}
```

Publish snapshots the managed API, named values, blue/green/pool backends,
fragments, and API policy before applying source. Absence is retained explicitly.
Backend Bicep always runs provider what-if before create. Restore requires the
snapshot's independently retained SHA-256 and restores prior values or deletes
only Riverside-managed objects that were absent before publication. Secret APIM
named values are rejected by publish and restore.

```powershell
./scripts/Publish-APIM.ps1 -ConfigPath <apim.json> -BackupDirectory <new-backup>
./scripts/Publish-APIM.ps1 -ConfigPath <apim.json> -BackupDirectory <new-backup> -Apply
./scripts/Restore-APIM.ps1 -ConfigPath <apim.json> -SnapshotPath <snapshot.json> -ExpectedSnapshotSha256 <sha256>
./scripts/Restore-APIM.ps1 -ConfigPath <apim.json> -SnapshotPath <snapshot.json> -ExpectedSnapshotSha256 <sha256> -Apply
```

## Azure Load Testing input

The load engine acquires its API token through `ManagedIdentityCredential`.
For `UserAssigned`, provide both the identity resource ID selected by Azure Load
Testing and its client ID used by the test process. For `SystemAssigned`, omit
both values.

```json
{
  "subscription_id": "<explicit-guid>",
  "tenant_id": "<explicit-guid>",
  "resource_group": "<explicit-name>",
  "load_test_resource": "<explicit-name>",
  "test_id": "<explicit-test-id>",
  "display_name": "<display-name>",
  "target_host": "<https-origin>",
  "token_scope": "<application-id-uri/.default>",
  "engine_instances": 2,
  "engine_identity_type": "<SystemAssigned|UserAssigned>",
  "engine_identity_resource_id": "<required-for-user-assigned>",
  "engine_identity_client_id": "<required-for-user-assigned>"
}
```

```powershell
./scripts/Materialize-AzureLoadTest.ps1 -ConfigPath <load.json> -OutputDirectory <generated>
./scripts/Start-AzureLoadTest.ps1 -ConfigPath <load.json> -MaterializedDirectory <generated> -TestRunId <lowercase-run-id>
./scripts/Start-AzureLoadTest.ps1 -ConfigPath <load.json> -MaterializedDirectory <generated> -TestRunId <lowercase-run-id> -TestOperation update -Apply
./scripts/Export-AzureLoadTestEvidence.ps1 -ConfigPath <load.json> -MaterializedDirectory <generated> -TestRunId <run-id> -OutputDirectory <new-evidence-dir>
./scripts/Export-AzureLoadTestEvidence.ps1 -ConfigPath <load.json> -MaterializedDirectory <generated> -TestRunId <run-id> -OutputDirectory <new-evidence-dir> -Apply
```

Export refuses a non-terminal run, never overwrites an evidence directory,
downloads result/report/log archives, exports `LoadTestRunMetrics` and
`EngineHealthMetrics`, and writes a SHA-256 inventory. A non-`Passed` run is
retained but fails the script. The exported service metric JSON is raw evidence;
transforming it into the normalized `result_parser.py` engine-health shape
remains an explicit reviewed step because the current preview CLI response shape
has not been cloud-validated here.

## Unvalidated behavior

Editor diagnostics are not execution evidence. These scripts have not been
parsed by an installed PowerShell runtime, invoked against Azure CLI extensions,
or tested against service API behavior. APIM policy expression acceptance,
snapshot round-trip fidelity, Azure ML asset upload/tag read-back, Azure Load
Testing managed-identity token acquisition, result archive names, metric response
shape, and all networking/RBAC/quota/region behavior require an authorized
non-production validation and retained evidence before production use.
