# Riverside Azure Infrastructure

This directory is the resource-group-scope Bicep composition for Riverside's
Azure production project. It creates shared foundations, can create Azure ML and
endpoint shells, and can either create or reference costly services. It does not
deploy model code, APIM policies, Databricks jobs, endpoint deployments, or
traffic weights.

## Modules

| Module | Responsibility |
|---|---|
| `identities.bicep` | Separate platform, Azure ML workspace, endpoint, and gateway user-assigned identities |
| `monitoring.bicep` | Log Analytics plus workspace-based Application Insights with local authentication disabled |
| `storage.bicep` | ADLS Gen2, private containers, OAuth-by-default access, RBAC, retention, and diagnostics |
| `key-vault.bicep` | RBAC-only Key Vault, purge protection, firewall controls, and audit diagnostics; no secrets are created |
| `machine-learning.bicep` | Azure ML workspace and optional Entra-authenticated managed online endpoint shell |
| `api-management.bicep` | Optional APIM creation or shared-service reference, managed identity, VNet settings, and diagnostics |
| `load-testing.bicep` | Optional Azure Load Testing resource and diagnostics |
| `databricks-integration.bicep` | Existing workspace, Unity Catalog, and Vector Search coordinates; no PAT or workspace creation |
| `private-endpoint.bicep` | Reusable Private Link endpoint with existing private DNS zone links |

`main.bicep` composes the modules and exports environment variables matching the
frozen config and deployment contracts. `azure.yaml` selects `infra/main.bicep`,
and `main.parameters.json` maps the `azd` environment name and location. Use only
the environment names `dev`, `staging`, or `production` so they remain valid
under `config/schema.json`.

## Identity And RBAC

The templates never accept a password, API key, connection string, SAS token,
Databricks PAT, or client secret.

| Principal | Narrow assignments created here |
|---|---|
| Platform identity | Storage Blob Data Contributor; Key Vault Secrets User |
| Azure ML workspace identity | Storage Blob Data Contributor; Key Vault Secrets Officer |
| Azure ML endpoint identity | Storage Blob Data Reader; Key Vault Secrets User |
| Gateway invoker | Endpoint-scoped AzureML Data Scientist, only when `assignGatewayInvokeRole` is enabled |

Azure ML currently requires the built-in AzureML Data Scientist role for managed
online endpoint invocation. Its scope is the endpoint, not the workspace or
resource group. When APIM is reused, set
`existingApiManagementGatewayPrincipalId` to the principal already attached to
that APIM service. The template grants that principal endpoint access; it does
not replace or mutate the existing APIM identity.

Existing Azure ML or APIM services must already satisfy their own associated
resource and identity requirements. Set diagnostic ownership flags deliberately
because a diagnostic setting with the same name is managed by this deployment.

## Network Modes

| Mode | Behavior |
|---|---|
| `public` | Service public endpoints are enabled. Storage and Key Vault accept all networks but still require Entra/RBAC authorization. |
| `restricted` | Storage and Key Vault default-deny and use the supplied CIDR/subnet allowlists. Azure ML remains public because its workspace firewall is represented by private connectivity rather than IP rules. |
| `private` | Storage, Key Vault, Azure ML workspace, and managed endpoint public access is disabled. Blob, DFS, vault, and Azure ML workspace private endpoints are created. Existing private DNS zones and a private-endpoint subnet are mandatory. |

For newly created APIM in private mode, `apiManagementVirtualNetworkType` must
be `Internal`, the selected SKU must support VNet injection, and a dedicated
subnet is required. Shared APIM networking is owned by that existing service.
APIM policy-level IP/token controls remain in the separately owned `apim/`
assets.

Private access for Log Analytics and Application Insights requires Azure Monitor
Private Link Scope, which is intentionally not created by this resource-group
composition. Keep `allowAzureMonitorPublicAccess` true with Entra authentication,
or set it false only after an existing AMPLS, DNS, and ingestion/query path are
verified.

## Parameter Examples

The files below are reviewable examples, not committed credentials or ready-to-
deploy production values:

- `config/dev.parameters.example.json`
- `config/staging.parameters.example.json`
- `config/production.parameters.example.json`

Replace every `replace-with-*`, documentation CIDR, all-zero UUID, and example
domain before use. The production example makes private DNS dependencies
explicit. The staging example enables Azure Load Testing; production reuses it
to avoid permanent duplicate test infrastructure.

## Diagnostics And Outputs

Storage data-plane logs, Key Vault audit logs, Azure ML logs, APIM logs, Load
Testing logs, and platform metrics target the shared Log Analytics workspace
when their diagnostic flags are enabled. Application logs use workspace-based
Application Insights. Telemetry attributes must still follow the bounded
allowlist in `contracts/v1/telemetry-attributes.schema.json`; infrastructure
diagnostics do not authorize prompt, completion, manuscript, tenant, user, or
request content as metric labels.

Important `azd` outputs include the gateway URL, Azure ML workspace and endpoint
names, blue/green deployment names, ADLS and artifact URIs, Key Vault URI,
monitoring resource IDs, managed-identity client IDs, load-testing resource ID,
and Databricks catalog/vector-search coordinates. No output contains a key or
token. Bicep outputs are written into the selected `azd` environment after an
authorized provision operation.

## Cost And Teardown

APIM dedicated tiers, Azure ML online deployments, Databricks clusters/SQL
warehouses/vector search, Log Analytics ingestion, retained telemetry, private
endpoints, and load-test engine minutes can dominate cost. This template defaults
APIM, Databricks, and Load Testing to reuse or no creation. Azure ML endpoint
deployments are not created here, so their instance count and accelerator cost
remain release-time decisions.

Before teardown, preserve only approved release evidence and confirm whether
ADLS artifacts or evaluation reports are records that must be retained. A normal
`azd down` targets resources managed by this environment; referenced APIM and
Databricks resources are not owned by the template. Key Vault and Storage have
soft-delete/data-retention behavior, so resource deletion does not prove data
purge. `azd down --purge` can permanently purge soft-deleted resources such as
Key Vault and requires explicit data-owner approval. Confirm endpoint deployments,
private DNS links, managed resource groups, diagnostic settings, and Azure ML
service-managed resources separately in teardown evidence.

## Validation Boundary

The files were authored with editor schema diagnostics only. No `bicep`, `az`,
`azd`, deployment, deployment-stack, or what-if command was run. Before an
authorized deployment, the owning team must run Bicep build/lint, tenant policy
review, provider registration checks, quota/region checks, what-if, secret scan,
and an independent RBAC/network/cost review.
