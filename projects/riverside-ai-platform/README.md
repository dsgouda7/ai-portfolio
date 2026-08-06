# Riverside AI Platform

Riverside is the Azure production composition for grounded editorial assistance
over authorized manuscript content. It joins frozen v1 contracts, governed
Databricks retrieval, an Azure ML custom-model endpoint, APIM policy, bounded
telemetry, evaluation gates, and operational assets without importing training or
ingestion implementation.

> **Evidence status (2026-08-05):** source and test assets exist. The non-cloud
> suite passed 142 tests with 5 cloud tests deselected, and the offline preflight
> passed 9 tests. No cloud test, deployment, Azure command, Databricks job, or live
> Azure/Databricks validation was performed. These local results are not evidence
> of production readiness or live cloud behavior.

## System boundary

```text
Editing client -> APIM -> FastAPI RAG host -> Databricks AI Search
                                  |
                                  +------------> Azure ML model endpoint
```

- APIM validates the client token and overwrites trusted Riverside identity
  headers.
- The FastAPI host is provisioned in an internal Azure Container Apps environment
  on a delegated subnet. Its ingress is exposed only through the environment's
  private load balancer. APIM must resolve and reach that private FQDN; public
  ingress is disabled by the environment design.
- The FastAPI host accepts the frozen `/v1/chat/completions` request, enforces the
  selected profile, retrieves through the `SearchIndex` protocol, and invokes the
  existing endpoint-client protocol.
- Databricks access uses Microsoft Entra managed/workload identity and the v1
  vector-record fields. Riverside does not import `rag-knowledge-pipeline` source.
- Release evaluation uses precomputed evidence. The release-gate package makes no
  model or cloud calls.
- Metric labels use the frozen bounded telemetry context. Prompt, completion,
  document, user, tenant, and request content are excluded.

Conceptual and data-plane owners:

- [Azure operational serving tutorial](../../learning/ai-infrastructure/09-azure-operational-llm-serving/README.md)
- [Databricks RAG knowledge pipeline](../rag-knowledge-pipeline/README.md)
- [Frozen v1 contracts](contracts/README.md)
- [Architecture and evidence boundary](docs/architecture.md)

## Project map

| Path | Responsibility |
|---|---|
| `src/app/` | FastAPI host, profile loading, Databricks contract adapter, runtime composition, request telemetry |
| `src/cli/` | `validate`, `evaluate`, and `report` operational commands |
| `src/artifact_validation/` | Immutable release verification and Azure ML serving lifecycle |
| `src/endpoint_client/` | Managed-identity Azure ML/Foundry clients and normalized v1 responses/errors |
| `src/rag_orchestrator/` | Authorization-aware retrieval, context assembly, citation resolution, refusal |
| `src/release_gates/` | Typed metrics, threshold policy, decisions, release reports |
| `src/telemetry/` | Bounded OTel dimensions, latency/token metrics, privacy-safe logging |
| `config/` | Schema plus Azure-only `dev`, `staging`, and `production` profiles |
| `azureml/`, `apim/`, `infra/` | Azure ML, API Management, Container Apps/ACR, Docker, and Bicep source assets |
| `evaluations/`, `load-tests/` | Versioned evaluation inputs and staged load assets |
| `tests/` | Unit, contract, cloud-free integration, and guarded cloud tests |

There is deliberately no local production profile. Local mechanism experiments
belong to the linked learning tutorial.

## Package and host

The following remain operator commands, not evidence of a deployed host.

```powershell
Set-Location projects/riverside-ai-platform
python -m pip install -e ".[test,telemetry]"
$env:RIVERSIDE_ENVIRONMENT = "dev"
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Routes:

| Route | Meaning |
|---|---|
| `GET /health` | Process liveness only |
| `GET /ready` | Profile composed and configured Databricks index resolved |
| `POST /v1/chat/completions` | Frozen v1 non-streaming response or SSE stream |

Azure ML readiness remains owned by its release verification, model load, and
warm-up probe. The RAG host does not generate a synthetic model request during
startup. Retrieval-backed streaming is currently contract-valid buffered SSE:
the orchestrator resolves citations after generation, then emits content and a
final usage/citation event. Do not claim token-by-token TTFT from this path.

## Environment profiles

Profiles are validated after whole-value `${NAME}` substitution. Inline template
fragments and defaults are not supported. Missing values fail closed. API keys,
Databricks personal access tokens, and static model keys are rejected.

The Bicep outputs supply these profile references:

| Variable | Purpose |
|---|---|
| `RIVERSIDE_RELEASE_MANIFEST_URI` | Immutable model release manifest |
| `RIVERSIDE_GATEWAY_BASE_URL` | Application-facing APIM origin |
| `RIVERSIDE_SERVING_ENDPOINT_NAME` | `riverside-<environment>` endpoint name |
| `RIVERSIDE_BLUE_DEPLOYMENT_NAME` | `riverside-<environment>-blue` |
| `RIVERSIDE_GREEN_DEPLOYMENT_NAME` | `riverside-<environment>-green` |
| `RIVERSIDE_EVALUATION_REPORT_URI` | Bound eight-domain release report |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | Approved OTLP collector |
| `RIVERSIDE_ORCHESTRATOR_NAME`, `RIVERSIDE_ORCHESTRATOR_URL` | Internal Container App deployment target and private backend origin |
| `AZURE_CONTAINER_REGISTRY_ENDPOINT` | Entra-only ACR used by `azd` image publication |

Initial provisioning uses the public Microsoft Container Apps hello-world image
only as a bootstrap revision so the private ACR does not need to contain an image
before it exists. `azd deploy rag-orchestrator` builds the repository Dockerfile,
pushes to the provisioned ACR, and updates the tagged Container App. Release
evidence must retain the resulting digest; the bootstrap revision is not an
application release.

The application runtime additionally requires:

| Variable | Purpose |
|---|---|
| `RIVERSIDE_CONFIG` | Optional profile path; defaults from `RIVERSIDE_ENVIRONMENT` |
| `RIVERSIDE_ENDPOINT_PROVIDER` | `azure_ml` or `foundry`; `apim` is rejected to prevent a routing loop |
| `RIVERSIDE_ENDPOINT_URL`, `RIVERSIDE_ENDPOINT_ROUTE` | Model endpoint origin and scoring route |
| `RIVERSIDE_ENDPOINT_TOKEN_SCOPE` | Entra scope ending in `/.default` |
| `RIVERSIDE_ENDPOINT_TIMEOUT_SECONDS` | Model deadline within the APIM profile deadline |
| `RIVERSIDE_AZUREML_DEPLOYMENT` | Optional physical Azure ML slot header |
| `RIVERSIDE_BACKEND_TENANT_ID` | Entra tenant allowed to issue the APIM backend token |
| `RIVERSIDE_BACKEND_AUDIENCE` | Exact audience used by APIM managed identity |
| `RIVERSIDE_APIM_PRINCIPAL_ID` | Only APIM managed-identity object ID trusted by the host |
| `DATABRICKS_HOST` | HTTPS workspace origin, with no path or credential |
| `RIVERSIDE_VECTOR_SEARCH_ENDPOINT` | Existing Databricks AI Search endpoint |
| `RIVERSIDE_EMBEDDING_ENDPOINT` | Managed embedding serving endpoint |
| `RIVERSIDE_EMBEDDING_DIMENSIONS` | Exact pinned vector dimensions |
| `RIVERSIDE_MODEL_RELEASE_ID` | Controlled telemetry release dimension |
| `RIVERSIDE_ACTIVE_DEPLOYMENT_NAME` | Configured blue or green telemetry dimension |
| `AZURE_CLIENT_ID` | Optional user-assigned managed-identity client ID |

Profile regions are `eastus2` for `dev`/`staging` and `uksouth` for the modeled
production parameters. Production selection is still unapproved and unvalidated.

### Name, region, and deadline reconciliation

- Profiles consume Bicep's `riverside-<environment>` endpoint and blue/green names
  through environment references. Deployment automation must override the
  component-owned Azure ML YAML `endpoint_name` and `RIVERSIDE_REGION` from the
  selected profile until those assets accept parameters directly.
- The application contract remains capped at 120 seconds. The enforced ordering is
  model/Databricks dependency timeout < FastAPI serving deadline < APIM deadline.
  Production uses 90 < 100 < 110 seconds; staging uses 70 < 80 < 90; dev uses
  40 < 50 < 60. The Azure ML 180-second container timeout is only an outer backend
  kill boundary and is not exposed as an application deadline.
- APIM and FastAPI both cap request bodies at 1,048,576 bytes. The FastAPI ASGI
  middleware counts actual streamed bytes and does not trust `Content-Length`.
  It is not exposed as an application request deadline.
- The index name and version in each profile are immutable logical inputs shared
  with the Databricks job. A deployment must prove that exact index exists before
  `/ready` succeeds.

## CLI

All commands operate on files and fail closed. They do not contact Azure or
Databricks.

```powershell
# Resolve and validate one profile
riverside validate --config config/staging.yaml

# Validate a JSON document against a frozen schema
riverside validate `
  --schema contracts/v1/app-chat-completion-request.schema.json `
  --document tests/fixtures/valid/app-chat-completion-request.json

# Recompute v1 gates and write the machine-readable release report
riverside evaluate `
  --metrics <approved-metrics.json> `
  --context <release-context.json> `
  --output <evaluation-release-report.json>

# Validate and render a concise review summary
riverside report `
  --report <evaluation-release-report.json> `
  --output <evaluation-release-report.md>
```

`evaluate` exits with code `2` for `hold` or `reject` after retaining the report.
Human approval and change authority remain mandatory even for `promote`.

## Test orchestration

The project `pyproject.toml` registers strict `integration` and `cloud` markers and
collects component, root release-gate, evaluation, and APIM contract tests.

```powershell
python -m pytest
python -m pytest tests/integration -m integration
$env:RIVERSIDE_CLOUD_TESTS = "1"
python -m pytest tests/cloud -m cloud --run-cloud
```

Cloud tests remain skipped unless both the flag and environment opt-in are set.
Production additionally requires `RIVERSIDE_ALLOW_PRODUCTION_CLOUD_TESTS=1`.
The runner supplies `RIVERSIDE_CLOUD_GATEWAY_URL` and
`RIVERSIDE_CLOUD_GATEWAY_SCOPE` for APIM chat tests, plus a network-restricted
`RIVERSIDE_CLOUD_BACKEND_URL` for liveness/readiness probes. The backend URL must
not be public.
The local non-cloud suite passed 142 tests with the 5 guarded cloud tests
deselected. The offline preflight suite passed 9 tests. No cloud test was run.

## Capability ledger

| Capability | Source state | Evidence state |
|---|---|---|
| FastAPI liveness, readiness, and v1 chat host | Implemented | Covered by the 142-passing non-cloud suite; no deployed-host validation |
| Config loading and strict environment substitution | Implemented | Covered by the 142-passing non-cloud suite and 9-passing offline preflight |
| Databricks embedding/vector-search adapter | Implemented against v1 fields and REST shape | No workspace call or filter validation |
| Azure ML/Foundry endpoint client composition | Implemented | No token, network, timeout, or streaming validation |
| Release-gate CLI and report rendering | Implemented | Local test coverage passed; no approved production metric bundle processed |
| Bounded request telemetry | Implemented through existing telemetry contract | Exporter, sampling, retention, and redaction unvalidated |
| Internal Container Apps host and ACR | Bicep, Dockerfile, managed identity, diagnostics, probes, scale bounds, outputs, and `azd` service mapping implemented | Bicep/build/deploy/network/DNS/image evidence absent |
| Blue/green names and region/deadline profiles | Parameterized | Azure ML YAML override/publish pipeline absent |
| Integration/cloud test orchestration | Implemented | Non-cloud suite: 142 passed; 5 cloud tests deselected; no cloud test executed |
| APIM, Azure ML, Bicep, load, rollback | Component source present | Static/live evidence absent |
| Production readiness, SLO, capacity, cost, residency, DR | Not established | Live reviewed evidence required |

Use [promise versus evidence](docs/promise-vs-evidence.md) and
[limitations](docs/limitations.md) for claim control.

## Remaining blockers

1. `azure.yaml` maps `rag-orchestrator` to the internal Container App and its
  Dockerfile. The APIM import/publish/restore pipeline is still absent, and no
  `azd` build or deployment has validated resource discovery or image publication.
2. Component-owned Azure ML endpoint/deployment/rollout YAML still hardcodes
   `riverside-smollm2-chat` and `eastus2`. Profiles define the authoritative values,
   but no deterministic command currently applies those overrides.
3. Registered model/environment creation and immutable model-package assembly are
   not implemented. Azure ML YAML references assets that must already exist.
4. The host validates the APIM managed-identity token signature, issuer, audience,
  expiry, and object ID before trusting internal headers. Live JWKS reachability,
  sovereign-cloud authority settings, identity rotation, and negative tests still
  require deployed evidence.
5. APIM forwards tenant, tier, and actor headers but not group claims. Restricted
  group ACLs therefore require a reviewed APIM claim-to-header extension. The host
  accepts `X-Riverside-Group-IDs` only after backend authentication; clients must
  never be allowed to reach the host directly or inject it through APIM.
6. The v1 vector-index record omits citation `title`, although the citation contract
   requires it. The adapter uses immutable `document_id` as a deterministic display
   fallback. A contract-versioned data-plane change is required for real titles.
7. Retrieval-backed SSE is buffered until citation validation completes. True
   token streaming needs an orchestrator streaming contract that can preserve
   citation correctness and normalized final metadata.
8. Databricks Direct Vector Access REST payloads, filter semantics, identity,
   deletion propagation, score shape, index readiness, region, quota, latency, and
   cost have no retained cloud result.
9. Release/evaluation URIs are validated as configuration values, but the RAG host
   does not download and re-verify artifact/report digests. Azure ML verifies its
   packaged release; an application-level evidence resolver remains a deployment
   design decision.
10. Container Apps has bounded HTTP-concurrency scaling in source, but scale-out,
  cold start, drain, and capacity are unvalidated. Shadow mirroring, atomic index
  rollback, load-result download, and retained evidence pipelines remain absent.

Do not deploy until the procedures in [deployment](docs/deployment.md),
[security and data boundaries](docs/security-and-data-boundaries.md), and
[operations runbook](docs/operations-runbook.md) are reviewed against one approved
environment and release.
