# Security and Data Boundaries

## Security posture

Riverside handles manuscript and editorial content that may be confidential or
restricted. The design assumes zero trust between client, gateway, orchestration,
data plane, model serving, telemetry, and build/deployment systems. Every crossing
needs authenticated identity, least-privilege authorization, bounded input/output,
and auditable configuration.

The contracts express intended fields and exclusions. They do not prove security
enforcement.

## Trust boundaries

| Boundary | Accepted input | Required control | Output limit |
|---|---|---|---|
| Client to APIM | Entra token plus v1 request | issuer/audience/signature, authorization, request/token/rate policy | normalized v1 response/error only |
| APIM to orchestrator | Trusted identity context and bounded request | managed identity, internal Container Apps ingress, duplicate 1 MiB body cap, deadline | no client-supplied tenant/ACL authority |
| Orchestrator to index | query plus server-derived authorization filters | tenant, ACL, region, classification, deletion filters | authorized chunks and immutable lineage only |
| Orchestrator to Azure ML | bounded prompt/context and stable alias | managed identity, private approved path, release binding | bounded stream/non-stream result |
| Components to telemetry | allowlisted operational attributes | redaction, sampling, TLS, diagnostic destination RBAC | no customer content/high-cardinality identity labels |
| CI/CD to Azure | reviewed artifact/IaC and workload identity | branch/change controls, least privilege, provenance | immutable deployment/evidence records |

## Identity and authorization

- Clients authenticate with Microsoft Entra ID. The application authorizes the
  operation and derives tenant context from trusted claims.
- APIM and workloads use managed identity in Azure; CI/CD uses workload identity.
- API keys and committed client secrets are outside the config contract.
- Use built-in roles where they fit, narrow scopes, and separate management-plane
  deploy rights from data-plane read/write rights.
- APIM policy editors are privileged: policy changes can use the APIM identity to
  reach allowed backends. Limit, review, and audit policy write access.
- Perform access reviews and negative tests for every identity path. Role assignment
  existence does not prove the path works or is least privilege.

## Data classification and records

The v1 data contracts require `tenant_id`, source/version/hash, ACL, region,
classification, pipeline lineage, and deletion state. Classification is one of
`public`, `internal`, `confidential`, or `restricted`; ACL visibility is `tenant`
or `restricted`.

These values travel with raw, parsed, chunk, and vector records. Producers must not
downgrade classification or broaden ACLs during parsing/chunking/indexing. Serving
must enforce filters before returning content. A schema-valid ACL is not proof that
the backend applied it.

`pending` and `deleted` states carry deletion timestamps. Deleted content must not
remain queryable. Propagation objectives, backup handling, and legal holds require
approved policy and live tests; none is committed yet.

Use the [deletion propagation rehearsal](checklists/deletion-rehearsal.md) to prove
source-to-index removal and that a pre-deletion backup cannot reintroduce content
before the deletion ledger is reapplied. The procedure is unexecuted and does not
establish a propagation objective.

## API and citation boundary

- The client selects a stable model alias, not a physical deployment.
- Message count, message length, input/output tokens, retrieval top-k, temperature,
  timeout, and retries are bounded. The gateway/runtime still tokenizes and enforces
  the aggregate limit.
- Citations contain lineage and answer spans, not copied source excerpts. A
  `source_uri` may itself be sensitive and is returned only to an authorized client;
  it is never a metric label.
- Errors expose normalized categories and safe messages. Backend exception text,
  credentials, stack traces, and internal resource identifiers stay internal and
  redacted.
- Streaming must enforce the same authorization, deadline, output, citation, and
  telemetry controls as non-streaming.

## Network boundaries

The Bicep source implements public/restricted/private modes, private endpoints for
Storage, Key Vault, and Azure ML, existing DNS-zone inputs, identity/RBAC modules,
optional APIM networking, and an internal Container Apps environment with a
deployable orchestrator host. The host is VNet-facing only through the environment's
private load balancer, runs as a non-root user, pulls from Entra-only ACR with
managed identity, and accepts trusted identity headers only after validating the
APIM backend token issuer, audience, expiry, and object ID. It does not create an
Azure Monitor Private Link Scope, a Databricks workspace/network, APIM private DNS,
or the APIM backend/policy publication. Source topology is not deployed topology.

Before production use, prove both positive reachability from approved origins and
negative reachability from denied origins. A private endpoint resource existing in
Azure is not enough; DNS and public-access configuration can still defeat intent.
Retain service configuration, DNS answers, observed path, identity tests, and
public/denied-origin failures with the
[private endpoint validation checklist](checklists/private-endpoint-validation.md).

## Secrets and keys

- Prefer identity-based data access; disable shared/key access where service support
  and the approved design permit.
- Store unavoidable certificates/secrets in Key Vault with purge protection,
  RBAC, rotation, expiry alerting, and private access.
- Never commit or log secrets, bearer tokens, connection strings, model registry
  credentials, or customer identifiers.
- Do not store sensitive values in Azure ML environment variables because platform
  diagnostics may collect environment-variable metadata/content.
- Secret scanning covers current source and history; a clean scan does not replace
  runtime log/telemetry review.

## Telemetry and debugging

Metrics use only the v1 allowlist. The host configures OTLP HTTP trace and metric
exporters during runtime composition and fails startup when required telemetry
configuration is missing or invalid. Prompts, completions, manuscript/document text,
source URIs, user/request/tenant/document/chunk IDs, authorization headers, and
tokens are excluded from metric dimensions. Logs deny request/response bodies by
default. Trace context may carry a trace ID, but no content.

Future content capture requires a separate ADR, purpose/legal basis, explicit access
model, encryption, region, retention, deletion, audit, and incident process. Debug
mode cannot silently weaken production controls.

Support personnel receive no implicit data or platform access. Review direct,
group, eligible, activated, vendor, break-glass, service-specific, and evidence-store
permissions with the [support-access audit](checklists/support-access-audit.md).
Require scoped purpose, approval, expiry, audit events, separation of duties, and
positive/negative tests; content access needs separate legal and data-owner approval.

## Supply chain and release

- Pin source commit, base revision, adapter/tokenizer/runtime versions, image digest,
  dependencies, and evaluation evidence.
- Verify SHA-256 digests before readiness and reject mutable labels such as `latest`.
- Build with isolated workload identity and retain provenance/SBOM/vulnerability
  results when the build pipeline is implemented.
- Promotion requires all eight release-report domains and human approval.
- Keep blue/green rollback targets immutable and compatible with retained index
  versions.

## Required security evidence

| Claim | Static evidence | Live Azure evidence |
|---|---|---|
| Contract rejects unsafe shapes | Valid/invalid fixtures and invariant tests | Gateway/runtime contract smoke |
| Identity is least privilege | IaC/RBAC review | positive and negative token/data-plane tests |
| Tenant/ACL isolation works | Adapter/filter unit tests | deployed cross-tenant and forbidden-document tests |
| Public paths are disabled | Bicep build/lint and policy review | network configuration export plus denied-origin tests |
| Only APIM can invoke chat | JWT validator and guarded direct-backend negative tests | valid APIM token succeeds; missing/invalid token and denied network origin fail |
| Telemetry excludes content | allowlist/redaction tests | sampled metrics/logs/traces reviewed under load/errors |
| Artifacts are immutable | digest/runtime validation tests | registry/deployment digest reconciliation |
| Deletion propagates | pipeline invariant tests | source-to-index-to-backup deletion rehearsal |
| Incident controls work | tabletop checklist | authorized staging containment/rollback exercise |

Until both columns are complete for the selected environment, the corresponding
security claim remains unproven.
