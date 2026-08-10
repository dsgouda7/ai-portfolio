# Riverside APIM AI Gateway

This directory contains static Azure API Management assets for the Riverside
application-facing chat API. It does not contain credentials, subscription
identifiers, deployed resource IDs, or live endpoints. Deployment tooling must
import the API definition, create the backend entities, create the named values,
publish each fragment, and finally apply the composed API policy.

## Asset map

| Asset | Responsibility |
|---|---|
| `api/openapi.json` | Importable OpenAPI 3.0 definition for `/v1/chat/completions`, aligned with the frozen v1 request and error contracts |
| `backends/backends.bicep` | Blue/green single backends, per-backend circuit breakers, and one priority/weighted pool |
| `parameters/named-values.json` | Non-secret deployment contract for every `{{named-value}}` referenced by policy |
| `policies/api-policy.xml` | Ordered composition of API-level policy fragments |
| `policies/fragments/*.xml` | Independently reviewable security, resilience, extension, and telemetry controls |

## Policy order and trust boundary

1. Correlation accepts only a valid W3C `traceparent`; otherwise it creates one.
2. `validate-azure-ad-token` validates tenant, client application, audience, and
   required delegated scope.
3. Tenant ID, tenant tier, and actor ID are derived only from the validated JWT.
   Client-supplied internal headers are overwritten before forwarding.
4. OpenAPI-backed content validation, request bytes, stable model alias, and
   declared input/output limits reject malformed or oversized requests.
5. Optional content safety runs before optional semantic cache lookup.
6. Token limits use a trusted tenant counter key and tier-specific TPM. Token
   estimation is also checked against the per-request input limit before the
   backend call.
7. The selected backend is an APIM pool. APIM distributes traffic by weight
   within a priority group and uses a lower-priority group only when every
   higher-priority backend is unavailable.
8. APIM obtains the backend token with its managed identity. No caller token or
   static backend credential is forwarded.
9. Non-streaming calls retry only `429`, `502`, `503`, and `504`, honor bounded
   delta-seconds or HTTP-date `Retry-After`, and stop at the configured count.
   Streaming calls are never replayed by the gateway.
10. Backend and policy failures are replaced by the v1 safe error envelope.

## Fragment catalog and conceptual owner

| Fragment | Gateway concept | Conceptual notebook section |
|---|---|---|
| `correlation.xml` | Trace continuity without metric-cardinality leakage | `learning/genai/12-llm-gateway/01-llm-gateway.ipynb`, Part 7: Assembling a Production-Style Gateway |
| `client-auth.xml` | Entra client authentication and audience/scope validation | Part 1: What Is an LLM Gateway, and Why Do We Need One? |
| `tenant-context.xml` | Trusted routing and quota context; no client-derived tenant tier | Part 3: Request Routing Strategies |
| `request-guardrails.xml` | Request-size, schema, stable alias, and per-request token bounds | Part 4: Rate Limiting and Throttling |
| `content-safety-extension.xml` | Optional prompt shield and request/completion safety policy | Part 7: Assembling a Production-Style Gateway |
| `cache-lookup-extension.xml` and `cache-store-extension.xml` | Disabled-by-default semantic cache with tenant and actor partitioning | Part 6: Cost Optimization Through Caching |
| `token-governance.xml` | Tenant-tier token limits and five allowlisted metric dimensions | Part 4: Rate Limiting and Throttling; Part 7: Assembling a Production-Style Gateway |
| `backend-identity.xml` | Managed-identity credential boundary | Part 8b: Real Gateway (LiteLLM Bridge), provider credential isolation |
| `bounded-retry.xml` | Retry budget, `Retry-After`, deadline, and stream replay boundary | Part 5: Fallback Strategies and Error Handling |
| `safe-response.xml` and `safe-policy-error.xml` | Stable, non-leaking application errors | Part 5: Fallback Strategies and Error Handling |
| `backends/backends.bicep` | Weighted/priority routing plus per-member circuit breaking | Part 3: Request Routing Strategies; Part 5: Fallback Strategies and Error Handling |

The notebook demonstrates provider-neutral mechanisms with deterministic
simulations. These files map those mechanisms to APIM constructs; they do not
claim deployed RBAC, quota, regional capacity, private networking, or live
circuit-breaker behavior.

## Parameterization

`parameters/named-values.json` is the authoritative list of policy parameters.
The entries are references to environment/IaC inputs, not values to paste into
source control. None is marked secret because bearer tokens and credentials are
never stored as named values: APIM obtains backend tokens at runtime through
managed identity.

Required deployment constraints:

- `riverside-max-retries` is `0` through `3` and must fit the v1 trace contract.
- `riverside-max-input-tokens` is at most `8192`; output is at most `2048`.
- backend timeout and maximum accepted retry delay are at most `120` and `300`
  seconds respectively.
- the request schema ID must identify the imported `ChatCompletionRequest` API
  schema.
- content safety and semantic caching default to `false` until their backends,
  managed-identity RBAC, privacy review, and negative tests are complete.

## Routing and circuit breaking

The Bicep module accepts backend URLs only as deployment parameters. Each single
backend has one circuit-breaker rule covering `429` and `5xx`, accepts
`Retry-After`, validates the TLS chain and host name, and belongs to the pool.

Defaults express a blue-primary/green-standby rollout (`priority` 1 and 2). To
run a weighted canary, assign both deployments the same priority and set weights
such as 95/5. APIM balancing and breaker state are approximate per gateway
instance; they are not a substitute for release gates or service-level tests.

## Extension points

Content safety is a real `llm-content-safety` policy guarded by a named-value
switch. Its backend must use managed identity with the minimum Content Safety
role. Semantic caching is also real but disabled by default. It partitions by
trusted tenant ID, actor ID, and immutable model release to prevent reuse across
authorization contexts. A future broader cache key requires an independently
validated authorization-scope claim and explicit leakage tests.

The token metric uses exactly five custom dimensions, the APIM maximum after
Azure Monitor default dimensions: environment, controlled release ID, stable
model alias, controlled deployment name, and allowlisted tenant tier. Request,
actor, tenant, prompt, completion, source, document, and trace identifiers are
forbidden dimensions. Token counts remain metric values; downstream telemetry
may add only attributes permitted by
`contracts/v1/telemetry-attributes.schema.json`.

## Static verification

`tests/contract/apim/` contains a standard-library policy inspection toolkit,
scenario manifest, and pytest-compatible tests. Those checks parse JSON/XML,
verify fragment composition and named-value coverage, inspect backend routing
and breaker markers, enforce the telemetry allowlist, and reject live endpoints
or secret-like literals. They do not evaluate APIM policy expressions or prove
service-side schema acceptance. Policy toolkit validation in an authorized
non-production APIM instance remains a separate cloud gate.

## Publish and restore workflow

`../scripts/Publish-APIM.ps1` accepts an explicit tenant, subscription, resource
group, service, API ID/path, every non-secret named value, backend URL/name/
weight/priority, and approved SHA-256 digests for the OpenAPI document, policy,
fragment directory, named-value contract, and backend Bicep. It fails before any
Azure call when an input or digest is missing.

Dry run is the default. With `-Apply`, the script verifies the current Azure CLI
identity, snapshots the Riverside-managed API, named values, blue/green/pool
backends, fragments, and API policy, runs provider what-if for backend Bicep,
then publishes in dependency order: API, named values, backends, fragments, and
API policy. Existing absence is part of the snapshot.

`../scripts/Restore-APIM.ps1` requires the independently retained snapshot
SHA-256, verifies target binding, restores prior resources in dependency order,
and deletes only Riverside-managed objects recorded as absent before publication.
Both scripts reject secret named values and use the current `az login`, workload
identity, or managed identity rather than credentials in files.

See `../scripts/README.md` for the input contract. APIM policy expression
acceptance, ARM snapshot round-trip fidelity, live identity, routing, deadlines,
retry behavior, and rollback timing remain unvalidated because neither script
was executed while authored.
