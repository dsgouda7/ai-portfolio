# Dependency Failure and Degraded-Mode Policy

## Status and scope

This is the draft operating policy for Riverside dependency failures. It records
the behavior implemented in source and the only operator responses allowed until
staging exercises approve additional modes. It is not evidence that a failure
exercise, fallback, circuit breaker, or recovery path has run.

Authorization, tenant/ACL filters, deletion filters, classification, region,
citations, output bounds, and telemetry redaction never become optional. There is
no approved answer-without-retrieval mode, alternate index, alternate model, or
cross-region failover. An operator must not invent one during an incident.

## Shared decision rules

1. Identify the failing boundary from correlated, content-free evidence; do not
   infer health from missing telemetry.
2. Freeze releases, index promotion, policy changes, and traffic increases.
3. Keep retries inside the committed APIM and endpoint-client budgets. Never add
   retries at another layer or replay a streaming request.
4. Fail closed when authorization context, authorized retrieval, release binding,
   or required readiness cannot be established.
5. Shed load with normalized `overloaded`, `timeout`, `backend_failure`, or
   `release_unavailable` responses rather than bypassing controls.
6. Use only a pre-approved, tested mitigation with a named owner, expiry, rollback
   target, and observation window. Otherwise restore the known-good state or keep
   the affected route unavailable.

## Boundary policy

| Boundary | Implemented behavior | Allowed degraded response | Prohibited response | Recovery evidence |
|---|---|---|---|---|
| APIM | Entra validation, request/token limits, bounded retries for non-streaming `429`, `502`, `503`, and `504`, backend circuit metadata, safe errors | Stop candidate routing; reject excess load with bounded `Retry-After`; route only to a pre-proven compatible known-good backend | Bypass token/tenant policy, unbounded retry, streaming replay, portal-only policy edit, or routing to an unverified backend | Policy/version export, backend/attempt counts, valid and invalid token checks, normalized errors, stable observation window |
| Orchestrator | Not-ready returns `release_unavailable`; retrieval or generation exceptions fail the request; empty authorized retrieval returns a refusal | Continue liveness for diagnosis; return safe per-request errors or an evidence-backed refusal only when retrieval completed successfully with no authorized chunks | Generate an answer after retrieval failure, trust client identity headers, disable ACL/deletion filters, or treat `/health` as readiness | `/ready`, contract smoke, cross-tenant/forbidden-document negatives, citation/refusal checks, release/index binding |
| Databricks retrieval and embedding | Managed-identity calls use bounded timeout; query filters include tenant, ACL scope, region, classification, active deletion state, and index version; exceptions fail the request | Pause index publication and return safe backend failure while the approved immutable index is restored | Unfiltered search, stale/alternate index substitution, removing deletion/region/classification filters, or answer-without-retrieval | Exact index/version and endpoint state, filter negatives, deletion check, representative retrieval/citation result, latency and error recovery |
| Azure ML | Managed-identity client retries bounded transient statuses and normalizes final failure; release readiness is separate from process liveness | Set candidate traffic to zero; serve only a named compatible known-good slot proven ready | Increase retries past profile bounds, send traffic to a failed/unverified slot, use mutable artifacts, or expose backend error text | Traffic export, artifact/deployment digests, readiness/warm-up, stream/non-stream smoke, latency/error stability |
| Telemetry | Required configuration is validated at runtime composition; OTLP export is asynchronous and content dimensions are denied | Keep serving only when independent health evidence exists and incident command explicitly accepts reduced observability for a bounded window | Treat missing signals as health, enable content capture, weaken redaction, or delete evidence to reduce cost | Exporter health, ingestion delay, alert path, bounded-cardinality/redaction sample, gap annotation and backfill decision |

## Degraded-mode authorization record

Before enabling any mode other than normal service or fail-closed rejection, record:

- incident/change ID, UTC start and expiry, environment, region, and affected tier;
- failed boundary and evidence, customer impact, and data/security assessment;
- exact mitigation, source/config/policy versions, and known-good rollback target;
- controls that remain enforced and tests proving each one;
- capacity, cost, and error-budget impact;
- authorizer, independent reviewer, observation window, and abort triggers.

An expired mode is removed or explicitly re-authorized. Recovery requires the
boundary-specific evidence above plus the validation in
[incident response](incident-response.md). Retain the completed record; this
template alone is not evidence.
