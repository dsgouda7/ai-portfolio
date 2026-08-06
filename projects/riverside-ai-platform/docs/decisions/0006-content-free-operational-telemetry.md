# ADR-0006: Content-Free Operational Telemetry

- **Status:** Accepted
- **Date:** 2026-08-05

## Context

Prompts, completions, manuscript text, source URIs, and document or tenant
identifiers can contain customer information. High-cardinality identifiers also
make metrics expensive and difficult to operate.

## Decision

Operational metrics use only the allowlist in
[`../../contracts/v1/telemetry-attributes.schema.json`](../../contracts/v1/telemetry-attributes.schema.json).
Do not place prompts, completions, document text, source URIs, user IDs, request
IDs, tenant IDs, document IDs, or chunk IDs in metric labels. Keep trace IDs in
trace context and the bounded API correlation envelope, not metric dimensions.

Application logs are structured, redacted, and deny content by default. Any future
content capture needs a separate threat model, purpose, consent/legal basis, access
model, retention, deletion path, and ADR.

## Consequences

- Operators diagnose content-specific failures through controlled evaluation and
  approved identifiers outside metric labels.
- Cardinality monitoring is still required for release and deployment values.
- Instrumentation and exporter configuration must be tested for accidental body,
  header, exception, or environment-variable capture.

## Evidence state

The telemetry allowlist, conventions, privacy sanitizer, instrumentation source,
unit tests, APIM bounded token metric, and diagnostic-setting Bicep are implemented
source assets. They were not executed in this task. No exporter result, deployed
data collection path, under-load redaction sample, cardinality result, or live
telemetry review is linked.
