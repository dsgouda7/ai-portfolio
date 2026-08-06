# ADR-0007: Databricks Direct Vector Index

- **Status:** Accepted
- **Date:** 2026-08-05

## Context

The remote data project now implements governed parsed-document to chunk/vector
processing, durable Delta contract tables, deletion propagation, and a Databricks
Direct Vector Access adapter. The adapter pins embedding/index versions and pushes
tenant, region, classification, deletion, and ACL-scope filters into AI Search,
then applies a second fail-closed authorization check in application code.

The local Chroma path lacks the complete production security/lineage boundary.
Azure AI Search remains a possible future adapter, but no equivalent source asset
or comparison evidence currently selects it.

## Decision

Use the Databricks Direct Vector Access implementation in
`projects/rag-knowledge-pipeline/phase2-vectorize/src/remote/` as the initial
production index target. Keep Riverside orchestration dependent on the v1
vector-index record and retrieval protocol rather than importing the data-pipeline
implementation or exposing a vendor SDK at the application boundary.

Index schema, embedding model/revision/dimensions, chunk strategy/version, and
logical index version are immutable release inputs. Publish a new index version
for incompatible changes and retain the prior target for rollback.

## Consequences

- The Databricks workspace, Unity Catalog, embedding endpoint, AI Search endpoint,
  index, identity, filters, deletion, latency, quota, region, and cost become live
  validation dependencies.
- Backend filter syntax differs by endpoint type and must be tested in the selected
  service, not inferred from cloud-free adapter tests.
- Azure AI Search requires a new adapter and ADR update or superseding decision.
- Chroma remains a local prototype backend only.

## Evidence state

Remote source, Databricks Asset Bundle definitions, fixed-schema table bootstrap,
quality reports, authorization/deletion logic, and cloud-free tests are implemented.
They were not executed in this task. No authorized Databricks workspace run, index
query, cross-tenant negative test, tombstone/deletion observation, load result,
residency review, or billing evidence is linked.
