# Databricks Index Update Assets

This bundle defines a two-task job. The first task creates fixed-schema Delta
tables for `document_chunk` v1 records, `vector_index_record` v1 records, and
durable quality reports. The second task reads `parsed_document` v1 rows,
chunks active text, calls the pinned embedding endpoint, updates a Direct Vector
Access index, and explicitly deletes stale or tombstoned record IDs.

The index schema is immutable. Change `index_version` and deploy a new index when
the backend schema, embedding revision, dimensions, or security metadata shape
changes. Route consumers to the new logical version only after its quality and
authorization evidence passes; deleting the old index is a separate operation.

Production jobs run as the configured service principal. No PAT, client secret,
or workspace token belongs in bundle variables or source control. Grant the job
identity only the required Unity Catalog table permissions, model-serving invoke
permission, and AI Search endpoint/index permissions.

`since` is an optional ISO-8601 watermark. Scheduled production runs should use
a durable orchestration watermark and overlap the prior window so retries are
safe. Contract merges, vector upserts, and primary-key deletes are idempotent.

These assets are statically authored. They do not prove workspace RBAC, OAuth,
regional model availability, endpoint capacity, private networking, Delta merge
behavior, AI Search filter semantics, latency, cost, or deletion completion in a
real Azure Databricks workspace. Those claims require authorized cloud tests.
