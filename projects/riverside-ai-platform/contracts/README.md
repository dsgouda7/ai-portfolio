# Riverside AI Platform Contracts

This directory defines the versioned boundaries shared by model promotion,
document ingestion, vector indexing, the application API, evaluation, and
telemetry. Version `v1` uses JSON Schema Draft 2020-12. Contract objects are
closed with `additionalProperties: false` unless a field is explicitly designed
as bounded metadata.

## Contract catalog

| Schema | Boundary |
|---|---|
| `v1/model-release-manifest.schema.json` | Immutable serving artifact plus training provenance and promotion evidence |
| `v1/raw-document.schema.json` | Governed reference to source bytes before parsing |
| `v1/parsed-document.schema.json` | Parsed text with parser provenance and source lineage |
| `v1/document-chunk.schema.json` | Versioned chunk, offsets, embedding target, and parent lineage |
| `v1/vector-index-record.schema.json` | Index-ready content, vector, security filters, and index version |
| `v1/app-chat-completion-request.schema.json` | OpenAI-compatible request with bounded generation and retrieval controls |
| `v1/app-chat-completion-response.schema.json` | Non-streaming OpenAI-compatible response plus citations, usage, trace, and deployment metadata |
| `v1/app-chat-completion-stream-event.schema.json` | Streaming chunk shape for server-sent events |
| `v1/app-error.schema.json` | Normalized application error envelope and retry guidance |
| `v1/citation.schema.json` | Content-free citation lineage and answer spans |
| `v1/deployment-metadata.schema.json` | Release, runtime, slot, region, index, and source commit returned by serving |
| `v1/evaluation-release-report.schema.json` | Machine-readable release evidence across all required gate domains |
| `v1/telemetry-attributes.schema.json` | Allowlisted bounded-cardinality OpenTelemetry metric attributes |
| `v1/common.schema.json` | Shared identifiers, digests, timestamps, ACLs, deletion state, and bounded metadata |

`../config/schema.json` is the initial environment-profile contract. It permits
only `dev`, `staging`, and `production`; the production project has no local
deployment profile. Secret values are not modeled. Variable configuration may
use `${UPPER_CASE_ENVIRONMENT_VARIABLE}` references, while authentication is
limited to managed identity or workload identity.

## Fine-tuning manifest mapping

The committed `checkpoints/*/experiment-manifest.json` files remain training
provenance. They are not release manifests and must not be rewritten to satisfy
the serving contract.

| Existing training field or artifact | Release contract destination | Rule |
|---|---|---|
| `model.id` | `base_model.id` | Copy exactly. |
| `model.revision` | `base_model.revision` and usually `tokenizer.revision` | Keep the pinned revision; do not replace it with `main` or `latest`. |
| `stage` | `adapter.stage` | Copy the training stage name. |
| `extra.parameter_strategy` and `adapter_config.json.peft_type` | `adapter.type` | Normalize `LORA` to `lora`; the release builder must reject disagreement. |
| `adapter_model.safetensors` | `adapter.uri` and `adapter.digest` | Register an immutable URI and compute SHA-256 outside the manifest. |
| `tokenizer.json` plus tokenizer configuration | `tokenizer.uri`, `tokenizer.revision`, and `tokenizer.digest` | Digest the exact deployed tokenizer bundle. |
| Entire `experiment-manifest.json` | `training_provenance.manifest_uri` and `manifest_digest` | Reference and digest the unchanged source manifest. |
| `extra.heldout_*`, `evaluation_files`, and later evaluator output | `evaluation-release-report.schema.json` metrics and evidence | Training evidence becomes one input to release evaluation, not an automatic promotion decision. |
| `packages` and `training_arguments` | Remain in the source manifest | Do not duplicate broad training internals into the serving release. |

The current manifests identify the SmolLM2 base revision, training stage, seed,
package versions, input file hashes, and training arguments. They do not provide
adapter, tokenizer, or manifest digests; immutable artifact URIs; a serving
runtime compatibility claim; or a complete release decision. Those values must
be produced by an artifact registration and evaluation workflow.

## RAG knowledge pipeline mapping

The existing local pipeline is a prototype adapter source, not a producer of
the complete production records.

| Existing RAG value | v1 destination | Required adapter behavior |
|---|---|---|
| Wikipedia row `id` | `document_id` | Namespace or otherwise stabilize it across sources and tenants. |
| Wikipedia row `title` | `parsed-document.title` | Preserve after parser normalization. |
| Wikipedia row `text` | `parsed-document.text` | Hash the normalized text into `content_hash`. |
| LangChain `Document.page_content` | `document-chunk.content` and `vector-index-record.content` | Preserve exact indexed text and compute its SHA-256. |
| LangChain metadata `id` | `document_id` and `parent_document_id` | Do not use it as the chunk ID; create a deterministic chunk ID. |
| LangChain metadata `title` | Citation `title` | Return display metadata without copying document text into the citation. |
| `local.embedding_model` or `remote.embedding_model` | `embedding.model_id` | Pin and record `model_revision` and actual vector dimensions as additional production inputs. |
| `chunk_size` and `chunk_overlap` | `chunking.strategy` and `chunking.version` | Version the complete strategy; numeric settings alone do not identify behavior. |
| Chroma collection | `index_name` and `index_version` | Assign an immutable logical index version independent of a mutable backend collection. |
| Query `question` | One `messages` item with role `user` | The application boundary is chat-completion compatible. |
| Query `top_k` and `temperature` | `retrieval.top_k` and `temperature` | Existing bounds remain compatible; v1 caps `top_k` at 20. |
| Response `answer` | `choices[0].message.content` | Normalize to the non-streaming response shape. |
| Response `sources_count` | No direct field | Replace the configured count with actual `citations`; `citations.length` is the supported-source count. |

The production ingestion adapter must additionally supply `tenant_id`, stable
source URI/version, raw and normalized content hashes, ACL, region,
classification, ingest timestamps, pipeline version, and deletion state. The
current Chroma path retains only `id` and `title`, so it cannot claim v1
compatibility until it preserves the required lineage and authorization fields.

## API and security decisions

- `model` is a stable alias. Clients do not select physical deployments or
  mutable model versions.
- Input is bounded by message count, message character length, and
  `max_input_tokens`; the gateway must still tokenize and enforce the actual
  aggregate input limit.
- Non-streaming and streaming responses are separate schemas so clients can
  validate each wire shape without ambiguous unions.
- Citations carry identifiers, hashes, versions, titles, and answer spans. They
  intentionally exclude source excerpts and document text.
- Tenant identity and authorization context come from authenticated request
  context, not client-provided retrieval filters.
- Errors expose normalized categories and safe messages. Backend exception
  text, credentials, and internal resource identifiers are not part of the
  contract.

## Telemetry boundary

`telemetry-attributes.schema.json` is an allowlist for metric attributes. It
permits service, environment, release, stable model alias, deployment, region,
route, outcome, cache result, token buckets, tenant tier, error category, and a
bounded retrieval bucket. It rejects all other labels by default.

Prompts, completions, document text, user IDs, request IDs, tenant IDs, source
URIs, and chunk/document IDs must never be metric labels. Trace IDs belong in
trace context and API correlation metadata, not metric attributes. Release and
deployment values are expected to come from controlled registries; operators
must monitor their cardinality when retention spans many releases.

## Validation and runtime invariants

A validator must load all `v1/*.schema.json` resources into one registry using
their `$id` values, enable Draft 2020-12 format assertions, and then validate the
fixture paired with each root schema. Fixture expectations are documented in
`../tests/fixtures/README.md`.

The following invariants require application checks because portable JSON
Schema does not compare sibling values or array lengths:

- `document_id` equals `parent_document_id` on chunk and vector records.
- `offsets.end` is greater than `offsets.start` and the span matches `content`.
- `vector.length` equals `embedding.dimensions`, and every value is finite.
- `usage.total_tokens` equals prompt plus completion tokens.
- Citation and answer spans are ordered, non-overlapping, and within their
  target strings.
- Metric threshold units equal observed units; uncertainty lower/upper bounds
  are ordered; recorded status agrees with the threshold operator.
- A promoted release report has no failed domain metric, and the release
  manifest decision and digest match that report.
- The release `model_profile` and `precision` are present in the serving
  runtime compatibility arrays.

## Versioning

Additive optional fields may be introduced in a new minor contract release only
after all consumers tolerate them. Because root objects are closed, adding a
field to an existing `v1` schema is otherwise breaking. Renames, removals,
meaning changes, new required fields, or relaxed security boundaries require a
new version directory and explicit migration. Immutable data and release
records retain the contract version under which they were written.

No schema, fixture, or URI in this directory contains a credential or a real
customer identifier.
