# Contract Fixture Matrix

Fixtures are paired with JSON Schema roots. Files under `valid/` are expected to
pass. Files under `invalid/` are expected to fail for the named primary reason.
The invalid fixtures are test inputs, not examples to copy into configuration or
data stores.

| Schema | Valid fixture | Invalid fixture | Primary invalid boundary |
|---|---|---|---|
| `contracts/v1/model-release-manifest.schema.json` | `valid/model-release-manifest.json` | `invalid/model-release-manifest.mutable-version.json` | `version` is the mutable label `latest`, not semantic versioning. |
| `contracts/v1/raw-document.schema.json` | `valid/raw-document.json` | `invalid/raw-document.missing-tenant-id.json` | Required tenant lineage is absent. |
| `contracts/v1/parsed-document.schema.json` | `valid/parsed-document.json` | `invalid/parsed-document.incomplete-deletion-state.json` | Deleted records require both request and deletion timestamps. |
| `contracts/v1/document-chunk.schema.json` | `valid/document-chunk.json` | `invalid/document-chunk.missing-parent-document-id.json` | Parent document lineage is absent. |
| `contracts/v1/vector-index-record.schema.json` | `valid/vector-index-record.json` | `invalid/vector-index-record.empty-vector.json` | An index record cannot contain an empty vector. |
| `contracts/v1/citation.schema.json` | `valid/citation.json` | `invalid/citation.embeds-source-text.json` | Closed citation metadata rejects copied source text. |
| `contracts/v1/deployment-metadata.schema.json` | `valid/deployment-metadata.json` | `invalid/deployment-metadata.local-environment.json` | The production project has no local deployment profile. |
| `contracts/v1/app-chat-completion-request.schema.json` | `valid/app-chat-completion-request.json` | `invalid/app-chat-completion-request.output-limit.json` | Requested output exceeds 2,048 tokens. |
| `contracts/v1/app-chat-completion-response.schema.json` | `valid/app-chat-completion-response.json` | `invalid/app-chat-completion-response.missing-citations.json` | Normalized responses always include the citations array, even when empty. |
| `contracts/v1/app-chat-completion-stream-event.schema.json` | `valid/app-chat-completion-stream-event.json` | `invalid/app-chat-completion-stream-event.wrong-object.json` | A stream event must identify itself as `chat.completion.chunk`. |
| `contracts/v1/app-error.schema.json` | `valid/app-error.json` | `invalid/app-error.overload-not-retryable.json` | Overload errors must be marked retryable and carry bounded retry guidance. |
| `contracts/v1/evaluation-release-report.schema.json` | `valid/evaluation-release-report.json` | `invalid/evaluation-release-report.missing-rollout-domain.json` | All eight release evidence domains are required. |
| `contracts/v1/telemetry-attributes.schema.json` | `valid/telemetry-attributes.json` | `invalid/telemetry-attributes.request-id-label.json` | Request IDs are forbidden metric labels. |
| `config/schema.json` | `valid/config.json` | `invalid/config.api-key-authentication.json` | Static API-key authentication is unsupported; use managed/workload identity. |

No fixture has been executed or schema-validated as part of its creation. A
future contract test runner should resolve schema `$id` values from a local
registry, enable format checking, assert every valid pair succeeds, and assert
every invalid pair fails with the expected keyword or required property.
