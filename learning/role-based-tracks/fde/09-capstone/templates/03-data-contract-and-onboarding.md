# Data Contract and Onboarding Plan

## Document control

| Field | Value |
|---|---|
| Artifact IDs | `DATA-CAP-01` to `DATA-CAP-04` |
| Version / status | `[TODO] / DRAFT` |
| Data owner / technical owner / reviewers | `[TODO]` |
| Architecture and source inputs | `[TODO]` |
| Scope / exclusions / revalidation trigger | `[TODO]` |

## Source inventory and purpose

| Source ID | Owner/authority | Approved purpose | Format/volume/freshness class | Classification/tenant/region | ACL model | Retention/deletion | Evidence class | Unknown/blocker |
|---|---|---|---|---|---|---|---|---|---|
| `SRC-*` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` |

## Source-to-contract mapping

| Source/record type | Stable identity/version | Required v1 fields | Mapping/coercion rules | Reject/quarantine conditions | Lineage retained | Validation owner |
|---|---|---|---|---|---|---|
| `[PDF/text/ERP/API]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` |

Explicitly cover superseded policy, layout-aware PDF parsing, duplicate pages,
low-confidence OCR, manuscript title scope, tombstones, ERP null territory,
pagination, schema drift, ambiguous committed writes, and disabled identities.

## Quality and quarantine gates

| Gate ID | Source/slice | Check and threshold/decision rule | Evidence class | Failure disposition | Owner | Downstream exposure blocked |
|---|---|---|---|---|---|---|
| `DQ-CAP-*` | `[TODO]` | `[TODO]` | `[TODO]` | `[reject/quarantine/conditional]` | `[TODO]` | `[TODO]` |

Passing schema checks does not establish source completeness, ownership,
authorization, retrieval relevance, answer quality, or deletion completion.

## ACL, sync, lineage, and deletion contract

| Concern | Contract | Negative/failure test | Evidence retained | Recovery/reconciliation | Owner | External validation |
|---|---|---|---|---|---|---|
| ACL projection and freshness | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` |
| Incremental cursor/watermark overlap | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` |
| Version/supersession/deduplication | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` |
| Tombstone and index deletion | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` |
| Schema/embedding/index migration | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` |

## Retrieval-readiness verdict

| Source ID | Verdict | Supporting gates | Blocking gaps | Allowed use | Prohibited use | Revalidate on |
|---|---|---|---|---|---|---|
| `SRC-*` | `[READY / CONDITIONAL / BLOCKED / EXCLUDED]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` |

Overall verdict: `[TODO]`

The overall decision is no stronger than the weakest source required by the
selected use case. Do not average a blocked security or lifecycle source into a
passing corpus score.
