# Riverside Platform and Data-Plane Mapping

## Document control

| Field | Value |
|---|---|
| Artifact ID | `MAP-CAP-01` |
| Version / status | `[TODO] / DRAFT` |
| Integration owner / reviewers | `[TODO]` |
| Requirements and architecture version | `[TODO]` |
| Scope / exclusions / revalidation trigger | `[TODO]` |

## Requirement-to-asset map

| Requirement/control ID | Project asset/contract/ADR | Source status | Evidence status | Gap or mismatch | Validation owner | Disposition |
|---|---|---|---|---|---|---|
| `[AC/SEC/DATA/EVAL/etc.]` | `[path and version]` | `[implemented source / absent / deferred]` | `[static pending/completed / live required]` | `[TODO]` | `[TODO]` | `[reuse / adapt / build / defer / remove]` |

Integrate through the v1 contracts. Do not import the RAG pipeline implementation
into serving or rewrite training manifests into release manifests.

## Mandatory project-gap dispositions

| Gap ID | Documented fact | Affected requirement/gate | Risk if ignored | Required evidence or change | Owner | Disposition/timing |
|---|---|---|---|---|---|---|
| `MAP-GAP-001` | Root README/package/environment/integration surfaces are missing | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[blocker/pre-canary/post-canary/accepted/out]` |
| `MAP-GAP-002` | Bicep and Azure ML endpoint names differ | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` |
| `MAP-GAP-003` | `uksouth` parameter and `eastus2` deployment metadata conflict | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` |
| `MAP-GAP-004` | 120-second contract and 180-second deployment timeout conflict | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` |
| `MAP-GAP-005` | Azure ML model/environment registration/package workflow is absent | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` |
| `MAP-GAP-006` | Deployable RAG-orchestrator host and `azd` service entries are absent | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` |
| `MAP-GAP-007` | APIM publish/restore pipeline and cloud smoke suite are absent | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` |
| `MAP-GAP-008` | Load-result download and composed release-gate job are absent | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` |
| `MAP-GAP-009` | Databricks identity/filter/deletion/load/region/cost behavior is unvalidated | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` |
| `MAP-GAP-010` | Atomic index rollback and shadow mirroring are not implemented/proven | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` |
| `MAP-GAP-011` | Single-region design has no regional failover/RTO/RPO evidence | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` |
| `MAP-GAP-012` | No retained static or live Azure/Databricks evidence is linked | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` |

## Contract and lineage mapping

| Release/domain | Producer | Contract | Immutable IDs/digests | Consumer | Compatibility check | Rollback target | Missing evidence |
|---|---|---|---|---|---|---|---|
| Training/model | `[TODO]` | `model-release-manifest` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` |
| Data/chunks/index | `[TODO]` | `raw/parsed/chunk/vector v1` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` |
| Prompt/application | `[TODO]` | `app request/response/error/citation v1` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` |
| Evaluation/release | `[TODO]` | `evaluation-release-report v1` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` |
| Telemetry/deployment | `[TODO]` | `telemetry/deployment metadata v1` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` |

## Promise-versus-evidence review

| Proposed customer wording | Project ledger status | Narrowest supported wording | Evidence needed to strengthen it | Owner |
|---|---|---|---|---|
| `[TODO]` | `[implemented source / modeled / live required / deferred]` | `[TODO]` | `[TODO]` | `[TODO]` |

Mapping verdict and launch blockers: `[TODO]`
