# Identity, Threat, Residency, and Controls Package

## Document control

| Field | Value |
|---|---|
| Artifact IDs | `SEC-CAP-01` to `SEC-CAP-04` |
| Version / status | `[TODO] / DRAFT` |
| Security/identity/data/privacy owners and reviewers | `[TODO]` |
| Architecture/data inputs | `[TODO]` |
| Scope / exclusions / revalidation trigger | `[TODO]` |

## Identity flow and trusted context

Trace `tenant_id`, `actor_id`, `role_ids`, `region_id`, `purpose`, `title_ids`,
and `trace_id` through every boundary.

| Boundary | Authentication source | Trusted context derivation | Authorization decision | Fail-closed condition | Audit evidence | Owner |
|---|---|---|---|---|---|---|
| Client/gateway | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` |
| Retrieval | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` |
| Model request | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` |
| Tool/action | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` |
| Response/telemetry | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` |

## RBAC and action matrix

| Role/service identity | Resource/action | Tenant/title/purpose scope | Allow condition | Explicit deny | Approval authority | Evidence |
|---|---|---|---|---|---|---|
| `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` |

## Threat model

| Threat ID | Asset and trust boundary | Abuse/failure path | Preconditions | Preventive/detective/response controls | Residual risk | Owner | Validation |
|---|---|---|---|---|---|---|---|
| `THR-CAP-*` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` |

Include cross-tenant retrieval, stale contractor access, role escalation, missing
purpose, title mismatch, regional misrouting, prompt injection through retrieved
content, prohibited tool use, content-bearing telemetry, deletion failure, and
supply-chain/release mismatch.

## Data-flow and residency map

| Data category | Source | Processing/storage/diagnostic path | Tenant/region rule | Backup/replication/support path | Decision owner | Evidence state | Open validation |
|---|---|---|---|---|---|---|---|
| Customer content | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[Modeled / External validation required]` | `[TODO]` |
| Identity/security metadata | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` |
| Model/evaluation artifacts | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` |
| Operational/incident data | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` |

A `region` field is policy metadata, not enforcement. Resource locations alone do
not prove residency. Record the current `uksouth`/`eastus2` project mismatch and
the owner/evidence required to resolve it.

## Controls and negative-test matrix

| Control ID | Objective | Implementation/design | Test case and expected deny/allow | Static/local evidence | Live/customer evidence required | Owner/status |
|---|---|---|---|---|---|---|
| `CTRL-CAP-*` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` |

Any successful forbidden access is a stop-ship result. A local expected denial is
not a deployed-control attestation.

## Security and residency gate

- [ ] Trusted context is derived server-side and survives every boundary.
- [ ] Cross-tenant, stale identity, role, region, purpose, title, and prohibited-tool
      cases fail closed in the evidence plan.
- [ ] Telemetry is useful without prohibited content or high-cardinality identity.
- [ ] Every legal, compliance, privacy, residency, cloud, and networking conclusion
      has an authorized external owner.
- [ ] Residual risks have exposure limits, triggers, and response owners.

Decision and conditions: `[TODO]`
