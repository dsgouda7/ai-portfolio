# Riverside Controls Matrix

**Status:** Source and design status only. `Implemented in source` does not mean
deployed, effective, least privilege, or compliant.

| Control ID | Objective | Control and boundary | Type | Local evidence | External evidence | Owner | Status |
|---|---|---|---|---|---|---|---|
| `CTL-RIV-001` | Trust actor and tenant claims | Gateway validates token and overwrites client authority headers | Preventive | APIM source mapping; local normalization design | Positive/negative deployed token tests | Security | Implemented in source; unproven live |
| `CTL-RIV-002` | Prevent role escalation | Effective roles come from active identity; requested roles only narrow | Preventive | `ISO-RIV-004` expected denial | IdP group/role mapping and bypass tests | Identity + Security | Modeled; not run |
| `CTL-RIV-003` | Block stale identities | Disabled status overrides nested-group membership | Preventive | `ISO-RIV-003`; `FACT-RIV-038` | Revocation, webhook, delta, full reconciliation, freshness test | Identity | Modeled; not run |
| `CTL-RIV-004` | Enforce tenant/title isolation | Server-derived filters plus post-retrieval check | Preventive | `ISO-RIV-002`, `ISO-RIV-007`; production adapter source | Deployed index and cache negative tests | Data + Security | Source/design; unproven live |
| `CTL-RIV-005` | Enforce purpose limitation | Gateway requires approved purpose independent of identity token | Preventive | `ISO-RIV-006`; `SEC-RIV-002` | Customer policy approval and deployed route tests | Security + Legal/Privacy | Modeled; not run |
| `CTL-RIV-006` | Enforce regional policy | Tenant allowed-region check at admission and retrieval | Preventive | `ISO-RIV-005`; `FACT-RIV-015/016` | Resource, processing, backup, diagnostics, support path evidence | Cloud + Legal/Privacy | Policy modeled; residency unproven |
| `CTL-RIV-007` | Bound tool authority | Concrete tool/action/payload policy; prohibited actions cannot be approved | Preventive | Rights-write expected denial; `FACT-RIV-018` | All deployed tool paths and service-identity scopes | Tool owner + Security | Modeled; not run |
| `CTL-RIV-008` | Bind human approval | Approval covers exact payload/state and business idempotency key | Preventive | Contract/design review | PageTurn integration replay and reconciliation | Applications owner | External validation required |
| `CTL-RIV-009` | Preserve decision evidence | Allow and deny branches write content-free audit event | Detective | Local audit schema in notebook | Destination RBAC, immutability, retention, legal hold, sampled events | Security + Legal | Modeled; not run |
| `CTL-RIV-010` | Prevent telemetry disclosure | Closed metric allowlist; no identity/content dimensions | Preventive/detective | Telemetry contract/source review | Sample metrics/logs/traces under load and errors | Operations + Security | Source/design; unproven live |
| `CTL-RIV-011` | Contain forbidden access | Disable affected tenant route; preserve evidence; notify owners | Responsive | `INC-RIV-003` tabletop mapping | Authorized staging containment/re-enable drill | Security | Procedure defined; not drilled here |
| `CTL-RIV-012` | Avoid secret exposure | Managed/workload identity; no committed keys or caller-token forwarding | Preventive | Config/contract/source review; directory scan | Runtime identity, Key Vault if needed, rotation/access tests | Cloud Security | Source/design; unproven live |
| `CTL-RIV-013` | Enforce deletion state | Deleted/pending records cannot be retrieved | Preventive | Record-state check in local policy | Source-to-index-cache-backup deletion rehearsal | Data + Legal/Privacy | Local mechanism only |
| `CTL-RIV-014` | Govern changes | Reviewed policy/IaC, immutable versions, negative regression gate | Preventive/detective | Versioned fixture/contracts | CI/CD workload identity, approvals, provenance, deployed regression | Platform owner | Partial source evidence |

## Decision rule

A control is ready for a customer assurance only when its policy intent is approved,
its implementation evidence is reviewed, its positive and negative behavior is
measured in the target environment, its operating owner accepts monitoring and
response, and its revalidation trigger is recorded. No local notebook result can
complete that chain by itself.
