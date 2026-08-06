# Riverside Identity and Isolation Threat Model

**Status:** `[Local-static]` threat analysis with `[External validation required]`
for deployed likelihood, control effectiveness, and legal/privacy conclusions.

## Scope and assets

Protected assets are unpublished manuscript content, rights records, title
assignments, identity/entitlement data, workflow authority, audit evidence, and
the integrity of tenant/region/purpose decisions. Trust boundaries are client to
gateway, gateway to orchestrator, orchestrator to index/model/tools, components
to telemetry/audit, and CI/CD to cloud.

## Threat ledger

| ID | Threat and abuse path | Impact | Preventive control | Detective/response control | Residual risk / external proof |
|---|---|---|---|---|---|
| `THR-RIV-001` | Caller supplies another tenant or broader role in request filters | Cross-tenant disclosure | Derive tenant/roles from validated identity; reject escalation | Negative token/retrieval tests; audit denial | Deployed gateway claim mapping and bypass testing |
| `THR-RIV-002` | Disabled contractor remains in stale nested group | Former-user access; `INC-RIV-003` | Enabled-status check; full reconciliation; fail closed on stale state | Entitlement-age alert; tenant-route disable | IdP webhook/delta/full-sync behavior and accepted freshness bound |
| `THR-RIV-003` | Model or retrieved text asks a tool to ignore policy | Unauthorized side effect | Deterministic tool policy outside model; exact payload schema | Deny audit; injection test set | All tool paths route through one enforceable gateway |
| `THR-RIV-004` | Human approval is reused after payload changes | TOCTOU workflow mutation | Approval hash binds tool, target, payload, and state version | Hash mismatch alert; re-approval | Live PageTurn state and retry behavior |
| `THR-RIV-005` | Backend ignores or partially applies retrieval filter | Restricted content returned | Mandatory server filter plus post-retrieval authorization | Forbidden-document canary; zero-tolerance gate | Deployed index query semantics and cache partitioning |
| `THR-RIV-006` | Region metadata is mistaken for residency enforcement | Unapproved processing or backup location | Region policy gate; closed environment config | Resource/config inventory and data-flow review | Service/SKU/feature, backup, diagnostic, support, subprocessor evidence |
| `THR-RIV-007` | Prompt/content/identity leaks through telemetry or errors | Confidentiality/privacy breach | Allowlist metrics; body logging off; safe error envelope | Sample logs/traces under success/failure/load | Diagnostic settings, exports, retention, support access |
| `THR-RIV-008` | Service identity has broad standing access | Large blast radius after compromise | Per-service identity, narrow scope, no caller-token forwarding | Access reviews; negative data-plane tests | Exact deployed assignments and credential lifecycle |
| `THR-RIV-009` | Policy engine unavailable and tool path fails open | Controls disappear during outage | Fail closed; only explicitly approved cached low-risk reads | Policy-health alert; blocked-call audit | Outage drill and break-glass governance |
| `THR-RIV-010` | Cache key omits authorization dimensions | Cross-user or cross-tenant response reuse | Partition by trusted tenant, actor, release, and authorization scope | Cache leakage scenarios; disable cache on uncertainty | Deployed cache implementation and invalidation behavior |

## Stop-ship conditions

- Any successful forbidden tenant, title, role, region, or purpose access.
- Any disabled identity accepted because a stale group says otherwise.
- Any autonomous publication, rights, payment, or unconfirmed workflow action.
- Any prompt, completion, manuscript, credential, or bearer token in telemetry.
- Any unapproved cross-region content path or silent fail-open authorization path.

## Review record

| Field | Value |
|---|---|
| Fixture/source | `RIV-FDE-1.0.0`; synthetic local scenarios |
| Local review status | Design complete; notebook execution not performed |
| Security approver | `PER-RIV-004` for case policy intent; actual reviewer required externally |
| Legal/privacy approver | `PER-RIV-003` for rights/legal scope; actual privacy authority required externally |
| Revalidation triggers | Identity, role, tool, index, cache, network, region, retention, model, or processor change |
