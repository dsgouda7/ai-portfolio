# Support and Escalation Matrix Template

> Example targets below come from frozen support model `SUP-RIV-001`. They are case inputs, not proof that a rota, paging path, notification process, or contract is operational.

## Support boundary

| Field | Riverside example/input | Accepted value/evidence |
|---|---|---|
| Support model ID | `SUP-RIV-001` | `<reference>` |
| Covered hours | 08:00-18:00 Europe/London, Monday-Friday excluding agreed holidays | `<Customer-validated decision>` |
| Hypercare | 30 days | `<start/end and exit rule>` |
| Out-of-hours coverage | Unresolved: `UNK-RIV-006` | `<funded decision or explicit fail-closed/queue behavior>` |
| Primary intake | `<not supplied>` | `<channel and owner>` |
| Status communications | `<not supplied>` | `<channel, audience, approver>` |
| Vendor escalation | `<not supplied>` | `<contracts, contacts, evidence required>` |
| Post-hypercare quality owner | Unresolved: `UNK-RIV-010` | `<accepted owner>` |

## Severity and response

| Severity | Frozen definition | Acknowledgement target | Customer update target | Incident commander | Outside-hours behavior | Notification authority |
|---|---|---:|---:|---|---|---|
| `SEV-1` | Confirmed/credible cross-tenant disclosure, prohibited consequential action, or broad outage with no safe workaround | 15 min | 30 min | `ROLE-SECURITY` | `<funded page or explicit safety posture>` | `<security/legal/comms authority>` |
| `SEV-2` | Material quality, data, tool, or availability failure affecting a cohort with safe containment/workaround | 60 min | 120 min | `ROLE-IT-OWNER` | `<behavior>` | `<comms authority>` |
| `SEV-3` | Limited defect with low immediate impact and no security boundary failure | 240 min | 480 min | `ROLE-IT-OWNER` | `<queue to covered hours or approved path>` | `<comms authority>` |

If impact is uncertain, begin at the higher plausible severity and downgrade with evidence.

## Capability ownership

| Capability | Accountable | Responsible | Consulted | Evidence required | Escalate when | Hypercare handback |
|---|---|---|---|---|---|---|
| Intake and known-issue triage | `PER-RIV-005` | `ROLE-SUPPORT-L1` | `PER-RIV-002` | Ticket, tenant tier, release ID, error code, redacted timestamp | Security signal, broad impact, no known issue | Customer-owned from start |
| Connector restart and application rollback | `PER-RIV-005` | `ROLE-SUPPORT-L2` | `PER-FDE-001` during hypercare | Runbook step, deployment ID, health check, rollback record | Ambiguous state, failed rollback, missing authority | Remove FDE consultation after passed handback drill |
| Policy/content correction | `PER-RIV-002` | `ROLE-EDITORIAL-DIRECTOR` | `PER-RIV-003`, `PER-RIV-004` | Source version, approval, reindex report, evaluation result | Policy authority conflict or security impact | Customer-owned |
| Identity containment/re-enablement | `PER-RIV-004` | `ROLE-SECURITY` | `PER-RIV-003`, `PER-RIV-005` | Entitlement snapshot, scope, negative tests, approval | Any false allow or uncertain disclosure | Customer-owned |
| Budget exception/cost review | `PER-RIV-006` | `ROLE-FINANCE` | `PER-RIV-001`, `PER-RIV-005` | Forecast, observed usage, allocation, decision | Spend envelope/forecast invalid | Customer-owned |
| Model/retrieval/release defect | `PER-FDE-001` during hypercare | `ROLE-FDE` during hypercare | Editorial, Security, IT | Release report, redacted trace, reproduction, fix evidence, handback | Fix changes scope/control or hypercare is ending | **Blocked until `UNK-RIV-010` is resolved** |

## Escalation path

| Boundary | L1 action | L2/service action | Incident/security/action authority | External/vendor action | Evidence returned to intake |
|---|---|---|---|---|---|
| Request/schema | Validate client version and bounds; do not retry invalid input | Identify client regression | Product/workflow owner for contract change | Client owner if external | Error code, release, safe timestamp |
| Identity/authorization | Do not bypass; escalate immediately on possible false allow | Isolate route and preserve evidence | Security controls re-enablement and disclosure decisions | Identity vendor as approved | Entitlement/filter decision references |
| Retrieval/model quality | Capture approved IDs/slices; no content in ticket | Separate index from release cause | Workflow/data/release owners decide scope and rollback | Model/index vendor as approved | Query-set/index/release evidence |
| Tool/workflow write | Pause writes on duplicate/ambiguous commit | Query target state and reconcile | Tool/business owner authorizes compensation | PageTurn owner/vendor | Business key, attempts, checkpoint |
| Capacity/provider | Stop retry amplification and ramp | Isolate quota/provider/tool/state bottleneck | Operations/finance approve capacity/cost change | Cloud/provider support | Offered/achieved load and dependency evidence |

## Explicit exclusions

Record items outside support, such as unsupported use cases, unapproved regions, direct production data repair, legal/compliance determination, autonomous publication/rights/payment changes, custom client defects, or unvalidated vendor behavior. Each exclusion names the receiving owner and safe customer path.

## Acceptance checks

- [ ] Covered and out-of-hours behavior are both explicit.
- [ ] Response targets have a staffed delivery path, not only a table.
- [ ] External communications and re-enablement authorities are named.
- [ ] Vendor escalation includes entitlement, channel, and evidence requirements.
- [ ] FDE responsibilities expire through explicit handback criteria.
- [ ] `UNK-RIV-010` is not silently resolved by leaving the FDE on call forever.
