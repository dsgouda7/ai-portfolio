# Riverside RBAC and Authority Matrix

**Status:** `[Local-static]` policy translation from `RIV-FDE-1.0.0`.
Customer Security and system owners must validate policy intent; deployed assignments
and data-plane behavior require external tests.

| Role | Tenant/resource scope | Read/retrieve | Draft/propose | Commit/approve | Explicit deny | Validation owner |
|---|---|---|---|---|---|---|
| `ROLE-EDITOR` | Assigned titles in own tenant; current policy | Assigned manuscript, current policy | Draft continuation; propose workflow transition | None through assistant | Publish, rights, payment, unassigned/cross-tenant content | Editorial + Security |
| `ROLE-SENIOR-EDITOR` | Assigned titles in own tenant | Same as editor; review output | Draft; propose | Confirm exact workflow transition | Rights and payment changes | Editorial + PageTurn owner |
| `ROLE-EDITORIAL-DIRECTOR` | Approved imprint titles | Imprint titles, policy, bounded rights lookup | Review output | Cohort approval; no autonomous record mutation | Rights/payment mutation | Editorial + Rights |
| `ROLE-RIGHTS-COUNSEL` | Approved rights records across named tenants | Rights records | Interpret and propose rights workflow | Approve rights workflow outside assistant mutation path | Grant rights through assistant | General Counsel |
| `ROLE-SECURITY` | Security evidence across named tenants | Security evidence, redacted decision records | Disable-route recommendation | Security gate, route disable, re-enablement | Unrelated manuscript use | Head of Information Security |
| `ROLE-FINANCE` | Cost/aggregate scope | Cost reports; masked financial fields | Budget exception proposal | Budget exception | Manuscript content | Finance owner |
| `ROLE-IT-OWNER` / `ROLE-SUPPORT-L2` | Operational metadata | Redacted trace, integration health | Rollback/restart proposal | Initiate approved rollback | Manuscript by default; security policy change | Applications + Security |
| `ROLE-SUPPORT-L1` | Ticket metadata | Ticket metadata only | Use triage runbook | None | Prompt/manuscript; tenant re-enable | Service desk owner |
| `ROLE-FDE` | Synthetic sandbox; redacted delivery evidence | Synthetic and approved redacted evidence | Technical gate recommendation | None; cannot approve own exception | Production content without time-bound approval | Customer Security |
| `ROLE-SERVICE-RETRIEVAL` | Request-scoped tenant/region/purpose/title | Filtered retrieval only | None | None | Context-free retrieval; source writes | Platform Security |
| `ROLE-SERVICE-WORKFLOW` | Exact approved transition | Workflow read | Submit exact idempotent transition | Only after bound human confirmation | Unconfirmed transition; rights/payment writes | PageTurn owner + Security |

## Separation-of-duty checks

- The FDE cannot approve the FDE's own security exception.
- The model has no role and receives no standing credential.
- Human confirmation does not turn a prohibited tool into an allowed one.
- Gateway policy editors cannot be treated as ordinary application operators.
- Route re-enablement belongs to `PER-RIV-004`, not the incident implementer.
- Management-plane deployment rights remain separate from data-plane access.

## Effective-role rule

`effective_roles = requested_roles intersect active_trusted_roles`

If requested roles contain anything outside active trusted roles, deny the request
as role escalation rather than silently ignoring the extra authority. This makes
an attempted escalation visible in audit and negative tests.
