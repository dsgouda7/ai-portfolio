# Hypercare and FDE Exit Criteria Template

## Principle

The FDE exits when customer operations can own the accepted scope and the remaining work has customer owners. The exit is not achieved by waiting 30 days or moving unresolved operational work into an unowned backlog.

## Control

| Field | Value |
|---|---|
| Artifact ID | `HOF-EXIT-01` |
| Hypercare window | `<start/end; Riverside input is 30 days>` |
| Accepted scope | `<package/acceptance reference>` |
| FDE owner | `PER-FDE-001` |
| Customer acceptance owner | `PER-RIV-005` or accepted successor |
| Business owner | `PER-RIV-001` |
| Status | `open / conditional / met / extended / rejected` |

## Exit gates

| Exit criterion | Required evidence | Customer owner after exit | Result | Blocker/remediation | Revalidate on |
|---|---|---|---|---|---|
| Package accepted for exact scope | `HOF-05` with no hidden critical gap | Acceptance owner | `<result>` | `<blocker>` | Scope change |
| Every recurring task/alert has owner and backup | Package and alert ownership checks | Operations/support | `<result>` | `<blocker>` | Rota/ownership change |
| Operators pass minimum drills | `HOF-04` measured records | Operations | `<result>` | `<blocker>` | Drill expiry/runbook change |
| Known-good rollback and re-enablement paths are understood | Timed drill and authority record | Release/incident owners | `<result>` | `<blocker>` | Release/architecture change |
| Data, identity, policy, cost, and vendor paths are handed back | Capability matrix | Named domain owners | `<result>` | `<blocker>` | Domain change |
| Post-hypercare model/retrieval quality owner accepted | Resolution for `UNK-RIV-010` | `<customer owner>` | `<result>` | `<blocker>` | Owner/scope change |
| Out-of-hours behavior accepted | Resolution for `UNK-RIV-006` | Business/support | `<result>` | `<blocker>` | Coverage/contract change |
| Temporary controls have owner and expiry | Incident/change records | Customer owner | `<result>` | `<blocker>` | Expiry/incident evidence |
| FDE-only access and secrets removed | Access review and revocation evidence | Security/IT | `<result>` | `<blocker>` | Access model change |
| Undocumented FDE knowledge removed | Independent operator exercise | Customer operations | `<result>` | `<blocker>` | Procedure change |
| Evidence-based backlog accepted | Backlog with priorities, owners, and blocked exposure | Product/operations | `<result>` | `<blocker>` | Review cadence |
| Recurring health/retirement review scheduled | Calendar/charter and first owner | Review chair | `<result>` | `<blocker>` | Cadence/owner change |

## Extension rule

Hypercare may be extended only with a reason, bounded scope, new end date, customer owner, FDE capacity agreement, exposure decision, and explicit plan to remove the dependency. Repeated extension is evidence of a failed ownership transfer, not a support strategy.

## Exit decision

| Decision | Scope | Evidence | Open conditions | Customer owner | FDE access removal date | Reopen trigger |
|---|---|---|---|---|---|---|
| `exit / extend / reduce exposure / reject handoff` | `<scope>` | `<references>` | `<conditions>` | `<owner>` | `<date>` | `<trigger>` |

## Honest close

Attach four lists:

1. measured results and limitations;
2. modeled assumptions and replacement-by-measurement owners;
3. customer-validated decisions and authority scope;
4. unknowns, blocked exposure, and next evidence owners.
