# Acceptance Sign-Off Template

> Acceptance records a bounded ownership decision. It does not erase failed drills, convert modeled claims into measurements, or authorize an approver outside their authority.

## Acceptance control

| Field | Value |
|---|---|
| Artifact ID | `HOF-05` |
| Package/version | `<HOF-PKG reference>` |
| Environment/release/index/policy scope | `<exact scope>` |
| Users/use cases/regions included | `<scope>` |
| Explicit exclusions | `<scope>` |
| Decision date/expiry | `<UTC dates>` |
| Decision | `accept / accept with conditions / reject` |

## Evidence summary

| Claim ID | Statement | Class | Evidence/approval reference | Scope | Limitation | Revalidate on |
|---|---|---|---|---|---|---|
| `<ID>` | `<bounded statement>` | `Measured / Modeled / Customer-validated` | `<reference>` | `<scope>` | `<limitation>` | `<trigger>` |

## Required acceptance gates

| Gate | Owner/authority | Evidence | Result | Condition/blocker |
|---|---|---|---|---|
| Operational readiness review | `PER-RIV-005` or accepted successor | `HOF-01` | `<result>` | `<condition>` |
| Business/workflow acceptance | `PER-RIV-001` with workflow owner input | `<criteria and scope>` | `<result>` | `<condition>` |
| Security/identity acceptance | `PER-RIV-004` | `<controls and negative tests>` | `<result>` | `<condition>` |
| Support and escalation | Business/support authorities | `HOF-03` | `<result>` | `UNK-RIV-006` disposition |
| Training and drills | Receiving operations owner | `HOF-04` | `<result>` | `<failed/expired drill>` |
| Post-hypercare ownership | Business and technical owners | Exit criteria | `<result>` | `UNK-RIV-010` disposition |
| Limitations and backlog | Relevant owners | Limitation/backlog references | `<result>` | `<critical unresolved item>` |

## Conditions

Every condition must include:

| Condition ID | Required evidence/action | Owner | Due date | Exposure limit while open | Automatic response if missed | Closure authority |
|---|---|---|---|---|---|---|
| `<COND-ID>` | `<bounded requirement>` | `<owner>` | `<date>` | `<cohort/use-case/region limit>` | `<pause, rollback, fail closed, or reject>` | `<authority>` |

## Known limitations and unresolved risks

List unsupported capabilities, deferred work, accepted residual risks, open unknowns, evidence expiry, and customer dependencies. Include the Riverside platform's lack of live Azure evidence and uncommitted SLO/capacity/cost/residency claims unless superseded by scoped retained evidence.

## Authority signatures

| Decision area | Authorized person/role | Authority basis | Decision and scope | Date | Conditions |
|---|---|---|---|---|---|
| Operations ownership | `<role>` | `<basis>` | `<decision>` | `<date>` | `<conditions>` |
| Business/workflow | `PER-RIV-001` or delegated authority | `<basis>` | `<decision>` | `<date>` | `<conditions>` |
| Security/identity | `PER-RIV-004` or delegated authority | `<basis>` | `<decision>` | `<date>` | `<conditions>` |
| Data/policy | `<roles>` | `<basis>` | `<decision>` | `<date>` | `<conditions>` |
| Support/commercial | `<roles>` | `<basis>` | `<decision>` | `<date>` | `<conditions>` |

## Automatic rejection conditions

- A critical drill failed or was never run.
- A recurring critical task or alert has no accepted owner and backup.
- A successful forbidden-access path exists or isolation evidence is missing for the accepted scope.
- Rollback/re-enablement authority or evidence is absent.
- Support commitments exceed staffed/funded coverage with no explicit safe posture.
- A modeled or source-only claim is presented as measured production evidence.
- FDE-only credentials, access, or undocumented knowledge remain required.

## Revalidation

Acceptance expires on material workflow, user, data, identity, region, policy, model, prompt, retrieval, architecture, vendor, support, or ownership change; contradictory incident evidence; failed health review; or the stated expiry date. Record who reopens which gate.
