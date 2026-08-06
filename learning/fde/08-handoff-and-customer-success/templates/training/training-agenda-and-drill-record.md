# Operator Training Agenda and Drill Record Template

> Training transfers decision capability. Attendance, slide completion, and a successful instructor demo do not prove an operator can act independently.

## Training control

| Field | Value |
|---|---|
| Artifact ID | `HOF-04` |
| Package/runbook versions | `<references>` |
| Target roles | L1 support, L2 operations, incident commander, security, data/policy, release owner, finance |
| Instructor/facilitator | `<role>` |
| Observers/assessors | `<independent roles>` |
| Environment/data | `<approved non-production environment and synthetic data>` |
| Session dates | `<UTC dates>` |
| Drill validity/expiry | `<date or material-change trigger>` |

## Agenda

| Block | Outcome | Method | Evidence |
|---|---|---|---|
| Scope and limitations | Operator can state accepted use cases, exclusions, unsupported claims, and support hours | Scenario questions | Scored responses |
| Architecture and version identity | Operator locates active release, slot, index, policy, runtime, region, and source commit | Guided inspection then independent repeat | Checklist and timestamps |
| Dashboard and alert triage | Operator separates gateway, retrieval, model, tool, policy, capacity, and telemetry failure | Seeded signals | Decision record |
| Evidence/privacy | Operator preserves useful records without copying customer content, secrets, or prohibited identifiers | Redaction exercise | Assessor checklist |
| Containment and recovery | Operator chooses fail closed, degraded mode, rollback, reconciliation, compensation, or escalation correctly | Tabletop plus timed drill | Measured drill record |
| Change control | Operator proposes a policy/threshold change with evaluation, approval, canary, rollback, and retained decision | Change simulation | Change record |
| Support and communications | Operator applies severity, hours, escalation, and communication authority | Role play | Timeline and update draft |
| Health and backlog | Operator turns trends/incidents/feedback into decisions and evidence-based backlog items | Review simulation | Health-review record |

## Minimum drills

| Drill ID | Scenario | Required operator behavior | Automatic failure |
|---|---|---|---|
| `DRL-RIV-TRACE` | Locate one failing request across gateway, retrieval/model, tool, policy, and audit boundaries | Correlate safe IDs/versions; identify missing trace context; avoid content leakage | Uses raw content/secrets or cannot identify the failed boundary |
| `DRL-RIV-ROLLBACK` | Candidate regression with a committed workflow action | Stop exposure; restore named known-good target; reconcile/compensate the committed action separately | Claims traffic rollback undoes the action |
| `DRL-RIV-DELETE` | Access revocation or deletion request | Apply approved path and verify downstream propagation and evidence | Restores deleted content or stops at source deletion without derived/index verification |
| `DRL-RIV-ISOLATION` | Seeded cross-tenant retrieval attempt | Fail closed, preserve evidence, engage Security, run negative tests before re-enable | Continues traffic, samples away evidence, or self-approves re-enablement |
| `DRL-RIV-THRESHOLD` | Evaluation or alert threshold change request | Use versioned policy/change process, representative slices, approval, canary, rollback, and retained decision | Directly edits production or copies an example threshold without evidence |

## Scoring rubric

Score each dimension separately; do not average away a critical failure.

| Score | Standard |
|---:|---|
| 0 | Unsafe action, authority bypass, evidence destruction/leakage, or no viable response |
| 1 | Finds the document but needs step-by-step FDE instruction |
| 2 | Executes the standard path with prompts; misses an edge condition or evidence field |
| 3 | Independently scopes, contains, preserves evidence, follows authority, validates, and communicates |

Required pass: score 3 for safety/authority/evidence handling and at least 2 for all other dimensions, with no automatic failure. Set stricter role-specific criteria where appropriate.

## Drill record

| Field | Entry |
|---|---|
| Drill ID/version/date | `<ID/version/UTC>` |
| Scenario and injected facts | `<bounded synthetic description>` |
| Operators and roles | `<names/roles>` |
| Observer/assessor | `<independent role>` |
| Time to detect/contain/decision/recovery | `<Measured values>` |
| Decisions and authorities used | `<references>` |
| Evidence handling result | `<result/reference>` |
| Scores by dimension | `<scores>` |
| Outcome | `pass / remediate and repeat / blocked` |
| Remediation, owner, due date | `<action>` |
| Retest evidence | `<reference or pending>` |
| Limitations | `<what the drill did not prove>` |

## Attendance and access

| Participant/role | Required modules | Attended | Drill(s) attempted | Result | Production access decision | Revalidate on |
|---|---|---|---|---|---|---|
| `<participant>` | `<blocks>` | `<yes/no>` | `<IDs>` | `<result>` | `<grant/retain/restrict>` | `<trigger>` |

## Training acceptance

- [ ] Every production role has a trained primary and backup.
- [ ] High-risk access follows passed drills, not attendance alone.
- [ ] Failed drills have remediation and retest records.
- [ ] Runbook or architecture changes expire affected drill evidence.
- [ ] Training materials expose limitations and support boundaries.
- [ ] Operators can complete the accepted path without private FDE context.
