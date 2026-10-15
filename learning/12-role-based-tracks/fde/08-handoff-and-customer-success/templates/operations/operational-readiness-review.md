# Operational Readiness Review Template

> Example fields below use the frozen Riverside case. They are not an approved review or readiness decision.

## Review control

| Field | Value |
|---|---|
| Artifact ID | `HOF-01` |
| Review version | `<version>` |
| Environment/release/index/policy scope | `<exact scope>` |
| Review date | `<UTC date>` |
| Review chair | `<person or role>` |
| Decision authority | `<authorized customer role>` |
| Status | `draft / in review / accepted with conditions / blocked / superseded` |
| Next review or revalidation trigger | `<date or material change>` |

## Decision rule

Readiness is blocked when any critical row lacks an accountable owner, required evidence, safe fallback, or accepted residual risk. An authored procedure, source asset, or planned test is not a passing result.

## Gate matrix

| Gate ID | Domain | Readiness question | Required evidence | Accountable owner | Reviewer/authority | Status | Blocker or condition | Revalidate on |
|---|---|---|---|---|---|---|---|---|
| `ORR-SVC-01` | Service | Are release, slot, index, policy, runtime, region, and source commit identifiable and mutually compatible? | Deployment metadata plus compatibility result | `<owner>` | `<release authority>` | `<status>` | `<condition>` | Any release/index/policy/runtime change |
| `ORR-OBS-01` | Observability | Can operators distinguish gateway, retrieval, model, tool, policy, capacity, and telemetry failures without prohibited content? | Dashboard review, trace drill, redaction check | `<owner>` | Security and operations | `<status>` | `<condition>` | Signal schema, sampling, retention, or exporter change |
| `ORR-ALT-01` | Alerts | Does every page have a severity, owner, first action, runbook, escalation, and false-positive path? | Alert catalog plus delivery drill | `<owner>` | Operations | `<status>` | `<condition>` | Alert rule, threshold, routing, or rota change |
| `ORR-RBK-01` | Recovery | Can operators contain, roll back, compensate, and re-enable without private FDE knowledge? | Timed drill records | `<owner>` | Incident/release authority | `<status>` | `<condition>` | Recovery path or dependency change |
| `ORR-DATA-01` | Data | Are freshness, schema, ACL, lineage, reindex, access change, retention, and deletion operations owned? | Data runbook and negative tests | `<owner>` | Data owner | `<status>` | `<condition>` | Source/schema/ACL/retention change |
| `ORR-SEC-01` | Identity/security | Are false allows contained and re-enabled only after negative isolation evidence? | Isolation drill and approval | `<owner>` | Security authority | `<status>` | `<condition>` | Identity, RBAC, tenant, purpose, or region change |
| `ORR-CHG-01` | Change | Do release, index, policy, prompt, model, and threshold changes retain evaluation and rollback evidence? | Change records and sampled approvals | `<owner>` | Change authority | `<status>` | `<condition>` | Change process revision |
| `ORR-SUP-01` | Support | Are hours, severity, response targets, communication authority, vendors, and exclusions accepted? | Support/escalation matrix | `<owner>` | Business/support authority | `<status>` | `<condition>` | Contract, rota, or vendor change |
| `ORR-TRN-01` | Training | Have named operators passed agreed high-risk drills? | Scored drill records | `<owner>` | Operations acceptance owner | `<status>` | `<condition>` | Owner/role/runbook change or drill expiry |
| `ORR-LIM-01` | Limitations | Are unsupported claims, deferred capabilities, and open unknowns visible at the decision point? | Limitations and claim register review | `<owner>` | Acceptance authority | `<status>` | `<condition>` | Contradictory or expired evidence |
| `ORR-RET-01` | Retirement | Can the service be disabled and data/artifacts handled under approved retention/deletion rules? | Retirement runbook and authority map | `<owner>` | Data/security/business owners | `<status>` | `<condition>` | Scope, retention, or contract change |

## Riverside authored-state blockers

Record these as blockers unless later evidence supersedes them:

| Blocker | Current evidence class | Required next evidence | Owner |
|---|---|---|---|
| No live Azure deployment or composed path result | Implemented source only | Authorized environment evidence package | `<release/platform owner>` |
| No committed SLO, error budget, RTO, RPO, capacity target, or budget | Unknown/modeled only | Approved service envelope with measurements | `<business, operations, finance>` |
| No rollback or incident rehearsal | Procedure only | Timed staging drills with retained evidence | `<operations/release owner>` |
| Region, endpoint name, and timeout profiles conflict | Source inconsistency | Reconciled configuration and validation | `<integration owner>` |
| Out-of-hours coverage is unresolved (`UNK-RIV-006`) | Unknown | Funded coverage decision and contact rota | `PER-RIV-001` |
| Post-hypercare quality ownership is unresolved (`UNK-RIV-010`) | Unknown | Accepted accountable owner and evidence duties | `PER-RIV-001` |

## Decision

| Field | Value |
|---|---|
| Decision | `accept / accept with conditions / block` |
| Exposure allowed | `<tenant, cohort, use case, region, dates>` |
| Conditions | `<owner, due date, evidence, automatic response if missed>` |
| Residual risks accepted | `<risk IDs and authority>` |
| Evidence references | `<immutable references>` |
| Decision recorded by | `<authorized role and date>` |

## Quick health check

- [ ] Every critical row has an accountable owner and authority.
- [ ] Every `pass` points to retained evidence, not source presence.
- [ ] Conditional acceptance includes exposure limit and automatic response.
- [ ] Open Riverside unknowns remain open unless a decision record resolves them.
- [ ] Failed or expired drills block readiness.
- [ ] Limitations are copied into acceptance, not buried in an appendix.
