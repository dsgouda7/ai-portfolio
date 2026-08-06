# SLA Tier Mapping

> Status: `[Draft / Customer-validated]`
> Artifact version: `[version]`
> Effective scope and dates: `[scope]`
> Decision owners: `[business, operations, security, commercial]`

## Service Definition

| Field | Value | Evidence / Owner |
|---|---|---|
| Covered request types | `[policy lookup / continuation / workflow assistance]` | `[source]` |
| Covered tenants and regions | `[scope]` | `[policy constraint]` |
| Covered hours and timezone | `[hours]` | `[support owner]` |
| Planned maintenance treatment | `[notice and exclusion rule]` | `[contract owner]` |
| Measurement point | `[client edge / gateway / service]` | `[operations owner]` |
| Success denominator | `[which outcomes count]` | `[operations owner]` |
| Explicit exclusions | `[customer dependency / force majeure / invalid request]` | `[contract owner]` |

## Tier Matrix

| Dimension | Pilot | Business Hours | Critical |
|---|---|---|---|
| Availability objective | No contractual SLA; measured pilot SLO | `[target]` during covered hours | `[target]` during agreed coverage |
| Latency by request type | Observe p50/p95/p99 | `[targets]` | `[targets]` |
| Architecture | Single active route; manual rollback | Redundant stateless app path; tested restore | Fault-domain isolation; tested failover where evidence supports it |
| Model/provider path | One approved route plus disable switch | Quota reservation or approved fallback | Independently validated fallback and degradation policy |
| Admission and quota | Cohort cap; hard token/request budgets | Tenant RPM/TPM/concurrency/spend limits | Reserved headroom; priority and load-shed policy |
| Monitoring | Dashboard and daily review | Alerts on burn, queue, throttling, retries, cost | 24x7 paging only if staffed and contracted |
| Rollout | Shadow or named cohort | Canary with ramp and rollback gates | Change windows, freeze rules, explicit re-enablement authority |
| Support | Best effort in named pilot hours | Severity response in covered hours | On-call rota, escalation, and response terms explicitly priced |
| Security constraints | Zero tolerance for forbidden access | Same; never traded for availability | Same; fail closed and isolate affected route |

## Error Budget and Burn

| Item | Definition |
|---|---|
| SLI | `[successful valid requests / valid requests, measured at ...]` |
| SLO window | `[rolling 30 days / calendar month / covered hours only]` |
| Allowed bad events | `[derive from target and eligible volume; do not round away small denominators]` |
| Fast-burn action | `[page / freeze ramp / shed load]` |
| Slow-burn action | `[investigate / capacity review / backlog]` |
| Security event treatment | Excluded from ordinary error-budget tradeoffs; follow security incident process |

## Validation Gates

- [ ] Load test covers request mix, token ranges, cache keys, retries, bursts, and long tails.
- [ ] Quota and regional capacity are confirmed for each approved route.
- [ ] Monitoring computes the exact contractual denominator and covered window.
- [ ] Failover and rollback are tested, including in-flight and stateful work.
- [ ] Support rota, severity definitions, response times, and re-enablement authority are accepted.
- [ ] Commercial owner approves incremental architecture and support cost.
- [ ] Legal/contract owner reviews exclusions, credits, and measurement language.

## Decision

`[Accept tier / remain at lower tier / collect more evidence]`

Rationale: `[Tie the decision to measured evidence, modeled risk, and unresolved validation.]`
