# Commercial Decision Record

## Record

| Field | Value |
|---|---|
| Customer / engagement | `[synthetic or approved identifier]` |
| Artifact version | `[version]` |
| Prepared on | `[date]` |
| Currency and tax basis | `[currency; included/excluded]` |
| Planning horizon | `[dates / months]` |
| Prepared by | `[role]` |
| Decision owners | `[finance, operations, security, product]` |
| Status | `[Draft estimate / Approved planning envelope / Quote pending / Rejected]` |

## Evidence Statement

This record contains `[Measured]`, `[Modeled]`, `[Policy constraint]`, `[Unknown]`, and `[External validation required]` entries. A calculation over modeled inputs remains modeled. This document is not a quote until the commercial owner changes the status after all quote prerequisites are complete.

## Demand and Capacity Range

| Scenario | Sessions / business day | Requests / month | Peak requests / hour | Peak TPM | Planned concurrency | Headroom | Evidence |
|---|---:|---:|---:|---:|---:|---:|---|
| Low | `[value]` | `[value]` | `[value]` | `[value]` | `[value]` | `[value]` | `[label/source]` |
| Expected | `[value]` | `[value]` | `[value]` | `[value]` | `[value]` | `[value]` | `[label/source]` |
| High | `[value]` | `[value]` | `[value]` | `[value]` | `[value]` | `[value]` | `[label/source]` |

## Monthly Cost Range

| Component | Low | Expected | High | Rate source/date | Validation owner |
|---|---:|---:|---:|---|---|
| Model input/output and retries | `[USD]` | `[USD]` | `[USD]` | `[source]` | `[owner]` |
| Application infrastructure | `[USD]` | `[USD]` | `[USD]` | `[source]` | `[owner]` |
| Retrieval and storage | `[USD]` | `[USD]` | `[USD]` | `[source]` | `[owner]` |
| Observability and evaluation | `[USD]` | `[USD]` | `[USD]` | `[source]` | `[owner]` |
| Other software | `[USD]` | `[USD]` | `[USD]` | `[source]` | `[owner]` |
| Support engineering | `[USD]` | `[USD]` | `[USD]` | `[source]` | `[owner]` |
| Risk / contingency reserve | `[USD]` | `[USD]` | `[USD]` | `[policy]` | `[owner]` |
| Total planning envelope | `[USD]` | `[USD]` | `[USD]` | Modeled | `[owner]` |

## Budget and Unit Economics

| Gate | Value | Result | Action |
|---|---:|---|---|
| Operating target | `[USD/month]` | `[within / exceeds]` | `[action]` |
| Hard planning ceiling | `[USD/month]` | `[within / exceeds]` | `[exception owner]` |
| Cost per successful request | `[range]` | `[status]` | `[measurement needed]` |
| Cost per active editor / accepted output | `[range]` | `[status]` | `[measurement needed]` |

## Sensitivity and Break-Even

Top drivers: `[support hours, output tokens, cache realization, retry rate, traffic, model rate, infrastructure tier]`.

Break-even decisions:

1. `[Decision and threshold; include quality/security guardrail.]`
2. `[Decision and threshold; include quality/security guardrail.]`
3. `[Decision and threshold; include quality/security guardrail.]`

## SLA Tier Consequence

Proposed tier: `[tier]`

Incremental architecture, quota, monitoring, rollout, and support cost: `[range]`.

Terms that cannot be offered yet: `[list unresolved evidence and why]`.

## Exclusions and Unknowns

- `[taxes, discounts, FX, data transfer, incident surge, service credits, legal fees, customer dependencies]`
- `[unknown demand shape, token distribution, cache safety, quota, region, failover, support staffing]`

## Quote Prerequisites

- [ ] Price book, discount, currency, tax, and billing units validated and dated.
- [ ] Regional model/service availability and quota validated.
- [ ] Load test validates latency, throughput, retries, cache, and failure behavior.
- [ ] Security and residency constraints map to an approved architecture.
- [ ] Support coverage, severity response, and escalation are staffed and priced.
- [ ] SLA measurement, exclusions, credits, and legal language are reviewed.
- [ ] Rollout scope, volume bands, overage handling, and change control are accepted.

## Decision

`[Approve planning envelope / Reduce scope / Change tier / Seek exception / Collect evidence / Reject]`

Conditions and expiry: `[conditions, owner, revalidation date or trigger]`.
