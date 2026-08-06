# Cost, Capacity, SLA, and Support Envelope

## Document control

| Field | Value |
|---|---|
| Artifact IDs | `CAP-CAP-01`, `COST-CAP-01`, `SVC-CAP-01` |
| Version / status | `[TODO] / DRAFT` |
| Capacity / finance / operations / commitment owners | `[TODO]` |
| Workload and architecture versions | `[TODO]` |
| Scope / exclusions / revalidation trigger | `[TODO]` |

## Input ledger

| Input ID | Definition/unit | Low | Expected | High/stress | Evidence class | Source/date | Confidence/limitation | Validation owner |
|---|---|---:|---:|---:|---|---|---|---|
| `ASM-RIV-* / CAPIN-CAP-*` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[Modeled / Measured]` | `[TODO]` | `[TODO]` | `[TODO]` |

Include arrival/burst shape, request mix, input/output token distributions,
retrieval top-k, service-time distributions, cache eligibility and realized hit
rate, retries, concurrency, blue/green overlap, growth, support hours, and failure
scenarios. Contract maxima are safety bounds, not average sizing inputs.

## Scenario calculations

| Scenario | Arrival and concurrency | RPM/TPM | Model/index/gateway capacity | Retry/cache effect | Headroom/quota | First breached gate | Evidence class |
|---|---|---|---|---|---|---|---|
| Low | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[Modeled]` |
| Expected | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[Modeled]` |
| High | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[Modeled]` |
| Stress/failure | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[Modeled]` |

Show formulas and units. State which input dominates each result. A Python or
spreadsheet calculation over modeled values remains modeled.

## Cost attribution

| Cost ID/category | Billing unit and rate | Low | Expected | High | Fixed/variable | Source/date/currency | Allocation rule | External validation |
|---|---|---:|---:|---:|---|---|---|---|
| Model serving | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` |
| Gateway/orchestration | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` |
| Databricks/index/storage | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` |
| Observability/network/security | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` |
| Evaluation/load/blue-green | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` |
| Support/hypercare | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` |

Normalize by successful work while retaining failed/rejected and idle cost. Do not
create a cheaper number by dropping requests or omitting fixed cost.

## Sensitivity and break-even

| Variable | Tested range | Effect on capacity/cost/SLO | Threshold or break-even | Decision implication | Measurement that replaces model |
|---|---|---|---|---|---|
| `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` |

At minimum vary sessions/arrival burst, token length, retry rate, cache realization,
service time, instance/unit rate, support hours, and blue/green overlap.

## Proposed service and support tiers

| Tier/use case | Quality/safety gate | Latency/availability proposal | Covered hours/support response | Capacity/quota | Rollout limit | Exclusions | Commitment authority/status |
|---|---|---|---|---|---|---|---|
| `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` |

Targets are not achieved results. A support response target is not a legal SLA or
service-credit term. Record who can make each commitment.

## External validation register

| Claim | Required environment/evidence | Owner/reviewer | Needed by | Consequence if absent | Status |
|---|---|---|---|---|---|
| Region/SKU/quota availability | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `OPEN` |
| Live price/billing reconciliation | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `OPEN` |
| Load, autoscale, overload, recovery | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `OPEN` |
| Support/contract terms | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `OPEN` |
| RTO/RPO/failover | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `OPEN` |

Envelope decision: `[TODO: PLANNING RANGE ONLY / HOLD / APPROVED FOR NAMED TEST]`
