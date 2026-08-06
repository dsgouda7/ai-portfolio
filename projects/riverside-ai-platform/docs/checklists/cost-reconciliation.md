# Cost Reconciliation Record

> **Status:** Workflow and blank evidence record only. No billing export, price,
> allocation, or reconciliation has been produced by this document.

## Window and inputs

1. Select a closed UTC billing window that can be aligned to release, workload,
   capacity, and incident/change timelines.
2. Export actual and amortized cost/usage with subscription, resource group,
   resource ID, meter, quantity, currency, pricing model, tags, and credits kept
   distinguishable. Record export query/version and immutable file hash.
3. Capture modeled inputs for the same window: price date/currency, reservations or
   discounts, instance/replica-hours, APIM units, Databricks compute, vector/index
   usage, storage/transactions, observability ingestion/retention, network/egress,
   evaluation/load testing, security, and support.
4. Capture successful requests/output tokens plus failed, rejected, retry, idle,
   blue/green overlap, and incident consumption from approved aggregate telemetry.
5. Define and version shared-cost allocation. Keep unallocated cost visible rather
   than forcing it into a tenant or request estimate.

## Reconcile

For each service/meter and for the total, record modeled cost, actual cost, variance,
variance percentage when the modeled denominator is nonzero, and cause. Separate
price variance, quantity variance, allocation variance, late-arriving charges,
credits, and unexplained variance.

| Service/meter | Modeled | Actual/amortized | Variance | Cause/evidence | Owner/action |
|---|---:|---:|---:|---|---|
| `<service/meter>` | `<currency>` | `<currency>` | `<currency/%>` | `<reference>` | `<owner/action>` |

Recompute cost per successful request and successful output token from the same
window. Report fixed/idle, failed/rejected, retry, rollout overlap, and incident
cost separately; never improve a unit cost by excluding valid actual charges or by
counting dropped work as success.

## Decision and closure

- [ ] Billing total ties to the authoritative export and currency.
- [ ] Resource/tag coverage and shared/unallocated cost are reviewed.
- [ ] Usage quantities reconcile to deployment and telemetry evidence or have a
  documented timing/measurement reason.
- [ ] Material variance threshold and anomaly owner are approved.
- [ ] Forecast, capacity assumption, allocation rule, or instrumentation is updated
  only through its owning change process.
- [ ] Finance/FinOps and service owner sign off; security reviews any evidence
  access or unexpected data exposure.

| Field | Value |
|---|---|
| Reconciliation ID/window/currency | `<values>` |
| Billing export query/hash | `<reference>` |
| Model/workload/release versions | `<references>` |
| Total modeled/actual/variance | `<values>` |
| Unit costs and excluded categories | `<values>` |
| Material anomalies and actions | `<references>` |
| Reviewers/decision/UTC date | `<values>` |
