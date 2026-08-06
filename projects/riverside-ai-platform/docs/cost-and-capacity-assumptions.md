# Cost and Capacity Assumptions

## Current confidence

No subscription, region, SKU, quota, request forecast, token distribution, corpus
size, SLO, retention period, or billing export is available. This document is a
measurement model, not a budget or capacity commitment. It contains no dollar
estimate because one would be decoration rather than evidence.

## Modeled architecture assumptions

| Assumption | Current value | Validation needed |
|---|---|---|
| Production topology | One approved Azure region | service/feature availability and residency review |
| Model backend | Azure ML managed online endpoint | artifact/runtime smoke, quota and SKU availability |
| Rollout | Blue/green overlap during candidate validation | duration, instance counts, quota, incremental cost |
| Gateway | APIM | tier capacity, policy latency, network and availability tests |
| Data plane | ADLS/Databricks with Direct Vector Access source implementation | workspace filter/deletion/load benchmark and ADR evidence |
| Orchestrator scaling | Container Apps HTTP concurrency rule; source floors/ceilings are dev 0/2, staging 1/4, production 2/10 | measured saturation signal, cooldown, cold-start, drain, recovery, and approved bounds |
| Demand | Unknown | business forecast plus observed arrival/concurrency distribution |
| Token workload | Contract maxima are 8,192 input and 2,048 output tokens; typical distribution unknown | production-safe aggregate telemetry and evaluation datasets |
| Headroom | Unset | approved failure/traffic margin based on bounded load |
| Availability target | No approved target; source load criteria are illustrative release inputs | business SLO, error budget, and representative cloud result |
| Retention | Unset | legal, security, operations, evaluation, and cost policy |

Contract maxima are safety bounds, not sizing averages. Capacity modeled only at
average prompt/output length will understate tail latency and memory/compute demand.

Riverside has no approved SLO or error budget. Use the
[draft SLO and error-budget decision](checklists/slo-error-budget-decision.md) to
propose measurable boundaries and approvals; it is explicitly unapproved until its
approval record and representative evidence are complete.

## Workload envelope

Define and version these inputs before a load run:

- requests per second by minute/hour/day and burst duration;
- concurrent streaming connections and client timeout;
- input/output token distributions including p50, p95, and maxima;
- retrieval top-k/search type and index/filter selectivity;
- cacheability and cache hit/miss policy;
- tenant-tier mix without customer identifiers;
- response success definition, refusal rate, and retry behavior;
- corpus/chunk/vector count, vector dimensions, update/deletion rate, and growth;
- candidate shadow/canary share and blue/green overlap window;
- regional failure/capacity scenario and approved degradation behavior.

The workload version belongs in every capacity and cost evidence set.

## Capacity model

Measure capacity at the first SLO or safety boundary, not at process failure.
For each deployment configuration, sweep concurrency until one of these occurs:

- deadline success or availability crosses its approved threshold;
- p95 TTFT/TPOT/total latency crosses threshold;
- overload/rejection or queueing crosses threshold;
- successful output tokens/second stops scaling acceptably;
- memory, compute, connection, index, APIM, or quota saturation appears;
- quality/safety changes because output is truncated or dependencies degrade.

Let $\lambda_{peak}$ be the approved peak arrival rate, $r_{safe}$ the measured
per-instance safe request rate for the versioned workload, and $h$ the approved
headroom fraction. A first planning estimate is:

$$
N_{planned} = \left\lceil \frac{\lambda_{peak}(1+h)}{r_{safe}} \right\rceil
$$

This is not a queueing guarantee. Validate the integer configuration under burst,
streaming, dependency throttling, one-instance loss, and recovery. When request
length varies materially, size against successful output tokens/second and memory
as well as requests/second.

## Cost model

Separate fixed and variable costs:

$$
C_{monthly} = C_{model} + C_{gateway} + C_{orchestration} + C_{index}
+ C_{databricks} + C_{storage} + C_{observability} + C_{network}
+ C_{evaluation} + C_{security} + C_{support}
$$

Track:

- Azure ML instance-hours, autoscale floor/ceiling, image/model storage, and
  blue/green overlap;
- APIM tier/units and policy/network overhead;
- orchestrator compute, minimum replicas, and scaling;
- dedicated Basic ACR storage/operations and private Container Apps environment
  infrastructure/network cost;
- vector-index storage, replicas, query/update units, and backup;
- Databricks jobs/SQL/serverless/classic compute and Unity Catalog storage;
- ADLS capacity, transactions, redundancy, versioning, soft delete, and backup;
- Application Insights/Log Analytics ingestion, retention, queries, alerts, and
  exports;
- private endpoints, DNS, firewall, egress/inter-region transfer;
- Azure Load Testing and evaluation compute/judge/model usage;
- Key Vault, Container Registry, build, vulnerability/provenance, and support plan.

Normalize only against successful work:

$$
C_{request} = \frac{C_{window}}{R_{successful}}, \qquad
C_{token} = \frac{C_{window}}{T_{successful\ output}}
$$

Report idle/fixed cost and failed/rejected request cost separately so a cheaper
number cannot be created by dropping work.

## Required load stages

1. **Test-engine calibration:** prove the generator reaches offered load and does
   not bottleneck first.
2. **Warm baseline:** stable low concurrency after artifact verification/warm-up.
3. **Step load:** identify slope changes and first gate breach.
4. **Expected peak:** sustain the approved peak for its business duration.
5. **Burst:** test short demand spikes, admission control, and retry behavior.
6. **Soak:** expose leaks, drift, throttling, and observability cost.
7. **Dependency delay/failure:** prove deadlines, retry bounds, circuit behavior,
   fallback policy, and recovery.
8. **Instance loss/scale event:** measure headroom, cold start, drain, and recovery.
9. **Blue/green overlap:** prove candidate rollout capacity and incremental cost.

Every stage records offered/achieved load, request/token distribution, instance
state, test-engine health, errors/rejections, latency, successful tokens, quality
sentinels, and cost window.

## Expected evidence commands

Locust/Azure Load Testing assets, criteria, synthetic requests, parser, and contract
tests are implemented source. The task ran none of them. Use the exact static and
normalization interfaces below from the repository root:

```powershell
python -m pytest projects/riverside-ai-platform/tests/contract/load_tests

python projects/riverside-ai-platform/load-tests/result_parser.py `
  --result "<engine-1-result.csv>" `
  --result "<engine-2-result.csv>" `
  --engine-health "<engine-health.json>" `
  --criteria projects/riverside-ai-platform/load-tests/success-criteria.json `
  --output "<normalized-load-evidence.json>"
```

Inspect the deployed model configuration before and after the run:

```powershell
az ml online-deployment list `
  --endpoint-name "<managed-online-endpoint-name>" `
  --resource-group "<approved-resource-group>" `
  --workspace-name "<azure-ml-workspace-name>" `
  --output table
```

Create and run the existing Azure Load Testing asset only in approved staging:

```powershell
$LoadTestResource = "<azure-load-testing-resource-name>"
$TestId = "riverside-chat-staged"
$TestRunId = "<unique-approved-test-run-id>"

az load test create `
  --resource-group "<approved-resource-group>" `
  --load-test-resource $LoadTestResource `
  --yaml-file projects/riverside-ai-platform/load-tests/azure-load-test.yaml

az load test run `
  --resource-group "<approved-resource-group>" `
  --load-test-resource $LoadTestResource `
  --test-id $TestId `
  --test-run-id $TestRunId

az load test run show `
  --resource-group "<approved-resource-group>" `
  --load-test-resource $LoadTestResource `
  --test-run-id $TestRunId
```

The committed YAML contains a non-routable host and Key Vault placeholder. Replace
them through the approved deployment pipeline; never commit a bearer token. Result
download/export remains a pipeline interface gap, so do not claim normalized load
evidence until every engine CSV and health record is retained.

## Budget and capacity gates

Before production, approve:

- monthly baseline and peak budget by environment/service;
- per-successful-request and per-successful-output-token thresholds;
- anomaly thresholds and owner;
- minimum/maximum instance counts, scale signal, cooldown, and manual override;
- quota headroom and escalation lead time;
- blue/green overlap budget and maximum retention;
- observability sampling/retention that still supports incident and audit needs;
- degradation order that preserves authorization and safety.

Do not reduce authorization, delete telemetry evidence, or raise retry counts to hit
a cost or capacity target.

The committed Container Apps values are safety bounds for review, not accepted
capacity. The HTTP rule targets 20 concurrent requests per replica; CPU is 0.5 and
memory is 1 GiB per replica. Production sets two minimum and ten maximum replicas
and zone redundancy in source. Validate regional support, subnet sizing, cold-start
behavior, dependency connection limits, and the cost of the minimum floor before
approval.

Until that evidence exists, every capacity intervention follows the
[manual capacity change record](checklists/manual-capacity-change.md). Operators
capture the bottleneck and all tier limits, approve quota and incremental cost,
change one control through reviewed source, observe the same versioned workload,
and revert on an abort threshold. Source min/max values do not authorize automatic
or manual scaling by themselves.

## Cost reconciliation workflow

For each closed billing period and after every material release, workload,
architecture, retention, or capacity change:

1. align an authoritative actual/amortized Azure cost export to the same UTC window
   as workload, release, capacity, and incident evidence;
2. reconcile every service and meter to the modeled quantity and price assumptions,
   preserving unallocated/shared cost and late charges;
3. separate price, quantity, allocation, credit, idle, failure/retry, incident, and
   blue/green overlap variance;
4. recompute cost per successful request and successful output token from the same
   population; and
5. assign material unexplained variance to an owner and update assumptions only
   after finance/FinOps and service review.

Use the [cost reconciliation record](checklists/cost-reconciliation.md) for the
query/hash, variance ledger, unit costs, decisions, and sign-off. This workflow has
not been run and no billing result is claimed.

## Live validation required

Retain regional quota/SKU evidence, load result, achieved workload, deployment
configuration, autoscale behavior, cost-management export, pricing date/currency,
discount/reservation assumptions, shared-cost allocation, and reviewer. Reconcile
modeled cost against billing after each release and material workload change.
