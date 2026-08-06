# SLA, Capacity, and Commercials

This chapter turns Riverside House's frozen demand, token, latency, availability, budget, and support assumptions into an auditable planning envelope. It is a planning exercise, not a vendor quote or proof that a production SLA is achievable.

## What You Build

- low, expected, and high demand scenarios;
- arrival-rate, concurrency, RPM, TPM, retry, cache, and headroom calculations;
- model, infrastructure, retrieval/storage, observability, software, and support cost attribution;
- sensitivity and break-even views;
- SLA tiers mapped to architecture, quota, monitoring, rollout, and support;
- a commercial decision record with explicit external validation owners.

The notebook intentionally makes two weak plans fail: a plan based only on daily averages and a plan that turns one expected-cost number into a quote.

## Evidence Boundary

| Label | Use in this chapter |
|---|---|
| `[Measured: synthetic fixture]` | Request traces observed in the committed AI Engineer fixture; useful for mechanics, not production extrapolation |
| `[Modeled]` | Calculations from frozen or chapter-owned planning assumptions |
| `[Policy constraint]` | Security, authorization, regional, or workflow rule that the design must preserve |
| `[Unknown]` | Missing input that needs discovery or a decision owner |
| `[External validation required]` | Live price, quota, regional capacity, cloud behavior, failover, or contractual commitment |

A modeled calculation does not become measured because Python produced it. A customer-approved target does not prove the system meets it.

## Inputs

- [Frozen Riverside FDE case](../shared/fixtures/riverside-engagement-v1.json)
- [Expected-facts ledger](../shared/fixtures/expected-facts-v1.json)
- [AI Engineer request traces](../../ai-engineer/shared/latency-cost/request-traces.jsonl)
- [Trace expected outcomes](../../ai-engineer/shared/latency-cost/EXPECTED_OUTCOMES.md)

The shared case is read-only. Chapter-owned assumptions such as requests per session, planning service time, unit rates, and cache realization are visibly labeled and include validation owners.

## Local Setup

Python 3.10 or newer is recommended. Setup is local only and does not contact a cloud service.

PowerShell:

```powershell
.\setup.ps1
```

Bash:

```bash
./setup.sh
```

Both scripts create `.venv`, install [requirements.txt](requirements.txt), and register the `fde-sla-capacity` kernel unless the skip-kernel option is used.

## Notebook

Open [sla-capacity-and-commercials.ipynb](sla-capacity-and-commercials.ipynb) and run it from top to bottom when you are ready to perform the exercise. Route validation executed the notebook successfully against committed synthetic fixtures, then cleared the generated outputs. All committed code cells therefore have empty outputs and null execution counts.

The verified run establishes that the local scenario, capacity, cost, and decision logic executes in the route environment. It does not establish live prices, quota, latency, capacity, support commitments, customer approval, legal terms, or production readiness.

## Templates

| Template | Purpose |
|---|---|
| [Capacity scenario](templates/capacity-scenario-template.csv) | Record low, expected, and high workload/capacity assumptions with evidence and owners |
| [Cost input ledger](templates/cost-input-template.csv) | Keep rates, units, provenance, date, and validation status visible |
| [SLA tier mapping](templates/sla-tier-mapping-template.md) | Bind service targets to architecture, quota, monitoring, rollout, and support |
| [Commercial decision record](templates/commercial-decision-record-template.md) | Separate estimate, ceiling, exclusions, approvals, and quote prerequisites |

Do not overwrite a template with a silent “final” value. Version the completed artifact and retain the source/date for every changed input.

## Honest Limits

This chapter cannot establish:

- current vendor prices, discounts, taxes, or exchange rates;
- model or regional quota availability;
- production p95/p99 latency or burst behavior;
- actual cacheability under authorization-scoped keys;
- failover RTO/RPO or cross-region data behavior;
- support staffing, response, or service-credit terms;
- legal enforceability of proposed SLA language.

Those items remain external validation gates before proposal, contract, or launch approval.

## Role boundary for commercial decisions

The FDE may elicit workload and support assumptions, build low/expected/high and stress scenarios, expose sensitivity and quality tradeoffs, identify quote prerequisites, and recommend technical options. The FDE does not independently negotiate discounts, service credits, liability, payment terms, support staffing, contract language, or a binding SLA unless the organization has explicitly delegated that authority.

Route price and discount inputs to finance or procurement, support commitments to the service owner, contract and liability language to commercial/legal owners, and final acceptance to the authorized customer and seller representatives. When those owners disagree, preserve the competing constraints and model the consequences; do not choose a contractual position for them.

## Downstream integration path

Use the chapter outputs to review the Riverside [cost and capacity assumptions](../../../projects/riverside-ai-platform/docs/cost-and-capacity-assumptions.md), [staged load-test assets](../../../projects/riverside-ai-platform/load-tests/README.md), [evaluation strategy](../../../projects/riverside-ai-platform/docs/evaluation-strategy.md), and [infrastructure choices](../../../projects/riverside-ai-platform/infra/README.md). A supervised practicum should replace modeled inputs with dated quotes and retained measurements, reconcile quota and region constraints, and return measured saturation, latency, recovery, and cost evidence to the claim register before any commitment. Source assets and proposed thresholds are not a quote or achieved service level.

## Related Material

- [Application Latency and Cost fixtures](../../ai-engineer/shared/latency-cost/EXPECTED_OUTCOMES.md)
- [LLM Gateway](../../genai/06-llm-gateway/06-llm-gateway.ipynb)
- [Inference Systems](../../ai-infrastructure/07-inference-systems/inference-systems.ipynb)
- [Production Scale and Capacity](../../agentic-ai-system-design/12-production-scale-and-capacity.md)
- [Azure Operational LLM Serving](../../ai-infrastructure/09-azure-operational-llm-serving/README.md)

The Azure chapter is a later validation path. No Azure claim or operation is required here.
