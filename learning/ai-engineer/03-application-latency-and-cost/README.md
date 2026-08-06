# Application Latency and Cost

> **Evidence banner:** `FIXTURE` inputs, `VALIDATED` deterministic outcomes,
> `OUTPUTS CLEARED`, `UNVALIDATED` production behavior.

This chapter follows Riverside House release `rel-riv-002` through a familiar
operational mistake: total average latency looks acceptable, so the team starts
tuning whichever component it already understands. The fixed traces force a
better sequence. You first reconcile request totals, then attribute stages,
separate p50 from p95, distinguish TTFT from TPOT, expose retries and cache
effects, normalize cost by successful work, and only then choose the next test.

The notebook reads the shared fixtures directly. It makes no provider call,
loads no model, and requires no credential.

## Start Here

1. Run `setup.ps1` on Windows or `setup.sh` on macOS/Linux.
2. Open `application-latency-and-cost.ipynb`.
3. Select `Python (AI Engineer 03 Latency and Cost .venv)`.
4. Run from the top when you are ready to produce local evidence.

The setup and notebook completed successfully in the unified FDE environment.
Notebook outputs and execution counts were then cleared for reuse.

## Failure-First Route

| Step | Failure exposed | Minimal fix |
|---|---|---|
| 1 | Total average latency hides tails, cache hits, and paid failure | Reconcile one request-stage ledger |
| 2 | End-to-end latency has no component owner | Attribute latency and cost by stage and attempt |
| 3 | p50/p95 still mixes first-token and streaming experience | Report TTFT and TPOT from final successful generation only |
| 4 | Reliability controls silently duplicate work | Measure retry amplification with a named denominator |
| 5 | Cache claims use hypothetical hit rates | Pair an exact hit with its recorded origin request |
| 6 | Requests/second and cheap averages reward dropped work | Normalize throughput, tokens, and cost by successful work |
| 7 | Dashboards invite tuning before diagnosis | Rank bottlenecks and name the next discriminating test |
| 8 | Detailed traces become unsafe metric labels | Map to bounded, content-free production telemetry |

## Deterministic Fixture Expectations

These are frozen expectations from the shared fixture contract, not results
produced during authoring:

| Metric | Expected value when run |
|---|---:|
| Success rate | 4 / 5 = 80% |
| Successful-request latency p50 / p95 | 165 ms / 285 ms |
| TTFT p50 / p95 | 50 ms / 60 ms |
| TPOT p50 / p95 | 15 ms / 20 ms |
| Successful uncached output throughput | 57.14 tokens/s |
| Retry amplification, generation-bearing requests | 1.50 attempts/request |
| Retry amplification, successful uncached requests | 1.33 attempts/request |
| Exact-hit observed savings | 158 ms and 1,030 micro-USD |
| Total observed cost | 6,420 micro-USD |
| Cost per successful request | 1,605 micro-USD |
| First bottleneck diagnosis | Generation, with retries amplifying tail and cost |

## Exact Health Checks

The notebook does not accept a dashboard conclusion until these checks pass:

1. Every fixture row satisfies the frozen request-trace schema.
2. For every request, stage latency sums to `total_latency_ms`.
3. For every request, stage cost sums to `total_cost_microusd`.
4. Cache hits have a valid origin, no billed tokens, no generation stage, and
   served tokens equal to the origin's served tokens.
5. TTFT and TPOT are computed only from final successful generation stages.
6. Both retry-amplification reports print their populations and denominators.
7. Cost per successful request includes failed-request and failed-attempt spend.
8. Billed input/output tokens and served output tokens reconcile independently.
9. The selected bottleneck has the highest stage p95 and a named next test.
10. Metric dimensions remain within the Riverside bounded allowlist; request IDs
    stay in trace context and never become metric labels.

## Completion Evidence

You have finished this chapter when you have:

- retained the verified fixture version and trace/schema digests;
- retained a zero-error request/stage latency and cost reconciliation;
- reported p50/p95 latency, TTFT, TPOT, useful throughput, retries, cache observations, token accounting, and cost with explicit populations;
- named the first bottleneck and the cheapest next discriminating test before proposing an optimization;
- recorded the hardware/software/workload boundary even when the input is a serial fixture;
- linked the report into a capstone evidence index as `LOCAL_FIXTURE`, not as a production SLO, capacity, billing, or Azure result.

## Files

| Path | Purpose |
|---|---|
| `application-latency-and-cost.ipynb` | Successfully executed failure-first analysis notebook, cleared for reuse |
| `requirements.txt` | Minimal local analysis and schema-validation dependencies |
| `setup.ps1`, `setup.sh` | Chapter-local environment and kernel setup |
| `artifacts/latency-cost-report.json` | Retained `LOCAL_FIXTURE` evidence from the successful unified FDE run |
| `../shared/latency-cost/request-traces.jsonl` | Frozen privacy-safe request and stage traces |
| `../shared/latency-cost/request-trace.schema.json` | Frozen JSON Schema contract |
| `../shared/latency-cost/EXPECTED_OUTCOMES.md` | Independent deterministic answer key |

## Conceptual Owners

This chapter composes existing mechanisms instead of reteaching them:

- [Gateway routing, retries, fallback, caching, and cost](../../genai/06-llm-gateway/06-llm-gateway.ipynb)
- [Inference TTFT, TPOT, batching, KV cache, and throughput](../../ai-infrastructure/07-inference-systems/inference-systems.ipynb)
- [Profiler-first bottleneck diagnosis](../../ai-infrastructure/03-profiling/pytorch-profiling.ipynb)
- [Azure-shaped operational serving and load gates](../../ai-infrastructure/09-azure-operational-llm-serving/README.md)
- [Riverside bounded telemetry contract](../../../projects/riverside-ai-platform/contracts/v1/telemetry-attributes.schema.json)
- [Content-free operational telemetry decision](../../../projects/riverside-ai-platform/docs/decisions/0006-content-free-operational-telemetry.md)
- [Cost and capacity assumptions](../../../projects/riverside-ai-platform/docs/cost-and-capacity-assumptions.md)
- [Repository authoring standard](../../../AUTHORING_GUIDE.md)

## Local Versus Production

The local notebook proves deterministic arithmetic over five serial synthetic
traces. It can teach attribution, denominator discipline, conservation checks,
and decision order. It cannot establish a production SLO, p99 behavior,
concurrency capacity, queueing, autoscaling, cold starts, cache correctness under
mutation, tokenizer accuracy, provider billing, observability overhead, exporter
delivery, cardinality, redaction under load, or Azure service behavior.

Production evidence requires a versioned workload, calibrated load generator,
warm baseline, step load, expected peak, burst, soak, dependency failure,
instance loss, recovery, and blue/green overlap. Retain achieved load, token
distributions, deployment state, errors, p50/p95/p99 TTFT/TPOT/total latency,
successful output throughput, cache outcomes, retry counts, billing export,
redaction samples, and reviewer sign-off. Do not turn request, user, tenant,
prompt, completion, document, or source identifiers into metric dimensions.

## Validation Status

The setup and notebook executed successfully in the unified FDE environment,
including the deterministic calculations and assertions. Notebook outputs were
then cleared, while `artifacts/latency-cost-report.json` was retained as local
fixture evidence. No production, provider, billing, capacity, or cloud behavior
was validated.
