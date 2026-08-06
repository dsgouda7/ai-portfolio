# Production Feedback and Drift Expected Outcomes

Each window contains six synthetic traces. The latency SLO is at most 140 ms; values above 140 ms are breaches. Query summaries are privacy-safe categories rather than retained raw prompts.

## Window Comparison

| Signal | Baseline | Current | Change |
| --- | ---: | ---: | ---: |
| Security traffic share | 1/6 = 16.7% | 3/6 = 50.0% | +33.3 percentage points |
| Retrieval hit rate | 5/6 = 83.3% | 3/6 = 50.0% | -33.3 percentage points |
| Quality pass rate | 5/6 = 83.3% | 3/6 = 50.0% | -33.3 percentage points |
| Mean latency | 100 ms | 150 ms | +50% |
| Mean cost | 1,000 micro-USD | 1,500 micro-USD | +50% |
| Latency SLO breaches | 0/6 | 3/6 | +50.0 percentage points |
| Policy correctness | 6/6 = 100% | 5/6 = 83.3% | -16.7 percentage points |

The current-window failure clusters are deterministic:

- `retrieval_miss`: 3 traces (`fb-009`, `fb-010`, `fb-012`)
- `quality_failure`: 3 traces (`fb-009`, `fb-010`, `fb-012`)
- `latency_slo_breach`: 3 traces (`fb-010`, `fb-011`, `fb-012`)
- `policy_false_allow`: 1 trace (`fb-010`)

## Review and Evaluation Candidate

The three reviewed traces map one-to-one to `evalcand-001`, `evalcand-002`, and `evalcand-003`. A versioned evaluation candidate should preserve the synthetic query summary, expected policy outcome, release lineage, review label, and originating `feedback_trace_id`. It must not copy raw production text because none is retained in this fixture.

## Expected Decision

The evidence supports two immediate actions:

1. **Retrieval/index update:** retrieval misses explain every current quality failure and are concentrated in finance/security queries.
2. **Guardrail change:** `fb-010` is a false allow on a confidential-data request; policy enforcement must fail closed independently of generation quality.

Investigate the latency/cost increase in parallel, especially the security route. Do not choose fine-tuning first: the fixture points to missing retrieval coverage and a policy-control defect, not a learned style or behavior gap. Re-run the versioned evaluation candidate after the index and guardrail changes; use no action only if the drift disappears under a representative follow-up window.
