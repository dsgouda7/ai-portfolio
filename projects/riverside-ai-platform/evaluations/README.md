# Riverside Evaluation Assets

These assets feed `src/release_gates` without invoking models or cloud services.
Each JSON dataset is immutable evidence input identified by `dataset_id`,
`version`, `source_digest`, and `evaluator_contract_version`. The v1 schema keeps
one domain per dataset so evaluators cannot silently mix populations or slices.

## Evaluation domains

| Domain | Required evidence families |
|---|---|
| Data quality | Parse/quarantine, completeness, duplicates, schema drift, ACL/classification, lineage, freshness, deletion propagation |
| Retrieval quality | Recall@k, MRR, nDCG, citation coverage, unsupported-query refusal, tenant and ACL leakage, source/format/tier/query slices |
| Generation and citations | One-sentence contract, groundedness, answer relevance, citation precision and correctness |
| Adaptation evidence | Held-out CPT perplexity, SFT complete-contract passes, DPO positive preference edges, general-language and safety retention |
| Safety and authorization | Unsafe completion, authorization bypass, and cross-tenant leakage adversaries |
| Operational SLOs | Readiness/warm-up, TTFT, TPOT, total latency, throughput, availability, deadlines, overload recovery, drain, rollback |
| Cost | Cost per successful request and successful output token |
| Rollout comparison | Baseline/candidate deltas plus owners, windows, abort criteria, rollback targets, and stage coverage |

The adaptation metrics intentionally follow the fine-tuning learning arc: CPT is
judged on held-out next-token likelihood, SFT on complete passes over unseen
request wording and contexts, and DPO on held-out relative preference evidence.
A training loss or artifact existing is not release evidence.

Retrieval and generation stay separate. Recall@k, MRR, and nDCG diagnose
retrieval; groundedness and citation metrics diagnose whether generated claims
are supported. The simple token-overlap demonstrations in the learning material
are mechanism examples, not production judges.

## Decision semantics

Thresholds are owned by the gate policy, not by candidate output. Metric units
must match their thresholds. Point estimates determine the v1 metric `status`;
required confidence intervals are then checked conservatively by the release
gate. A point pass whose interval crosses the threshold produces `hold`, not
`promote`.

- `promote`: every required domain and slice is present, all point thresholds
  pass, and required uncertainty/sample-size checks clear.
- `hold`: evidence is missing, underpowered, or does not conservatively clear a
  threshold.
- `reject`: a blocking threshold fails or an unexpected supplied metric fails.

A report generator recomputes the decision from typed metric records. It rejects
a claimed `promote` containing a failed metric and a claimed `reject` with no
failed metric.

## External evaluators

`ExternalEvaluator` is a protocol. The default `DisabledExternalEvaluator`
raises before any call. A judge model, hosted metric, or human-review system must
be injected explicitly and must return metrics carrying its own versioned
identity. Release gates never make LLM, model, network, or cloud calls.

## Layout

- `schemas/v1/evaluation-dataset.schema.json`: closed, versioned dataset schema.
- `datasets/v1/`: representative fixtures for all eight domains.
- `examples/v1/`: machine-readable metric and release-report examples.
- `fixtures/invalid/`: deliberately contradictory evidence used by contract tests.
