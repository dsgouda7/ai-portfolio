# AI Engineer Shared Fixtures

This directory contains immutable, synthetic fixtures shared by the five AI Engineer notebooks. Notebook-specific code may read these files but must not rewrite them.

## Version and Integrity

[`VERSION`](VERSION) names the fixture contract version. [`fixture-manifest.json`](fixture-manifest.json) pins the SHA-256 digest of every fixture, schema, and expected-outcome file consumed by the notebooks. Each notebook verifies its owned files against this manifest before parsing them.

A digest mismatch means the local fixture bytes are stale or were modified. Do not update the expected digest merely to make a notebook continue. Restore the pinned bytes, or intentionally create a new fixture version and review every affected schema, expected outcome, assertion, plan, and capstone dependency.

## Contract

- All data is fictional and privacy-safe. Names, organizations, manuscript details, and operational events belong to the Riverside House teaching scenario.
- Deliberate PII examples use reserved domains and fictional North American `555-01xx` numbers. They are detection probes, not real personal data.
- IDs are stable, lowercase, and never derived from row order. Consumers should join on IDs rather than array positions.
- Timestamps are fixed UTC values. Costs are integer micro-US-dollars (`cost_microusd`) so arithmetic is exact: 1,000,000 micro-US-dollars equals 1 USD.
- Latencies are integer milliseconds. Percentile expectations use the nearest-rank method unless a fixture-specific document says otherwise.
- JSONL schemas apply to each line independently. JSON schemas apply to the complete JSON document.
- Schema validation is necessary but not sufficient. Cross-record and semantic invariants are documented in each `EXPECTED_OUTCOMES.md`.

## Fixture Map

| Notebook | Fixture directory | Stable join keys | Purpose |
| --- | --- | --- | --- |
| Training data quality and lineage | `training-data/` | `example_id`, `comparison_group_id` | Duplicate leakage, template validity, PII, provenance, contamination, and label disagreement |
| Prompt release and experimentation | `prompt-release/` | `prompt_release_id`, `eval_case_id` | Versioned candidates, paired comparisons, slice gates, and rollback evidence |
| Application latency and cost | `latency-cost/` | `request_id`, `release_id` | Stage latency, retries, cache savings, token usage, and cost attribution |
| Release registry and lineage | `release-lineage/` | `release_id`, artifact IDs | Base/adapter/prompt/index compatibility, promotion evidence, and rollback lineage |
| Production feedback and drift | `feedback-drift/` | `feedback_trace_id`, `request_id`, `release_id` | Windowed traffic, retrieval, quality, latency, cost, policy, and review signals |

## ID Namespaces

| Prefix | Entity |
| --- | --- |
| `td-` | Training example |
| `cg-` | Preference comparison group |
| `prompt-riv-` | Prompt release |
| `ec-` | Paired evaluation case |
| `req-` | Application request |
| `rel-riv-` | Application release manifest |
| `fb-` | Production feedback trace |
| `evalcand-` | Candidate evaluation case derived from reviewed feedback |

## Change Policy

These fixtures are version 1 contracts. Correcting a typo that changes expected arithmetic or semantic labels requires a schema-version bump and corresponding expected-outcome update. Additive records should receive new IDs; existing IDs must not be recycled.
