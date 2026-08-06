# Expected Deliverables

## Submission Layout

Create a candidate evidence directory outside `templates/` with this shape:

```text
candidate-evidence/
  evidence-index.json
  release-manifest.json
  data-quality-report.json
  retrieval-generation-evaluation.json
  prompt-comparison.json
  local-operational-slo-report.json
  azure-mapping.json
  drift-iteration-decision.json
  unsupported-claims-ledger.json
  decision-record.md
```

Start from the matching files under [`templates/`](templates/). Preserve upstream source artifacts unchanged and reference them by repository-relative path or governed URI plus digest.

The [worked teaching example](worked-teaching-example.md) shows a deliberately incomplete source-only `hold`. It is not a candidate package, is not passing evidence, and must not be cited as though its values were produced by your run.

## Cross-Artifact Invariants

Every file must satisfy these rules:

1. `candidate_release_id` is identical everywhere.
2. The baseline and rollback target resolve to known accepted releases and predate the candidate.
3. Every referenced artifact uses an immutable ID/version and a SHA-256 digest where bytes exist.
4. Every result has an evidence class, environment, source, and limitation.
5. Retrieval and generation metrics use the same versioned case set or explicitly document the difference.
6. Prompt comparison names every changed behavior field and holds all other fields fixed.
7. Local SLO evidence identifies hardware, software, workload, percentile method, populations, and reconciliation status.
8. Azure mapping names the live test needed for each cloud claim; `IMPLEMENTED_SOURCE` or `MODELED` is never relabeled `LIVE_AZURE`.
9. Drift decisions cite reviewed cases and state why each unselected intervention is not supported first.
10. The unsupported-claims ledger contains every absent, stale, modeled, contradictory, or out-of-scope claim used in the decision narrative.

## Deliverable Contracts

### `evidence-index.json`

The package table of contents. It records package version, candidate release, source commit, creation time, author/reviewer roles, each artifact path/digest/evidence class, and unresolved contradictions. A reviewer should be able to start here and locate every claim.

### `release-manifest.json`

The capstone integration object. It references, rather than copies, data, model, adapter, tokenizer, prompt, index, evaluator, operational, Azure-mapping, drift, and claims-ledger artifacts. It records compatibility gates, report binding, rollback target, and final release decision.

Use the local teaching contract in [`../shared/release-lineage/release-manifests.schema.json`](../shared/release-lineage/release-manifests.schema.json) for the application graph and map model-serving fields to the platform [`model-release-manifest.schema.json`](../../../projects/riverside-ai-platform/contracts/v1/model-release-manifest.schema.json). Do not rewrite a training manifest into a serving manifest.

### `data-quality-report.json`

Required content:

- source and candidate dataset IDs/digests;
- schema and semantic validation;
- duplicate and split-leakage findings;
- template/label validity;
- PII, provenance, rights, and contamination findings;
- task/split/slice counts and thin-slice warnings;
- preference disagreement and shortcut-risk findings;
- row-level issue ledger and explicit curation actions;
- gate decision, blockers, warnings, limitations, and evidence class.

A report that only says `pass` or `fail` is incomplete.

### `retrieval-generation-evaluation.json`

Required content:

- immutable dataset and evaluator versions;
- predeclared thresholds and critical slices;
- retrieval metrics such as recall@k, MRR, nDCG, authorization leakage, unsupported-query refusal, and citation coverage where applicable;
- generation metrics such as task correctness, groundedness, citation correctness, refusal, and safety/policy outcome;
- slice-level baseline/candidate comparisons;
- a gold-context or equivalent ablation that localizes at least one failure;
- evaluator validity, uncertainty, sample size, and limitations;
- separate retrieval and generation gate decisions.

### `prompt-comparison.json`

Required content:

- baseline and candidate prompt-release IDs and bundle digests;
- prompt, tool, retrieval, model, and evaluator pins;
- declared and measured changed fields;
- paired outcomes, slice metrics, uncertainty, and sample size;
- non-compensating critical regressions;
- shadow/A-B/canary state labeled as planned, locally simulated, or live;
- immutable rollback evidence and decision.

### `local-operational-slo-report.json`

Required content:

- local environment and workload identity;
- request/stage reconciliation for latency and cost;
- success population and denominator;
- p50/p95 total latency, TTFT, TPOT, useful throughput, retry amplification, cache observations, token accounting, and cost per successful request;
- bottleneck and next discriminating test;
- SLO thresholds and status, if predeclared;
- explicit statements that local serial fixture results do not prove concurrency capacity, p99, autoscaling, billing, exporter behavior, or Azure performance.

### `azure-mapping.json`

Map logical capabilities and contracts to the intended services:

- Azure ML managed online endpoints for custom model artifacts and blue/green deployments;
- API Management for the application boundary, bounded policies, and managed-identity backend access;
- Microsoft Entra ID and managed/workload identity for authentication;
- ADLS Gen2 and Databricks governed ingestion, with Databricks Direct Vector Access as the accepted initial index target;
- Azure Monitor/Application Insights for operational telemetry under the bounded allowlist;
- Key Vault and Container Registry where required by the platform design;
- Microsoft Foundry projects/evaluations/models only as an explicitly scoped mapping, not an assumed custom-artifact backend.

For each row, name the local evidence, target Azure resource, identity boundary, network/data boundary, required live validation, current evidence class, owner, and unsupported claim.

### `drift-iteration-decision.json`

Required content:

- baseline/current windows and releases;
- privacy-safe sampling policy;
- traffic, data, retrieval, quality, latency, cost, and policy signals with counts and denominators;
- uncertainty and critical-event handling;
- multi-label failure clusters and reviewed case lineage;
- versioned evaluation-candidate reference;
- explicit decisions for prompt, retrieval/index, guardrail, fine-tuning, and no action;
- selected intervention, rejected first actions, owner, next test, follow-up window, and rollback target.

### `unsupported-claims-ledger.json`

List every desired claim that the package does not prove. Each entry needs requested wording, safe wording, evidence currently available, missing evidence, scope, owner, revalidation trigger, and status. At minimum assess production readiness, Azure deployment, SLO attainment, capacity, cost, security/tenant isolation, private networking, residency, rollback, disaster recovery, telemetry privacy, and drift representativeness.

### `decision-record.md`

One concise final review:

- decision: `promote`, `hold`, or `reject`;
- candidate, baseline, rollback target, environment, and requested next stage;
- gate table with evidence links;
- failed/non-compensating gates;
- chosen intervention and next tests;
- known contradictions and accepted residual risk;
- the five strongest claims the package supports;
- the five most important claims it does not support;
- author, reviewers, approver, and timestamps.

## Completeness Check

The package is complete only when every rubric row maps to at least one artifact and every artifact maps to a rubric row. Extra dashboards, prose, or screenshots do not compensate for a missing contract, identity, digest, denominator, threshold, or limitation.

For a generalization pass, create a separate package with a new immutable candidate and newly declared policies. A path change or ID substitution over Riverside evidence is not generalization.
