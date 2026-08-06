# Evaluation Strategy

## Release question

Evaluation answers one question: does this immutable candidate have enough evidence
to advance one rollout stage for one environment? It does not answer whether an
artifact exists or whether one aggregate score improved.

The v1 release report requires eight domains. Every metric records dataset and
evaluator versions, slice, threshold/operator/unit, observed value/unit, status,
and uncertainty when applicable. Thresholds are release policy inputs. This
document does not invent them.

## Evidence tiers

| Tier | Environment | What it may establish |
|---|---|---|
| Fixture/static | No live Azure data plane | Schema and invariant behavior, deterministic checks, redaction rules |
| Offline candidate | Approved versioned datasets | Quality/safety comparison under the evaluator's stated limits |
| Cloud smoke | `dev` or `staging` Azure | Identity, network, deployment, contract, readiness, and basic telemetry path |
| Bounded load | `staging` Azure | Latency/throughput/rejection/recovery behavior within tested load and duration |
| Shadow | Production path, no candidate response served | Candidate observations on approved mirrored traffic without client impact |
| Canary | Production, bounded candidate share | Comparative behavior for the stated population and observation window |
| Broad rollout | Production | Current release behavior; still not a permanent guarantee |

Local learning measurements explain mechanisms. They are not release-tier Azure
evidence.

## Domain gates

### Data quality

Measure parse success/quarantine rate, required-field completeness, exact and
near-duplicate rate, schema drift, ACL/classification coverage, lineage completeness,
freshness, and deletion propagation. Slice by approved source, format, tenant tier,
classification, and region without exposing customer identifiers.

Block when required security/lineage fields are missing, deleted content remains
retrievable past the approved propagation objective, or the evaluation corpus does
not represent the candidate index version.

### Retrieval quality

Measure recall@k, MRR, nDCG, citation coverage, unsupported-query refusal, and
tenant/ACL leakage. Keep retrieval and generation results separate. Include negative
queries where the correct result is no authorized evidence.

The implemented production target is Databricks Direct Vector Access. Cloud-free
adapter tests are not evidence that Databricks filter syntax, deletion, identity,
quality, latency, or cost works in the selected workspace.

### Generation and citation quality

Measure task-specific deterministic contracts, groundedness, answer relevance,
citation precision/correctness, and refusal when evidence is absent. Evaluate
citations against immutable document/chunk hashes and index version, not copied
source text in the citation object.

Report automatic metrics, judge metrics, and human review separately. A judge score
without evaluator version, calibration, and uncertainty is insufficient.

### Adaptation evidence

Compare base and candidate behavior directly and reference held-out CPT, SFT, and
DPO evidence from the unchanged source training manifests. Verify general-language
retention and targeted behavior. Training evidence is an input, not promotion.

### Safety and authorization

Test authentication failures, forbidden tenants/ACLs, policy violations, prompt
injection against retrieved content, cross-tenant leakage, disallowed data classes,
token limits, safe error messages, and telemetry/log content exclusion. Any
confirmed cross-tenant disclosure blocks release.

### Operational SLOs

Measure readiness after digest verification and warm-up, p50/p95 TTFT, TPOT and
total latency, throughput, successful output tokens/second, availability, deadline
success, rejections, 429/overload behavior, retry amplification, and recovery after
overload. Record test-engine health and achieved concurrency; offered load alone is
not throughput evidence.

### Cost

Measure cost per successful request and per successful output token. Separate fixed
idle cost, candidate overlap, model compute, APIM, retrieval/index, Databricks,
storage, observability, load testing, and network transfer. Use billing exports for
live evidence; list-price arithmetic remains modeled.

### Rollout comparison

Compare candidate and baseline under the same window, traffic definition, index
version or explicitly controlled index change, and approved slices. Record sample
size and uncertainty. A canary with too little traffic produces `hold`, not `pass`.

## Dataset and evaluator controls

- Version datasets immutably and record content digests, inclusion rules, source
  authorization, split policy, and index version.
- Keep customer content out of git. Store approved evaluation data in governed
  storage and reference it by URI/digest.
- Pin evaluator code/model/prompt/version and calibration evidence.
- Preserve untouched test sets; do not tune thresholds against the final holdout.
- Report every required slice, including regressions hidden by an aggregate.
- Record confidence intervals or another justified uncertainty method where sampling
  matters.
- Require human review for safety-critical, authorization, and ambiguous editorial
  judgments.

## Expected command interface

Versioned datasets and a library-level release-gate engine are implemented. There
is no release-gate CLI or cloud-evaluation suite. These are the exact available
static test interfaces, run from the repository root:

```powershell
python -m pytest tests/unit/release_gates
python -m pytest tests/contract/evaluations
python -m pytest projects/riverside-ai-platform/tests/unit/artifact_validation
python -m pytest projects/riverside-ai-platform/tests/contract/application_api
```

The test source expects failures for missing domains, contradictory decisions,
unsafe evidence URIs/digests, incompatible artifacts, and contract violations. This
task did not run those tests. Production evaluation still needs a composition
command/job that loads all approved datasets/evaluators, produces and stores one
report, binds it to the manifest, and returns nonzero for `hold`/`reject` where
promotion was requested.

## Promotion decision

`promote` means all required metrics pass and the candidate may advance only to the
named next stage. `hold` means evidence is incomplete, uncertain, or awaiting
review. `reject` means a release boundary failed. The manifest and report decisions
and digests must agree.

Automatic promotion from a training notebook is prohibited. Production advancement
requires the named release approver and change authority even when the machine-
readable decision is `promote`.

## Retention

Retain the manifest, report, evidence URIs/digests, dataset/evaluator versions,
raw aggregate results permitted by policy, deployment metadata, traffic allocation,
approvals, and rollback target for the governed audit period. Do not retain prompts,
completions, manuscript text, or customer identifiers merely because an evaluator
used them.
