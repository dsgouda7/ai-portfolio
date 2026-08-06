# Assessment Rubric

## Scoring Principles

The capstone is scored out of 100. Evidence quality, traceability, and honest limits matter more than a favorable release outcome. A justified `hold` or `reject` can receive 100 points.

No points are awarded merely because a file exists, a notebook was run, a command is shown, or an architecture diagram names Azure services.

## Rubric

| Area | Points | Full-credit evidence |
|---|---:|---|
| C0. Scope and evidence index | 8 | One immutable candidate ID, baseline, rollback target, source commit, environment/workload, package index, artifact digests, evidence classes, owners, and contradictions |
| C1. Data quality and lineage | 12 | Structural and semantic checks, leakage, PII, provenance/rights, contamination, slices, preference risk, issue ledger, curation actions, source/candidate digests, and gate decision |
| C2. Retrieval evaluation | 10 | Versioned cases/evaluator, predeclared thresholds, recall/ranking/authorization/unsupported-query slices, baseline/candidate comparison, uncertainty, and retrieval decision |
| C2. Generation and citation evaluation | 10 | Task, groundedness, citation, refusal, safety/policy metrics; gold-context ablation; evaluator limits; separate generation decision |
| C3. Prompt comparison and release control | 10 | Complete pinned bundles, measured diff, paired outcomes, critical slices, uncertainty, exposure-state honesty, rollback evidence, and decision |
| C4. Local operational SLO report | 10 | Reconciled stage ledger, named populations and percentile method, TTFT/TPOT, useful throughput, retry/cache/token/cost economics, bottleneck, next test, and local-only limits |
| C5. Release manifest and compatibility | 10 | Immutable references across model/adapter/tokenizer/data/prompt/index/evaluators/runtime, AND-composed gates, report binding, request lineage, and valid rollback graph |
| C5. Azure architecture mapping | 8 | Contract-to-service map, Azure ML/APIM/identity/data/index/monitoring boundaries, live checks, ownership, region/residency status, and no cloud overclaim |
| C6. Drift and iteration decision | 10 | Seven drift lenses, counts/denominators, uncertainty, multi-label clusters, reviewed case lineage, five-way action decision, follow-up test/window, and rollback target |
| C6. Unsupported-claims ledger | 7 | Complete claim inventory, evidence-safe wording, missing evidence, owner, scope, expiry/revalidation trigger, and explicit unvalidated status |
| C7. Final decision and defense | 5 | Decision follows all non-compensating gates, cites evidence, exposes residual risk, and names the cheapest tests that could change it |

## Evidence Quality Scale

Apply this scale within each rubric area:

| Level | Credit | Description |
|---|---:|---|
| Trustworthy | 100% | Complete, internally consistent, reproducible/auditable, correctly classified, and explicit about limits |
| Useful with gaps | 70% | Main evidence is present, but one important identity, digest, slice, denominator, threshold, or limitation is weak |
| Directional | 40% | Some analysis exists, but evidence cannot support the claimed decision without substantial reviewer inference |
| Unsupported | 0% | Missing, contradictory, fabricated, mislabeled, or stronger than its evidence |

## Honesty Credits

These behaviors earn full rather than reduced credit when technically justified:

- rejecting a candidate that improves aggregate quality but fails a critical slice;
- returning `hold` because sample size, reviewer agreement, or evaluator validity is insufficient;
- marking a promised upstream artifact absent when the source does not currently produce it;
- distinguishing expected fixture outcomes from executed local evidence;
- distinguishing local fixture, local measured, modeled, static, and live Azure results;
- selecting multiple parallel actions when failures have different owners;
- choosing no configuration change until a discriminating test isolates the bottleneck;
- refusing to call a source asset deployed, secure, scalable, compliant, cost-optimized, or production ready.

## Score Caps

Apply the lowest relevant cap after calculating points:

| Condition | Maximum score |
|---|---:|
| Missing unsupported-claims ledger | 59 |
| Claims Azure deployment, production readiness, security, residency, capacity, cost, or rollback without matching live evidence | 49 |
| Missing release manifest or no immutable candidate release ID across artifacts | 59 |
| Retrieval and generation results are combined so failure ownership cannot be determined | 69 |
| No gold-context/equivalent ablation and no other discriminating localization test | 79 |
| Critical safety/authorization regression is averaged away or treated as compensable | 49 |
| Data blockers are silently deleted or relabeled without a curation ledger | 59 |
| Local fixture/simulation numbers are presented as production SLO or Azure performance | 49 |
| Final decision contradicts a known failed non-compensating gate | 49 |
| Evidence contains secrets, credentials, raw customer content, or prohibited identifiers | 0 pending incident review |

## Automatic Rejection Findings

The submission cannot recommend promotion while any of these remains true:

- unresolved cross-split leakage, unredacted PII, unknown/unapproved rights, eval-reserved training content, invalid chat targets, or unresolved preference disagreement;
- incompatible base, adapter, tokenizer, model profile, precision, or serving runtime;
- missing or contradictory required evaluation domain;
- confirmed cross-tenant/ACL disclosure, policy false allow, or critical safety regression;
- mutable or dangling rollback target;
- unreconciled latency or cost ledger used for an operational claim;
- report/manifest release ID, decision, version, or digest mismatch;
- evidence class is `UNVALIDATED` for a gate required by the requested rollout stage.

## Reviewer Questions

Use these questions during the final defense:

1. Which single artifact lets me identify the exact release that served a request?
2. Which check would falsify your data gate?
3. Show one failure localized between retrieval and generation.
4. Which critical slice can block an aggregate improvement?
5. What denominator is used for retry amplification and cost per success?
6. Which Azure statement is only modeled, and what live test would promote it?
7. Why is the selected drift intervention smaller or safer than fine-tuning first?
8. What result would change your final decision?
