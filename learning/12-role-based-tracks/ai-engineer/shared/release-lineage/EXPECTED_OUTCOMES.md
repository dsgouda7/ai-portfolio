# Release Lineage Expected Outcomes

## Semantic Invariants

A manifest is promotable only when all of the following hold:

1. Every referenced artifact and report has a stable ID and SHA-256 or report ID.
2. `adapter.compatible_base_model_id` equals `base_model.artifact_id`.
3. Every gate result is true.
4. A non-initial accepted release has an accepted rollback target.
5. A blocked release records at least one reason and must not become the active deployment.

The repeated hexadecimal digests are deterministic teaching values, not claims about files in the repository.

## Expected Decisions

| Release | Expected decision | Reason |
| --- | --- | --- |
| `rel-riv-001` | Accept as initial release | All gates pass; no prior rollback target exists |
| `rel-riv-002` | Accept as current release | All gates pass; rollback target is `rel-riv-001` |
| `rel-riv-003` | Block | Adapter requires `model-smollm2-360m-instruct`, but the manifest supplies `model-smollm2-135m-instruct` |

The passing evaluator report on `rel-riv-003` must not override compatibility failure. Promotion is an AND across gates, not a majority vote.

## Lineage Queries

- Every request in `../latency-cost/request-traces.jsonl` was answered or attempted by `rel-riv-002`.
- `rel-riv-002` resolves to base `model-smollm2-135m-instruct` at `rev-001`, adapter `adapter-riv-sft-002` at `v2`, dataset `dataset-riv-curated-002`, prompt `prompt-riv-001`, index `index-riv-policy-002`, and evaluator report `eval-report-riv-002`.
- The immediate rollback target for `rel-riv-002` is `rel-riv-001`.
- `rel-riv-003` must never answer a request because its base/adapter pair is incompatible.

Azure ML or Microsoft Foundry may store these entities in different registry resources, but the local manifest remains the portable evidence contract. Cloud registration is not validated by this fixture.
