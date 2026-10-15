# Worked Teaching Example: Source-Only Hold

> **Teaching-only, not passing evidence.** This example demonstrates classification and decision wording from the repository's current source state. It is not a learner submission, contains no executed notebook result, and must not be copied into `candidate-evidence/` as proof.

## Review Scope

| Field | Teaching value |
|---|---|
| Candidate | `rel-riv-002` from the synthetic Riverside scenario |
| Requested stage | Local evidence review only |
| Fixture contract | `ai-engineer-fixtures.v1` |
| Runtime state | No AI Engineer notebook result retained by this example |
| Cloud state | No live Azure result retained by this example |

## Minimal Evidence Index

| Claim | Source | Correct class | Supported wording |
|---|---|---|---|
| Shared fixture bytes are pinned | `../shared/VERSION` and `../shared/fixture-manifest.json` | `IMPLEMENTED_SOURCE` | A version and digest manifest exist in source |
| Prompt fixture is expected to reject the candidate | `../shared/prompt-release/EXPECTED_OUTCOMES.md` | `IMPLEMENTED_SOURCE` | The authored fixture contract specifies a critical-slice rejection; this example did not execute it |
| Riverside has Azure ML, APIM, contract, evaluation, and IaC assets | `../../../projects/riverside-ai-platform/` | `IMPLEMENTED_SOURCE` | Production-shaped source assets exist |
| The candidate passes local gates | No retained run | `UNVALIDATED` | No passing local claim is supported |
| The candidate works on Azure | No retained cloud evidence | `UNVALIDATED` | No Azure behavior claim is supported |

## Gate Decision

| Gate | Status | Reason | Next discriminating test |
|---|---|---|---|
| Data, prompt, operations, lineage, drift | `hold` | Expected outcomes and notebook source are not executed evidence | Run the owning notebook in an approved local environment and retain versioned outputs |
| Retrieval and generation | `hold` | No release-bound retrieval/generation report is attached | Produce separate reports plus a gold-context ablation |
| Azure mapping | `hold` for live claims | Source mapping exists; runtime behavior is absent | Run the exact authorized smoke, policy, load, telemetry, and teardown checks required for the requested stage |

## Decision Record

**Decision: `hold`.** The repository contains enough source to define the review, but this teaching example contains no retained execution evidence. It therefore cannot recommend promotion, cannot receive a passing capstone score, and cannot assert local gate success or Azure behavior.

The important pattern is the chain: source existence supports `IMPLEMENTED_SOURCE`; an authored expected outcome remains source; only a retained run can support `LOCAL_FIXTURE`, `LOCAL_MEASURED`, `STATIC_VALIDATION`, or `LIVE_AZURE` as applicable.
