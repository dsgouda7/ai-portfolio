# Evaluation, Rollout, Rollback, and Change Plan

## Document control

| Field | Value |
|---|---|
| Artifact IDs | `EVAL-CAP-01`, `ROL-CAP-01`, `ROL-CAP-02` |
| Version / status | `[TODO] / DRAFT` |
| Evaluation / release / workflow / operations owners | `[TODO]` |
| Candidate, baseline, dataset, evaluator, index, and release IDs | `[TODO]` |
| Scope / exclusions / revalidation trigger | `[TODO]` |

## Evaluation contract

| Domain | Dataset/slices | Metric or decision rule | Threshold fixed before scoring | Uncertainty/reviewer | Critical/non-compensating? | Failure owner | Evidence tier |
|---|---|---|---|---|---|---|---|
| Data quality | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` |
| Retrieval | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` |
| Generation/citation | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` |
| Adaptation | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` |
| Safety/authorization | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `YES` | `[TODO]` | `[TODO]` |
| Operational SLO | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` |
| Cost | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` |
| Rollout comparison | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` |

Separate retrieval from generation with a gold-context or equivalent ablation.
Include unsupported/answerability, tenant/ACL, current/superseded, regional policy,
rights-evidence, deletion, and committed-action slices. A small or biased sample
produces `HOLD`, not optimistic promotion.

## Evidence-tier plan

| Tier | Population/environment | Question answered | Entry gate | Exit evidence | What it cannot prove |
|---|---|---|---|---|---|
| Static/fixture | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` |
| Offline candidate | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` |
| Cloud smoke | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` |
| Bounded load | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` |
| Shadow | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` |
| Canary | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` |

## Rollout cohorts and change controls

| Stage/cohort | Users/traffic/mode | Entry gate | Observation window | Health/adoption signals | Exit/ramp rule | Abort/rollback trigger | Decision authority |
|---|---|---|---|---|---|---|---|
| Offline | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` |
| Shadow | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` |
| Champion canary | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` |
| Imprint/regional canary | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` |
| Broad availability | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` |

`GO` advances only to the named next stage. `HOLD` preserves current exposure.
`ABORT` stops advancement. `ROLLBACK` routes new work to a named prior state.

## Rollback, reconciliation, and compensation

| Failure | Stop new exposure | Known-good release/index/policy | Already committed action? | Reconcile/correct/compensate | Evidence preserved | Owner/approver | Revalidation |
|---|---|---|---|---|---|---|---|
| `[TODO]` | `[TODO]` | `[TODO]` | `[YES/NO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` |

Do not present traffic rollback as reversal of a PageTurn write or other external
side effect.

## Customer change and communication plan

| Audience/cohort | Message purpose | Known facts | Unknowns/limitations | Training/support | Feedback path | Communication owner | Next update/decision |
|---|---|---|---|---|---|---|---|
| `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` |

## Go/no-go record

| Field | Decision |
|---|---|
| Named next stage | `[TODO]` |
| Evidence package and claim IDs | `[TODO]` |
| Passed gates | `[TODO]` |
| Holds/failures and dispositions | `[TODO]` |
| Exposure limits and automatic stop conditions | `[TODO]` |
| Known-good rollback target | `[TODO]` |
| Authorities and approvals | `[TODO]` |
| Decision / expiry / revalidation trigger | `[TODO]` |
