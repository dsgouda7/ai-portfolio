# FPL squad optimizer: design and evidence

This document describes the implemented local web application and the isolated
opposition experiment. Operational commands are in [README.md](README.md).
An implemented experiment is not the same thing as a validated improvement.

## 1. Current architecture

```text
Archive + official completed player histories + next fixtures
    -> SQLite source tables and runtime metadata
    -> shifted player form, fixture strength, market and external context
    -> fingerprinted model_feature_cache
    -> position-specific XGBoost / CatBoost / LambdaRank / GRU
    -> per-fixture point forecasts and market-value prior
    -> optional weighted ensemble, still at fixture granularity
    -> validated opposition adjustment, if a matching policy exists
    -> sum fixtures into Gameweek points; normalize position scores
    -> eligibility, squad/transfer selection, XI, captain and bench
    -> read-only advice and append-only local squad revisions
```

Training and inference share [model_registry.py](model_registry.py). A manifest
records the completed-data cutoff and feature order. The GRU uses all available
distinct player-Gameweek history; separate target fixtures get separate
inference sequences. Blank Gameweeks project zero points. Double Gameweeks sum
the individual fixture forecasts.

The base models already receive opponent ID, home/away, team and opponent Elo,
Elo difference, recent team/opponent goals for and against, rest and congestion.
The opposition experiment therefore tests an **additional** interaction, not
the introduction of opponent awareness from scratch.

### Selection modes

- Normal new-team generation uses a MILP to maximize the 15-player score, then
  controlled stochastic variations subject to squad/XI quality floors.
- Full advisor generation uses the shared
  [squad_optimizer.py](squad_optimizer.py) MILP to jointly choose a legal squad
  and XI. Its objective is XI points plus `0.001 * squad points`; bench quality
  is a small secondary preference, not an exact lexicographic guarantee.
- The captain and vice-captain are the highest two forecasts in the chosen XI.
  The objective does **not** explicitly optimize the doubled captain score or
  model captaincy uncertainty.
- Existing-squad advice applies beneficial sequential transfers subject to
  budget, club and transfer-count constraints. Its XI helper uses positional
  minimums and normalized flex scores; it is not the joint full-advisor solver.

The web application never submits changes to an official FPL account.

## 2. Decision: reuse opposition-player forecasts at the final scoring step

**Chosen approach:** reuse the same checkpoint's baseline forecasts for both
clubs. Do not add another trained model, change training targets, or call the
player model repeatedly in a feedback loop.

Implementation: [opposition.py](opposition.py).

### Inputs and order

1. Score the entire fixture snapshot with the selected model. For an ensemble,
   blend the component fixture scores first. Market-value weighting has already
   happened inside each component, exactly once.
2. Freeze these baseline scores. Never derive opposition strength from scores
   that have already been dampened.
3. For each club in each fixture, choose a projected legal XI using baseline
   points: one goalkeeper, 3-5 defenders, 2-5 midfielders, 1-3 forwards.
   This is a forecast-based representative XI, **not** a predicted official
   starting lineup or a minutes-probability model.
4. Derive attacking and defensive forecast units from those XIs.
5. Apply one bounded adjustment to each fixture prediction, then sum fixtures
   and perform ordinary team selection.

No account data, actual Gameweek points, realized starts, or realized minutes
are read by this adjustment. Current eligibility filtering happens afterward;
the projected opposition XI itself uses baseline forecasts, not a separate
live lineup service.

### Opposition statistic

For each selected player, divide their nonnegative baseline forecast by the
mean forecast of projected-XI players at the same position. If that position's
mean is zero, use a neutral relative score of one.

- **Attacking unit:** mean relative score of the projected midfielders/forwards.
- **Defensive unit:** mean relative score of the goalkeeper/defenders.
- **Pressure:** the opposing unit divided by the mean corresponding unit across
  all projected club-fixtures in the snapshot.

Goalkeepers/defenders face the opposing attacking unit; midfielders/forwards
face the opposing defensive unit. Position normalization avoids interpreting
an extra midfielder as stronger opposition merely because midfielders have a
different FPL point scale.

These are **FPL-forecast proxies**, not pure football attack/defence ratings.
A defender's attacking returns, save points, minutes expectations and existing
fixture context all affect them. Their incremental value must be measured.

### Dampening rule

```text
excess_pressure = max(opposition_pressure - 1, 0)
factor = max(1 - max_reduction, 1 / (1 + strength * excess_pressure))
adjusted_points = baseline_points * factor       if baseline_points > 0
                  baseline_points                otherwise
```

The first registered candidate uses `strength=0.5` and `max_reduction=0.25`.
Those are explicitly experimental settings, not learned or calibrated
probabilities. Weaker-than-average opposition does not boost scores, negative
scores are not made less negative, and there is no recursive rescoring.

Both sides of a fixture must have enough forecasts for a legal projected XI.
Incomplete coverage, duplicate player/fixture rows, invalid IDs and non-finite
scores are explicit errors, not neutral success-shaped fallbacks.

### Explainability and persistence

When active, the resulting player rows contain:

- `opposition_base_points`: the pre-adjustment Gameweek total.
- `opposition_factor`: adjusted total divided by positive baseline total;
  for multiple fixtures this is a score-weighted effective factor.
- `opposition_fixtures`: opponent and fixture IDs, baseline points, pressure,
  factor and adjusted points for each fixture.
- `opposition_version`: the adjustment algorithm version.

The web header labels the layer `disabled`, `active` or `stale`; player hover
cards expose the baseline and effective factor. Draft/commit JSON retains the
audit fields. Normal selection, transfers, replacements and captain ranking
consume the same final `predicted_points`.

## 3. Validation and rollout contract

The base models already encode opposition context. Applying another difficulty
factor can double-count it and make predictions worse. **No matching validated
policy means no opposition adjustment.**

[evaluate_opposition.py](simulations/evaluate_opposition.py) is a separate
evaluation command; it does not alter the training loop, retrain checkpoints,
refresh source data, or change official/private account state.

### Prospective registration

Before a Gameweek deadline, register a fixed candidate and freeze the fixture
forecasts, eligible player IDs, model/component SHA-256 artifact fingerprint, feature-cache
timestamp, training cutoff and official deadline. The input cache must match
its source fingerprint. Registration refuses to overwrite a frozen forecast.

After FPL marks the event finished and data-checked, score that saved forecast
against official results. Baseline and adjusted teams use the **same**
full-advisor solver, player pool, prices, budget and club constraints. Actual
scoring includes captain/vice fallback and automatic substitutions, with no
chips or transfer hits for these independently generated fresh squads.

The report recomputes player squared errors over the same eligible pool and
records both selected teams. Do not choose a setting after inspecting test
results and then call those same results untouched validation.

### Enablement gate

A candidate can be published only when:

1. At least **five distinct untouched Gameweeks** have completed.
2. Candidate registration, frozen forecasts, model training and frozen feature
   creation all precede each evaluated deadline.
3. Every evaluated snapshot follows its checkpoint's completed training cutoff.
4. Adjusted teams have strictly more total actual FPL points.
5. Total captain bonus points do not regress.
6. Pooled player RMSE does not regress.

Retrospective replays are diagnostics and do not count toward those thresholds.
Five Gameweeks is a minimum operational safeguard, **not** statistical proof
or a guarantee of future performance. Longer, multi-season comparisons remain
desirable. Fresh-squad results also do not establish improved transfer strategy.

The ledger cannot mix candidate settings or checkpoint/component fingerprints.
Use a frozen checkpoint set for a trial; weekly retraining starts a new trial.
Retain each pre-deadline snapshot or registered forecast rather than rebuilding
it after results are known. This deliberately favors verifiable evidence over
convenient retrospective backfilling.

`--publish-if-passed` writes a model-specific sidecar such as
`models_current_ensemble_opposition.json` only after the gate passes. Serving
recomputes the gate and checks the checkpoint/component fingerprint. A retrain
makes an old policy stale: the app logs a warning, exposes that status and uses
baseline predictions until a new policy is validated. Malformed or rejected
policy files produce explicit errors.

### Current trial

The first ensemble candidate has frozen **562 GW8 fixture forecasts** before
the official `2026-10-23T17:30:00Z` deadline. The local evaluation ledger is
`simulations/results/opposition_ensemble_trial_sha256.json`; the frozen forecasts are
`simulations/results/opposition_gw8_forecast_sha256.json`. These are ignored local
artifacts, not portable performance claims.

There are no completed untouched Gameweeks in that trial yet, so the layer
remains disabled. The former GW7 cache/checkpoints were replaced by a normal
local refresh before replay; reconstructing GW7 from the new through-GW7
checkpoint would be an in-sample evaluation and is rejected.

## 4. Documentation reconciliation and known limitations

The following distinctions matter when interpreting the project:

| Older documentation claim | Actual implementation |
|---|---|
| No opposition features | Elo, recent goals and fixture context already feed every position model |
| All selection is greedy | Web generation uses MILP; legacy command-line/simulation selectors remain greedy |
| Full advisor is exactly captain-optimal | It optimizes XI points with a small squad-score preference, then assigns captaincy |
| All features are rolling form | Fixture, price, historical trajectory, market and external features also exist |
| Full-time form density is 1 | `roll5_minutes / 450` gives 0.2 for a 90-minute rolling mean |
| Sparsity interaction is a model input | `tm_market_value_x_sparsity` is computed but is not in current `POS_FEATURES` |
| Every new player is conservatively given zero form | Sparse rolling features are blended toward active positional peers |
| Ensemble quality is measured blend performance | Its current metrics average component training metrics; weights use in-sample RMSE |
| Legacy simulator proves current held-out performance | It uses a fixed checkpoint without enforcing its cutoff, a lagged snapshot, and hindsight XI selection |
| Oracle is an exact maximum | Legacy simulation's oracle squad selector is greedy |
| Every page load refreshes fixture data | It refreshes live eligibility; persisted fixtures/data/models require explicit update |

The legacy simulator also joins actuals by round/element without a season
filter or fixture aggregation. Its historical numbers are not a trustworthy
rollout gate for the current multi-season pipeline or this extension.

Other limitations remain: captaincy has no explicit recent-benching safeguard;
feature-history provenance needs care in retrospective studies; shared
fixture/club outcomes correlate player scores; the dampening proxy does not
model those correlations or a probability distribution of squad points.

No Azure infrastructure is included. The Docker web entry point still uses
Flask's local development bind; production/container serving needs an explicit
network/WSGI configuration. Documentation corrections do not imply that these
unrelated deployment or legacy-evaluation issues have been repaired.
