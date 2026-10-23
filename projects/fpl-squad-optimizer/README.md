# EPL Fantasy Team Generator

For the implemented architecture, opposition-adjustment decision, rollout gate
and documentation/implementation reconciliation, see [DESIGN.md](DESIGN.md).

## Problem statement

**Can a machine learning model — trained entirely on community-sourced, freely available data and running on a personal CPU — pick a Fantasy Premier League squad that performs at least as well as the average human manager?**

This project asks whether position-specific models, given a £100M budget and
FPL squad rules, can produce useful next-Gameweek advice. Manager averages vary
substantially by Gameweek; comparisons should use that event's official average,
not a fixed human-performance band.

**Constraints we set for ourselves:**
- Only freely available, community-maintained data — no paid APIs, no live odds, no premium scouting feeds
- CPU-only training and inference — no GPU, no cloud compute
- Completed-Gameweek training cutoffs and pre-deadline inputs are required;
  retrospective evaluation must additionally verify feature provenance.

**Historical report, not a current validated result:** an earlier experiment
reported 70.7 XI points and 52.5% oracle capture. The legacy simulator has
cutoff, hindsight-lineup and actuals-join limitations, so those figures do not
establish current live accuracy. See [evaluation limitations](#simulation-and-traintest-split).

> **Learning project** — I built this to get hands-on with the full ML lifecycle
> end-to-end: data wrangling, feature engineering, model selection, evaluation,
> and live inference. The domain (football) is one I care about, which kept me
> honest — I could immediately see when the output was wrong and had to fix it
> for real rather than paper over it.
>
> **What I drove:** the core problem framing, the decision to split models by
> position rather than use a single model with a position feature, the rule-based
> eligibility design (using FPL's own `status` and `chance_of_playing_next_round`
> fields), and all feature selection decisions.
>
> **Where Copilot leaned in:** scaffolding the rolling-feature pipeline without
> data leakage, writing the greedy squad-selection with FPL constraints, building
> the pitch UI with hover cards, debugging the FPL player ID reuse bug between
> seasons (Akanji was ID 341 in 2023-24; Karl Darlow inherited that ID in 2024-25).

## Quick start

```powershell
# 1. clone the FPL dataset and set up the venv
.\setup.ps1

# 2. train — downloads every available live player-GW history and trains
#    all five model families from the same completed-Gameweek cutoff
.\.venv\Scripts\python.exe train\train.py --model all

# 3. start the web app and open it in your browser
.\start.ps1

# or start it without automatically opening a browser
.\.venv\Scripts\python.exe fpl-generator\web.py    # http://localhost:5000

# optional legacy report; NOT the opposition rollout validation
.\.venv\Scripts\python.exe simulations\simulate_season.py --test-from 15 --test-to 37
```

`start.ps1` is the recurring one-command launcher. All account input and team
generation remain in the web interface: enter your FPL entry ID under **Load
Public FPL Squad** and press **Import squad and analyze**. The app estimates the
available free transfers from published history and generates that many
beneficial transfer recommendations when possible. Current-Gameweek ownership
remains private until the deadline, so recommendations use the latest public
squad.

> **Update cadence:** data and checkpoints are not refreshed on a timer. Live
> eligibility is fetched whenever a team is generated, while persisted player
> histories, prices, enrichments, and models stay at their last successful
> training cutoff. The vaastav archive generally updates daily during the
> season, and the official FPL API publishes completed data after each
> Gameweek. Refresh after FPL has marked the latest Gameweek as checked.

### Weekly workflow

1. After FPL marks the previous Gameweek data as checked, press **Update &
  retrain** in the web header or run `python train/train.py --model all`.
  Training downloads all current-season
  player histories, refreshes official weekly prices and real historical
  Transfermarkt valuations, rebuilds the SQLite feature cache when needed, and
  writes all five model artifacts with the same temporal cutoff. The web button
  runs one background job at a time against staged copies, keeps the current
  data/models available while it runs, and publishes the database and complete
  artifact set only after every model trains successfully. Progress appears
  beside the button as a phase-labelled percentage bar, and the page reloads
  after publication.
  **Full advisor run** performs the same complete update and training batch,
  then forces the weighted ensemble and jointly optimizes the legal 15-player
  squad plus starting XI. It maximizes combined-model XI points, uses bench
  quality as a small secondary preference, and assigns captain and vice-captain
  to the two highest ensemble projections in that XI.
2. Run `./start.ps1` to start the server and open `/generate-team` in your
  browser. The app loads the selected checkpoint (XGBoost by default), validates
  its cutoff, loads the model-ready SQLite cache, scores the next fixture
  snapshot, and applies live availability rules from the public FPL API.
3. If a local draft or committed squad already exists, the app opens the newest
  revision and rolls it into the target Gameweek. It refreshes prices,
  predictions, injuries, suspensions, bank calculations, and transfer advice
  without changing the committed historical revision.
4. Enter the FPL **entry ID** under **Load Public FPL Squad** and press **Import
  squad and analyze**.
  Completed public Gameweeks, picks, transfers, chip use, points, ranks, bank,
  and squad value are refreshed in SQLite. The latest permanent 15-player squad
  becomes a local draft and is re-scored for the next Gameweek. The app estimates
  the free-transfer allowance from public history and produces a sequential list
  of beneficial legal moves up to that allowance. Free Hit picks are retained
  for history but do not replace permanent ownership.
5. Review the complete recommendation list and the read-only next-Gameweek
  lineup beneath it. The lineup applies the recommended transfers first, then
  selects the best legal starting XI, captain, vice-captain, and ordered bench.
  To explore a different move, use **Find a Replacement** to select any player
  from the current locally recorded squad and see the three highest-projected
  legal same-position alternatives within the available bank and club limit.
  If you already made a transfer in the official portal before the deadline,
  use **Record Portal Transfer Locally** because FPL does not publish that
  current-Gameweek change yet. The local draft, remaining transfer allowance,
  recommendations, and lineup are recalculated after the update.
6. Reproduce the transfers and lineup in **FPL My Team**. This project does not
  submit private account changes.
7. After the next deadline, sync the same public entry ID again. The official
  public picks become the new ownership baseline, and `/performance-review`
  compares committed recommendations with actual points, the official average,
  and the Dream Team.

### What happens when the web app opens

`GET /` redirects to `GET /generate-team`. That request follows this order:

1. Verify the SQLite database and at least one model artifact exist.
2. Resolve `?model=xgboost|catboost|lambdarank|rnn|ensemble`, load the checkpoint, and reject a checkpoint
  whose training rows exceed its completed-Gameweek cutoff.
3. Load `model_feature_cache` if its complete source fingerprint still matches;
  otherwise rebuild and persist the feature frame from the source tables.
4. Score individual upcoming fixtures, apply a matching validated opposition
  policy if present, sum each player's Gameweek points, and remove players who are
  no longer in the league or currently unavailable.
5. Load the newest local draft/commit. If it belongs to an earlier Gameweek, roll
  it forward without mutating the earlier revision. If no state exists, create
  and persist an initial controlled-random draft.
6. Re-score the owned squad, choose a legal model-ranked XI, calculate transfer
  suggestions, flag unavailable squad members, and render the pitch and manager
  controls.

The model selector changes how players are scored; it does not silently replace
an existing owned squad. **New Team** is different: it deliberately creates a
fresh controlled-random 15-player draft under the £100m, 2/5/5/3, and maximum-
three-per-club constraints. Only alternatives retaining at least 95% of the
deterministic optimum's squad and XI predicted totals are eligible. **Replay**
recreates a generated draft from its persisted seed.

For an ongoing official FPL entry, prefer **Sync** and transfer editing over
**New Team**: synchronization preserves actual ownership, acquisition-price
history, bank, public chip use, and the latest official permanent squad.

**New Team** and **Sync** both create drafts. Neither replaces the latest
committed local revision until **Save squad** is pressed.

### Squad persistence and season history

SQLite is the source of truth. The existing FPL database contains two append-only
tables:

| Table | Purpose |
|---|---|
| `squad_versions` | One row per draft or committed save, including season, Gameweek, revision, bank, free transfers, point hit, active chip, and full state JSON |
| `squad_version_players` | The 15 player rows for each version, including lineup/bench order, captaincy, purchase/current/selling prices, prediction, status, and news |
| `fpl_entry_sync` | Public entry ID plus non-personal synchronization metadata |
| `fpl_entry_gameweeks` | Public points, ranks, bank, value, transfers, bench points, and chip by completed Gameweek |
| `fpl_entry_picks` | All 15 public picks per completed Gameweek, including order, multipliers, and captaincy |
| `fpl_entry_transfers` | Public transfer ledger with player IDs, prices, event, and timestamp |

Revisions increase within each season/Gameweek. Loading the page resumes the
working draft/current state using the latest persisted version. A legacy
`saved_squad` table is imported once as the first committed ledger revision.

Readable JSON mirrors are exported to `squads/current_squad.json`,
`squads/draft_squad.json`, and `squads/history/`, but the app always loads from
SQLite. These exports are ignored by Git because they contain local season state.

The history can also be inspected through:

```text
GET /api/squad
GET /api/squad/history
GET /api/squad/history?season=2026-27
```

The state validator enforces current FPL rules: 15 unique players, 2 GK/5 DEF/
5 MID/3 FWD, no more than three from one club, legal starting formations, captain
and vice-captain in the XI, ordered substitutes, up to five saved free transfers,
four-point extra-transfer costs, the 20-transfer weekly limit, current selling
price rules, and one active chip.

## Dataset

| Source | What it provides | How it's used |
|---|---|---|
| [vaastav/Fantasy-Premier-League](https://github.com/vaastav/Fantasy-Premier-League) | Community-maintained archive of FPL player stats. The current season's per-player GW CSV files are the primary training rows; prior seasons contribute rolled-up history features (season-over-season trajectory via `history.csv`) | Primary training data — ingested into SQLite on first run; the FPL API supplements any GWs the submodule hasn't published yet |
| [FPL Bootstrap API](https://fantasy.premierleague.com/api/bootstrap-static/) | Live player list with current status, injury news, price, and FPL's own expected-points estimate | Eligibility filtering at prediction time; prunes non-EPL players from training pool |
| [Transfermarkt historical dataset](https://github.com/dcaribou/transfermarkt-datasets) via [Reep](https://github.com/withqwerty/reep) | CC0 dated valuation changes in EUR, cross-referenced via stable FPL player code | Historical as-of quality prior; missing players remain unavailable and are never imputed |

All three sources are free and require no authentication. The vaastav archive updates daily during the season; re-running `setup.ps1` + `train.py` picks up new data automatically.

## How it works

**Selection target:** predict each player's points in the *next* game week, rank the full pool by predicted points, then apply FPL squad constraints (£100M budget, 2GK+5DEF+5MID+3FWD squad, max 3 players per club) to pick the best 15. Starting XI is chosen by FPL formation rules (1GK, 3+ DEF, 2+ MID, 1+ FWD); the unconstrained flex slot is filled by the highest-scoring player regardless of position.

Upcoming fixtures are scored individually and then summed into each player's
Gameweek projection. A player scheduled twice therefore receives both fixture
opportunities, while a player whose club has no fixture receives zero projected
points. The pitch marks multi-fixture players with a **DGW ×2** badge.

**Models** — choose XGBoost, CatBoost, calibrated pairwise LambdaRank, a deep GRU,
or their inverse-RMSE ensemble in the team-page dropdown. XGBoost and CatBoost
predict point totals from the current pre-fixture row. LambdaRank optimizes
within-Gameweek player ordering and uses later held-out Gameweeks to calibrate
rank scores back onto the points scale. The ranking implementation uses
XGBoost's stable `rank:pairwise` objective. The position-specific two-layer GRUs
consume every available distinct player-Gameweek row and use recurrent dropout
plus gradient clipping. Every checkpoint enforces the same completed-data
cutoff and frozen feature order.

Training writes XGBoost, CatBoost, LambdaRank, GRU and ensemble artifacts.
XGBoost uses `models.joblib` when configured/present, otherwise
`models_current.joblib`; other families use the corresponding suffix or their
`FPL_<MODEL>_MODELS_FILE` override. Each artifact persists its season, completed
Gameweek, internal cutoff, training timestamp, feature order, price coverage and
Transfermarkt provenance. GRU artifacts additionally describe their sequence
policy.

The model-ready temporal frame is persisted in SQLite as `model_feature_cache`, with source fingerprint and row/column audit data in `model_feature_cache_metadata`. Training and web inference reuse it while the underlying player-GW, roster, attribute, Transfermarkt, and refresh metadata remain unchanged; any source change invalidates and rebuilds the cache automatically. `rnn_sequence_index` records every training sample's player, first and last internal Gameweek, sequence length, cutoff, and `all_available` policy. GRU tensors are padded in memory so SQLite stays canonical and independent of a particular scaler or model checkpoint.

Official FPL `value` exists on every historical player-Gameweek row. Transfermarkt valuation changes are loaded from the weekly CC0 `player_valuations` dataset and joined only when the valuation date is no later than the fixture kickoff. The upstream file provides dates rather than publication times, so values become available at end-of-day to prevent same-day lookahead. Players without a real dated valuation remain missing; no mean or synthetic value is used.

`tm_market_value` and `tm_value_available` are direct inputs to every position model. The first is `log10(EUR)` only for a real dated valuation; the second is `1` when that value exists and `0` otherwise. Dense GRU tensors use a neutral standardized placeholder where the value is missing, but the availability mask tells the model that no Transfermarkt observation exists; the placeholder is never treated or persisted as a valuation.

### Deduplicated enrichment

The enriched frame adds each signal once at its closest authoritative source:

- The FPL player-Gameweek feed supplies ownership, transfers in/out/balance,
  price, BPS, recoveries, tackles, defensive contributions, and expected stats.
- The same FPL fixture rows generate pre-match Elo, rolling team attack/defence,
  rest days, and 14-day congestion, avoiding a duplicate fixture provider.
- The Transfermarkt CC0 appearance dataset contributes only mapped **non-Premier-
  League** appearances. Premier League rows are excluded because FPL already
  contains them. Strict pre-kickoff 90/365-day aggregates help new signings.
- Transfermarkt historical valuations remain a separate dated as-of quality
  signal; no valuation or appearance is imputed.

The feature cache is schema-versioned as well as source-fingerprinted, so changes
to formulas invalidate SQLite even when the downloaded source tables are stable.

**Selection** — web new-team generation uses a MILP maximizing the 15-player
score, followed by controlled variations. **Full advisor run** jointly optimizes
squad and XI using `XI points + 0.001 * squad points`; it assigns captaincy
afterward. Existing-squad/simple XI helpers fill formation minimums and then use
position-normalized flex scores. The older CLI/simulation selectors are greedy.

### Isolated opposition experiment

The final-stage layer in [opposition.py](opposition.py) reuses the same model's
frozen forecasts for the opposing club's projected XI. Attacking forecasts
create pressure on your GK/DEF; defensive forecasts create pressure on your
MID/FWD. The experimental factor is bounded to at most a 25% reduction by
default. It is applied once per fixture, after ensemble blending and before
Gameweek aggregation. There is no new training model or recursive prediction.

**It is disabled unless a matching policy passes the prospective validation
gate.** Existing Elo and recent-goals features remain unchanged. See
[the design and limitations](DESIGN.md#2-decision-reuse-opposition-player-forecasts-at-the-final-scoring-step)
for the exact statistic, formula and double-counting risk.

Register a candidate **before** its deadline; use the current target Gameweek:

```powershell
.\.venv\Scripts\python.exe simulations\evaluate_opposition.py `
  --model ensemble --gameweek 8 --register `
  --forecast simulations\results\opposition_gw8_forecast_sha256.json `
  --report simulations\results\opposition_ensemble_trial_sha256.json
```

The initial GW8 registration has already been saved locally. Registration
refuses to overwrite it. After FPL marks that event finished and data-checked:

```powershell
.\.venv\Scripts\python.exe simulations\evaluate_opposition.py `
  --model ensemble --gameweek 8 `
  --forecast simulations\results\opposition_gw8_forecast_sha256.json `
  --report simulations\results\opposition_ensemble_trial_sha256.json `
  --publish-if-passed
```

Publication requires at least five untouched Gameweeks, higher actual team
points, no captain-bonus regression and no pooled player-RMSE regression.
Use a frozen checkpoint set and fixed candidate settings throughout a trial;
different checkpoint/component fingerprints cannot be merged into one ledger.
A retrain invalidates an older policy. One good Gameweek or a retrospective
replay is insufficient. No model or database is modified by these commands;
only forecasts, evaluation reports and a gate-approved policy are written.

The model-specific policy is stored beside its checkpoint, for example
`models_current_ensemble_opposition.json`. The web header shows its status;
active hover cards expose pre-adjustment points and the effective factor.

**Confidence margin** — each selected player shows their predicted points minus the next available player at the same position. Margin < 0.5 means the model was essentially guessing between two players.

### Determinism

Model scoring is repeatable for a frozen feature cache and checkpoint, but every
explicit **New Team** request performs controlled stochastic selection. Player
scores are perturbed using the position model's 80th-percentile absolute error
(falling back to RMSE for older checkpoints), multiple legal squads are
solved, and only variants retaining at least 95% of both the deterministic squad
and starting-XI predicted totals are accepted. The selected seed, quality ratios,
attempt count, and number of accepted alternatives are persisted with the draft,
so a recommendation is varied at generation time but auditable afterward. The
floor is relative because XGBoost and GRU prediction scales differ; human-level
actual performance remains an out-of-sample evaluation target, not a per-GW
guarantee. Configure with `FPL_VARIATION_QUALITY_FLOOR`,
`FPL_VARIATION_NOISE_SCALE`, and `FPL_VARIATION_ATTEMPTS`. The default noise is
10% of positional uncertainty; exploratory MILP failures are skipped, and generation
falls back explicitly to the deterministic optimum if no alternative clears the
floor. Each draft exposes a replay link for its persisted seed.
The same recommendation can also be replayed directly with
`/generate-team?new_team=1&model=<model>&seed=<seed>`.

## Features and why we chose them

Player-form features use **five distinct player-Gameweek totals, shifted one
Gameweek before windowing**. Fixture context belongs to the target fixture;
price, market, external-appearance and historical-trajectory features are not
all rolling averages. `ict_index` is excluded from model inputs; its components
are available separately.

### Goalkeeper
| Feature | Why |
|---|---|
| `saves` (roll5) | Direct FPL points source — each save has a chance of earning a save bonus |
| `expected_goals_conceded` (roll5) | Proxy for defensive pressure; high xGC means fewer clean sheet opportunities |
| `penalties_saved` (roll5) | Rare but high-value event; predictable only through keeper profile |
| `clean_sheets` (roll5) | Rolling clean-sheet rate is the single strongest predictor for GK points |
| `goals_conceded` (roll5) | Actual goals let in, separating luck from xGC |
| `minutes` (roll5) | Playing time — a keeper not starting scores zero |
| `was_home` | Home keepers face fewer shots historically |
| `value` | FPL price as a proxy for consensus quality |
| `opponent_team` | Model learns per-club attacking strength from historical data |

### Defender
| Feature | Why |
|---|---|
| `expected_goals_conceded` (roll5) | Clean sheet probability — the primary points driver for defenders |
| `expected_goals` (roll5) | Set-piece and attacking threat; some defenders score regularly |
| `expected_assists` (roll5) | Fullbacks who create are worth significantly more |
| `clean_sheets` (roll5) | Rolling rate, separates consistent starters from rotation risks |
| `creativity` (roll5) | Attacking contribution — overlapping fullbacks score high here |
| `goals_scored`, `assists` (roll5) | Actual involvement, not just expected |
| `minutes`, `starts` (roll5) | Rotation filter — a defender who starts 60% of games is high-risk |
| `was_home`, `value`, `opponent_team` | Standard context features |
| `tm_market_value` | Quality prior — a world-class CB returning from injury has low form stats but high underlying value |
| `form_data_density` | `roll5_minutes / 450` — quantifies how much data the rolling window has. Low = injury returnee, high = nailed |
| `tm_market_value_x_sparsity` | Computed diagnostic column, not a current `POS_FEATURES` model input |

### Midfielder
| Feature | Why |
|---|---|
| `expected_goals`, `expected_assists` (roll5) | Core attacking output signal |
| `creativity` (roll5) | Key pass frequency — midfielders rank highest on this metric |
| `influence` (roll5) | Overall involvement in high-impact moments |
| `goals_scored`, `assists`, `bonus` (roll5) | Actual returns; bonus points capture involvement not in xG/xA |
| `minutes`, `starts` (roll5) | Rotation is the biggest risk for midfielders — Mo Salah benched is worthless |
| `was_home`, `value`, `opponent_team` | Context |
| `tm_market_value`, `form_data_density` | Quality prior and recent-minutes proxy |

### Forward
| Feature | Why |
|---|---|
| `expected_goals` (roll5) | The primary forward value signal — shot volume × quality |
| `expected_assists` (roll5) | Forwards who also create (Firmino-type) have higher floors |
| `threat` (roll5) | FPL's own shot-threat score — forward-specific, captures speculative shots xG misses |
| `influence` (roll5) | Involvement in decisive moments |
| `goals_scored`, `assists`, `bonus` (roll5) | Actual returns |
| `minutes`, `starts` (roll5) | Forwards are the most-rotated position in the squad |
| `was_home`, `value`, `opponent_team` | Context |
| `tm_market_value`, `form_data_density` | Quality prior and recent-minutes proxy |

## Metrics

The tables below preserve an earlier reported experiment, not a validated
benchmark for the current models. In particular, the legacy simulator chooses
the evaluated XI using actual points. Do not use these figures or the legacy
validation-report page to approve the opposition layer.

| Metric | Value | What it means |
|---|---|---|
| **Reported oracle-capture rate** | **52.5% avg** (33.6–76.8% per GW) | Legacy reported ratio against a heuristic hindsight squad, not a proven optimum or human-manager benchmark |
| Reported XI actual pts/GW | 70.7 pts | Legacy experiment; current simulator chooses this XI with hindsight |
| Reported oracle XI pts/GW | 134.1 pts | Legacy greedy hindsight reference, not a proven performance ceiling |
| Per-player prediction RMSE | 4.28 pts | Average prediction error per player per GW. Consistent with FPL's inherent volatility (a single rotation or clean-sheet flip swings 4–6 pts) |
| Per-GW prediction RMSE | 15.6 pts | How far the team-level total prediction was from reality |

**Before/after: effect of evidence-based tree count (200 → 44)**

| Metric | 200 trees (original) | 44 trees (tuned) | Change |
|---|---|---|---|
| Oracle-capture rate | 51.0% | 52.5% | +1.5 pp |
| Avg our XI pts/GW | 68.7 | 70.7 | +2.0 pts |
| Per-player RMSE | 5.01 pts | 4.28 pts | −15% |
| Per-GW prediction RMSE | 38.1 pts | 15.6 pts | −59% |

An earlier checkpoint reported `form_data_density` as its leading feature.
Feature importance varies by model, position and training run; this observation
is not a general accuracy or causal claim.

## How we chose the number of trees

The original setting of 200 trees was a reasonable starting default. To determine whether it was actually optimal we ran a **temporal walk-forward sweep** using XGBoost's built-in early stopping:

- For each position (GK/DEF/MID/FWD), train on the first 70% of game weeks, validate on the next 10%, step forward 5 GWs, repeat for 5 folds.
- Each fold uses `n_estimators=1000` with `early_stopping_rounds=30` — XGBoost halts automatically when validation RMSE stops improving.
- Record the `best_iteration` for every fold.

Results (see `train/tune_n_estimators.py`):

| Position | Median best_n | p90 best_n | Mean val RMSE |
|---|---|---|---|
| GK | 20 | 22 | 1.85 pts |
| DEF | 35 | 42 | 2.37 pts |
| MID | 22 | 22 | 2.31 pts |
| FWD | 30 | 43 | 2.67 pts |

Early stopping halted at **14–47 trees** across all folds and positions — confirming that 200 trees was overfitting the training set by 4–9×. The **p90 across all positions = 44**, which gives a small buffer above the median without allowing unnecessary extrapolation. That value is now `N_ESTIMATORS` in `train/trainer.py`.

To reproduce:

```powershell
.venv\Scripts\python.exe train/tune_n_estimators.py
```

## Eligibility filter

**What the FPL API gives you for free:**

Every player in `bootstrap-static/elements` has:
- `status` — `a` (available), `d` (doubtful), `i` (injured), `s` (suspended),
  `u` (unavailable / loaned / sold), `n` (not in squad)
- `chance_of_playing_next_round` — 0–100 or null
- `news` — FPL's own injury/availability note (e.g. "Knee injury — Expected back
  24 May", "Has joined Porto on loan for the rest of the season.")
- `ep_next` — FPL's own expected points for the next round (surfaced in the UI
  alongside the model's prediction as a sanity check)

That covers roughly 95% of eligibility decisions without any external data source.

**Eligibility rules:**

```
status in {u, n}                       → definitely out (loaned/sold)
status = i or s                        → definitely out
chance_of_playing_next_round >= 50     → in
chance_of_playing_next_round < 25      → out
everything else (doubtful, no chance given)
    → treated as available (conservative default)
```

The rules are deterministic and require no LLM. Fetching current status still
requires the official FPL API. A failed live eligibility fetch logs a warning
and the web path currently proceeds without that filter.

**New/transferred players and data sparsity:**

New arrivals can have few or no Premier League history rows. The current
feature builder explicitly blends sparse rolling features toward active
positional peers. Real dated Transfermarkt values provide an additional
quality prior; missing market values are not imputed. Low minutes therefore do
not automatically force a zero projection or establish a safe starting role.

The `training_registry` table tracks which players were in each training run,
so `new_players_since_last_run()` returns the IDs that need GW history backfilled
before the next re-train. `ingest_from_fpl_api(new_ids)` does that backfill.

## Simulation and train/test split

An earlier experiment described a 2025-26 GW1-14 training/GW15-37 test split.
The current `simulations/simulate_season.py` does **not** enforce or reproduce
that split. It loads whichever XGBoost checkpoint is configured and keeps it
fixed; it does not retrain per Gameweek.

Important implementation limitations:

- It scores `Game_Week == gw - 1`, whereas current model targets and live
  inference use the target fixture row with already-shifted rolling features.
- It does not verify that evaluated Gameweeks follow the checkpoint cutoff.
- Its actuals lookup filters by round, not season, and does not aggregate
  duplicate/double-fixture player rows.
- `_xi_actual_total` chooses the evaluated XI using **actual** points, which is
  hindsight rather than a forecast-time lineup.
- Its squad and "oracle" selectors are greedy, so the oracle is not an exact
  mathematical optimum.
- It generates independent fresh squads without a transfer/chip strategy.

Consequently its CSV dashboard is a **legacy diagnostic**, not proof of
out-of-sample live performance, and is not used for opposition-policy
publication. These code paths have not been silently rewritten as part of
the opposition extension.

Run the simulation yourself:

```powershell
.venv\Scripts\python.exe simulations/simulate_season.py --test-from 15 --test-to 37 --overwrite
.venv\Scripts\python.exe simulations/metrics.py
```

Historical in-sample training snapshot (not the current checkpoint):

| Position | n rows | Top feature |
|---|---|---|
| GK | 2,464 | form_data_density |
| DEF | 7,290 | form_data_density |
| MID | 8,866 | form_data_density |
| FWD | 2,295 | form_data_density |

Current model-quality cards are in-sample diagnostics. Ensemble weights use
component in-sample inverse squared RMSE, and ensemble metrics currently average
component metrics rather than measuring the blended predictions. Neither
establishes unseen-Gameweek accuracy.

## Limitations

FPL scores are inherently noisy: clean sheets flip on a single goal, bonus points
involve a committee, and rotation keeps minutes unpredictable. The model's job is
to pick better players than naive form-averaging, not to nail exact point totals.

Opponent awareness already includes Elo and recent team/opponent goals as well
as IDs and home/away. Its usefulness still needs out-of-sample measurement.
The new forecast-derived opposition layer is gated because it could double-count
this context. Captaincy lacks an explicit recent-benching safeguard; reported
point forecasts are not calibrated haul probabilities. Shared club/fixture
outcomes also make players' scores correlated.

## Transfermarkt market value

The model includes real dated Transfermarkt values as a quality signal beyond
recent FPL form, both as direct features and a small post-model prior. It is a
proxy for player quality, not evidence that a returning player will start.

### ID mapping

Transfermarkt IDs are resolved via [Reep](https://github.com/withqwerty/reep)
— a public football entity register that cross-references identities across 30+
providers. The key bridge:

```
FPL player.code  →  Reep key_opta_numeric  →  Reep key_transfermarkt
```

`player.code` is the stable FPL identifier (unlike `id`, which changes at season
rollover). Reep's `data/people.csv` is downloaded once, cached locally, and
refreshed every 7 days. ~95% of EPL players have a Transfermarkt ID in Reep.

### New DB table: `player_transfer_values`

| Column | Type | Description |
|---|---|---|
| `fpl_id` | INTEGER PK | FPL player `id` for the current season |
| `fpl_code` | INTEGER | Stable FPL `code` used for Reep lookup |
| `tm_id` | TEXT | Transfermarkt player ID |
| `tm_value_eur` | REAL | Market value in EUR |
| `tm_value_date` | TEXT | ISO date of Transfermarkt's own "last updated" stamp |
| `tm_fetched_at` | TEXT | ISO datetime when we last fetched from Transfermarkt |
| `tm_value_imputed` | INTEGER | Legacy compatibility flag; production rows must always be `0` |
| `tm_retry` | INTEGER | 1 = re-attempt fetch on every subsequent training run |

### Refresh logic (`transfer_values.py`)

`ensure_transfer_values(conn)` is called automatically by `train.py` after
`ensure_new_players()`. It performs three steps:

**Step 1 — ID population (`ensure_tm_ids`):**
Loads Reep `data/people.csv`, builds a `fpl_code → tm_id` lookup, and upserts
Transfermarkt IDs for all players in `players_raw`. No network calls for players
whose `tm_id` is already set and non-null.

**Step 2 — Historical valuation refresh (`refresh_transfer_value_history_from_bulk`):**
- Downloads the weekly CC0 `player_valuations.csv.gz` publication.
- Keeps only mapped Transfermarkt player IDs and real positive EUR values.
- Persists every dated valuation change in `player_transfer_value_history`.
- Converts dates to conservative end-of-day timestamps for leakage-safe joins.

**Step 3 — Current value refresh (`refresh_transfer_values`):**
- Default mode (no flag): fetches only players where `tm_fetched_at IS NULL`
  or `tm_retry = 1`. Skips players with fresh data.
- `--force` CLI flag: re-fetches **all** players regardless of cached date.
- Per-player cadence: 1 second sleep between requests (polite crawl rate).
- **Only updates stored value if the new `tm_value_date` is strictly newer
  than what is already stored** — avoids regressing to stale data.
- **No fallback for missing data**: unavailable values remain `NULL` with
  `tm_retry = 1`; no mean, zero, or other synthetic market value is stored.

### CLI

```powershell
# Check current state
python transfer_values.py --status

# Fetch only new / retry players (default, called by train.py)
python transfer_values.py

# Re-fetch all players regardless of cache
python transfer_values.py --force

# Rebuild the Reep ID mapping from scratch
python transfer_values.py --rebuild-ids
```

### Feature integration

`tm_market_value` (log₁₀ EUR, added to all 4 position feature sets) gives the
models a position-independent quality signal. Values are joined strictly as of
each fixture date. Missing observations remain missing and are paired with an
explicit availability flag; no current value is imputed into historical rows.

#### Edge case: players with sparse rolling stats

A fully fit Salah has 5 rich GW rows — the model should trust his rolling form
almost exclusively. For sparse histories, the implementation blends rolling
features toward active positional peers and may apply a real market-value prior.
Tree models can learn feature interactions; the explicit sparsity interaction
below is computed for diagnostics but is not in current `POS_FEATURES`.

```
form_data_density         = roll5_minutes / (5 × 90)  # 90-minute mean gives 0.2
tm_market_value_x_sparsity = tm_market_value × (1 − form_data_density)
```

| Scenario | density | x_sparsity |
|---|---|---|
| Elite player, just returned from injury | 0.0 | = `tm_market_value` (full signal) |
| Rotation player, 45-minute rolling mean | 0.1 | = 0.9 × `tm_market_value` |
| Full-time starter, 90-minute rolling mean | 0.2 | = 0.8 × `tm_market_value` |
| Player with no TM data | any | Missing, with an explicit availability mask |

`form_data_density` is a recent-minutes proxy, not a literal count of available
observations. Sparse-form blending uses `density * 5`, giving a full-time
starter a blending weight of one.


## Model selection

The original evaluation established XGBoost as the point-regression baseline:

| Approach | Outcome |
|---|---|
| **Linear regression** | Fast and interpretable, but the FPL scoring surface is highly non-linear — a clean sheet is a step function, bonus points follow a committee decision, and the interaction between minutes played and all other stats is multiplicative. Underfit systematically on high-variance scorers and produced negative point predictions for rare events. |
| **Neural network (MLP)** | Marginally better calibration in controlled tests, but required 10–15× the wall-clock training time for negligible accuracy gain on the same features. The training set (~8,000–9,000 rows per position) is too small to avoid overfitting without heavy regularisation that erased the gain. Compute cost vs ROI was unfavourable on a CPU-only dev machine. |
| **XGBoost** | Strong baseline accuracy across all four positions. Handles missing values natively, trains quickly, and exposes feature importances. |

The production selector now also includes CatBoost, calibrated pairwise ranking,
position-specific two-layer GRUs, and an inverse-RMSE weighted ensemble. These
models share the same cutoff manifest and leakage-safe feature cache.

Position z-scores remain part of simple existing-squad/legacy XI selection.
Full-advisor joint optimization instead uses absolute predicted points under
explicit formation constraints.

The per-GW breakdown and interactive per-position RMSE are in the `/validation-report` web UI. Run the simulation first, then open `http://localhost:5000/validation-report`.

## Web UI — routes

| Route | Description |
|---|---|
| `GET /generate-team` | Recommended FPL squad for the current GW on an interactive pitch with predicted scores, confidence margins, and player health cards |
| `GET /api/squad` | Latest committed and draft squad states loaded from SQLite |
| `GET /api/squad/history` | Append-only season/Gameweek revision ledger; accepts an optional `season` query parameter |
| `POST /api/fpl-entry/sync` | Localhost-only synchronization of public history and the latest publicly visible squad using only `entry_id` and `model` |
| `POST /api/training/start` | Locally start one background latest-data refresh and all-model training job; accepts an empty JSON object |
| `GET /api/training/status` | Poll current refresh/training state and the latest bounded progress log |
| `POST /api/squad/draft` | Validate and append a draft revision |
| `POST /api/squad/commit` | Validate and append a committed revision plus readable JSON export |
| `GET /performance-review` | Scores each final committed Gameweek squad with actual results, captaincy, chips, transfer hits, and automatic substitutions; compares it with the official average and Dream Team |
| `GET /validation-report` | Side-by-side pitch: model team vs oracle optimal per simulated GW, with full metrics dashboard. Requires simulation results. |
| `GET /` | Redirects to `/generate-team` |

### Syncing to an official FPL account

This project does **not** collect an email, password, MFA secret, refresh token,
or session cookie, and it does not submit changes to the official FPL website.
There is no documented public FPL write API. The application deliberately uses
only public entry endpoints and does not accept access tokens or other private
credentials.

Keep account credentials confidential and review the current FPL terms before
using automated integrations. The supported workflow is human-in-the-loop: review the recommendation
locally, then reproduce the confirmed transfers, lineup, captaincy, and chip in
the official FPL app or website. Do not provide passwords, MFA secrets, access
tokens, refresh tokens, or Premier League cookies to this project.

The **Load Public FPL Squad** panel accepts a public numeric FPL entry ID. After each
deadline it downloads the public entry history, transfers, and all published
Gameweek picks; stores normalized records in `fpl_entry_gameweeks`,
`fpl_entry_picks`, and `fpl_entry_transfers`; imports the latest permanent squad
as a model-rescored local draft; and preserves official XI, bench order,
captaincy, bank, and public chip history in the sync tables. It estimates the
next free-transfer allowance from that history, then generates sequential
beneficial transfers under the budget and three-per-club constraints. Those
transfers are applied to a projected squad before the app selects and displays
the next legal XI, ordered bench, captain, and vice-captain from current model
scores. If the latest event used Free Hit, ownership comes from the preceding
non-Free-Hit event.

FPL does not publicly expose pre-deadline picks or the exact private live free-
transfer balance. The app therefore labels the imported pitch as the last public
squad and the free-transfer count as an estimate. Portal transfers made after
that public Gameweek can be recorded explicitly in the local draft without
altering the last-public-squad view; this consumes one estimated transfer and
recalculates advice, but does not claim that FPL confirmed the private change.
Personal profile names, email, password, tokens, cookies, and MFA data are never
requested or persisted. The sync endpoint rejects private credential fields,
non-local and cross-origin requests, and requests repeated within a short
cooldown.

The validation report page shows:
- Interactive GW selector (prev/next buttons or click any GW badge)
- Side-by-side pitch view: our generated team on the left, oracle optimal on the right — each player circle coloured by position with actual GW points shown
- Per-GW score row: our XI pts, oracle pts, gap, % of oracle
- Summary metrics panel (aggregate over all simulated GWs)
- Per-position prediction RMSE chips
- Scrollable per-GW results table with click-to-select

The server checks all prerequisites on every request and returns a styled error page with the exact fix command if any setup step has been skipped.

## Container deployment

The project is structured as four independent container images — three pipeline steps and one long-running web service — sharing a common I/O contract via environment variables.

```
┌────────────────────────────────────────────────────────────────────┐
│                        pipeline-data volume                        │
│                                                                    │
│  ┌─────────┐    fantasy_football.db    ┌───────┐    models.joblib  │
│  │ ingest  │ ──────────────────────► │ train │ ──────────────►    │
│  └─────────┘                          └───────┘                    │
│                                           │                        │
│                                           │ models.joblib +        │
│                                           │ fantasy_football.db    │
│                                           ▼                        │
│                                      ┌──────────┐                 │
│                                      │ simulate │ ── results/ ──► │
│                                      └──────────┘                 │
│                                                                    │
│  ┌───────┐ ◄── DB + models + results ──────────────────────────── │
│  │ serve │  http://…:5000/generate-team                           │
│  └───────┘  http://…:5000/validation-report                       │
└────────────────────────────────────────────────────────────────────┘
```

### Run locally with Docker Compose

```powershell
# Full pipeline (ingest → train → simulate) then start web server
docker compose up

# Re-train only (ingest already ran, DB exists on volume)
docker compose run --rm train

# Web server only (pipeline artifacts already on volume)
docker compose up serve

# Reset everything and start fresh
docker compose down -v && docker compose up
```

### Images and env vars

| Image | Dockerfile | Key inputs | Key outputs |
|---|---|---|---|
| `fpl-ingest` | `pipeline/ingest/Dockerfile` | `FPL_VAASTAV_DIR` | `FPL_DB_FILE` |
| `fpl-train` | `pipeline/train/Dockerfile` | `FPL_DB_FILE` | `FPL_MODELS_FILE` |
| `fpl-simulate` | `pipeline/simulate/Dockerfile` | `FPL_DB_FILE`, `FPL_MODELS_FILE`, `FPL_SIM_FROM/TO` | `FPL_RESULTS_DIR` |
| `fpl-serve` | `fpl-generator/Dockerfile` | `FPL_DB_FILE`, `FPL_MODELS_FILE`, `FPL_RESULTS_DIR` | HTTP on port 5000 |

All env vars default to local paths so local runs need no configuration.

### Hooking up to Azure ML

The pipeline steps could be wrapped in
[Azure ML command components](https://learn.microsoft.com/en-us/azure/machine-learning/component-reference/component-reference).
That integration is not implemented or verified here. It requires orchestration,
identity, artifact mounting and serving configuration; the web entry point
currently binds Flask's local development server, not a production/container
network interface.

**Mapping: container step → AML component**

```
pipeline/steps/ingest.py   →  CommandComponent (environment: fpl-ingest image in ACR)
                               inputs:  vaastav_dir  (FileDataset, mounted)
                               outputs: db_dir       (PipelineData)
                               env:     FPL_VAASTAV_DIR=${{inputs.vaastav_dir}}
                                        FPL_DB_FILE=${{outputs.db_dir}}/fantasy_football.db

pipeline/steps/train.py    →  CommandComponent (environment: fpl-train image in ACR)
                               inputs:  db_dir       (from ingest output)
                               outputs: models_dir   (PipelineData)

pipeline/steps/simulate.py →  CommandComponent (environment: fpl-simulate image in ACR)
                               inputs:  db_dir, models_dir
                               outputs: results_dir  (PipelineData)
```

**Deployment steps (infra not included here):**
1. Build and push images to Azure Container Registry: `az acr build --registry <acr> --image fpl-ingest .`
2. Register the vaastav clone as an AML `FileDataset` (or point at an Azure Blob mount)
3. Wrap each `pipeline/steps/*.py` in a `CommandComponent` YAML referencing the ACR image and wiring `inputs`/`outputs` to the env vars above
4. Define an `@pipeline` function composing the three components in sequence
5. Submit the pipeline; AML handles compute provisioning, data lineage, and run tracking
6. Deploy `fpl-serve` as an Azure Container App (or AKS deployment) reading artifacts from the AML-registered output datasets

The web app (`fpl-serve`) is intentionally kept outside the pipeline — it is a long-running service, not a batch step, and should be deployed separately once the pipeline has produced its artifacts.

## Authorship

This project was co-authored with AI (GitHub Copilot / Claude).

**Core ML and design decisions were driven by me:**
- Problem framing: predict next-GW FPL points per player, not match outcomes
- Model architecture decision: 4 separate XGBRegressors by position rather than a single model with a position feature — eliminates cross-position noise
- Evaluation design: oracle-capture rate as the primary held-out metric — measures how close we get to the theoretically best squad, not just raw prediction error
- Eligibility design: rule-based filter using FPL's own `status` + `chance_of_playing_next_round` fields, covering 100% of cases deterministically
- Feature set curation per position: rejected `ict_index` (linear combination of sub-features already present), chose `form_data_density` × `tm_market_value` cross-term to handle injury returnees
- Decision to hold out GWs 15–37 and compare against oracle baseline rather than only reporting training-set metrics

**AI provided:**
- Scaffolding the rolling-feature pipeline (shift-before-window to prevent leakage)
- Writing greedy squad selection with FPL constraints (budget, per-club cap, minimum formation rules)
- Building the pitch UI with interactive hover cards and bench
- Debugging the FPL player ID reuse bug (Akanji→Darlow, `id` changes at season rollover; `code` is stable)
- Designing the three-layer eligibility filter loop (DB prune → training filter → live inference filter)
- Building the simulation framework (`simulations/simulate_season.py`, `simulations/metrics.py`)
- Building the validation report web UI (`/validation-report`)

We also considered linear regression and neural networks before settling on XGBoost. Linear regression underfit the non-linear scoring surface; neural networks showed marginal gains at 10–15× compute cost on a dataset too small (~9K rows per position) to avoid overfitting. XGBoost gave the best accuracy / compute-cost / interpretability trade-off for this problem.
