# EPL Fantasy Team Generator

## Problem statement

**Can a machine learning model — trained entirely on community-sourced, freely available data and running on a personal CPU — pick a Fantasy Premier League squad that performs at least as well as the average human manager?**

The average FPL manager scores roughly 50–55 points per game week. The best managers — those in the top 10k globally — score closer to 65–70. Both groups make decisions based on intuition, recency bias, and partial information. This project sets a concrete, measurable target: build a model that, given the same £100M budget and the same FPL squad rules, assembles a team that captures at least as much of the theoretically optimal score as an experienced human manager would.

**Constraints we set for ourselves:**
- Only freely available, community-maintained data — no paid APIs, no live odds, no premium scouting feeds
- CPU-only training and inference — no GPU, no cloud compute
- No lookahead: the model is only ever trained on data from game weeks that have already been played; prediction uses only information available before the GW kicks off

**Result:** across 23 held-out game weeks (GW 15–37), the model averaged **68.7 actual points per XI** — capturing **51% of the oracle-optimal squad**, right in the range of an experienced human FPL manager (45–60%). See the [validation report](#simulation-and-traintest-split) for the full breakdown.

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
./setup.ps1

# 2. train — fetches live EPL members, prunes non-EPL players from the DB,
#    trains 4 XGBRegressors, saves models + EPL snapshot to models.joblib
python train/train.py

# 3. run the web app
python fpl-generator/web.py    # → http://localhost:5000

# optional: walk-forward out-of-sample evaluation
python simulations/simulate_season.py --test-from 15 --test-to 37
```

> **Re-train whenever the FPL dataset updates.** The vaastav repo updates daily
> during the season. `setup.ps1` runs `git pull` on it; then `python train/train.py`
> re-prunes the DB and rebuilds the models in one step.

## Dataset

| Source | What it provides | How it's used |
|---|---|---|
| [vaastav/Fantasy-Premier-League](https://github.com/vaastav/Fantasy-Premier-League) | Community-maintained archive of FPL player stats. The current season's per-player GW CSV files are the primary training rows; prior seasons contribute rolled-up history features (season-over-season trajectory via `history.csv`) | Primary training data — ingested into SQLite on first run; the FPL API supplements any GWs the submodule hasn't published yet |
| [FPL Bootstrap API](https://fantasy.premierleague.com/api/bootstrap-static/) | Live player list with current status, injury news, price, and FPL's own expected-points estimate | Eligibility filtering at prediction time; prunes non-EPL players from training pool |
| [Transfermarkt via Reep](https://github.com/withqwerty/reep) | Market value in EUR for each player, cross-referenced via FPL player code | Quality prior signal for the model — especially useful for injury returnees with sparse rolling stats |

All three sources are free and require no authentication. The vaastav archive updates daily during the season; re-running `setup.ps1` + `train.py` picks up new data automatically.

## How it works

**Selection target:** predict each player's points in the *next* game week, rank the full pool by predicted points, then apply FPL squad constraints (£100M budget, 2GK+5DEF+5MID+3FWD squad, max 3 players per club) to pick the best 15. Starting XI is chosen by FPL formation rules (1GK, 3+ DEF, 2+ MID, 1+ FWD); the unconstrained flex slot is filled by the highest-scoring player regardless of position.

**Models** — four XGBRegressors, one per position. Using separate models rather than a single model with a position feature eliminates cross-position noise (`saves` is always zero for outfield players; xGC is irrelevant for a striker). Target: `total_points` for the next game week.

**Selection** — greedy by predicted points. Flex spots (positions not constrained by formation minimums) are filled by z-score within position (`predicted_points_norm`), so an exceptional defender is correctly preferred over an average forward for the last outfield slot.

**Confidence margin** — each selected player shows their predicted points minus the next available player at the same position. Margin < 0.5 means the model was essentially guessing between two players.

## Features and why we chose them

All features use a **5-game-week rolling average, shifted one GW before the window** — the model never sees data from the GW it's predicting. `ict_index` is excluded because it's a linear combination of creativity, threat, and influence, all of which are already present separately.

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
| `tm_market_value_x_sparsity` | `market_value × (1 − density)` — the market value signal only matters when form data is sparse |

### Midfielder
| Feature | Why |
|---|---|
| `expected_goals`, `expected_assists` (roll5) | Core attacking output signal |
| `creativity` (roll5) | Key pass frequency — midfielders rank highest on this metric |
| `influence` (roll5) | Overall involvement in high-impact moments |
| `goals_scored`, `assists`, `bonus` (roll5) | Actual returns; bonus points capture involvement not in xG/xA |
| `minutes`, `starts` (roll5) | Rotation is the biggest risk for midfielders — Mo Salah benched is worthless |
| `was_home`, `value`, `opponent_team` | Context |
| `tm_market_value`, `form_data_density`, `tm_market_value_x_sparsity` | Same quality-prior pattern as DEF |

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
| `tm_market_value`, `form_data_density`, `tm_market_value_x_sparsity` | Quality prior — crucial for elite strikers returning from long injuries |

## Metrics

| Metric | Value | What it means |
|---|---|---|
| **Oracle-capture rate** | **51% avg** (45–82% per GW) | % of the theoretically best squad score the model recovers. Human managers in the 45–60% range. |
| Avg our XI actual pts/GW | 68.7 pts | Actual points scored by the model's starting XI, averaged over 23 held-out GWs |
| Avg oracle XI pts/GW | 134.1 pts | Hindsight-best squad under same constraints — the performance ceiling |
| Per-player prediction RMSE | 5.01 pts | Average prediction error per player per GW. Consistent with FPL's inherent volatility (a single rotation or clean-sheet flip swings 4–6 pts) |
| Per-GW prediction RMSE | 38.1 pts | How far the team-level total prediction was from reality |

Top feature in all 4 models: **`form_data_density`** — confirming that data sparsity (who is newly returned, newly arrived, or rotating) is more predictive than any single stat.

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

This covers the full player pool deterministically — no external services required.

**New/transferred players and data sparsity:**

Players who joined mid-season appear in the live API but have sparse GW history
(few or zero rows in `player_gw`). The rolling features for those players default
to zero or near-zero. The model then predicts low points for them — which is the
correct conservative behaviour: don't confidently pick a player you have no form
data on. The 5-GW window naturally provides a data-density signal without any
explicit handling needed.

The `training_registry` table tracks which players were in each training run,
so `new_players_since_last_run()` returns the IDs that need GW history backfilled
before the next re-train. `ingest_from_fpl_api(new_ids)` does that backfill.

## Simulation and train/test split

**Train:** GWs 1–14 of the 2025-26 season (~8,400 player-GW rows across 4 position models).
**Test (held-out):** GWs 15–37 — 23 game weeks the model never saw during training.

For each held-out GW the model builds the feature pool from all prior GWs, selects a squad under
FPL constraints (£100M budget, 3-per-club cap, formation rules), and also selects an oracle squad
using actual points as the target. The oracle is the hindsight upper bound — the best squad
possible under the same constraints with perfect knowledge. The gap is attributable to prediction
error alone.

**Simulation assumptions (what this is and isn't):**

| Assumption | What we did | What a stricter setup would do |
|---|---|---|
| **Static model** | `models.joblib` is trained once on GWs 1–14, then held fixed for all 23 test GWs | Re-train at each GW using only data up to GW-1 (true rolling walk-forward) |
| **Lag-1 features only** | All rolling features (form, bps, xG, etc.) are built from GW-1 data — no lookahead leakage | Same |
| **No transfers or chips** | Each GW picks a fresh squad of 15 independently; no carry-forward team, no transfer cost, no wildcards or bench boost | Model a transfer budget and penalise switching costs |
| **Oracle uses same constraints** | Oracle selects hindsight-best squad under the same £100M + 3-per-club + formation rules | Same — this is intentional so the gap measures prediction error, not rule-gaming |
| **Single season** | Trained and evaluated entirely within 2025-26 | Multi-season rolling window to account for style/manager changes |

The static-model assumption is the most consequential: because the model is not retrained per GW, later test GWs benefit from a model that has already seen the season's statistical distribution during training. The oracle-capture metric is therefore a slightly optimistic upper bound on what a live deployment would achieve.

Run the simulation yourself:

```powershell
.venv\Scripts\python.exe simulations/simulate_season.py --test-from 15 --test-to 37 --overwrite
.venv\Scripts\python.exe simulations/metrics.py
```

In-sample training quality (reported by the model on its own training data):

| Position | n rows | Top feature |
|---|---|---|
| GK | 2,464 | form_data_density |
| DEF | 7,290 | form_data_density |
| MID | 8,866 | form_data_density |
| FWD | 2,295 | form_data_density |

> `form_data_density` being the top feature across all positions confirms that
> data sparsity (injury returnees, new signings with few GW rows) is the most
> powerful discriminator in the model — players with thin rolling-window data
> are correctly deprioritised.

## Limitations

FPL scores are inherently noisy: clean sheets flip on a single goal, bonus points
involve a committee, and rotation keeps minutes unpredictable. The model's job is
to pick better players than naive form-averaging, not to nail exact point totals.

No fixture difficulty weighting. The model sees `opponent_team` as a raw integer
ID, so it learns relative difficulty from historical scores against each club —
but it doesn't separate "hard fixture" from "player wasn't fit". Adding an FDR
feature is the obvious next improvement.

## Transfermarkt market value

The model currently lacks a **player quality signal** that is independent of
recent FPL form. A player returning from injury with 5 blank GW rows will look
cheap on rolling features despite being world-class. Transfermarkt market value
gives the model a single continuous proxy for real-world perceived quality that
survives blank GWs, rotation, and data sparsity for new arrivals.

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
| `tm_value_imputed` | INTEGER | 1 if value was imputed (mean fallback, no TM data found) |
| `tm_retry` | INTEGER | 1 = re-attempt fetch on every subsequent training run |

### Refresh logic (`transfer_values.py`)

`ensure_transfer_values(conn)` is called automatically by `train.py` after
`ensure_new_players()`. It performs two steps:

**Step 1 — ID population (`ensure_tm_ids`):**
Loads Reep `data/people.csv`, builds a `fpl_code → tm_id` lookup, and upserts
Transfermarkt IDs for all players in `players_raw`. No network calls for players
whose `tm_id` is already set and non-null.

**Step 2 — Value refresh (`refresh_transfer_values`):**
- Default mode (no flag): fetches only players where `tm_fetched_at IS NULL`
  or `tm_retry = 1`. Skips players with fresh data.
- `--force` CLI flag: re-fetches **all** players regardless of cached date.
- Per-player cadence: 1 second sleep between requests (polite crawl rate).
- **Only updates stored value if the new `tm_value_date` is strictly newer
  than what is already stored** — avoids regressing to stale data.
- **Fallback for missing data**: if Transfermarkt returns no value (player not
  found or blocked), assigns the mean `tm_value_eur` of all players who do have
  a value, sets `tm_value_imputed = 1` and `tm_retry = 1` so every subsequent
  training run retries the live fetch until real data is found.

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
model a position-independent quality signal. It is zero-filled if the table is
not yet populated, so training still completes without errors.

#### Edge case: players with sparse rolling stats

A fully fit Salah has 5 rich GW rows — the model should trust his rolling form
almost exclusively. A world-class player returning from a 10-week injury has
zero or near-zero rolling stats — the model should lean heavily on his
market value as a quality prior. XGBoost can't be told this directly (it has
no concept of "trust feature A more when feature B is low"), so it's encoded
explicitly via two derived features:

```
form_data_density         = roll5_minutes / (5 × 90)  ∈ [0, 1]
tm_market_value_x_sparsity = tm_market_value × (1 − form_data_density)
```

| Scenario | density | x_sparsity |
|---|---|---|
| Elite player, just returned from injury | 0.0 | = `tm_market_value` (full signal) |
| Rotation player, half fit | 0.5 | = ½ × `tm_market_value` |
| Fully fit starter (played every minute) | ~1.0 | ≈ 0 (model uses rolling form) |
| Player with no TM data | any | 0 (graceful zero-fill) |

`form_data_density` is also useful on its own: a low value tells the model
that rolling averages are computed from very little data and should be
treated cautiously, regardless of TM value.


## Model selection — why XGBoost

We evaluated three model families before settling on XGBoost:

| Approach | Outcome |
|---|---|
| **Linear regression** | Fast and interpretable, but the FPL scoring surface is highly non-linear — a clean sheet is a step function, bonus points follow a committee decision, and the interaction between minutes played and all other stats is multiplicative. Underfit systematically on high-variance scorers and produced negative point predictions for rare events. |
| **Neural network (MLP)** | Marginally better calibration in controlled tests, but required 10–15× the wall-clock training time for negligible accuracy gain on the same features. The training set (~8,000–9,000 rows per position) is too small to avoid overfitting without heavy regularisation that erased the gain. Compute cost vs ROI was unfavourable on a CPU-only dev machine. |
| **XGBoost** | Best accuracy across all four positions. Handles missing values natively — critical for sparse rolling windows on injury returnees. Trains in seconds. Feature importances are directly interpretable (`form_data_density` is the top feature in all four models). The position-split design keeps each ensemble small and avoids cross-position noise. Chose this. |

The two-tier flex-spot normalisation (z-score within position for the unconstrained slots) was a post-model design choice to prevent the selection algorithm from always filling flex spots with midfielders simply because they score the most absolute points in the dataset.

The per-GW breakdown and interactive per-position RMSE are in the `/validation-report` web UI. Run the simulation first, then open `http://localhost:5000/validation-report`.

## Web UI — routes

| Route | Description |
|---|---|
| `GET /generate-team` | Recommended FPL squad for the current GW on an interactive pitch with predicted scores, confidence margins, and player health cards |
| `GET /validation-report` | Side-by-side pitch: model team vs oracle optimal per simulated GW, with full metrics dashboard. Requires simulation results. |
| `GET /` | Redirects to `/generate-team` |

The validation report page shows:
- Interactive GW selector (prev/next buttons or click any GW badge)
- Side-by-side pitch view: our generated team on the left, oracle optimal on the right — each player circle coloured by position with actual GW points shown
- Per-GW score row: our XI pts, oracle pts, gap, % of oracle
- Summary metrics panel (aggregate over all simulated GWs)
- Per-position prediction RMSE chips
- Scrollable per-GW results table with click-to-select

The server checks all prerequisites on every request and returns a styled error page with the exact fix command if any setup step has been skipped.

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

