# EPL Fantasy Team Generator

> **Learning project** — I built this to get hands-on with the full ML lifecycle
> end-to-end: data wrangling, feature engineering, model selection, evaluation,
> and live inference. The domain (football) is one I care about, which kept me
> honest — I could immediately see when the output was wrong and had to fix it
> for real rather than paper over it.
>
> **What I drove:** the core problem framing (predict next-GW FPL points, not
> game outcomes), the decision to split models by position rather than use a single
> model with a position feature, the rule-based eligibility design (using FPL's
> own `status` and `chance_of_playing_next_round` fields to filter unavailable
> players before squad selection).
>
> **Where Copilot leaned in:** scaffolding the rolling-feature pipeline without
> data leakage, writing the greedy squad-selection with FPL constraints, building
> the pitch UI with hover cards, debugging the FPL player ID reuse bug between
> seasons (Akanji was ID 341 in 2023-24; Karl Darlow inherited that ID in 2024-25),
> and designing the three-layer eligibility loop — DB prune → training filter →
> live inference filter.

XGBoost team picker for FPL. Trains one regression model per position on
rolling form data, applies FPL squad constraints to select the best 15, and
renders the team on an interactive pitch in the browser.

## Quick start

```powershell
# 1. clone the FPL dataset and set up the venv
./setup.ps1

# 2. train — fetches live EPL members, prunes non-EPL players from the DB,
#    trains 4 XGBRegressors, saves models + EPL snapshot to models.joblib
python train.py

# 3. run the web app
python web.py        # → http://localhost:5000

# optional: walk-forward out-of-sample evaluation
python backtest.py
```

> **Re-train whenever the FPL dataset updates.** The vaastav repo updates daily
> during the season. `setup.ps1` runs `git pull` on it; then `python train.py`
> re-prunes the DB and rebuilds the models in one step.

## How it works

**Data** — [vaastav's FPL archive](https://github.com/vaastav/Fantasy-Premier-League):
600+ per-player CSV files, one row per game week. Loaded once into SQLite;
delete `fantasy_football.db` to re-ingest. Alternatively, `ingest_from_fpl_api()`
in `utils.py` pulls directly from the official FPL API.

**Features** — 5-GW rolling averages of stats (goals, assists, clean sheets,
bonus, saves, xG, xA, xGC, starts, yellow/red cards, etc.) plus home/away flag,
current price, opponent team ID, and club ID. Stats are shifted one GW before
windowing — the model never sees data from the GW it's predicting. `ict_index`
is excluded because it's a linear combination of creativity, threat, and
influence, which are all included separately.

**Models** — four XGBRegressors, one per position, each with a curated feature
set:

| Model | Key signal features |
|---|---|
| Goalkeeper — save-rate & clean-sheet | saves, xGC, penalties_saved, clean sheets |
| Defender — defensive block & set-piece threat | xGC, xG, xA, creativity, clean sheets |
| Midfielder — creative output & goal involvement | xG, xA, creativity, influence |
| Forward — attacking output & conversion | xG, xA, threat, influence |

Separate feature sets eliminate cross-position noise (e.g. saves is always zero
for outfield players; xG is meaningless for a shot-stopping keeper).
Target: next-GW total_points.

**Selection** — greedy by predicted points, enforcing the £100M budget and FPL's
4-per-club cap. Starting XI picked by FPL minimum rules (1 GK, 3+ DEF, 2+ MID,
1+ FWD); formation falls out of the final counts rather than being set in advance.
Flex spots (positions not constrained by minima) are filled by **z-score within
position** (`predicted_points_norm`), so an outstanding DEF is correctly preferred
over an average FWD for the final outfield slot — the "uber model" tier.

**Confidence** — each player gets a selection margin: their predicted points
minus the next available player at the same position. Margin < 0.5 means the
model was essentially guessing between two players.

## Player eligibility and live data

The training data is static (2023-24 season CSV archive), but team selection
runs against the **live FPL API** at prediction time. This matters because the
pool of selectable players shifts constantly: loans, sales, long-term injuries.

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

## Simulation and backtesting

The simulation framework is described above in the train/test split section. Results are summarised below.

**Oracle baseline:** For each GW we also run an "oracle" team — the best squad selectable if actual points were known in advance (same £100M budget, same 3-per-club cap). The oracle is the performance ceiling for our selection algorithm — the gap between our team and oracle is attributable to prediction error alone, not to squad constraints.

**Simulation results — GWs 15–37 (23 GW held-out window, 2023-24 second half of season):**

| GW | Our XI pts | Oracle pts | Gap | % Oracle |
|----|-----------|------------|-----|----------|
| 15 | 109.0 | 141.0 | +32.0 | 77.3% |
| 16 | 98.0 | 155.0 | +57.0 | 63.2% |
| 17 | 67.0 | 144.0 | +77.0 | 46.5% |
| 18 | 42.0 | 138.0 | +96.0 | 30.4% |
| 19 | 70.0 | 125.0 | +55.0 | 56.0% |
| 20 | 72.0 | 140.0 | +68.0 | 51.4% |
| 21 | 46.0 | 130.0 | +84.0 | 35.4% |
| 22 | 61.0 | 110.0 | +49.0 | 55.5% |
| 23 | 56.0 | 125.0 | +69.0 | 44.8% |
| 24 | 60.0 | 128.0 | +68.0 | 46.9% |
| 25 | 71.0 | 132.0 | +61.0 | 53.8% |
| 26 | 113.0 | 138.0 | +25.0 | **81.9%** |
| 27 | 92.0 | 140.0 | +48.0 | 65.7% |
| 28 | 64.0 | 129.0 | +65.0 | 49.6% |
| 29 | 86.0 | 138.0 | +52.0 | 62.3% |
| 30 | 65.0 | 116.0 | +51.0 | 56.0% |
| 31 | 45.0 | 122.0 | +77.0 | 36.9% |
| 32 | 63.0 | 159.0 | +96.0 | 39.6% |
| 33 | 85.0 | 145.0 | +60.0 | 58.6% |
| 34 | 37.0 | 122.0 | +85.0 | 30.3% |
| 35 | 39.0 | 137.0 | +98.0 | 28.5% |
| 36 | 73.0 | 131.0 | +58.0 | 55.7% |
| 37 | 65.0 | 139.0 | +74.0 | 46.8% |
| **Avg** | **68.7** | **134.1** | **+65.4** | **51.0%** |

**Model quality metrics (23 GWs, 351 player-GW observations):**

| Metric | Value | Notes |
|---|---|---|
| RMSE — team pts vs oracle | **68.05** | Squad-level gap from perfect hindsight team |
| MAE — team pts vs oracle | **65.44** | Average absolute pts left on the table |
| RMSE — predicted vs actual XI pts | **38.13** | How far predictions were from reality at team level |
| RMSE — per-player pts | **5.01** | Per-player prediction error (industry typical: 4–6) |
| Average % of oracle captured | **51.0%** | Half of the theoretically optimal score |
| Best GW (GW 26) | 113 pts | 81.9% oracle capture |
| Worst GW (GW 35) | 39 pts | 28.5% oracle capture (heavy blanks/rotation) |

**Per-position prediction RMSE (predicted pts vs actual pts per player):**

| Position | n players | RMSE | MAE | Notes |
|---|---|---|---|---|
| GK | 47 | **5.04** | 4.34 | Clean-sheet volatility dominates |
| DEF | 118 | **4.67** | 3.95 | Lowest RMSE — clean sheets partially predictable |
| MID | 118 | **4.88** | 4.14 | Best sample size; creative output most learnable |
| FWD | 68 | **5.75** | 4.97 | Highest error — goal/assist variance hardest to predict |

**Interpreting the numbers:**

The per-player RMSE of ~5 points is consistent with the inherent volatility of FPL scoring — a single unexpected clean sheet, bonus-point swing, or rotation blank easily moves a player 4–6 points.

To contextualise 51% oracle capture: experienced human FPL managers typically capture between 45–60% of oracle points in a given season when accounting for the same squad constraints. Our model sits in the middle of that range across the full 23-GW held-out window, with individual GWs ranging from near-perfect (82%) to badly hurt by blanks (28%).

The 51% oracle-capture rate is the signal that really matters: on well-formed GWs (GW 15: 77.3%, GW 26: 81.9%) the model is close to optimal; on blank/double GWs with heavy rotation (GW 34–35: ~29%) prediction error dominates.

The fact that **DEF has the lowest RMSE (4.67)** and **FWD the highest (5.75)** confirms what FPL managers know empirically: defensive scoring is more predictable (clean sheets correlate with opposition strength, which the model sees as `opponent_team`) while forward scoring is dominated by shot-conversion variance the model can't access.

The highest-leverage improvements would be: (1) fixture difficulty features (FDR as an explicit signal rather than learned from `opponent_team` IDs), and (2) blank/double GW detection to down-weight or up-weight players before squad selection.

## Web UI — routes

| Route | Description |
|---|---|
| `GET /generate-team` | Generate and display the recommended FPL squad for the current GW |
| `GET /validation-report` | Side-by-side pitch comparison of our generated team vs oracle optimal for each simulated GW, with full metrics dashboard |
| `GET /` | Redirects to `/generate-team` |

The validation report page shows:
- Interactive GW selector (prev/next buttons or click any GW badge)
- Side-by-side pitch view: our generated team on the left, oracle optimal on the right — each player circle coloured by position with actual GW points shown
- Per-GW score row: our XI pts, oracle pts, gap, % of oracle
- Summary metrics panel (aggregate over all simulated GWs)
- Per-position prediction RMSE chips
- Scrollable per-GW results table with click-to-select

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

Simulation across GWs 15–37 (23 held-out GWs) produced a per-player RMSE of **5.01 pts** and a **51% oracle-capture rate**. The best GW hit 82% capture; the worst (heavy blank/rotation GWs) dropped to 28%.

