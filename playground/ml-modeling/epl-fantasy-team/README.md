# EPL Fantasy Team Generator

> **Learning project** — I built this to get hands-on with the full ML lifecycle
> end-to-end: data wrangling, feature engineering, model selection, evaluation,
> and live inference. I also used it as a forcing function to understand where LLMs
> actually add value versus where a simple rule is better. The domain (football) is
> one I care about, which kept me honest — I could immediately see when the output
> was wrong and had to fix it for real rather than paper over it.
>
> **What I drove:** the core problem framing (predict next-GW FPL points, not
> game outcomes), the decision to split models by position rather than use a single
> model with a position feature, the two-tier eligibility design (rule-based for
> the 95% that are deterministic, LLM only for genuinely ambiguous news strings),
> and the choice to anchor on Spearman ρ rather than RMSE as the evaluation metric
> that actually matters for team selection.
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

**Rule-based first, LLM for the remainder:**

```
status in {u, n}                       → definitely out (loaned/sold)
status = i or s                        → definitely out
chance_of_playing_next_round >= 50     → in
chance_of_playing_next_round < 25      → out
everything else (status=d, chance 25–49, or no chance given)
    → pass news string to LLM
```

The LLM (local `qwen2.5-coder:7b` via Ollama) reads the `news` field and returns
AVAILABLE or UNAVAILABLE with a one-sentence rationale. It only fires on the
~20–30 genuinely ambiguous players per GW — not on the full 841-player pool.
If Ollama isn't running the filter degrades gracefully to rule-based with
doubtful players treated as available.

Why a code model for a football question? The `news` field is already structured
text with a consistent grammar ("Injury type — Expected back DD Mon"). A 7B
reasoning model handles this accurately without needing football-specific
fine-tuning. We're not asking it to predict whether a player will score; we're
asking it to parse a sentence.

**What we're not doing:** external news APIs (Sky Sports, BBC Sport). The FPL
`news` field aggregates the same information with a lag of at most a few hours,
and it's already clean, structured, and free. The marginal coverage gain from
scraping sports news doesn't justify the complexity or rate-limit overhead for
this use case.

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

## Walk-forward backtest

For each GW from 15 onwards, train on all prior GWs and evaluate on that GW's
actuals. Same information boundary as live deployment — no future data leaks in.

```
python backtest.py
```

Results (GW 15–37, 2023-24 season):

```
pos    RMSE     R²      ρ     base_RMSE  base_R²  base_ρ      n
-----  ------  ------  -----  ---------  -------  ------  -----
GK       --      --      --       --        --       --      --
DEF      --      --      --       --        --       --      --
MID      --      --      --       --        --       --      --
FWD      --      --      --       --        --       --      --
ALL      --      --      --       --        --       --      --
```

> Run `python backtest.py` after ingesting the dataset to populate this table.

Baseline predicts each player's mean total_points from all training GWs.
ρ (Spearman rank correlation) is the metric that matters here — absolute point
prediction is noisy, but ranking players correctly is what drives team selection.

## Limitations

R² will look low. FPL scores are inherently noisy: clean sheets flip on a single
goal, bonus points involve a committee, and rotation keeps minutes unpredictable.
The model's job is to rank players better than naive averaging, not to nail point
totals. Check ρ vs baseline_ρ, not R² in isolation.

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


