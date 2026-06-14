# EPL Fantasy Team Generator

XGBoost team picker for FPL. Trains one regression model per position on
rolling form data, applies FPL squad constraints to select the best 15, and
renders the team on an interactive pitch in the browser.

![pitch UI](docs/pitch-preview.png)

## How it works

**Data** — [vaastav's FPL archive](https://github.com/vaastav/Fantasy-Premier-League):
600+ per-player CSV files, one row per game week. Loaded once into SQLite;
delete `fantasy_football.db` to re-ingest. Alternatively, `ingest_from_fpl_api()`
in `utils.py` pulls directly from the official FPL API.

**Features** — 5-GW rolling averages of 12 stats (goals, assists, clean sheets,
ICT index, saves, bonus, etc.) plus home/away flag, current price, and opponent
team ID. Stats are shifted one GW before windowing — the model never sees data
from the GW it's predicting.

**Models** — four XGBRegressors, one per position (GK / DEF / MID / FWD).
Separate models because saves and goals conceded are the primary signal for
a goalkeeper and mostly noise for a forward. Target: next-GW total_points.

**Selection** — greedy by predicted points, enforcing the £100M budget and FPL's
4-per-club cap. Starting XI picked by FPL minimum rules (1 GK, 3+ DEF, 2+ MID,
1+ FWD); formation falls out of the final counts rather than being set in advance.

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

## Setup

```
git clone https://github.com/vaastav/Fantasy-Premier-League
./setup.ps1          # creates venv, installs deps

python train.py      # ingest CSV → SQLite, train models
python web.py        # http://localhost:5000

python backtest.py   # optional: walk-forward evaluation
```
