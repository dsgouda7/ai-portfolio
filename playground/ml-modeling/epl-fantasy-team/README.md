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
