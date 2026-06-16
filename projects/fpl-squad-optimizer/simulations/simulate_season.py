"""
simulate_season.py — Walk-forward GW simulation for FPL team selection.

For each test GW the script:
  1. Builds the player-feature snapshot for that GW (using GW-1 rolling data).
  2. Runs the same greedy squad-selection as the live generator.
  3. Records predicted and *actual* GW points (looked up from player_gw).
  4. Computes the *oracle optimal* team for that GW (best possible team given
     actual points — a hindsight upper bound).
  5. Saves one row per GW to   simulations/results/simulation_results.csv
     and one row per player per GW to simulations/results/player_rows.csv.

Usage
-----
    # default: test on last 10 GWs of the season
    python simulations/simulate_season.py

    # explicit test window
    python simulations/simulate_season.py --test-from 20 --test-to 29

    # wipe previous results and re-run
    python simulations/simulate_season.py --overwrite

The results are consumed by:
    python simulations/metrics.py          → pretty-printed metric report
    flask /validation-report               → per-GW side-by-side pitch UI
"""

import argparse
import os
import sqlite3
import sys
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# ── project root on sys.path ──────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from utils import (
    DB_FILE,
    MODELS_FILE,
    POS_FEATURES,
    build_features,
    normalize_pool_scores,
    pick_starting_xi,
)

RESULTS_DIR = Path(os.environ.get(
    'FPL_RESULTS_DIR',
    str(Path(__file__).parent / 'results'),
))
SIM_CSV     = RESULTS_DIR / 'simulation_results.csv'   # one row per GW
PLAYER_CSV  = RESULTS_DIR / 'player_rows.csv'           # one row per player×GW

STRUCTURE = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
MAX_PER_TEAM = 3
MAX_SPEND    = 1000   # in tenths of a £M (1000 = £100M)


# ── helpers ───────────────────────────────────────────────────────────────────

def _build_pool(all_data: pd.DataFrame, models: dict, gw: int) -> pd.DataFrame:
    """Score every player using their GW-1 feature snapshot."""
    snapshot = all_data[all_data["Game_Week"] == gw - 1].copy()
    parts = []
    for pos, model in models.items():
        sub = snapshot[snapshot["element_type"] == pos].copy()
        if sub.empty:
            continue
        X = sub[POS_FEATURES[pos]].fillna(0)
        sub["predicted_points"] = model.predict(X)
        parts.append(sub)
    if not parts:
        return pd.DataFrame()
    pool = pd.concat(parts, ignore_index=True)
    return normalize_pool_scores(pool)


def _select_squad(pool: pd.DataFrame, score_col: str = "predicted_points") -> pd.DataFrame:
    """Greedy squad selection enforcing FPL constraints."""
    squad, spend, counts = [], 0, Counter()
    for pos, n in STRUCTURE.items():
        candidates = pool[pool["element_type"] == pos].sort_values(
            score_col, ascending=False
        )
        picked = 0
        cur_ids = {p["id"] for p in squad}
        for _, p in candidates.iterrows():
            if picked == n:
                break
            if (
                p["id"] not in cur_ids
                and counts[int(p["team"])] < MAX_PER_TEAM
                and spend + int(p["value"]) <= MAX_SPEND
            ):
                squad.append(p.to_dict())
                spend += int(p["value"])
                counts[int(p["team"])] += 1
                cur_ids.add(p["id"])
                picked += 1
    return pd.DataFrame(squad)


def _get_actual_points(db_file: str, gw: int) -> pd.DataFrame:
    """Return player_id → actual GW points and minutes for all players in a GW."""
    conn = sqlite3.connect(db_file)
    df = pd.read_sql(
        "SELECT element AS player_id, total_points AS actual_points, minutes "
        "FROM player_gw WHERE round = ?",
        conn,
        params=[gw],
    )
    conn.close()
    return df.set_index("player_id")


def _add_actual(squad: pd.DataFrame, actual_df: pd.DataFrame) -> pd.DataFrame:
    """Join actual points onto a squad DataFrame."""
    squad = squad.copy()
    squad["player_id"] = squad["id"].astype(int)
    act = actual_df[["actual_points"]].copy()  # only bring in actual_points
    squad = squad.join(act, on="player_id")
    squad["actual_points"] = squad["actual_points"].fillna(0).astype(float)
    return squad


def _xi_actual_total(squad_with_actual: pd.DataFrame) -> float:
    """Pick the best starting XI (FPL rules) and sum actual points."""
    if squad_with_actual.empty or "actual_points" not in squad_with_actual.columns:
        return 0.0
    # borrow pick_starting_xi but replace predicted_points with actual for selection
    tmp = squad_with_actual.copy()
    tmp["predicted_points"] = tmp["actual_points"]   # use actual for selection order
    tmp["predicted_points_norm"] = tmp["actual_points"]
    try:
        starters, _, _ = pick_starting_xi(tmp)
        return float(starters["actual_points"].sum())
    except Exception:
        return float(tmp.nlargest(11, "actual_points")["actual_points"].sum())


def _oracle_total(pool_with_actual: pd.DataFrame) -> tuple[float, pd.DataFrame]:
    """Select the hindsight-optimal squad using actual points, return XI total and squad."""
    oracle_squad = _select_squad(pool_with_actual, score_col="actual_points")
    oracle_squad = _add_actual(oracle_squad, pool_with_actual.set_index("id")[["actual_points"]]) \
        if "actual_points" not in oracle_squad.columns else oracle_squad
    total = _xi_actual_total(oracle_squad)
    return total, oracle_squad


def _fetch_fpl_dream_team(gw: int) -> list[int]:
    """Try to fetch FPL official Dream Team player IDs for this GW. Returns [] on failure."""
    try:
        import urllib.request, json
        url = f"https://fantasy.premierleague.com/api/dream-team/{gw}/"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=6) as resp:
            data = json.loads(resp.read())
        return [e["id"] for e in data.get("dream_team", [])]
    except Exception:
        return []


# ── main simulation loop ──────────────────────────────────────────────────────

def run_simulation(test_from: int, test_to: int, overwrite: bool = False) -> pd.DataFrame:
    RESULTS_DIR.mkdir(exist_ok=True)

    # Load model and features once
    print(f"Loading model from {MODELS_FILE} ...")
    checkpoint   = joblib.load(MODELS_FILE)
    models_map   = checkpoint["models"]
    epl_members  = checkpoint.get("epl_members")

    print(f"Building features from {DB_FILE} ...")
    all_data = build_features(DB_FILE)

    # Determine available GWs
    available_gws = sorted(all_data["Game_Week"].unique())
    test_gws      = [gw for gw in range(test_from, test_to + 1) if gw in available_gws]
    if not test_gws:
        print(f"No data available for GWs {test_from}–{test_to}. Aborting.")
        return pd.DataFrame()

    print(f"Simulating GWs {test_gws[0]}–{test_gws[-1]} ({len(test_gws)} game weeks)")

    gw_rows, player_rows = [], []

    for gw in test_gws:
        print(f"\n  GW {gw} ...", end="  ")

        # 1. Build scored pool (uses GW-1 snapshot, so no data leakage)
        pool = _build_pool(all_data, models_map, gw)
        if pool.empty:
            print("pool empty, skipping")
            continue

        # 2. EPL filter (same as live)
        if epl_members:
            from eligibility import player_name_key
            pool = pool[pool.apply(
                lambda r: player_name_key(r.get("first_name", ""), r.get("second_name", "")) in epl_members,
                axis=1,
            )].copy()

        # 3. Get actual points for this GW
        actual_df = _get_actual_points(DB_FILE, gw)

        # 4. Our generated squad (predicted points → greedy selection)
        our_squad = _select_squad(pool)
        if our_squad.empty:
            print("squad selection failed, skipping")
            continue
        our_squad = _add_actual(our_squad, actual_df)
        our_xi_pts = _xi_actual_total(our_squad)

        # 5. Oracle squad (actual points → greedy selection — hindsight upper bound)
        pool_with_actual = pool.copy()
        pool_with_actual = pool_with_actual.join(actual_df[["actual_points"]], on="id")
        pool_with_actual["actual_points"] = pool_with_actual["actual_points"].fillna(0).astype(float)
        oracle_xi_pts, oracle_squad = _oracle_total(pool_with_actual)

        # 6. FPL Dream Team (best 11 regardless of budget — optional)
        dream_ids = _fetch_fpl_dream_team(gw)
        dream_pts: float = 0.0
        if dream_ids:
            dm = actual_df.reindex(dream_ids)
            dream_pts = float(dm["actual_points"].fillna(0).sum())

        our_pred = float(our_squad["predicted_points"].sum())
        print(
            f"our XI actual={our_xi_pts:.1f}  oracle={oracle_xi_pts:.1f}"
            + (f"  dream={dream_pts:.1f}" if dream_ids else "")
        )

        gw_rows.append({
            "gw":              gw,
            "our_predicted":   round(our_pred, 2),
            "our_actual_xi":   round(our_xi_pts, 2),
            "oracle_xi":       round(oracle_xi_pts, 2),
            "dream_pts":       round(dream_pts, 2),
            "gap_to_oracle":   round(oracle_xi_pts - our_xi_pts, 2),
            "pct_of_oracle":   round(100 * our_xi_pts / oracle_xi_pts, 1) if oracle_xi_pts else 0.0,
        })

        # Per-player rows (our squad)
        for _, p in our_squad.iterrows():
            player_rows.append({
                "gw":               gw,
                "player_id":        int(p.get("id", p.get("player_id", 0))),
                "first_name":       p.get("first_name", ""),
                "second_name":      p.get("second_name", ""),
                "element_type":     p.get("element_type", ""),
                "team":             int(p.get("team", 0)),
                "value":            int(p.get("value", 0)),
                "predicted_points": round(float(p.get("predicted_points", 0)), 3),
                "actual_points":    round(float(p.get("actual_points", 0)), 1),
                "source":           "our",
            })
        # Per-player rows (oracle squad)
        for _, p in oracle_squad.iterrows():
            player_rows.append({
                "gw":               gw,
                "player_id":        int(p.get("id", p.get("player_id", 0))),
                "first_name":       p.get("first_name", ""),
                "second_name":      p.get("second_name", ""),
                "element_type":     p.get("element_type", ""),
                "team":             int(p.get("team", 0)),
                "value":            int(p.get("value", 0)),
                "predicted_points": round(float(p.get("predicted_points", 0)), 3),
                "actual_points":    round(float(p.get("actual_points", 0)), 1),
                "source":           "oracle",
            })

    gw_df     = pd.DataFrame(gw_rows)
    player_df = pd.DataFrame(player_rows)

    if overwrite or not SIM_CSV.exists():
        gw_df.to_csv(SIM_CSV, index=False)
        player_df.to_csv(PLAYER_CSV, index=False)
    else:
        # Merge: replace rows for simulated GWs, keep the rest
        existing_gw = pd.read_csv(SIM_CSV)
        existing_pl = pd.read_csv(PLAYER_CSV)
        gw_df     = pd.concat([existing_gw[~existing_gw["gw"].isin(gw_df["gw"])], gw_df]).sort_values("gw")
        player_df = pd.concat([existing_pl[~existing_pl["gw"].isin(player_df["gw"])], player_df]).sort_values(["gw", "source"])
        gw_df.to_csv(SIM_CSV, index=False)
        player_df.to_csv(PLAYER_CSV, index=False)

    print(f"\nResults written to {SIM_CSV}")
    return gw_df


# ── CLI entry point ───────────────────────────────────────────────────────────

def _cli():
    parser = argparse.ArgumentParser(description="FPL season simulator")
    parser.add_argument("--test-from", type=int, default=None,
                        help="First GW to simulate (default: last 10 GWs)")
    parser.add_argument("--test-to",   type=int, default=None,
                        help="Last GW to simulate (default: max GW in data)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Wipe existing results and re-run")
    args = parser.parse_args()

    # Determine max GW from data
    import sqlite3 as _sq
    conn = _sq.connect(DB_FILE)
    max_gw = conn.execute("SELECT MAX(round) FROM player_gw").fetchone()[0] or 38
    conn.close()

    test_to   = args.test_to   or max_gw
    test_from = args.test_from or max(1, test_to - 9)

    results = run_simulation(test_from, test_to, overwrite=args.overwrite)

    if not results.empty:
        from simulations.metrics import print_report
        print_report(results)


if __name__ == "__main__":
    _cli()
