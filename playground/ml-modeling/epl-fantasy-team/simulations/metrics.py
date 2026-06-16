"""
metrics.py — Load simulation results and print/return a metric report.

Usage
-----
    # after running simulate_season.py
    python simulations/metrics.py

    # or import in another script
    from simulations.metrics import load_results, compute_metrics, print_report
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

RESULTS_DIR = Path(__file__).parent / "results"
SIM_CSV     = RESULTS_DIR / "simulation_results.csv"
PLAYER_CSV  = RESULTS_DIR / "player_rows.csv"


# ── loaders ───────────────────────────────────────────────────────────────────

def load_results() -> pd.DataFrame:
    """Load per-GW simulation results. Returns empty DataFrame if missing."""
    if not SIM_CSV.exists():
        return pd.DataFrame()
    return pd.read_csv(SIM_CSV)


def load_player_rows() -> pd.DataFrame:
    """Load per-player simulation rows. Returns empty DataFrame if missing."""
    if not PLAYER_CSV.exists():
        return pd.DataFrame()
    return pd.read_csv(PLAYER_CSV)


# ── metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(gw_df: pd.DataFrame) -> dict:
    """
    Compute summary metrics from the per-GW simulation DataFrame.

    Key columns expected:
        gw, our_predicted, our_actual_xi, oracle_xi, dream_pts, gap_to_oracle
    """
    if gw_df.empty:
        return {}

    our  = gw_df["our_actual_xi"].values
    ora  = gw_df["oracle_xi"].values
    pred = gw_df["our_predicted"].values

    rmse_vs_oracle     = float(np.sqrt(np.mean((our - ora) ** 2)))
    mae_vs_oracle      = float(np.mean(np.abs(our - ora)))
    rmse_pred_vs_actual= float(np.sqrt(np.mean((pred - our) ** 2)))
    avg_our            = float(np.mean(our))
    avg_oracle         = float(np.mean(ora))
    avg_gap            = float(np.mean(ora - our))
    avg_pct_of_oracle  = float(np.mean(gw_df["pct_of_oracle"]))
    n_gws              = len(gw_df)

    # Per-player prediction RMSE
    pl_df = load_player_rows()
    per_player_rmse = None
    if not pl_df.empty:
        ours_pl = pl_df[pl_df["source"] == "our"].copy()
        if not ours_pl.empty:
            per_player_rmse = float(
                np.sqrt(np.mean((ours_pl["predicted_points"] - ours_pl["actual_points"]) ** 2))
            )

    return {
        "n_gws":                   n_gws,
        "avg_our_xi_pts":          round(avg_our, 2),
        "avg_oracle_xi_pts":       round(avg_oracle, 2),
        "avg_gap_to_oracle":       round(avg_gap, 2),
        "avg_pct_of_oracle":       round(avg_pct_of_oracle, 1),
        "rmse_team_vs_oracle":     round(rmse_vs_oracle, 3),
        "mae_team_vs_oracle":      round(mae_vs_oracle, 3),
        "rmse_pred_vs_actual_xi":  round(rmse_pred_vs_actual, 3),
        "rmse_per_player_pts":     round(per_player_rmse, 3) if per_player_rmse else None,
    }


def compute_per_position_rmse(pl_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per-position RMSE of predicted vs actual points (our selections)."""
    if pl_df is None:
        pl_df = load_player_rows()
    if pl_df.empty:
        return pd.DataFrame()
    ours = pl_df[pl_df["source"] == "our"].copy()
    rows = []
    for pos in ["GK", "DEF", "MID", "FWD"]:
        sub = ours[ours["element_type"] == pos]
        if sub.empty:
            continue
        rmse = float(np.sqrt(np.mean((sub["predicted_points"] - sub["actual_points"]) ** 2)))
        mae  = float(np.mean(np.abs(sub["predicted_points"] - sub["actual_points"])))
        rows.append({"position": pos, "n": len(sub), "rmse": round(rmse, 3), "mae": round(mae, 3)})
    return pd.DataFrame(rows)


# ── pretty printer ────────────────────────────────────────────────────────────

def print_report(gw_df: pd.DataFrame | None = None):
    if gw_df is None:
        gw_df = load_results()
    if gw_df is None or gw_df.empty:
        print("No simulation results found.  Run simulate_season.py first.")
        return

    m = compute_metrics(gw_df)

    sep = "─" * 54
    print(f"\n{'FPL SIMULATION METRICS':^54}")
    print(sep)
    print(f"  GWs simulated              : {m['n_gws']}")
    print(f"  Avg our XI actual pts / GW : {m['avg_our_xi_pts']}")
    print(f"  Avg oracle XI pts  / GW    : {m['avg_oracle_xi_pts']}")
    print(f"  Avg gap to oracle  / GW    : {m['avg_gap_to_oracle']}")
    print(f"  % of oracle captured (avg) : {m['avg_pct_of_oracle']}%")
    print(sep)
    print(f"  RMSE (team pts vs oracle)  : {m['rmse_team_vs_oracle']}")
    print(f"  MAE  (team pts vs oracle)  : {m['mae_team_vs_oracle']}")
    print(f"  RMSE (pred vs actual XI)   : {m['rmse_pred_vs_actual_xi']}")
    if m["rmse_per_player_pts"] is not None:
        print(f"  RMSE (per-player pts)      : {m['rmse_per_player_pts']}")
    print(sep)

    # Per-GW table
    print(f"\n{'GW':>4}  {'Our XI':>8}  {'Oracle':>8}  {'Gap':>8}  {'% Opt':>7}")
    print("─" * 44)
    for _, row in gw_df.sort_values("gw").iterrows():
        print(
            f"{int(row['gw']):>4}  "
            f"{row['our_actual_xi']:>8.1f}  "
            f"{row['oracle_xi']:>8.1f}  "
            f"{row['gap_to_oracle']:>+8.1f}  "
            f"{row['pct_of_oracle']:>6.1f}%"
        )
    print("─" * 44)
    print(
        f"{'AVG':>4}  "
        f"{m['avg_our_xi_pts']:>8.1f}  "
        f"{m['avg_oracle_xi_pts']:>8.1f}  "
        f"{m['avg_gap_to_oracle']:>+8.1f}  "
        f"{m['avg_pct_of_oracle']:>6.1f}%"
    )

    # Per-position RMSE
    pos_df = compute_per_position_rmse()
    if not pos_df.empty:
        print(f"\n{'Per-position prediction RMSE (predicted vs actual pts)':^54}")
        print("─" * 40)
        print(f"{'Pos':>4}  {'n':>5}  {'RMSE':>8}  {'MAE':>8}")
        print("─" * 40)
        for _, row in pos_df.iterrows():
            print(f"{row['position']:>4}  {int(row['n']):>5}  {row['rmse']:>8.3f}  {row['mae']:>8.3f}")


if __name__ == "__main__":
    print_report()
