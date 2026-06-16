"""
backtest.py -- walk-forward evaluation of the per-position XGBoost models.

For each GW from MIN_TEST_GW to GAME_WEEK-1, this script trains fresh models
on all earlier GWs and evaluates on that GW's actuals. The model never sees
future data -- same information boundary as live deployment.

Run:
    python train/backtest.py

Prints a per-position results table and saves raw predictions to
backtest_results.csv. Slower than train.py (retrains N times) but that's
the point -- in-sample R² from train.py is optimistic.
"""
import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import r2_score
from tabulate import tabulate
from xgboost import XGBRegressor

from utils import DB_FILE, POS_FEATURES, GAME_WEEK, build_features

MIN_TEST_GW = 15  # need enough GW history for rolling features to stabilise
N_ESTIMATORS = 200


def _baseline(train_df, test_df):
    """Predict each player's mean total_points from training history."""
    player_means = train_df.groupby("id")["total_points"].mean()
    fallback = train_df["total_points"].mean()
    return test_df["id"].map(player_means).fillna(fallback).values


def run_backtest(df):
    records = []

    for test_gw in range(MIN_TEST_GW, GAME_WEEK):
        train = df[(df["Game_Week"] < test_gw - 1) & df["target"].notna()].copy()
        test = df[(df["Game_Week"] == test_gw - 1) & df["target"].notna()].copy()

        if train.empty or test.empty:
            continue

        for pos in ["GK", "DEF", "MID", "FWD"]:
            pos_train = train[train["element_type"] == pos]
            pos_test = test[test["element_type"] == pos]

            if len(pos_train) < 20 or pos_test.empty:
                continue

            model = XGBRegressor(
                n_estimators=N_ESTIMATORS,
                max_depth=4,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbosity=0,
            )
            model.fit(pos_train[POS_FEATURES[pos]].fillna(0), pos_train["target"])

            y_pred = model.predict(pos_test[POS_FEATURES[pos]].fillna(0))
            y_true = pos_test["target"].values
            y_base = _baseline(pos_train, pos_test)

            for yt, yp, yb, pid in zip(
                y_true, y_pred, y_base, pos_test["id"].values
            ):
                records.append(
                    {"gw": test_gw, "pos": pos, "player_id": pid,
                     "y_true": yt, "y_pred": yp, "y_base": yb}
                )

        print(f"  GW {test_gw} done")

    return pd.DataFrame(records)


def _metrics(sub):
    rmse     = float(np.sqrt(((sub["y_true"] - sub["y_pred"]) ** 2).mean()))
    r2       = float(r2_score(sub["y_true"], sub["y_pred"]))
    rho      = float(spearmanr(sub["y_true"], sub["y_pred"]).statistic)
    b_rmse   = float(np.sqrt(((sub["y_true"] - sub["y_base"]) ** 2).mean()))
    b_r2     = float(r2_score(sub["y_true"], sub["y_base"]))
    b_rho    = float(spearmanr(sub["y_true"], sub["y_base"]).statistic)
    return rmse, r2, rho, b_rmse, b_r2, b_rho


def summarise(df):
    rows = []
    for pos in ["GK", "DEF", "MID", "FWD", "ALL"]:
        sub = df if pos == "ALL" else df[df["pos"] == pos]
        if sub.empty:
            continue
        rmse, r2, rho, b_rmse, b_r2, b_rho = _metrics(sub)
        rows.append([
            pos,
            f"{rmse:.3f}", f"{r2:.3f}", f"{rho:.3f}",
            f"{b_rmse:.3f}", f"{b_r2:.3f}", f"{b_rho:.3f}",
            len(sub),
        ])

    print("\nWalk-forward results (out-of-sample):")
    print(tabulate(
        rows,
        headers=["pos", "RMSE", "R²", "ρ", "base_RMSE", "base_R²", "base_ρ", "n"],
        tablefmt="simple",
    ))
    print("\nBaseline: predict each player's mean total_points from all prior GWs.")
    print("ρ = Spearman rank correlation -- what actually matters for team selection.")


df = build_features(DB_FILE)
print(f"Running walk-forward backtest (GW {MIN_TEST_GW} to {GAME_WEEK - 1})...")
results = run_backtest(df)
summarise(results)

results.to_csv("backtest_results.csv", index=False)
print("\nRaw predictions saved to backtest_results.csv")
