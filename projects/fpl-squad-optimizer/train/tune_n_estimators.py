"""
tune_n_estimators.py — find the optimal number of XGBoost trees via
temporal walk-forward validation.

Usage:
    python train/tune_n_estimators.py

Method:
    For each position (GK/DEF/MID/FWD):
      - Sort GWs chronologically.
      - Walk-forward splits: train on first 70% of GWs, validate on the next
        10%, step forward by 5 GWs, repeat for 5 folds.
      - For each fold, use XGBoost early stopping (patience=30) with a large
        upper bound (1 000 trees) — XGBoost stops automatically when val loss
        plateaus.
      - Record best_iteration and val RMSE per fold.
    Then print a per-position summary and the recommended n_estimators.

The recommended value is the 90th-percentile best_iteration across all folds
and positions (gives headroom without over-fitting).
"""
import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import warnings
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import root_mean_squared_error, r2_score

from utils import DB_FILE, GAME_WEEK, POS_FEATURES, build_features, MODELS_FILE

warnings.filterwarnings('ignore', category=UserWarning)

MAX_TREES    = 1_000   # upper bound; early stopping will find the true optimum
EARLY_STOP   = 30      # rounds without improvement before stopping
TRAIN_FRAC   = 0.70    # fraction of GWs used for training in each fold
VAL_FRAC     = 0.10    # fraction of GWs used for validation in each fold
STEP_GWS     = 5       # how many GWs to step forward between folds
N_FOLDS      = 5       # number of walk-forward folds


def _load_df() -> pd.DataFrame:
    df = build_features(DB_FILE)
    return df


def _cast_features(X: pd.DataFrame) -> pd.DataFrame:
    for c in X.select_dtypes(include='object').columns:
        X[c] = pd.to_numeric(X[c], errors='coerce').fillna(0).astype(int)
    return X


def tune_position(df: pd.DataFrame, pos: str) -> dict:
    pos_df = df[df['element_type'] == pos].copy()
    gws    = sorted(pos_df['Game_Week'].unique())
    n_gws  = len(gws)

    if n_gws < 10:
        print(f"  [{pos}] not enough GWs ({n_gws}) — skipping")
        return {}

    features = POS_FEATURES[pos]
    fold_results = []

    for fold in range(N_FOLDS):
        # Walk-forward: each fold starts STEP_GWS later
        offset       = fold * STEP_GWS
        train_end_i  = int(n_gws * TRAIN_FRAC) + offset
        val_end_i    = train_end_i + int(n_gws * VAL_FRAC)

        if val_end_i > n_gws:
            break   # ran out of data

        train_gws = gws[:train_end_i]
        val_gws   = gws[train_end_i:val_end_i]

        train_rows = pos_df[pos_df['Game_Week'].isin(train_gws) & pos_df['target'].notna()]
        val_rows   = pos_df[pos_df['Game_Week'].isin(val_gws)   & pos_df['target'].notna()]

        if len(train_rows) < 50 or len(val_rows) < 10:
            continue

        X_tr = _cast_features(train_rows[features].fillna(0).copy())
        y_tr = train_rows['target']
        X_va = _cast_features(val_rows[features].fillna(0).copy())
        y_va = val_rows['target']

        model = XGBRegressor(
            n_estimators=MAX_TREES,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=0,
            early_stopping_rounds=EARLY_STOP,
            eval_metric='rmse',
        )
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

        best_n   = model.best_iteration + 1   # XGBoost is 0-indexed
        val_pred = model.predict(X_va, iteration_range=(0, best_n))
        val_rmse = float(root_mean_squared_error(y_va, val_pred))
        val_r2   = float(r2_score(y_va, val_pred))

        fold_results.append({
            'fold':      fold + 1,
            'train_gws': len(train_gws),
            'val_gws':   len(val_gws),
            'n_train':   len(train_rows),
            'n_val':     len(val_rows),
            'best_n':    best_n,
            'val_rmse':  round(val_rmse, 4),
            'val_r2':    round(val_r2, 4),
        })
        print(f"    fold {fold+1}: train_gws={len(train_gws):3d}  n_train={len(train_rows):5,d}"
              f"  best_n={best_n:4d}  val_rmse={val_rmse:.4f}  val_r²={val_r2:.4f}")

    return {'pos': pos, 'folds': fold_results}


def main():
    print(f"Loading features from {DB_FILE} (GW={GAME_WEEK})...")
    df = _load_df()
    df = df[df['target'].notna()]
    print(f"  {len(df):,} rows with targets across GW {df['Game_Week'].min()}–{df['Game_Week'].max()}\n")

    all_best_ns = []
    summary_rows = []

    for pos in ['GK', 'DEF', 'MID', 'FWD']:
        print(f"[{pos}]")
        result = tune_position(df, pos)
        if not result or not result['folds']:
            continue

        folds     = result['folds']
        best_ns   = [f['best_n']   for f in folds]
        val_rmses = [f['val_rmse'] for f in folds]
        val_r2s   = [f['val_r2']   for f in folds]

        median_n   = int(np.median(best_ns))
        p90_n      = int(np.percentile(best_ns, 90))
        mean_rmse  = round(float(np.mean(val_rmses)), 4)
        mean_r2    = round(float(np.mean(val_r2s)), 4)
        all_best_ns.extend(best_ns)

        summary_rows.append({
            'pos':       pos,
            'median_n':  median_n,
            'p90_n':     p90_n,
            'mean_val_rmse': mean_rmse,
            'mean_val_r2':   mean_r2,
            'folds':     len(folds),
        })
        print(f"  → median best_n={median_n}  p90={p90_n}  "
              f"mean_val_rmse={mean_rmse}  mean_val_r²={mean_r2}\n")

    # ── final recommendation ──────────────────────────────────────────────
    if not all_best_ns:
        print("No results — check DB_FILE and that data is ingested.")
        return

    recommended = int(np.percentile(all_best_ns, 90))

    print("=" * 68)
    print("SUMMARY")
    print("=" * 68)
    print(f"{'POS':<6} {'median_n':>8} {'p90_n':>7} {'val_rmse':>10} {'val_r²':>8} {'folds':>6}")
    print("-" * 68)
    for r in summary_rows:
        print(f"{r['pos']:<6} {r['median_n']:>8} {r['p90_n']:>7} "
              f"{r['mean_val_rmse']:>10.4f} {r['mean_val_r2']:>8.4f} {r['folds']:>6}")
    print("=" * 68)
    print(f"\nCurrent N_ESTIMATORS  : 200")
    print(f"Recommended (p90 all) : {recommended}")
    print(f"\nTo apply: edit train/trainer.py → N_ESTIMATORS = {recommended}")


if __name__ == '__main__':
    main()
