"""
train.py -- ingest the FPL dataset and train one XGBRegressor per position.

Run this whenever the dataset is updated:
    python train/train.py

On each run it:
  1. git pulls the vaastav Fantasy-Premier-League submodule (latest GW data)
  2. Auto-detects the current season and last played GW from the submodule
  3. Ingests players_raw.csv + per-player gw.csv + history.csv into SQLite
  4. Fetches Understat attributes for any new players not yet cached
  5. Trains 4 XGBRegressors (one per position) with form + DNA + trajectory
  6. Saves models, metrics, EPL snapshot, and feature metadata to models.joblib

Delete fantasy_football.db to force a full re-ingest of the CSV files.
"""
import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import sqlite3
import subprocess

import joblib
import numpy as np
import pandas as pd
import requests
from sklearn.metrics import r2_score, root_mean_squared_error
from xgboost import XGBRegressor

from utils import (
    DB_FILE, MODELS_FILE, PLAYERS_DIR, RAW_DATA_PATH, GAME_WEEK, SEASON,
    FPL_API_BASE, POS_FEATURES, MODEL_NAMES,
    ingest, ingest_from_fpl_api, supplement_gw_from_api,
    build_features, register_trained_players,
)
from eligibility import get_eligibility, get_epl_members
from player_attributes import ensure_new_players
from transfer_values import ensure_transfer_values

N_ESTIMATORS = 200  # boosting rounds; 200 converges well at lr=0.1 on ~20k rows

# ---------------------------------------------------------------------------
# Step 1: Pull latest data from the vaastav submodule
# ---------------------------------------------------------------------------
print(f"Season: {SEASON}  |  GAME_WEEK: {GAME_WEEK}")
print("Syncing Fantasy-Premier-League submodule...")
try:
    result = subprocess.run(
        ['git', '-C', 'Fantasy-Premier-League', 'pull', '--ff-only'],
        capture_output=True, text=True, timeout=60,
    )
    msg = (result.stdout + result.stderr).strip()
    print(f"  {msg or 'already up to date'}")
except Exception as exc:
    print(f"  Warning: submodule pull failed ({exc}) — proceeding with local data")


def train_models(df, game_week):
    """
    Train one XGBRegressor per position on all GWs before game_week-1.

    Each model uses only the features relevant to its position (POS_FEATURES),
    removing cross-position noise (e.g. saves for outfielders, threat for GKs).
    Returns (models, metrics) dicts keyed by position; metrics includes r2,
    rmse, n, top_feature, model_name, and n_features.
    """
    train = df[(df['Game_Week'] < game_week - 1) & df['target'].notna()].copy()
    models = {}
    metrics = {}

    for pos in ['GK', 'DEF', 'MID', 'FWD']:
        pos_features = POS_FEATURES[pos]
        pos_train = train[train['element_type'] == pos]
        X = pos_train[pos_features].fillna(0)
        # FPL API and vaastav can return team/opponent_team as object dtype.
        # Cast any remaining object columns to int so XGBoost accepts them.
        for _c in X.select_dtypes(include='object').columns:
            X[_c] = pd.to_numeric(X[_c], errors='coerce').fillna(0).astype(int)
        y = pos_train['target']

        model = XGBRegressor(
            n_estimators=N_ESTIMATORS,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=0,
        )
        model.fit(X, y)
        models[pos] = model

        y_pred = model.predict(X)
        top_feat = pos_features[model.feature_importances_.argmax()]
        metrics[pos] = {
            'r2':          round(float(r2_score(y, y_pred)), 4),
            'rmse':        round(float(root_mean_squared_error(y, y_pred)), 4),
            'n':           len(pos_train),
            'top_feature': top_feat,
            'model_name':  MODEL_NAMES[pos],
            'n_features':  len(pos_features),
        }
        m = metrics[pos]
        print(f"  [{MODEL_NAMES[pos]}]")
        print(f"    n={m['n']:,}  features={m['n_features']}  R²={m['r2']:.4f}  RMSE={m['rmse']:.4f}  top={top_feat}")

    return models, metrics


print("Fetching EPL member list from FPL API (filters non-EPL players from DB)...")
try:
    epl_members = get_epl_members()
    print(f"  {len(epl_members)} current EPL players found")
except Exception as exc:
    print(f"  Warning: EPL member fetch failed ({exc}), proceeding without DB filter")
    epl_members = None

ingest(PLAYERS_DIR, RAW_DATA_PATH, DB_FILE, epl_members=epl_members)

# ---------------------------------------------------------------------------
# Step 2: Supplement from FPL API if vaastav is stale
# ---------------------------------------------------------------------------
# vaastav is only updated ~3x/year.  After each gameweek, the FPL API has
# results before vaastav does.  We detect this by comparing GAME_WEEK (the
# last GW in the submodule's CSV files) to the last finished GW from the
# live FPL API.  If the API is ahead, we call ingest_from_fpl_api() which
# makes one summary request per player and upserts the result into the DB.
game_week = GAME_WEEK  # may be bumped up if FPL API has newer data
try:
    _events = requests.get(f'{FPL_API_BASE}/bootstrap-static/', timeout=20).json().get('events', [])
    _finished = [e['id'] for e in _events if e.get('finished')]
    _live_gw  = max(_finished) if _finished else GAME_WEEK
    if _live_gw > GAME_WEEK:
        print(f"Vaastav is {_live_gw - GAME_WEEK} GW(s) stale "
              f"(submodule=GW{GAME_WEEK}, FPL live=GW{_live_gw}) "
              f"— supplementing GW history from FPL API (this may take a few minutes)...")
        # Supplement player_gw only — do NOT replace players_raw.
        # The vaastav player list is correct for the season; only the GW rows
        # are incomplete.  Using the FPL API players_raw would risk pulling
        # next-season data (IDs change at season rollover) and breaking the
        # inner join with the current season's GW history.
        with sqlite3.connect(DB_FILE) as _conn:
            _pids = pd.read_sql("SELECT id FROM players_raw", _conn)['id'].tolist()
        _fetched_gw = supplement_gw_from_api(DB_FILE, _pids)
        game_week = _fetched_gw if _fetched_gw > 0 else _live_gw
        print(f"  Training will use GW{game_week} as current GW.")
    else:
        print(f"Vaastav is current (GW{GAME_WEEK}).")
except Exception as exc:
    import traceback
    print(f"Warning: FPL staleness check failed ({exc}) — training on vaastav data only")
    traceback.print_exc()

# Fetch EA FC / SOFIFA attributes for any player not yet in player_attributes.
# No-ops if everyone is already cached.  Only hits the network for new arrivals.
with sqlite3.connect(DB_FILE) as _attr_conn:
    ensure_new_players(_attr_conn)

# Fetch Transfermarkt market values for new / retry players.
# Uses the Reep cross-provider register (FPL code -> TM ID) then calls the
# Transfermarkt scraper at 1 req/s.  Only updates stored values when TM's own
# last_updated date is strictly newer.  Missing players get mean imputation
# and are retried on every subsequent train run until real data is found.
with sqlite3.connect(DB_FILE) as _tm_conn:
    ensure_transfer_values(_tm_conn)

print("Building features...")
all_data = build_features(DB_FILE)

print("Training XGBoost models (one per position)...")
models, metrics = train_models(all_data, game_week)

joblib.dump({'models': models, 'metrics': metrics}, MODELS_FILE)
print(f"Models + metrics saved to {MODELS_FILE}")

# Save the EPL member set captured above (status != 'u' at train time).
# This is used as a tier-1 pool pre-filter at inference: it removes players
# who have left the league since training, without touching injured/suspended
# players (those are handled by the tier-2 live API check).
checkpoint = joblib.load(MODELS_FILE)
checkpoint['epl_members'] = epl_members
checkpoint['model_names'] = MODEL_NAMES
checkpoint['pos_features'] = {pos: list(feats) for pos, feats in POS_FEATURES.items()}
joblib.dump(checkpoint, MODELS_FILE)
if epl_members is not None:
    print(f"EPL member snapshot ({len(epl_members)} players) saved to {MODELS_FILE}")
print(f"Model names and per-position feature sets saved to {MODELS_FILE}")

print("Registering trained players in DB...")
register_trained_players(DB_FILE, all_data)
