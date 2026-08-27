"""
train.py -- ingest temporal FPL data and train selectable model families.

Run this whenever the dataset is updated:
    python train/train.py

On each run it:
  1. git pulls the vaastav Fantasy-Premier-League submodule (latest GW data)
  2. Auto-detects the current season and last played GW from the submodule
  3. Ingests players_raw.csv + per-player gw.csv + history.csv into SQLite
    4. Fetches Understat attributes for any new players not yet cached
    5. Persists or reuses model-ready temporal features in local SQLite
    6. Trains XGBoost and/or full-history GRU position models
    7. Saves separate checkpoints with metrics and temporal cutoff manifests

Delete fantasy_football.db to force a full re-ingest of the CSV files.
"""
import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import argparse
import sqlite3
import subprocess

import requests

from utils import (
    DB_FILE, PLAYERS_DIR, RAW_DATA_PATH, GAME_WEEK, SEASON,
    FPL_API_BASE,
    ingest, refresh_live_season_data,
    build_features, register_trained_players, get_runtime_context,
)
from eligibility import get_epl_members
from player_attributes import ensure_new_players
from transfer_values import ensure_transfer_values
from train.model_training import train_and_save_models
from feature_cache import load_or_build_feature_cache

parser = argparse.ArgumentParser()
parser.add_argument(
    '--model', choices=('xgboost', 'rnn', 'all'), default='all',
    help='Model artifact(s) to train (default: all).',
)
args = parser.parse_args()

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


print("Fetching EPL member list from FPL API (filters non-EPL players from DB)...")
try:
    epl_members = get_epl_members()
    print(f"  {len(epl_members)} current EPL players found")
except Exception as exc:
    print(f"  Warning: EPL member fetch failed ({exc}), proceeding without DB filter")
    epl_members = None

ingest(PLAYERS_DIR, RAW_DATA_PATH, DB_FILE, epl_members=epl_members)

# ---------------------------------------------------------------------------
# Step 2: Replace current-season data with complete live FPL histories
# ---------------------------------------------------------------------------
# The archive provides prior-season context, but the live API is authoritative
# for the current season. Fetch every active player's full available history so
# both model families train from the same complete set of finished Gameweeks.
try:
    _bootstrap_response = requests.get(
        f'{FPL_API_BASE}/bootstrap-static/', timeout=20
    )
    _bootstrap_response.raise_for_status()
    _refresh = refresh_live_season_data(DB_FILE, _bootstrap_response.json())
    print(
        f"Live FPL history: {_refresh.get('live_rows', 0):,} rows; "
        f"completed cutoff={_refresh.get('completed_internal_index')}; "
        f"snapshot={_refresh.get('snapshot_game_week')}"
    )
except Exception as exc:
    raise RuntimeError(
        'Full live FPL history refresh failed; refusing to train from a '
        'partial or ambiguously dated dataset.'
    ) from exc

# Fetch EA FC / SOFIFA attributes for any player not yet in player_attributes.
# No-ops if everyone is already cached.  Only hits the network for new arrivals.
with sqlite3.connect(DB_FILE) as _attr_conn:
    ensure_new_players(_attr_conn)

# Fetch Transfermarkt market values for new / retry players.
# Uses the Reep cross-provider register (FPL code -> TM ID) then calls the
# Transfermarkt scraper at 1 req/s.  Only updates stored values when TM's own
# last_updated date is strictly newer. Missing players remain unavailable and
# are retried; synthetic market values are never stored.
with sqlite3.connect(DB_FILE) as _tm_conn:
    ensure_transfer_values(_tm_conn)

print("Building features...")
all_data, cache_hit = load_or_build_feature_cache(DB_FILE, build_features)
print(f"Feature cache: {'hit' if cache_hit else 'rebuilt in SQLite'}")
runtime = get_runtime_context(DB_FILE)
train_and_save_models(
    all_data,
    runtime['completed_internal_index'],
    args.model,
    epl_members,
    DB_FILE,
)

print("Registering trained players in DB...")
register_trained_players(DB_FILE, all_data)
