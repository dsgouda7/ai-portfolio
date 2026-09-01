"""
Pipeline Step 1 — Ingest
========================
Reads the vaastav Fantasy-Premier-League CSV archive (mounted or local),
calls the live FPL Bootstrap API to supplement any stale game weeks,
enriches with player attributes (SOFIFA) and Transfermarkt market values,
and writes the result to a SQLite database.

I/O contract (env vars)
-----------------------
  FPL_VAASTAV_DIR   path to the Fantasy-Premier-League/data/ directory
                    Default: <project_root>/Fantasy-Premier-League/data
  FPL_DB_FILE       full path to write fantasy_football.db
                    Default: <project_root>/fantasy_football.db

In an Azure ML pipeline this step would:
  - Mount the vaastav AML dataset at FPL_VAASTAV_DIR
  - Write to a pipeline Output at the directory containing FPL_DB_FILE

Local run (no container):
  python pipeline/steps/ingest.py

Container run:
  docker run --rm \\
    -e FPL_VAASTAV_DIR=/data/vaastav \\
    -e FPL_DB_FILE=/data/output/fantasy_football.db \\
    -v $(pwd)/Fantasy-Premier-League:/data/vaastav:ro \\
    -v pipeline_data:/data/output \\
    fpl-ingest
"""
import os
import pathlib
import sqlite3
import sys

import requests

# Make project root importable regardless of cwd
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))

from utils import (
    DB_FILE, PLAYERS_DIR, RAW_DATA_PATH, GAME_WEEK, SEASON,
    FPL_API_BASE,
    ingest, refresh_live_season_data,
)
from eligibility import get_epl_members
from player_attributes import ensure_new_players
from transfer_values import ensure_transfer_values
from external_appearances import refresh_external_appearances


def main() -> None:
    print(f"[ingest] Season: {SEASON}  |  GW: {GAME_WEEK}")
    print(f"[ingest] vaastav root : {os.environ.get('FPL_VAASTAV_DIR', '(local submodule)')}")
    print(f"[ingest] DB target    : {DB_FILE}")

    # Ensure the output directory exists (needed when DB_FILE is on a mounted volume)
    pathlib.Path(DB_FILE).parent.mkdir(parents=True, exist_ok=True)

    # Step 1a — fetch current EPL member list from FPL API (used to prune non-EPL
    #           players from the training pool; non-fatal if the API is unavailable)
    try:
        epl_members = get_epl_members()
        print(f"  EPL API: {len(epl_members)} current players")
    except Exception as exc:
        print(f"  Warning: EPL member fetch failed ({exc}) — proceeding without filter")
        epl_members = None

    # Step 1b — ingest vaastav CSVs into SQLite
    ingest(PLAYERS_DIR, RAW_DATA_PATH, DB_FILE, epl_members=epl_members)

    # Step 1c — replace the current-season slice with every player-GW row
    # available from the live FPL API and append a separate future snapshot.
    try:
        response = requests.get(f'{FPL_API_BASE}/bootstrap-static/', timeout=20)
        response.raise_for_status()
        refresh = refresh_live_season_data(DB_FILE, response.json())
        print(
            f"  Live FPL history: {refresh.get('live_rows', 0):,} rows; "
            f"completed cutoff={refresh.get('completed_internal_index')}; "
            f"snapshot={refresh.get('snapshot_game_week')}"
        )
    except Exception as exc:
        raise RuntimeError(
            'Full live FPL history refresh failed; refusing to train from a '
            'partial or ambiguously dated dataset.'
        ) from exc

    # Step 1d — enrich with SOFIFA player attributes (no-ops for cached players)
    with sqlite3.connect(DB_FILE) as conn:
        ensure_new_players(conn)

    # Step 1e — enrich with Transfermarkt market values (no-ops for fresh cache)
    with sqlite3.connect(DB_FILE) as conn:
        ensure_transfer_values(conn)
        external_rows = refresh_external_appearances(conn)
        print(f"  External non-PL appearances: {external_rows:,}")

    print("[ingest] Done.")


if __name__ == "__main__":
    main()
