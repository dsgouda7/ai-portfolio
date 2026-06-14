import os
import sqlite3

import pandas as pd
import requests

PLAYERS_DIR      = './Fantasy-Premier-League/data/2023-24/players/'
RAW_DATA_PATH    = './Fantasy-Premier-League/data/2023-24/players_raw.csv'
DB_FILE          = 'fantasy_football.db'
MODELS_FILE      = 'models.joblib'
GAME_WEEK        = 38
MAX_PLAYERS_PER_TEAM = 4
MAX_SPEND        = 1000
FORM_WINDOW      = 5

ROLL_COLS = [
    'total_points', 'minutes', 'goals_scored', 'assists',
    'clean_sheets', 'bonus', 'ict_index', 'creativity', 'threat',
    'influence', 'goals_conceded', 'saves',
]

# features passed to XGBoost:
#   roll5_* = rolling FORM_WINDOW-GW mean of each stat (lagged by 1, no leakage)
#   was_home, value, opponent_team = current-GW context
#   element_type_enc = position as integer (1=GK 2=DEF 3=MID 4=FWD)
FEATURES = [f'roll5_{c}' for c in ROLL_COLS] + ['was_home', 'value', 'opponent_team', 'element_type_enc']

# no auth required; mirrors data from the vaastav CSV archive
FPL_API_BASE = 'https://fantasy.premierleague.com/api'


def ingest(players_dir, raw_data_path, db_file):
    if os.path.exists(db_file):
        print(f"DB found at {db_file}, skipping ingest. Delete it to re-ingest.")
        return

    print("Ingesting data into SQLite...")
    conn = sqlite3.connect(db_file)

    pd.read_csv(raw_data_path).to_sql('players_raw', conn, if_exists='replace', index=False)

    frames = []
    for folder in os.scandir(players_dir):
        if not folder.is_dir():
            continue
        _, player_id = folder.name.rsplit('_', 1)
        df = pd.read_csv(os.path.join(folder, 'gw.csv'))
        df['player_id'] = int(player_id)
        # row position == game week index (matches original index-as-GW convention)
        df['Game_Week'] = range(len(df))
        frames.append(df)

    pd.concat(frames, ignore_index=True).to_sql('player_gw', conn, if_exists='replace', index=False)
    conn.close()
    print("Ingest complete.")


def build_features(db_file):
    """
    Merge player metadata with GW history and compute rolling features.

    Roll stats are shifted by 1 before windowing so target-GW data never
    leaks into the features. Returns one row per (player, GW) with a
    'target' column containing next-GW total_points.
    """
    conn = sqlite3.connect(db_file)
    raw = pd.read_sql(
        "SELECT element_type, team, second_name, first_name, id FROM players_raw",
        conn,
    )
    gw_cols = ', '.join([
        'player_id', 'Game_Week', 'total_points', 'minutes', 'goals_scored',
        'assists', 'clean_sheets', 'bonus', 'ict_index', 'creativity', 'threat',
        'influence', 'goals_conceded', 'saves', 'was_home', 'value', 'opponent_team',
    ])
    all_gw = pd.read_sql(f"SELECT {gw_cols} FROM player_gw", conn)
    conn.close()

    # keep element_type as both a human-readable label and a numeric feature
    raw['element_type_enc'] = raw['element_type']   # 1/2/3/4 -- XGBoost reads this directly
    raw['element_type'] = raw['element_type'].map({1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'})

    df = raw.merge(all_gw, left_on='id', right_on='player_id', how='inner')
    df = df.sort_values(['id', 'Game_Week']).reset_index(drop=True)

    for col in ROLL_COLS:
        df[f'roll5_{col}'] = (
            df.groupby('id')[col]
            .transform(lambda x: x.shift(1).rolling(FORM_WINDOW, min_periods=1).mean())
        )

    df['target'] = df.groupby('id')['total_points'].shift(-1)
    return df



def _ensure_registry(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS training_registry (
            run_id        TEXT,
            player_id     INTEGER,
            first_name    TEXT,
            second_name   TEXT,
            element_type  TEXT,
            team          INTEGER
        )
    """)
    conn.commit()


def register_trained_players(db_file, df_trained):
    """
    Record which players were included in the current training run.

    Each call appends a new batch tagged with the current UTC timestamp so
    you can diff runs and spot newly arrived players whose history the model
    has never seen.
    """
    import datetime
    run_id = datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')

    id_col = 'id' if 'id' in df_trained.columns else 'player_id'
    registry = df_trained[[
        id_col, 'first_name', 'second_name', 'element_type', 'team'
    ]].drop_duplicates(subset=[id_col]).copy()
    registry = registry.rename(columns={id_col: 'player_id'})
    registry['run_id'] = run_id

    conn = sqlite3.connect(db_file)
    _ensure_registry(conn)
    registry.to_sql('training_registry', conn, if_exists='append', index=False)
    conn.close()
    print(f"  Registered {len(registry)} players under run {run_id}")


def new_players_since_last_run(db_file):
    """
    Return player_ids in players_raw that are NOT in the most recent
    training_registry run -- these need GW history backfilled before
    re-training.
    """
    conn = sqlite3.connect(db_file)
    _ensure_registry(conn)
    last_run = pd.read_sql(
        "SELECT player_id FROM training_registry WHERE run_id = "
        "(SELECT MAX(run_id) FROM training_registry)",
        conn,
    )
    current = pd.read_sql("SELECT id FROM players_raw", conn)
    conn.close()
    return set(current['id']) - set(last_run['player_id'])



def _table_exists(conn, name):
    cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (name,))
    return cur.fetchone() is not None


def ingest_from_fpl_api(db_file, player_ids=None):
    """
    Fetch player metadata and GW history from the official FPL API and
    upsert into the local SQLite DB.

    player_ids: optional list of ints. If None, fetches ALL players
    (full refresh). Pass new_players_since_last_run() to only backfill
    missing players -- much faster for daily updates.
    """
    print("Fetching bootstrap-static from FPL API...")
    resp = requests.get(f'{FPL_API_BASE}/bootstrap-static/', timeout=30)
    resp.raise_for_status()
    players_raw = pd.DataFrame(resp.json()['elements'])

    if player_ids is not None:
        players_raw = players_raw[players_raw['id'].isin(player_ids)]

    conn = sqlite3.connect(db_file)
    existing = pd.read_sql("SELECT * FROM players_raw", conn) if _table_exists(conn, 'players_raw') else pd.DataFrame()
    if not existing.empty and player_ids is not None:
        existing = existing[~existing['id'].isin(player_ids)]
    pd.concat([existing, players_raw], ignore_index=True).to_sql('players_raw', conn, if_exists='replace', index=False)
    conn.close()

    frames = []
    print(f"Fetching GW history for {len(players_raw)} players...")
    for pid in players_raw['id'].tolist():
        try:
            r = requests.get(f'{FPL_API_BASE}/element-summary/{pid}/', timeout=15)
            r.raise_for_status()
            history = r.json().get('history', [])
            if not history:
                continue
            df = pd.DataFrame(history)
            df['player_id'] = pid
            df['Game_Week'] = range(len(df))
            # API returns ICT/creativity/threat/influence as strings
            for col in ('ict_index', 'creativity', 'threat', 'influence'):
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            frames.append(df)
        except requests.RequestException as exc:
            print(f"  Warning: could not fetch player {pid}: {exc}")

    if frames:
        gw_new = pd.concat(frames, ignore_index=True)
        conn = sqlite3.connect(db_file)
        existing_gw = pd.read_sql("SELECT * FROM player_gw", conn) if _table_exists(conn, 'player_gw') else pd.DataFrame()
        if not existing_gw.empty and player_ids is not None:
            existing_gw = existing_gw[~existing_gw['player_id'].isin(player_ids)]
        pd.concat([existing_gw, gw_new], ignore_index=True).to_sql('player_gw', conn, if_exists='replace', index=False)
        conn.close()
        print(f"  Upserted GW history for {len(players_raw)} players.")
