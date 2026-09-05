import json
import os
import pathlib
import re
import sqlite3
from collections import Counter
from contextlib import closing

import numpy as np
import pandas as pd
import requests

_ROOT = pathlib.Path(__file__).parent

# ---------------------------------------------------------------------------
# Season auto-detection
# ---------------------------------------------------------------------------
# Reads the Fantasy-Premier-League submodule and picks the latest season
# folder (e.g. 2025-26), counts the GW CSV files to find the last played GW,
# and exposes SEASON, PLAYERS_DIR, RAW_DATA_PATH, GAME_WEEK as module-level
# constants. Falls back to 2023-24 / GW38 if the submodule is missing (e.g.
# before first setup).

def _autodetect_season(
    base: str = None,
) -> tuple[str, str, str, int]:
    """
    Return (season, players_dir, raw_data_path, game_week).
    season   : '2025-26'
    game_week: last played GW number (= snapshot GW used for prediction).
    """
    if base is None:
        base = str(_ROOT / 'Fantasy-Premier-League' / 'data')
    if not os.path.isdir(base):
        return ('2023-24',
                str(_ROOT / 'Fantasy-Premier-League' / 'data' / '2023-24' / 'players') + os.sep,
                str(_ROOT / 'Fantasy-Premier-League' / 'data' / '2023-24' / 'players_raw.csv'),
                38)

    seasons = sorted(
        d for d in os.listdir(base)
        if os.path.isdir(os.path.join(base, d)) and re.match(r'\d{4}-\d{2}$', d)
    )
    if not seasons:
        return ('2023-24',
                './Fantasy-Premier-League/data/2023-24/players/',
                './Fantasy-Premier-League/data/2023-24/players_raw.csv',
                38)

    season = seasons[-1]
    gw_nums = []
    for candidate in reversed(seasons):
        gws_dir = os.path.join(base, candidate, 'gws')
        candidate_gws = []
        if os.path.isdir(gws_dir):
            for f in os.listdir(gws_dir):
                m = re.match(r'gw(\d+)\.csv', f)
                if m:
                    candidate_gws.append(int(m.group(1)))
        if candidate_gws:
            season = candidate
            gw_nums = candidate_gws
            break

    season_dir = os.path.join(base, season)

    game_week = max(gw_nums) if gw_nums else 38
    return (
        season,
        os.path.join(season_dir, 'players') + os.sep,
        os.path.join(season_dir, 'players_raw.csv'),
        game_week,
    )


SEASON, PLAYERS_DIR, RAW_DATA_PATH, GAME_WEEK = _autodetect_season(
    # FPL_VAASTAV_DIR lets the ingest container mount vaastav as a volume.
    # Falls back to the submodule path used by local setup.ps1 runs.
    base=os.environ.get('FPL_VAASTAV_DIR'),
)

# ---------------------------------------------------------------------------
# Pipeline I/O paths
# ---------------------------------------------------------------------------
# Each path is controlled by an env var so the same source tree runs both
# locally (defaults below) and inside a containerised pipeline step, where
# the runner mounts directories and injects the env vars.
#
#   Local default         Container / AML mount
#   ─────────────────     ──────────────────────────────────────────────────
#   fantasy_football.db   $FPL_DATA_DIR/fantasy_football.db
#   models.joblib         $FPL_MODELS_DIR/models.joblib
#
# To run a step in a container:
#   docker run --env FPL_DATA_DIR=/mnt/data --env FPL_MODELS_DIR=/mnt/models …
def _artifact_path(env_name: str, primary: str, fallback: str) -> str:
    configured = os.environ.get(env_name)
    if configured:
        return configured
    primary_path = _ROOT / primary
    if primary_path.exists() and primary_path.stat().st_size > 0:
        return str(primary_path)
    return str(_ROOT / fallback)


DB_FILE = _artifact_path(
    'FPL_DB_FILE', 'fantasy_football.db', 'fantasy_football_current.db'
)
MODELS_FILE = _artifact_path(
    'FPL_MODELS_FILE', 'models.joblib', 'models_current.joblib'
)

MAX_PLAYERS_PER_TEAM = 3
MAX_SPEND        = 1000
FORM_WINDOW      = 5
MARKET_VALUE_WEIGHT_PL_HISTORY = 0.10
MARKET_VALUE_WEIGHT_NO_HISTORY = 0.30

ROLL_COLS = [
    'total_points', 'minutes', 'goals_scored', 'assists',
    'clean_sheets', 'bonus', 'bps', 'ict_index', 'creativity', 'threat',
    'influence', 'goals_conceded', 'saves',
    # StatsBomb expected stats + discipline (v2 feature expansion)
    'expected_goals', 'expected_assists', 'expected_goals_conceded',
    'starts', 'yellow_cards', 'red_cards', 'penalties_saved',
    'defensive_contribution', 'recoveries', 'tackles',
    'clearances_blocks_interceptions',
]

MARKET_FEATURES = [
    'ownership_log', 'ownership_change_log',
    'transfers_in_log', 'transfers_out_log',
    'transfer_balance_log', 'transfer_momentum_per_owner',
    'price_change_1', 'price_change_3',
]

FIXTURE_CONTEXT_FEATURES = [
    'rest_days', 'rest_days_available', 'matches_previous_14d',
    'team_elo_pre', 'opponent_elo_pre', 'elo_difference',
    'team_goals_for_roll5', 'team_goals_against_roll5',
    'opponent_goals_for_roll5', 'opponent_goals_against_roll5',
]

EXTERNAL_APPEARANCE_FEATURES = [
    'external_minutes_90d', 'external_appearances_90d',
    'external_goal_involvements_365d', 'external_minutes_365d',
    'external_data_available',
]

# Human-readable description for each position model (stored in checkpoint)
MODEL_NAMES = {
    'GK':  'Goalkeeper \u2014 save-rate & clean-sheet model',
    'DEF': 'Defender \u2014 defensive block & set-piece threat model',
    'MID': 'Midfielder \u2014 creative output & goal involvement model',
    'FWD': 'Forward \u2014 attacking output & conversion model',
}

# Per-position feature sets.  Each model is trained and scores players with
# only the features relevant to that position, removing noise (e.g. saves is
# meaningful for GKs, pure zeros for everyone else).  element_type_enc is
# dropped since each model IS one position.
POS_FEATURES = {
    'GK': [
        'roll5_total_points', 'roll5_minutes', 'roll5_clean_sheets',
        'roll5_goals_conceded', 'roll5_expected_goals_conceded',
        'roll5_saves', 'roll5_penalties_saved',
        'roll5_bonus', 'roll5_bps', 'roll5_influence',
        'roll5_starts', 'roll5_yellow_cards', 'roll5_red_cards',
        'was_home', 'value', 'opponent_team', 'team',
        'tm_market_value', 'tm_value_available',
        *MARKET_FEATURES, *FIXTURE_CONTEXT_FEATURES, *EXTERNAL_APPEARANCE_FEATURES,
        'form_data_density',           # fraction of last 5 GWs played (0=none, 1=all)
        'player_age_years',            # fractional age at kickoff date
        # Season-over-season trajectory
        'hist_prev_pts_per90', 'hist_prev_cs_per90', 'hist_prev_saves_per90',
        'hist_career_seasons', 'hist_yoy_pts_delta',
    ],
    'DEF': [
        'roll5_total_points', 'roll5_minutes', 'roll5_clean_sheets',
        'roll5_goals_conceded', 'roll5_expected_goals_conceded',
        'roll5_goals_scored', 'roll5_assists',
        'roll5_expected_goals', 'roll5_expected_assists',
        'roll5_creativity', 'roll5_influence',
        'roll5_bonus', 'roll5_bps', 'roll5_starts',
        'roll5_defensive_contribution', 'roll5_recoveries',
        'roll5_tackles', 'roll5_clearances_blocks_interceptions',
        'roll5_yellow_cards', 'roll5_red_cards',
        'was_home', 'value', 'opponent_team', 'team',
        'tm_market_value', 'tm_value_available',
        *MARKET_FEATURES, *FIXTURE_CONTEXT_FEATURES, *EXTERNAL_APPEARANCE_FEATURES,
        'form_data_density',           # fraction of last 5 GWs played (0=none, 1=all)
        'player_age_years',            # fractional age at kickoff date
        # Season-over-season trajectory
        'hist_prev_pts_per90', 'hist_prev_cs_per90',
        'hist_prev_goals_per90', 'hist_prev_assists_per90',
        'hist_career_seasons', 'hist_yoy_pts_delta',
    ],
    'MID': [
        'roll5_total_points', 'roll5_minutes',
        'roll5_goals_scored', 'roll5_assists', 'roll5_clean_sheets',
        'roll5_expected_goals', 'roll5_expected_assists',
        'roll5_creativity', 'roll5_influence',
        'roll5_bonus', 'roll5_bps', 'roll5_starts',
        'roll5_defensive_contribution', 'roll5_recoveries', 'roll5_tackles',
        'roll5_yellow_cards', 'roll5_red_cards',
        'was_home', 'value', 'opponent_team', 'team',
        'tm_market_value', 'tm_value_available',
        *MARKET_FEATURES, *FIXTURE_CONTEXT_FEATURES, *EXTERNAL_APPEARANCE_FEATURES,
        'form_data_density',           # fraction of last 5 GWs played (0=none, 1=all)
        'player_age_years',            # fractional age at kickoff date
        # Season-over-season trajectory
        'hist_prev_pts_per90', 'hist_prev_goals_per90',
        'hist_prev_assists_per90', 'hist_prev_xg_per90', 'hist_prev_xa_per90',
        'hist_career_seasons', 'hist_yoy_pts_delta',
    ],
    'FWD': [
        'roll5_total_points', 'roll5_minutes',
        'roll5_goals_scored', 'roll5_assists',
        'roll5_expected_goals', 'roll5_expected_assists',
        'roll5_threat', 'roll5_influence',
        'roll5_bonus', 'roll5_bps', 'roll5_starts',
        'roll5_yellow_cards', 'roll5_red_cards',
        'was_home', 'value', 'opponent_team', 'team',
        'tm_market_value', 'tm_value_available',
        *MARKET_FEATURES, *FIXTURE_CONTEXT_FEATURES, *EXTERNAL_APPEARANCE_FEATURES,
        'form_data_density',           # fraction of last 5 GWs played (0=none, 1=all)
        'player_age_years',            # fractional age at kickoff date
        # Season-over-season trajectory
        'hist_prev_pts_per90', 'hist_prev_goals_per90',
        'hist_prev_assists_per90', 'hist_prev_xg_per90',
        'hist_career_seasons', 'hist_yoy_pts_delta',
    ],
}

# Player attribute columns.
# ea_* columns are a placeholder (SOFIFA blocked); tm_* columns are active.
ATTR_COLS: list[str] = [
    'tm_market_value',          # log10(EUR) from Transfermarkt via Reep ID bridge
    'tm_value_available',       # 1 only when a real dated valuation exists
    'tm_market_value_x_sparsity',  # tm_market_value * (1 - form_data_density)
                                    # Amplifies TM signal when rolling stats are thin
    'form_data_density',        # roll5_minutes / (FORM_WINDOW*90); 0=no recent
                                # minutes (new/injured), ~0.2=full-time regular.
                                # Used as blend weight after * FORM_WINDOW.
    'player_age_years',         # fractional age at each GW's kickoff date
                                # computed from birth_date in players_raw
]

# Season-over-season history features (from history.csv inside each player
# folder in the vaastav submodule).  Represent player trajectory / DNA:
#   hist_prev_*_per90  — per-90 rate from the most recent prior season
#   hist_yoy_pts_delta — change in pts/90 from 2 seasons ago to last season
#   hist_career_seasons — total EPL seasons on record (experience proxy)
HIST_COLS = [
    'hist_prev_pts_per90',
    'hist_prev_goals_per90',
    'hist_prev_assists_per90',
    'hist_prev_cs_per90',
    'hist_prev_saves_per90',
    'hist_prev_xg_per90',
    'hist_prev_xa_per90',
    'hist_career_seasons',
    'hist_yoy_pts_delta',
]

# no auth required; mirrors data from the vaastav CSV archive
FPL_API_BASE = 'https://fantasy.premierleague.com/api'


def _table_exists(conn, name):
    cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (name,))
    return cur.fetchone() is not None


def _season_from_events(events: list[dict]) -> str:
    """Derive an FPL season label from the first available event deadline."""
    deadlines = pd.to_datetime(
        [event.get('deadline_time') for event in events if event.get('deadline_time')],
        errors='coerce',
        utc=True,
    )
    deadlines = deadlines[~pd.isna(deadlines)]
    if len(deadlines) == 0:
        return SEASON
    first = min(deadlines)
    start_year = first.year if first.month >= 7 else first.year - 1
    return f'{start_year}-{str(start_year + 1)[-2:]}'


def ingest(players_dir, raw_data_path, db_file, epl_members=None):
    conn = sqlite3.connect(db_file)

    if not _table_exists(conn, 'players_raw'):
        conn.close()
        print("Ingesting data into SQLite...")
        conn = sqlite3.connect(db_file)

        raw_df = pd.read_csv(raw_data_path)
        if epl_members is not None:
            before = len(raw_df)
            raw_df = raw_df[
                raw_df.apply(
                    lambda r: (str(r['first_name']).lower(), str(r['second_name']).lower()) in epl_members,
                    axis=1,
                )
            ]
            print(f"  EPL filter: removed {before - len(raw_df)} non-EPL players from players_raw")

        raw_df.to_sql('players_raw', conn, if_exists='replace', index=False)
        valid_ids = set(raw_df['id'])

        frames = []
        for folder in os.scandir(players_dir):
            if not folder.is_dir():
                continue
            _, player_id = folder.name.rsplit('_', 1)
            if int(player_id) not in valid_ids:
                continue
            df = pd.read_csv(os.path.join(folder, 'gw.csv'))
            df['player_id'] = int(player_id)
            df['Game_Week'] = range(len(df))
            frames.append(df)

        pd.concat(frames, ignore_index=True).to_sql('player_gw', conn, if_exists='replace', index=False)

        # Also ingest per-season history for each player.  history.csv lives
        # alongside gw.csv in every player folder and contains one row per
        # prior season — the source of our season-over-season trajectory
        # features (hist_prev_pts_per90, hist_yoy_pts_delta, etc.).
        hist_frames = []
        for folder in os.scandir(players_dir):
            if not folder.is_dir():
                continue
            _, player_id = folder.name.rsplit('_', 1)
            if int(player_id) not in valid_ids:
                continue
            hist_path = os.path.join(folder.path, 'history.csv')
            if os.path.exists(hist_path):
                hdf = pd.read_csv(hist_path)
                hdf['player_id'] = int(player_id)
                hist_frames.append(hdf)
        if hist_frames:
            pd.concat(hist_frames, ignore_index=True).to_sql(
                'player_history', conn, if_exists='replace', index=False
            )

        conn.close()
        print("Ingest complete.")
        return

    # DB already exists — prune players who have since left the EPL
    if epl_members is not None:
        players = pd.read_sql("SELECT id, first_name, second_name FROM players_raw", conn)
        gone = players[
            ~players.apply(
                lambda r: (str(r['first_name']).lower(), str(r['second_name']).lower()) in epl_members,
                axis=1,
            )
        ]
        if not gone.empty:
            gone_ids = gone['id'].tolist()
            placeholders = ','.join('?' * len(gone_ids))
            conn.execute(f"DELETE FROM players_raw WHERE id IN ({placeholders})", gone_ids)
            conn.execute(f"DELETE FROM player_gw WHERE player_id IN ({placeholders})", gone_ids)
            conn.commit()
            names = [f"{r['first_name']} {r['second_name']}" for _, r in gone.iterrows()]
            print(f"  Pruned {len(gone)} non-EPL players from DB: {', '.join(names[:10])}"
                  + (' ...' if len(names) > 10 else ''))
        else:
            print("  DB up to date — no non-EPL players found")
    else:
        print(f"DB found at {db_file}, skipping prune (no EPL member list provided).")

    conn.close()


def refresh_current_roster(db_file: str, bootstrap: dict = None, fixtures: list = None) -> dict:
    """Create a preseason inference snapshot from the current official FPL roster."""
    if bootstrap is None:
        response = requests.get(f'{FPL_API_BASE}/bootstrap-static/', timeout=30)
        response.raise_for_status()
        bootstrap = response.json()

    events = bootstrap.get('events', [])
    finished = [event['id'] for event in events if event.get('finished')]
    target_event = next((event for event in events if not event.get('finished')), None)
    if finished or target_event is None:
        return {'rolled_over': False, 'reason': 'current season is not in preseason'}

    current = pd.DataFrame(bootstrap.get('elements', []))
    if current.empty:
        raise ValueError('FPL bootstrap response contained no players')
    current = current[
        current['code'].notna()
        & current.get('status', pd.Series('a', index=current.index)).ne('u')
        & ~current.get('removed', pd.Series(False, index=current.index)).fillna(False)
    ].copy()
    current['code'] = current['code'].astype(int)
    current['id'] = current['id'].astype(int)
    for column in current.select_dtypes(include=['object', 'str']).columns:
        current[column] = current[column].map(
            lambda value: json.dumps(value)
            if isinstance(value, (dict, list)) else value
        )

    if fixtures is None:
        response = requests.get(
            f"{FPL_API_BASE}/fixtures/?event={int(target_event['id'])}", timeout=30
        )
        response.raise_for_status()
        fixtures = response.json()

    fixture_by_team = {}
    for fixture in fixtures:
        home = int(fixture['team_h'])
        away = int(fixture['team_a'])
        fixture_by_team[home] = (away, True, fixture.get('kickoff_time'))
        fixture_by_team[away] = (home, False, fixture.get('kickoff_time'))

    with closing(sqlite3.connect(db_file)) as conn:
        old_raw = pd.read_sql('SELECT * FROM players_raw', conn)
        old_gw = pd.read_sql('SELECT * FROM player_gw', conn)
        if _table_exists(conn, 'app_metadata'):
            existing_metadata = dict(
                conn.execute('SELECT key, value FROM app_metadata').fetchall()
            )
            existing_snapshot = existing_metadata.get('snapshot_game_week')
            if existing_snapshot is not None:
                old_gw = old_gw[
                    old_gw['Game_Week'] != int(existing_snapshot)
                ].copy()
        old_history = (
            pd.read_sql('SELECT * FROM player_history', conn)
            if _table_exists(conn, 'player_history') else pd.DataFrame()
        )

        old_id_to_code = dict(zip(old_raw['id'].astype(int), old_raw['code'].astype(int)))
        code_to_current_id = dict(zip(current['code'], current['id']))
        old_to_current_id = {
            old_id: code_to_current_id[code]
            for old_id, code in old_id_to_code.items()
            if code in code_to_current_id
        }

        historical_gw = old_gw[old_gw['player_id'].isin(old_to_current_id)].copy()
        historical_gw['player_id'] = historical_gw['player_id'].map(old_to_current_id)
        snapshot_index = int(historical_gw['Game_Week'].max()) + 1

        snapshot_rows = []
        for player in current.to_dict('records'):
            row = {column: 0 for column in old_gw.columns}
            opponent, was_home, kickoff = fixture_by_team.get(
                int(player['team']), (0, False, target_event.get('deadline_time'))
            )
            row.update({
                'element': int(player['id']),
                'player_id': int(player['id']),
                'Game_Week': snapshot_index,
                'round': int(target_event['id']),
                'fixture': 0,
                'kickoff_time': kickoff,
                'opponent_team': opponent,
                'was_home': was_home,
                'value': int(player.get('now_cost') or 0),
                'modified': False,
            })
            snapshot_rows.append(row)

        current_gw = pd.DataFrame(snapshot_rows, columns=old_gw.columns)
        pd.concat([historical_gw, current_gw], ignore_index=True).to_sql(
            'player_gw', conn, if_exists='replace', index=False
        )
        current.to_sql('players_raw', conn, if_exists='replace', index=False)

        if not old_history.empty:
            current_history = old_history[
                old_history['player_id'].isin(old_to_current_id)
            ].copy()
            current_history['player_id'] = current_history['player_id'].map(old_to_current_id)
            current_history.to_sql('player_history', conn, if_exists='replace', index=False)

        for table in (
            'player_attributes', 'player_id_registry', 'player_transfer_values',
            'saved_squad',
        ):
            conn.execute(f'DROP TABLE IF EXISTS {table}')

        seasons = sorted(
            path.name for path in (_ROOT / 'Fantasy-Premier-League' / 'data').glob('*-*')
            if re.fullmatch(r'\d{4}-\d{2}', path.name)
        )
        current_season = seasons[-1] if seasons else SEASON
        pd.DataFrame([
            {'key': 'season', 'value': current_season},
            {'key': 'target_game_week', 'value': str(int(target_event['id']))},
            {'key': 'latest_completed_game_week', 'value': '0'},
            {'key': 'live_start_index', 'value': str(snapshot_index)},
            {'key': 'completed_internal_index', 'value': str(snapshot_index - 1)},
            {'key': 'snapshot_game_week', 'value': str(snapshot_index)},
            {'key': 'fpl_refreshed_at', 'value': pd.Timestamp.now(tz='UTC').isoformat()},
        ]).to_sql('app_metadata', conn, if_exists='replace', index=False)

    return {
        'rolled_over': True,
        'season': current_season,
        'target_game_week': int(target_event['id']),
        'snapshot_game_week': snapshot_index,
        'players': len(current),
        'retained': len(old_to_current_id),
        'arrivals': len(current) - len(old_to_current_id),
        'departed': len(old_raw) - len(old_to_current_id),
    }


def refresh_live_season_data(
    db_file: str,
    bootstrap: dict = None,
) -> dict:
    """Download all current-season player-GW rows and create the next snapshot.

    Prior-season rows are retained as sequence context and remapped through the
    stable FPL ``code``. Current-season rows replace only the previous live
    snapshot/current-season slice. The returned internal indexes deliberately
    separate the latest completed row from the future inference snapshot.
    """
    if bootstrap is None:
        response = requests.get(f'{FPL_API_BASE}/bootstrap-static/', timeout=30)
        response.raise_for_status()
        bootstrap = response.json()

    events = bootstrap.get('events', [])
    current_season = _season_from_events(events)
    completed_events = [
        int(event['id']) for event in events
        if event.get('finished') and event.get('data_checked')
    ]
    latest_completed = max(completed_events) if completed_events else 0
    target_event = next(
        (event for event in events if int(event['id']) == latest_completed + 1),
        None,
    )
    if target_event is None:
        target_event = next((event for event in events if not event.get('finished')), None)
    if target_event is None:
        return {'refreshed': False, 'reason': 'no future FPL event is available'}

    current = pd.DataFrame(bootstrap.get('elements', []))
    if current.empty:
        raise ValueError('FPL bootstrap response contained no players')
    current = current[current['code'].notna()].copy()
    current['id'] = current['id'].astype(int)
    current['code'] = current['code'].astype(int)
    for column in current.select_dtypes(include=['object', 'str']).columns:
        current[column] = current[column].map(
            lambda value: json.dumps(value) if isinstance(value, (dict, list)) else value
        )

    with closing(sqlite3.connect(db_file)) as conn:
        old_raw = pd.read_sql('SELECT * FROM players_raw', conn)
        old_gw = pd.read_sql('SELECT * FROM player_gw', conn)
        metadata = (
            dict(conn.execute('SELECT key, value FROM app_metadata').fetchall())
            if _table_exists(conn, 'app_metadata') else {}
        )

    if 'live_start_index' in metadata:
        historical_boundary = int(metadata['live_start_index'])
    elif 'snapshot_game_week' in metadata:
        # A preseason rollover snapshot follows retained prior-season rows.
        historical_boundary = int(metadata['snapshot_game_week'])
    elif SEASON == current_season:
        # Fresh in-season ingest: archive rows and API rows describe the same
        # season, so the live API must replace rather than duplicate them.
        historical_boundary = 0
    else:
        historical_boundary = (
            int(old_gw['Game_Week'].max()) + 1 if not old_gw.empty else 0
        )
    historical = old_gw[old_gw['Game_Week'] < historical_boundary].copy()

    old_id_to_code = dict(zip(old_raw['id'].astype(int), old_raw['code'].astype(int)))
    code_to_current_id = dict(zip(current['code'], current['id']))
    historical['player_id'] = historical['player_id'].map(
        lambda old_id: code_to_current_id.get(old_id_to_code.get(int(old_id)))
    )
    historical = historical[historical['player_id'].notna()].copy()
    historical['player_id'] = historical['player_id'].astype(int)
    if 'element' in historical.columns:
        historical['element'] = historical['player_id']

    live_start_index = (
        int(historical['Game_Week'].max()) + 1 if not historical.empty else 0
    )
    frames = []
    failed_player_ids = []
    player_ids = current['id'].tolist()
    total_history_players = len(player_ids)
    for player_number, player_id in enumerate(player_ids, start=1):
        for attempt in range(3):
            try:
                response = requests.get(
                    f'{FPL_API_BASE}/element-summary/{player_id}/', timeout=20
                )
                response.raise_for_status()
                rows = response.json().get('history', [])
                if not rows:
                    break
                player_history = pd.DataFrame(rows)
                player_history = player_history[
                    player_history['round'].astype(int) <= latest_completed
                ].copy()
                if player_history.empty:
                    break
                player_history['player_id'] = int(player_id)
                player_history['Game_Week'] = (
                    live_start_index + player_history['round'].astype(int) - 1
                )
                for column in ('ict_index', 'creativity', 'threat', 'influence'):
                    if column in player_history.columns:
                        player_history[column] = pd.to_numeric(
                            player_history[column], errors='coerce'
                        )
                frames.append(player_history)
                break
            except (requests.RequestException, ValueError, KeyError, TypeError):
                if attempt == 2:
                    failed_player_ids.append(int(player_id))
        if player_number % 50 == 0 or player_number == total_history_players:
            print(
                f'FPL history download: {player_number}/{total_history_players} players',
                flush=True,
            )

    if failed_player_ids:
        raise RuntimeError(
            'Incomplete FPL history download after three attempts for player IDs: '
            + ', '.join(map(str, failed_player_ids[:20]))
        )

    live_history = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    downloaded_rounds = (
        set(live_history['round'].astype(int)) if not live_history.empty else set()
    )
    missing_rounds = set(completed_events) - downloaded_rounds
    if missing_rounds:
        raise RuntimeError(
            'FPL history download is missing completed Gameweeks: '
            + ', '.join(map(str, sorted(missing_rounds)))
        )
    completed_internal_index = (
        live_start_index + latest_completed - 1 if latest_completed else live_start_index - 1
    )
    snapshot_index = completed_internal_index + 1

    fixture_response = requests.get(
        f"{FPL_API_BASE}/fixtures/?event={int(target_event['id'])}", timeout=30
    )
    fixture_response.raise_for_status()
    fixture_by_team = {}
    for fixture in fixture_response.json():
        home, away = int(fixture['team_h']), int(fixture['team_a'])
        fixture_by_team[home] = (away, True, fixture.get('kickoff_time'))
        fixture_by_team[away] = (home, False, fixture.get('kickoff_time'))

    columns = list(old_gw.columns)
    for column in live_history.columns:
        if column not in columns:
            columns.append(column)
    active = current[
        current.get('status', pd.Series('a', index=current.index)).ne('u')
        & ~current.get('removed', pd.Series(False, index=current.index)).fillna(False)
    ]
    snapshot_rows = []
    total_players = int(bootstrap.get('total_players') or 0)
    for player in active.to_dict('records'):
        row = {column: 0 for column in columns}
        opponent, was_home, kickoff = fixture_by_team.get(
            int(player['team']), (0, False, target_event.get('deadline_time'))
        )
        row.update({
            'element': int(player['id']),
            'player_id': int(player['id']),
            'Game_Week': snapshot_index,
            'round': int(target_event['id']),
            'fixture': 0,
            'kickoff_time': kickoff or target_event.get('deadline_time'),
            'opponent_team': opponent,
            'was_home': was_home,
            'value': int(player.get('now_cost') or 0),
            'selected': round(
                float(player.get('selected_by_percent') or 0) * total_players / 100
            ),
            'transfers_in': int(player.get('transfers_in_event') or 0),
            'transfers_out': int(player.get('transfers_out_event') or 0),
            'transfers_balance': (
                int(player.get('transfers_in_event') or 0)
                - int(player.get('transfers_out_event') or 0)
            ),
            'modified': False,
        })
        snapshot_rows.append(row)
    snapshot = pd.DataFrame(snapshot_rows, columns=columns)

    combined = pd.concat(
        [historical, live_history, snapshot], ignore_index=True, sort=False
    ).reindex(columns=columns)
    for column in current.columns:
        if column not in old_raw.columns:
            old_raw[column] = None

    metadata_rows = [
        {'key': 'season', 'value': current_season},
        {'key': 'latest_completed_game_week', 'value': str(latest_completed)},
        {'key': 'target_game_week', 'value': str(int(target_event['id']))},
        {'key': 'live_start_index', 'value': str(live_start_index)},
        {'key': 'completed_internal_index', 'value': str(completed_internal_index)},
        {'key': 'snapshot_game_week', 'value': str(snapshot_index)},
        {'key': 'fpl_refreshed_at', 'value': pd.Timestamp.now(tz='UTC').isoformat()},
    ]
    with closing(sqlite3.connect(db_file)) as conn:
        current.to_sql('players_raw', conn, if_exists='replace', index=False)
        combined.to_sql('player_gw', conn, if_exists='replace', index=False)
        pd.DataFrame(metadata_rows).to_sql(
            'app_metadata', conn, if_exists='replace', index=False
        )
        conn.commit()

    return {
        'refreshed': True,
        'season': current_season,
        'latest_completed_game_week': latest_completed,
        'target_game_week': int(target_event['id']),
        'live_start_index': live_start_index,
        'completed_internal_index': completed_internal_index,
        'snapshot_game_week': snapshot_index,
        'players': len(current),
        'snapshot_players': len(active),
        'live_rows': len(live_history),
    }


def get_runtime_context(db_file: str) -> dict:
    """Return the season, displayed GW, and internal inference snapshot index."""
    context = {
        'season': SEASON,
        'target_game_week': GAME_WEEK,
        'snapshot_game_week': GAME_WEEK - 1,
        'completed_internal_index': GAME_WEEK - 1,
        'latest_completed_game_week': GAME_WEEK,
    }
    try:
        with closing(sqlite3.connect(db_file)) as conn:
            metadata = dict(conn.execute('SELECT key, value FROM app_metadata').fetchall())
        target_game_week = int(metadata.get(
            'target_game_week', context['target_game_week']
        ))
        snapshot_game_week = int(metadata.get(
            'snapshot_game_week', context['snapshot_game_week']
        ))
        context.update({
            'season': metadata.get('season', context['season']),
            'target_game_week': target_game_week,
            'snapshot_game_week': snapshot_game_week,
            'completed_internal_index': int(metadata.get(
                'completed_internal_index', snapshot_game_week - 1
            )),
            'latest_completed_game_week': int(metadata.get(
                'latest_completed_game_week', max(0, target_game_week - 1)
            )),
        })
    except (sqlite3.Error, TypeError, ValueError):
        pass
    return context


def build_features(db_file):
    """
    Merge player metadata with GW history and compute rolling features.

    Roll stats are shifted by 1 before windowing so target-GW data never
    leaks into the features. Returns one row per (player, GW) with a
    'target' column containing that row's total_points. The rolling inputs are
    shifted, so they contain only information available before that fixture.
    """
    conn = sqlite3.connect(db_file)
    raw = pd.read_sql(
        "SELECT element_type, team, second_name, first_name, id, code, birth_date FROM players_raw",
        conn,
    )
    all_gw = pd.read_sql("SELECT * FROM player_gw", conn).drop_duplicates()
    # Ensure all ROLL_COLS exist — DBs created before the v2 feature expansion
    # will be missing expected_*/starts/discipline columns.  Fill with 0 so
    # training degrades gracefully rather than crashing.
    for _col in ROLL_COLS:
        if _col not in all_gw.columns:
            all_gw[_col] = 0.0

    # Join EA FC player-DNA attributes if the table has been populated.
    # Attributes are keyed by FPL player ID.  Left-join so players without
    # a fuzzy match keep all their rows (ea_* cols will be NaN, filled below).
    _attr_meta = {
        'fpl_id', 'ea_name', 'ea_team', 'ea_position',
        'season', 'last_updated',
        # backward compat for old FBref / Understat schema DBs
        'fb_name', 'fb_team', 'fb_position', 'fb_minutes',
        'us_name', 'us_team', 'us_position', 'us_games',
        'fifa_name', 'position',
    }
    try:
        attr_df = pd.read_sql("SELECT * FROM player_attributes", conn)
        _attr_val_cols = [c for c in attr_df.columns if c not in _attr_meta]
        attr_df = attr_df[['fpl_id'] + _attr_val_cols]
        # Keep columns as-is (ea_overall etc.) -- POS_FEATURES references them directly
    except Exception:
        attr_df = pd.DataFrame()  # table not yet created -- graceful degradation

    # Load dated Transfermarkt values for a leakage-safe as-of join below.
    try:
        from transfer_values import load_transfer_value_history
        tm_df = load_transfer_value_history(conn)
    except Exception:
        tm_df = pd.DataFrame()  # module or table not yet available

    try:
        from external_appearances import load_external_appearances
        external_appearances = load_external_appearances(conn)
    except Exception:
        external_appearances = pd.DataFrame()

    # Load season-over-season history and compute trajectory features.
    try:
        hist_raw = pd.read_sql("SELECT * FROM player_history", conn)
        hist_features = _compute_history_features(hist_raw)
    except Exception:
        hist_features = pd.DataFrame()  # table not yet ingested

    conn.close()

    # keep element_type as both a human-readable label and a numeric feature
    raw['element_type_enc'] = raw['element_type']   # 1/2/3/4 -- XGBoost reads this directly
    raw['element_type'] = raw['element_type'].map({1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'})

    df = raw.merge(all_gw, left_on='id', right_on='player_id', how='inner')
    df = df.sort_values(['id', 'Game_Week']).reset_index(drop=True)
    df['_prior_pl_rows'] = df.groupby('id').cumcount()

    df = _add_temporal_context_features(df)
    df = _add_fixture_strength_features(df)

    for col in ROLL_COLS:
        df[f'roll5_{col}'] = (
            df.groupby('id')[col]
            .transform(lambda x: x.shift(1).rolling(FORM_WINDOW, min_periods=1).mean())
        )

    # Rolling form is already shifted one row, while fixture, opponent and
    # price belong to this row. The prediction target is therefore this GW's
    # points. An additional shift would pair GW t context with GW t+1 results.
    df['target'] = df['total_points']

    # Compute player_age_years: fractional age at each GW's kickoff date.
    # birth_date is 'YYYY-MM-DD'; kickoff_time is 'YYYY-MM-DDThh:mm:ssZ'.
    # Using kickoff_time gives exact per-GW age so a player who turns 30 in
    # January is correctly older in the second half of the season.
    # Falls back to the season start date (Aug 1) if kickoff_time is missing.
    _birth = pd.to_datetime(raw.set_index('id')['birth_date'], errors='coerce')
    df['_birth_dt'] = df['id'].map(_birth)
    _ko = pd.to_datetime(df['kickoff_time'], errors='coerce', utc=True).dt.tz_localize(None)
    _season_start = pd.Timestamp(f'{SEASON[:4]}-08-01')
    _ref_date = _ko.where(_ko.notna(), _season_start)
    df['player_age_years'] = ((_ref_date - df['_birth_dt']).dt.days / 365.25).round(2)
    df.drop(columns=['_birth_dt'], inplace=True)

    # Attach EA FC attributes (one row per player, broadcast to all their GWs)
    if not attr_df.empty:
        df = df.merge(attr_df, left_on='id', right_on='fpl_id', how='left')
        df.drop(columns=['fpl_id'], errors='ignore', inplace=True)

    # Attach only values observed on or before each fixture's kickoff time.
    from transfer_values import attach_transfer_values_asof
    df = attach_transfer_values_asof(df, tm_df)
    df['tm_value_available'] = df['tm_market_value'].notna().astype(float)
    from external_appearances import attach_external_appearance_features
    df = attach_external_appearance_features(df, external_appearances)

    # Attach season history trajectory features (one row per player)
    if not hist_features.empty:
        df = df.merge(hist_features, left_on='id', right_on='player_id', how='left')
        df.drop(columns=['player_id'], errors='ignore', inplace=True)

    # Keep unavailable market values missing. The model input builder combines
    # the explicit availability mask with a neutral standardized placeholder;
    # the placeholder is not interpreted as a Transfermarkt valuation.
    for _col in ATTR_COLS + HIST_COLS:
        if _col not in df.columns:
            df[_col] = float('nan') if _col.startswith('tm_') else 0.0
        else:
            if not _col.startswith('tm_'):
                df[_col] = df[_col].fillna(0.0)

    df['has_pl_history'] = (
        (df['_prior_pl_rows'] > 0) | (df['hist_career_seasons'] > 0)
    ).astype(int)
    df.drop(columns=['_prior_pl_rows'], inplace=True)

    # Compute form_data_density: fraction of last 5 GWs a player was on the
    # pitch.  roll5_minutes is already the 5-GW rolling mean of minutes played;
    # dividing by (FORM_WINDOW * 90) scales it so that a full-time regular who
    # averages 90 min/GW produces density = 90/(5*90) = 0.2.  This is the
    # natural upper bound given the rolling-mean formulation.
    # A new signing or returning injury has density = 0.
    df['form_data_density'] = (df['roll5_minutes'] / (FORM_WINDOW * 90)).clip(0.0, 1.0)

    # --- Sparse player feature blending ---
    # Problem: new signings and long-term injured players have roll5_* = 0,
    # making the model predict them as terrible even when their Transfermarkt
    # value signals elite quality.
    #
    # Solution: for players with sparse recent data, blend their rolling form
    # features toward the per-GW position mean of fully-active players.
    # tm_market_value is intentionally NOT blended — it remains the signal
    # that differentiates a £100M new signing from a £5M one.
    #
    # blend_weight = (form_data_density * FORM_WINDOW).clip(0, 1)
    #              = roll5_minutes / 90   (fraction of a full 90-min game/GW)
    #
    #   weight = 0.0  →  100% position mean  (no recent minutes: new/injured)
    #   weight = 0.5  →  50/50              (rotation player, ~45 min/GW avg)
    #   weight = 1.0  →  100% actual        (regular starter, no blending)
    _blend_cols = [c for c in df.columns if c.startswith('roll5_')]
    _blend_w    = (df['form_data_density'] * FORM_WINDOW).clip(0.0, 1.0)

    for _pos in ['GK', 'DEF', 'MID', 'FWD']:
        _pos_mask    = df['element_type'] == _pos
        _sparse_mask = _pos_mask & (_blend_w < 1.0)
        _dense_mask  = _pos_mask & (_blend_w >= 0.8)  # avg >= 72 min/GW

        if not _sparse_mask.any():
            continue

        # Per-GW reference means from established players.
        # Falls back to the full position pool if there are fewer than 5
        # established players (early-season or very small positions).
        _ref_mask  = _dense_mask if _dense_mask.sum() >= 5 else _pos_mask
        _gw_means  = df.loc[_ref_mask].groupby('Game_Week')[_blend_cols].mean()
        _glob_mean = df.loc[_ref_mask, _blend_cols].mean()

        _sidx = df.index[_sparse_mask]
        _ref_vals = pd.DataFrame(
            [_gw_means.loc[gw] if gw in _gw_means.index else _glob_mean
             for gw in df.loc[_sidx, 'Game_Week']],
            index=_sidx,
        )

        _w = _blend_w.loc[_sidx].to_numpy()[:, None]   # shape (N, 1)
        df.loc[_sidx, _blend_cols] = (
            _w       * df.loc[_sidx, _blend_cols].to_numpy()
            + (1.0 - _w) * _ref_vals.to_numpy()
        )

    # Interaction: amplify TM market value when form data is sparse.
    # When density=0 (player has no recent minutes), this equals tm_market_value
    # and gives the model a strong prior on player quality.
    # When density=1 (fully fit starter), this equals 0 and the model relies
    # entirely on rolling form.
    df['tm_market_value_x_sparsity'] = (
        df['tm_market_value'] * (1.0 - df['form_data_density'])
    ).round(4)

    return df


def _signed_log1p(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors='coerce').fillna(0.0)
    return np.sign(numeric) * np.log1p(numeric.abs())


def _add_temporal_context_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add pre-fixture market momentum, price trend, rest and congestion."""
    enriched = df.copy()
    for column in ('selected', 'transfers_in', 'transfers_out', 'transfers_balance', 'value'):
        if column not in enriched:
            enriched[column] = 0.0
        enriched[column] = pd.to_numeric(enriched[column], errors='coerce').fillna(0.0)

    enriched['ownership_log'] = np.log1p(enriched['selected'].clip(lower=0))
    previous_selected = enriched.groupby('id')['selected'].shift(1)
    enriched['ownership_change_log'] = _signed_log1p(
        enriched['selected'] - previous_selected
    )
    enriched['transfers_in_log'] = np.log1p(enriched['transfers_in'].clip(lower=0))
    enriched['transfers_out_log'] = np.log1p(enriched['transfers_out'].clip(lower=0))
    enriched['transfer_balance_log'] = _signed_log1p(enriched['transfers_balance'])
    enriched['transfer_momentum_per_owner'] = (
        enriched['transfers_balance'] / enriched['selected'].clip(lower=1)
    ).clip(-5.0, 5.0)
    enriched['price_change_1'] = (
        enriched['value'] - enriched.groupby('id')['value'].shift(1)
    ).fillna(0.0)
    enriched['price_change_3'] = (
        enriched['value'] - enriched.groupby('id')['value'].shift(3)
    ).fillna(0.0)

    kickoff = pd.to_datetime(enriched.get('kickoff_time'), errors='coerce', utc=True)
    previous_kickoff = kickoff.groupby(enriched['id']).shift(1)
    raw_rest = (kickoff - previous_kickoff).dt.total_seconds() / 86400
    enriched['rest_days_available'] = raw_rest.notna().astype(float)
    enriched['rest_days'] = raw_rest.clip(lower=0, upper=30).fillna(14.0)
    enriched['matches_previous_14d'] = 0.0
    for _, indexes in enriched.groupby('id', sort=False).groups.items():
        index_array = np.asarray(list(indexes), dtype=int)
        player_kickoff = kickoff.iloc[index_array]
        valid_indexes = np.flatnonzero(player_kickoff.notna().to_numpy())
        if not len(valid_indexes):
            continue
        valid_times = player_kickoff.iloc[valid_indexes].astype('int64').to_numpy()
        horizon = 14 * 24 * 60 * 60 * 1_000_000_000
        left = np.searchsorted(valid_times, valid_times - horizon, side='left')
        enriched.loc[
            index_array[valid_indexes], 'matches_previous_14d'
        ] = np.arange(len(valid_times)) - left
    return enriched


def _fixture_rows_from_player_history(df: pd.DataFrame) -> pd.DataFrame:
    required = {
        'fixture', 'opponent_team', 'was_home', 'team_h_score',
        'team_a_score', 'kickoff_time', 'Game_Week',
    }
    if not required.issubset(df.columns):
        return pd.DataFrame()
    source = df[
        pd.to_numeric(df['fixture'], errors='coerce').fillna(0).gt(0)
        & pd.to_numeric(df['opponent_team'], errors='coerce').fillna(0).gt(0)
    ].copy()
    kickoff = pd.to_datetime(source['kickoff_time'], errors='coerce', utc=True)
    source['_season_key'] = np.where(
        kickoff.dt.month >= 7, kickoff.dt.year, kickoff.dt.year - 1
    )
    records = []
    for (season_key, fixture_id), group in source.groupby(['_season_key', 'fixture']):
        home_rows = group[group['was_home'].astype(bool)]
        away_rows = group[~group['was_home'].astype(bool)]
        if home_rows.empty or away_rows.empty:
            continue
        first = group.iloc[0]
        records.append({
            'season_key': int(season_key),
            'fixture': int(fixture_id),
            'Game_Week': int(first['Game_Week']),
            'kickoff_time': first['kickoff_time'],
            'home_team': int(pd.to_numeric(away_rows['opponent_team']).mode().iloc[0]),
            'away_team': int(pd.to_numeric(home_rows['opponent_team']).mode().iloc[0]),
            'home_goals': float(first.get('team_h_score') or 0),
            'away_goals': float(first.get('team_a_score') or 0),
        })
    return pd.DataFrame(records).drop_duplicates(['season_key', 'fixture'])


def _add_fixture_strength_features(df: pd.DataFrame) -> pd.DataFrame:
    """Attach team strength known before each row's fixture kickoff."""
    enriched = df.copy()
    defaults = {
        'team_elo_pre': 1500.0, 'opponent_elo_pre': 1500.0,
        'elo_difference': 0.0,
        'team_goals_for_roll5': 1.25, 'team_goals_against_roll5': 1.25,
        'opponent_goals_for_roll5': 1.25, 'opponent_goals_against_roll5': 1.25,
    }
    for column, value in defaults.items():
        enriched[column] = value
    fixtures = _fixture_rows_from_player_history(enriched)
    if fixtures.empty:
        return enriched
    fixtures['_kickoff'] = pd.to_datetime(fixtures['kickoff_time'], errors='coerce', utc=True)
    fixtures = fixtures.sort_values(['_kickoff', 'fixture'])
    ratings: dict[tuple[int, int], float] = {}
    goals_for: dict[tuple[int, int], list[float]] = {}
    goals_against: dict[tuple[int, int], list[float]] = {}
    contexts: dict[tuple[int, int], dict[str, float]] = {}
    latest: dict[tuple[int, int], dict[str, float]] = {}
    for fixture in fixtures.itertuples(index=False):
        season = int(fixture.season_key)
        home = (season, int(fixture.home_team))
        away = (season, int(fixture.away_team))
        home_elo, away_elo = ratings.get(home, 1500.0), ratings.get(away, 1500.0)
        home_for = float(np.mean(goals_for.get(home, [1.25])[-5:]))
        home_against = float(np.mean(goals_against.get(home, [1.25])[-5:]))
        away_for = float(np.mean(goals_for.get(away, [1.25])[-5:]))
        away_against = float(np.mean(goals_against.get(away, [1.25])[-5:]))
        contexts[(season, int(fixture.fixture))] = {
            'home_team_elo': home_elo, 'away_team_elo': away_elo,
            'home_goals_for': home_for, 'home_goals_against': home_against,
            'away_goals_for': away_for, 'away_goals_against': away_against,
        }
        expected_home = 1 / (1 + 10 ** ((away_elo - (home_elo + 65)) / 400))
        actual_home = 1.0 if fixture.home_goals > fixture.away_goals else (
            0.0 if fixture.home_goals < fixture.away_goals else 0.5
        )
        delta = 20 * (actual_home - expected_home)
        ratings[home], ratings[away] = home_elo + delta, away_elo - delta
        goals_for.setdefault(home, []).append(fixture.home_goals)
        goals_against.setdefault(home, []).append(fixture.away_goals)
        goals_for.setdefault(away, []).append(fixture.away_goals)
        goals_against.setdefault(away, []).append(fixture.home_goals)
        latest[home] = {
            'elo': ratings[home],
            'goals_for': float(np.mean(goals_for[home][-5:])),
            'goals_against': float(np.mean(goals_against[home][-5:])),
        }
        latest[away] = {
            'elo': ratings[away],
            'goals_for': float(np.mean(goals_for[away][-5:])),
            'goals_against': float(np.mean(goals_against[away][-5:])),
        }

    for index, row in enriched.iterrows():
        kickoff_value = pd.to_datetime(row.get('kickoff_time'), errors='coerce', utc=True)
        season = (
            kickoff_value.year if pd.notna(kickoff_value) and kickoff_value.month >= 7
            else kickoff_value.year - 1 if pd.notna(kickoff_value) else 0
        )
        fixture_value = pd.to_numeric(pd.Series([row.get('fixture')]), errors='coerce').iloc[0]
        fixture_id = int(fixture_value) if pd.notna(fixture_value) else 0
        context = contexts.get((season, fixture_id))
        was_home = bool(row.get('was_home'))
        if context:
            team_prefix, opponent_prefix = ('home', 'away') if was_home else ('away', 'home')
            enriched.at[index, 'team_elo_pre'] = context[f'{team_prefix}_team_elo']
            enriched.at[index, 'opponent_elo_pre'] = context[f'{opponent_prefix}_team_elo']
            enriched.at[index, 'team_goals_for_roll5'] = context[f'{team_prefix}_goals_for']
            enriched.at[index, 'team_goals_against_roll5'] = context[f'{team_prefix}_goals_against']
            enriched.at[index, 'opponent_goals_for_roll5'] = context[f'{opponent_prefix}_goals_for']
            enriched.at[index, 'opponent_goals_against_roll5'] = context[f'{opponent_prefix}_goals_against']
        else:
            team_value = pd.to_numeric(pd.Series([row.get('team')]), errors='coerce').iloc[0]
            opponent_value = pd.to_numeric(pd.Series([row.get('opponent_team')]), errors='coerce').iloc[0]
            team = int(team_value) if pd.notna(team_value) else 0
            opponent = int(opponent_value) if pd.notna(opponent_value) else 0
            team_latest = latest.get((season, team), {})
            opponent_latest = latest.get((season, opponent), {})
            enriched.at[index, 'team_elo_pre'] = team_latest.get('elo', 1500.0)
            enriched.at[index, 'opponent_elo_pre'] = opponent_latest.get('elo', 1500.0)
            enriched.at[index, 'team_goals_for_roll5'] = team_latest.get('goals_for', 1.25)
            enriched.at[index, 'team_goals_against_roll5'] = team_latest.get('goals_against', 1.25)
            enriched.at[index, 'opponent_goals_for_roll5'] = opponent_latest.get('goals_for', 1.25)
            enriched.at[index, 'opponent_goals_against_roll5'] = opponent_latest.get('goals_against', 1.25)
        enriched.at[index, 'elo_difference'] = (
            enriched.at[index, 'team_elo_pre'] - enriched.at[index, 'opponent_elo_pre']
        )
    return enriched


def apply_market_value_weighting(scored: pd.DataFrame) -> pd.DataFrame:
    """Blend a calibrated market-value prior into model predictions.

    Market value contributes 10% for players with Premier League history and
    30% for history-free arrivals. Imputed values contribute nothing. Within
    each position, value percentile is mapped onto the model-score distribution
    before blending so both signals share the same points scale.
    """
    weighted = scored.copy()
    weighted['model_predicted_points'] = weighted['predicted_points'].astype(float)
    weighted['market_value_prior'] = weighted['model_predicted_points']
    weighted['market_value_weight'] = 0.0

    for position in weighted['element_type'].dropna().unique():
        mask = weighted['element_type'] == position
        group = weighted.loc[mask]
        real_value = (
            group['tm_market_value'].fillna(0).gt(0)
            & group.get(
                'tm_value_imputed', pd.Series(0, index=group.index)
            ).fillna(0).eq(0)
        )
        if not real_value.any():
            continue

        value_percentile = group.loc[real_value, 'tm_market_value'].rank(
            method='average', pct=True
        )
        sorted_scores = group['model_predicted_points'].sort_values().to_numpy()
        prior_indexes = (
            value_percentile.mul(len(sorted_scores) - 1).round().astype(int)
        )
        market_prior = pd.Series(
            sorted_scores[prior_indexes.to_numpy()], index=value_percentile.index
        )
        history_weight = group.loc[real_value].get(
            'has_pl_history', pd.Series(0, index=value_percentile.index)
        ).fillna(0).astype(bool).map({
            True: MARKET_VALUE_WEIGHT_PL_HISTORY,
            False: MARKET_VALUE_WEIGHT_NO_HISTORY,
        })

        weighted.loc[value_percentile.index, 'market_value_prior'] = market_prior
        weighted.loc[value_percentile.index, 'market_value_weight'] = history_weight

    weight = weighted['market_value_weight']
    weighted['predicted_points'] = (
        (1.0 - weight) * weighted['model_predicted_points']
        + weight * weighted['market_value_prior']
    ).round(4)
    return weighted


def _compute_history_features(hist_df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive season-over-season trajectory features from the player_history table.

    Returns one row per player_id with columns in HIST_COLS.
    All rates are per-90 so they are independent of games played / rotation.
    Players with < 270 minutes in the prior season get 0.0 (data too sparse).
    """

    def _season_year(name: str) -> int:
        m = re.match(r'(\d{4})/', str(name))
        return int(m.group(1)) if m else 0

    def _p90(total, minutes: float) -> float:
        return round(float(total) / minutes * 90, 4) if minutes >= 270 else 0.0

    hist_df = hist_df.copy()
    hist_df['_yr'] = hist_df['season_name'].apply(_season_year)

    rows = []
    for pid, grp in hist_df.groupby('player_id'):
        grp = grp.sort_values('_yr', ascending=False).reset_index(drop=True)

        prev  = grp.iloc[0].to_dict() if len(grp) > 0 else {}
        prev2 = grp.iloc[1].to_dict() if len(grp) > 1 else {}

        mins  = float(prev.get('minutes',  0) or 0)
        mins2 = float(prev2.get('minutes', 0) or 0)

        pts_p90  = _p90(prev.get('total_points',  0), mins)
        pts_p90_2 = _p90(prev2.get('total_points', 0), mins2)

        rows.append({
            'player_id':              pid,
            'hist_prev_pts_per90':    pts_p90,
            'hist_prev_goals_per90':  _p90(prev.get('goals_scored',       0), mins),
            'hist_prev_assists_per90':_p90(prev.get('assists',            0), mins),
            'hist_prev_cs_per90':     _p90(prev.get('clean_sheets',       0), mins),
            'hist_prev_saves_per90':  _p90(prev.get('saves',              0), mins),
            'hist_prev_xg_per90':     _p90(prev.get('expected_goals',     0), mins),
            'hist_prev_xa_per90':     _p90(prev.get('expected_assists',   0), mins),
            'hist_career_seasons':    len(grp),
            'hist_yoy_pts_delta':     round(pts_p90 - pts_p90_2, 4),
        })

    return pd.DataFrame(rows)


def normalize_pool_scores(pool):
    """
    Add ``predicted_points_norm``: z-score of predicted_points within each
    position group.

    The four position models operate on different feature sets and predict on
    different scales — a raw score of 4.5 from the DEF model and 4.5 from
    the FWD model are not directly comparable.  Normalising within position
    translates each score into 'sigmas above/below the position mean', so
    flex formation slots in ``pick_starting_xi`` can be filled by the most
    exceptional player regardless of position.
    """
    pool = pool.copy()
    pool['predicted_points_norm'] = 0.0
    for pos in pool['element_type'].unique():
        mask = pool['element_type'] == pos
        pts = pool.loc[mask, 'predicted_points']
        sigma = pts.std()
        pool.loc[mask, 'predicted_points_norm'] = (
            ((pts - pts.mean()) / sigma) if sigma > 0 else (pts - pts.mean())
        ).round(4)
    return pool


# ---------------------------------------------------------------------------
# Squad persistence & weekly transfer logic
# ---------------------------------------------------------------------------

_SQUAD_TABLE = 'saved_squad'


def save_squad(db_file: str, squad_df: pd.DataFrame, game_week: int, season: str) -> None:
    """Overwrite the saved_squad table with the current 15 players."""
    id_col = 'player_id' if 'player_id' in squad_df.columns else 'id'
    to_save = squad_df[[id_col, 'first_name', 'second_name', 'element_type', 'team', 'value']].copy()
    to_save = to_save.rename(columns={id_col: 'player_id'})
    to_save['gw_saved'] = game_week
    to_save['season']   = season
    conn = sqlite3.connect(db_file)
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {_SQUAD_TABLE} (
            player_id    INTEGER PRIMARY KEY,
            first_name   TEXT,
            second_name  TEXT,
            element_type TEXT,
            team         INTEGER,
            value        REAL,
            gw_saved     INTEGER,
            season       TEXT
        )
    """)
    conn.execute(f'DELETE FROM {_SQUAD_TABLE}')
    to_save.to_sql(_SQUAD_TABLE, conn, if_exists='append', index=False)
    conn.commit()
    conn.close()


def load_squad(db_file: str):
    """Return saved squad DataFrame or None if none exists."""
    conn = sqlite3.connect(db_file)
    if not _table_exists(conn, _SQUAD_TABLE):
        conn.close()
        return None
    df = pd.read_sql(f'SELECT * FROM {_SQUAD_TABLE}', conn)
    conn.close()
    return df if not df.empty else None


def score_squad_from_pool(saved_squad: pd.DataFrame, scored_pool: pd.DataFrame) -> pd.DataFrame:
    """
    Join saved squad members against the current-GW scored pool.
    Members absent from the pool (left EPL / no GW data) get predicted_points=0
    and retain their saved metadata.  Always returns a DataFrame with 'id' column.
    """
    pool_by_id = scored_pool.set_index('id')
    rows = []
    for _, sp in saved_squad.iterrows():
        pid = int(sp['player_id'])
        if pid in pool_by_id.index:
            pool_row = pool_by_id.loc[pid]
            if isinstance(pool_row, pd.DataFrame):
                pool_row = pool_row.iloc[0]
            row = pool_row.to_dict()
            row['id'] = pid
        else:
            row = {
                'id':                    pid,
                'first_name':            str(sp.get('first_name', '')),
                'second_name':           str(sp.get('second_name', '')),
                'element_type':          str(sp.get('element_type', '')),
                'team':                  int(sp.get('team', 0)),
                'value':                 float(sp.get('value', 0)),
                'predicted_points':      0.0,
                'predicted_points_norm': 0.0,
                'selection_margin':      0.0,
            }
        rows.append(row)
    df = pd.DataFrame(rows)
    for col in ('selection_margin', 'predicted_points_norm'):
        if col not in df.columns:
            df[col] = 0.0
    return df.reset_index(drop=True)


def pick_starting_xi(squad: pd.DataFrame) -> tuple:
    """
    Pick best starting 11 from a squad using FPL rules:
    exactly 1 GK, min 3 DEF, min 2 MID, min 1 FWD, 11 total.
    Uses predicted_points_norm for flex-slot tie-breaking.
    Returns (starters_df, bench_df, formation_str).
    """
    gk_sorted  = squad[squad['element_type'] == 'GK'].sort_values('predicted_points', ascending=False)
    starter_gk = gk_sorted.iloc[[0]]
    bench_gk   = gk_sorted.iloc[[1]] if len(gk_sorted) > 1 else pd.DataFrame()

    outfield = squad[squad['element_type'] != 'GK']
    mins     = {'DEF': 3, 'MID': 2, 'FWD': 1}
    counts   = {'DEF': 0, 'MID': 0, 'FWD': 0}
    starters_out, used_ids = [], set()

    # Pass 1 — fill position minimums
    for pos, min_n in mins.items():
        for _, p in outfield[outfield['element_type'] == pos].sort_values(
            'predicted_points', ascending=False
        ).iterrows():
            if counts[pos] < min_n:
                starters_out.append(p)
                counts[pos] += 1
                used_ids.add(p['id'])

    # Pass 2 — fill remaining outfield slots by normalised cross-position score
    sort_col = 'predicted_points_norm' if 'predicted_points_norm' in outfield.columns else 'predicted_points'
    for _, p in outfield[~outfield['id'].isin(used_ids)].sort_values(sort_col, ascending=False).iterrows():
        if len(starters_out) == 10:
            break
        starters_out.append(p)
        counts[p['element_type']] += 1
        used_ids.add(p['id'])

    starters  = pd.concat([starter_gk, pd.DataFrame(starters_out)], ignore_index=True)
    bench_out = outfield[~outfield['id'].isin(used_ids)].sort_values('predicted_points', ascending=False)
    bench = (
        pd.concat([bench_gk, bench_out], ignore_index=True)
        if not bench_gk.empty else bench_out.reset_index(drop=True)
    )
    formation = f"{counts['DEF']}-{counts['MID']}-{counts['FWD']}"
    return starters, bench, formation


def suggest_transfer(
    scored_squad: pd.DataFrame,
    eligible_pool: pd.DataFrame,
    max_per_team: int = MAX_PLAYERS_PER_TEAM,
    max_spend: int = 1000,
):
    """
    Find the single best transfer (one OUT → one IN) that maximises
    predicted-points gain, subject to budget and per-club cap.
    Returns dict { 'out', 'in_', 'gain', 'new_spend' } or None.
    """
    squad_ids   = set(scored_squad['id'].astype(int))
    squad_spend = int(scored_squad['value'].sum())
    team_counts = Counter(scored_squad['team'].astype(int))

    best_gain, best_out, best_in_ = float('-inf'), None, None

    for _, out_p in scored_squad.iterrows():
        freed    = int(out_p['value'])
        budget   = max_spend - squad_spend + freed
        tc_after = Counter(team_counts)
        tc_after[int(out_p['team'])] -= 1

        cands = eligible_pool[
            (eligible_pool['element_type'] == out_p['element_type'])
            & (~eligible_pool['id'].astype(int).isin(squad_ids - {int(out_p['id'])}))
        ].sort_values('predicted_points', ascending=False)

        for _, in_p in cands.iterrows():
            if int(in_p['id']) == int(out_p['id']):
                continue
            if int(in_p['value']) > budget:
                continue
            if tc_after.get(int(in_p['team']), 0) >= max_per_team:
                continue
            gain = float(in_p['predicted_points']) - float(out_p['predicted_points'])
            if gain > best_gain:
                best_gain, best_out, best_in_ = gain, out_p.copy(), in_p.copy()
            break  # sorted desc — first valid = best for this out_p

    if best_out is None:
        return None
    return {
        'out':       best_out.to_dict(),
        'in_':       best_in_.to_dict(),
        'gain':      round(best_gain, 2),
        'new_spend': squad_spend - int(best_out['value']) + int(best_in_['value']),
    }


def find_ineligible_replacements(
    ineligible_squad: pd.DataFrame,
    eligible_pool: pd.DataFrame,
    full_squad: pd.DataFrame,
    max_per_team: int = MAX_PLAYERS_PER_TEAM,
    max_spend: int = 1000,
) -> list:
    """
    For each ineligible squad member, find the best available same-position
    replacement within budget and per-club cap.
    Returns list of dicts: { 'out', 'replacement' (or None), 'gain' }.
    """
    full_ids    = set(full_squad['id'].astype(int))
    squad_spend = int(full_squad['value'].sum())
    team_counts = Counter(full_squad['team'].astype(int))
    results = []
    for _, out_p in ineligible_squad.iterrows():
        freed    = int(out_p.get('value', 0))
        budget   = max_spend - squad_spend + freed
        tc_after = Counter(team_counts)
        tc_after[int(out_p.get('team', 0))] -= 1
        cands = eligible_pool[
            (eligible_pool['element_type'] == out_p['element_type'])
            & (~eligible_pool['id'].astype(int).isin(full_ids - {int(out_p['id'])}))
        ].sort_values('predicted_points', ascending=False)
        replacement = None
        for _, in_p in cands.iterrows():
            if int(in_p['value']) <= budget and tc_after.get(int(in_p['team']), 0) < max_per_team:
                replacement = in_p.to_dict()
                break
        results.append({
            'out':         out_p.to_dict(),
            'replacement': replacement,
            'gain':        round(
                float(replacement['predicted_points']) - float(out_p['predicted_points']), 2
            ) if replacement else 0.0,
        })
    return results


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



def supplement_gw_from_api(db_file: str, player_ids: list) -> int:
    """
    Supplement player_gw with GW history from the FPL API for the given
    player IDs.  Does NOT touch players_raw -- the vaastav player list is
    kept intact.  Useful when the vaastav submodule is stale: we keep the
    known player set but top up their GW rows.

    Returns the highest GW number seen in the fetched data (1-indexed).
    """
    frames = []
    print(f"Fetching GW history for {len(player_ids)} players from FPL API...")
    for pid in player_ids:
        try:
            r = requests.get(f"{FPL_API_BASE}/element-summary/{pid}/", timeout=15)
            r.raise_for_status()
            history = r.json().get("history", [])
            if not history:
                continue
            df = pd.DataFrame(history)
            df["player_id"] = pid
            df["Game_Week"] = range(len(df))
            for col in ("ict_index", "creativity", "threat", "influence"):
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            frames.append(df)
        except requests.RequestException as exc:
            print(f"  Warning: could not fetch player {pid}: {exc}")

    if not frames:
        return 0

    gw_new = pd.concat(frames, ignore_index=True)
    conn = sqlite3.connect(db_file)
    if _table_exists(conn, "player_gw"):
        existing_gw = pd.read_sql("SELECT * FROM player_gw", conn)
        existing_gw = existing_gw[~existing_gw["player_id"].isin(player_ids)]
        pd.concat([existing_gw, gw_new], ignore_index=True).to_sql(
            "player_gw", conn, if_exists="replace", index=False
        )
    else:
        gw_new.to_sql("player_gw", conn, if_exists="replace", index=False)
    conn.close()

    live_gw = int(gw_new["Game_Week"].max()) + 1  # Game_Week is 0-indexed
    print(f"  GW history supplemented: {len(frames)} players, up to GW{live_gw}.")
    return live_gw


def ingest_from_fpl_api(db_file, player_ids=None):
    """
    Fetch player metadata and GW history from the official FPL API and
    upsert into the local SQLite DB.

    player_ids: optional list of ints.
      - None  => full replace: overwrites players_raw and player_gw entirely
                 with the latest FPL API data for all current players.
      - list  => partial upsert: replaces only the specified player IDs,
                 keeping existing rows for other players untouched.
    """
    print("Fetching bootstrap-static from FPL API...")
    resp = requests.get(f"{FPL_API_BASE}/bootstrap-static/", timeout=30)
    resp.raise_for_status()
    players_raw = pd.DataFrame(resp.json()["elements"])

    if player_ids is not None:
        players_raw = players_raw[players_raw["id"].isin(player_ids)]

    conn = sqlite3.connect(db_file)
    if player_ids is not None and _table_exists(conn, "players_raw"):
        # Partial upsert: keep rows for players NOT in the update list
        existing = pd.read_sql("SELECT * FROM players_raw", conn)
        existing = existing[~existing["id"].isin(player_ids)]
        pd.concat([existing, players_raw], ignore_index=True).to_sql(
            "players_raw", conn, if_exists="replace", index=False
        )
    else:
        # Full replace (player_ids=None or table doesn't exist yet)
        players_raw.to_sql("players_raw", conn, if_exists="replace", index=False)
    conn.close()

    frames = []
    print(f"Fetching GW history for {len(players_raw)} players...")
    for pid in players_raw["id"].tolist():
        try:
            r = requests.get(f"{FPL_API_BASE}/element-summary/{pid}/", timeout=15)
            r.raise_for_status()
            history = r.json().get("history", [])
            if not history:
                continue
            df = pd.DataFrame(history)
            df["player_id"] = pid
            df["Game_Week"] = range(len(df))
            for col in ("ict_index", "creativity", "threat", "influence"):
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            frames.append(df)
        except requests.RequestException as exc:
            print(f"  Warning: could not fetch player {pid}: {exc}")

    if frames:
        gw_new = pd.concat(frames, ignore_index=True)
        conn = sqlite3.connect(db_file)
        if player_ids is not None and _table_exists(conn, "player_gw"):
            existing_gw = pd.read_sql("SELECT * FROM player_gw", conn)
            existing_gw = existing_gw[~existing_gw["player_id"].isin(player_ids)]
            pd.concat([existing_gw, gw_new], ignore_index=True).to_sql(
                "player_gw", conn, if_exists="replace", index=False
            )
        else:
            gw_new.to_sql("player_gw", conn, if_exists="replace", index=False)
        conn.close()
        print(f"  Upserted GW history for {len(players_raw)} players.")
