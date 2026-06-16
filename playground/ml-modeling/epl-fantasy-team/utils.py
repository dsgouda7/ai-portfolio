import os
import pathlib
import re
import sqlite3
from collections import Counter

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

    season     = seasons[-1]
    season_dir = os.path.join(base, season)
    gws_dir    = os.path.join(season_dir, 'gws')

    gw_nums = []
    if os.path.isdir(gws_dir):
        for f in os.listdir(gws_dir):
            m = re.match(r'gw(\d+)\.csv', f)
            if m:
                gw_nums.append(int(m.group(1)))

    game_week = max(gw_nums) if gw_nums else 38
    return (
        season,
        os.path.join(season_dir, 'players') + os.sep,
        os.path.join(season_dir, 'players_raw.csv'),
        game_week,
    )


SEASON, PLAYERS_DIR, RAW_DATA_PATH, GAME_WEEK = _autodetect_season()
DB_FILE          = str(_ROOT / 'fantasy_football.db')
MODELS_FILE      = str(_ROOT / 'models.joblib')
MAX_PLAYERS_PER_TEAM = 4
MAX_SPEND        = 1000
FORM_WINDOW      = 5

ROLL_COLS = [
    'total_points', 'minutes', 'goals_scored', 'assists',
    'clean_sheets', 'bonus', 'ict_index', 'creativity', 'threat',
    'influence', 'goals_conceded', 'saves',
    # StatsBomb expected stats + discipline (v2 feature expansion)
    'expected_goals', 'expected_assists', 'expected_goals_conceded',
    'starts', 'yellow_cards', 'red_cards', 'penalties_saved',
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
        'roll5_bonus', 'roll5_influence',
        'roll5_starts', 'roll5_yellow_cards', 'roll5_red_cards',
        'was_home', 'value', 'opponent_team', 'team',
        'tm_market_value',             # player quality proxy (log-scaled EUR)
        'tm_market_value_x_sparsity',  # TM value * (1 - density): high when elite + sparse data
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
        'roll5_bonus', 'roll5_starts',
        'roll5_yellow_cards', 'roll5_red_cards',
        'was_home', 'value', 'opponent_team', 'team',
        'tm_market_value',             # player quality proxy (log-scaled EUR)
        'tm_market_value_x_sparsity',  # TM value * (1 - density): high when elite + sparse data
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
        'roll5_bonus', 'roll5_starts',
        'roll5_yellow_cards', 'roll5_red_cards',
        'was_home', 'value', 'opponent_team', 'team',
        'tm_market_value',             # player quality proxy (log-scaled EUR)
        'tm_market_value_x_sparsity',  # TM value * (1 - density): high when elite + sparse data
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
        'roll5_bonus', 'roll5_starts',
        'roll5_yellow_cards', 'roll5_red_cards',
        'was_home', 'value', 'opponent_team', 'team',
        'tm_market_value',             # player quality proxy (log-scaled EUR)
        'tm_market_value_x_sparsity',  # TM value * (1 - density): high when elite + sparse data
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


def build_features(db_file):
    """
    Merge player metadata with GW history and compute rolling features.

    Roll stats are shifted by 1 before windowing so target-GW data never
    leaks into the features. Returns one row per (player, GW) with a
    'target' column containing next-GW total_points.
    """
    conn = sqlite3.connect(db_file)
    raw = pd.read_sql(
        "SELECT element_type, team, second_name, first_name, id, birth_date FROM players_raw",
        conn,
    )
    all_gw = pd.read_sql("SELECT * FROM player_gw", conn)
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

    # Join Transfermarkt market values (log-scaled EUR).
    try:
        from transfer_values import load_transfer_values
        tm_df = load_transfer_values(conn)
    except Exception:
        tm_df = pd.DataFrame()  # module or table not yet available

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

    for col in ROLL_COLS:
        df[f'roll5_{col}'] = (
            df.groupby('id')[col]
            .transform(lambda x: x.shift(1).rolling(FORM_WINDOW, min_periods=1).mean())
        )

    df['target'] = df.groupby('id')['total_points'].shift(-1)

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

    # Attach Transfermarkt market value (one row per player)
    if not tm_df.empty:
        df = df.merge(tm_df, left_on='id', right_on='fpl_id', how='left')
        df.drop(columns=['fpl_id'], errors='ignore', inplace=True)

    # Attach season history trajectory features (one row per player)
    if not hist_features.empty:
        df = df.merge(hist_features, left_on='id', right_on='player_id', how='left')
        df.drop(columns=['player_id'], errors='ignore', inplace=True)

    # Ensure every attr_ column referenced in POS_FEATURES exists — fills with
    # 0.0 if the player_attributes table has never been populated or a player
    # had no fuzzy match.  XGBoost treats 0 as "unknown ability" for these cols.
    for _col in ATTR_COLS + HIST_COLS:
        if _col not in df.columns:
            df[_col] = 0.0
        else:
            df[_col] = df[_col].fillna(0.0)

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
    max_per_team: int = 4,
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
    max_per_team: int = 4,
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
