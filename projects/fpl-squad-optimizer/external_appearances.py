from __future__ import annotations

import io
import sqlite3
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

_APPEARANCES_URL = (
    'https://pub-e682421888d945d684bcae8890b0ec20.r2.dev/'
    'data/appearances.csv.gz'
)
_PREMIER_LEAGUE_COMPETITION_ID = 'GB1'

_DDL = """
CREATE TABLE IF NOT EXISTS player_external_appearances (
    appearance_id TEXT PRIMARY KEY,
    fpl_code INTEGER NOT NULL,
    tm_id INTEGER NOT NULL,
    game_id INTEGER NOT NULL,
    appearance_date TEXT NOT NULL,
    competition_id TEXT NOT NULL,
    minutes_played INTEGER NOT NULL,
    goals INTEGER NOT NULL,
    assists INTEGER NOT NULL,
    yellow_cards INTEGER NOT NULL,
    red_cards INTEGER NOT NULL,
    source TEXT NOT NULL,
    synced_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_external_appearances_player_date
ON player_external_appearances (fpl_code, appearance_date);
"""


def refresh_external_appearances(
    connection: sqlite3.Connection,
    timeout: int = 120,
) -> int:
    """Persist mapped non-PL appearances, avoiding overlap with FPL rows."""
    response = requests.get(_APPEARANCES_URL, timeout=timeout)
    response.raise_for_status()
    appearances = pd.read_csv(
        io.BytesIO(response.content),
        compression='gzip',
        usecols=[
            'appearance_id', 'game_id', 'player_id', 'date', 'competition_id',
            'yellow_cards', 'red_cards', 'goals', 'assists', 'minutes_played',
        ],
    )
    mappings = pd.read_sql(
        """
        SELECT fpl_code, tm_id FROM player_transfer_values
        WHERE fpl_code IS NOT NULL AND tm_id IS NOT NULL
        """,
        connection,
    )
    mappings['player_id'] = pd.to_numeric(mappings['tm_id'], errors='coerce')
    mappings = mappings.dropna(subset=['player_id']).copy()
    mappings['player_id'] = mappings['player_id'].astype(int)
    mapped = appearances.merge(mappings, on='player_id', how='inner')
    mapped = mapped[mapped['competition_id'] != _PREMIER_LEAGUE_COMPETITION_ID].copy()
    mapped['date'] = pd.to_datetime(mapped['date'], errors='coerce')
    mapped = mapped.dropna(subset=['date', 'appearance_id']).drop_duplicates('appearance_id')
    for column in ('minutes_played', 'goals', 'assists', 'yellow_cards', 'red_cards'):
        mapped[column] = pd.to_numeric(mapped[column], errors='coerce').fillna(0).astype(int)
    synced_at = datetime.now(timezone.utc).isoformat(timespec='seconds')
    rows = [
        (
            str(row.appearance_id), int(row.fpl_code), int(row.player_id),
            int(row.game_id), row.date.strftime('%Y-%m-%d'),
            str(row.competition_id), int(row.minutes_played), int(row.goals),
            int(row.assists), int(row.yellow_cards), int(row.red_cards),
            'transfermarkt_appearances_cc0', synced_at,
        )
        for row in mapped.itertuples(index=False)
    ]
    connection.executescript(_DDL)
    connection.execute('DELETE FROM player_external_appearances')
    connection.executemany(
        'INSERT INTO player_external_appearances VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
        rows,
    )
    connection.commit()
    return len(rows)


def load_external_appearances(connection: sqlite3.Connection) -> pd.DataFrame:
    try:
        return pd.read_sql(
            """
            SELECT fpl_code, appearance_date, competition_id, minutes_played,
                   goals, assists, yellow_cards, red_cards
            FROM player_external_appearances
            ORDER BY fpl_code, appearance_date
            """,
            connection,
        )
    except (sqlite3.Error, pd.errors.DatabaseError):
        return pd.DataFrame()


def attach_external_appearance_features(
    rows: pd.DataFrame,
    appearances: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate only external appearances strictly before each FPL kickoff."""
    enriched = rows.copy()
    feature_names = [
        'external_minutes_90d', 'external_appearances_90d',
        'external_goal_involvements_365d', 'external_minutes_365d',
        'external_data_available',
    ]
    for feature in feature_names:
        enriched[feature] = 0.0
    if appearances.empty or 'code' not in enriched:
        return enriched

    external = appearances.copy()
    external['_date'] = pd.to_datetime(external['appearance_date'], errors='coerce', utc=True)
    external = external.dropna(subset=['_date'])
    kickoff = pd.to_datetime(enriched['kickoff_time'], errors='coerce', utc=True)
    codes = pd.to_numeric(enriched['code'], errors='coerce')
    for code, row_indexes in enriched[codes.notna()].groupby(codes[codes.notna()]).groups.items():
        history = external[external['fpl_code'] == int(code)].sort_values('_date')
        if history.empty:
            continue
        index_array = np.asarray(list(row_indexes), dtype=int)
        row_kickoffs = kickoff.loc[index_array]
        valid = row_kickoffs.notna().to_numpy()
        if not valid.any():
            continue
        history_times = history['_date'].astype('int64').to_numpy()
        minutes = pd.to_numeric(history['minutes_played'], errors='coerce').fillna(0).to_numpy()
        involvements = (
            pd.to_numeric(history['goals'], errors='coerce').fillna(0)
            + pd.to_numeric(history['assists'], errors='coerce').fillna(0)
        ).to_numpy()
        minute_prefix = np.concatenate(([0.0], np.cumsum(minutes)))
        involvement_prefix = np.concatenate(([0.0], np.cumsum(involvements)))
        target_indexes = index_array[valid]
        target_times = row_kickoffs.iloc[np.flatnonzero(valid)].astype('int64').to_numpy()
        ends = np.searchsorted(history_times, target_times, side='left')
        starts_90 = np.searchsorted(
            history_times, target_times - 90 * 86400 * 1_000_000_000, side='left'
        )
        starts_365 = np.searchsorted(
            history_times, target_times - 365 * 86400 * 1_000_000_000, side='left'
        )
        enriched.loc[target_indexes, 'external_minutes_90d'] = (
            minute_prefix[ends] - minute_prefix[starts_90]
        )
        enriched.loc[target_indexes, 'external_appearances_90d'] = ends - starts_90
        enriched.loc[target_indexes, 'external_goal_involvements_365d'] = (
            involvement_prefix[ends] - involvement_prefix[starts_365]
        )
        enriched.loc[target_indexes, 'external_minutes_365d'] = (
            minute_prefix[ends] - minute_prefix[starts_365]
        )
        enriched.loc[target_indexes, 'external_data_available'] = (ends > 0).astype(float)
    return enriched
