from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

from utils import DB_FILE, FPL_API_BASE


class FplEntrySyncError(RuntimeError):
    pass


_DDL = """
CREATE TABLE IF NOT EXISTS fpl_entry_sync (
    entry_id       INTEGER PRIMARY KEY,
    started_event  INTEGER,
    current_event  INTEGER,
    synced_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS fpl_entry_gameweeks (
    entry_id             INTEGER NOT NULL,
    event                INTEGER NOT NULL,
    points               INTEGER NOT NULL,
    total_points         INTEGER NOT NULL,
    event_rank           INTEGER,
    overall_rank         INTEGER,
    bank                 INTEGER NOT NULL,
    squad_value          INTEGER NOT NULL,
    event_transfers      INTEGER NOT NULL,
    event_transfers_cost INTEGER NOT NULL,
    points_on_bench      INTEGER NOT NULL,
    active_chip          TEXT,
    synced_at            TEXT NOT NULL,
    PRIMARY KEY (entry_id, event)
);

CREATE TABLE IF NOT EXISTS fpl_entry_picks (
    entry_id        INTEGER NOT NULL,
    event           INTEGER NOT NULL,
    player_id       INTEGER NOT NULL,
    pick_position   INTEGER NOT NULL,
    multiplier      INTEGER NOT NULL,
    is_captain      INTEGER NOT NULL,
    is_vice_captain INTEGER NOT NULL,
    element_type    INTEGER NOT NULL,
    PRIMARY KEY (entry_id, event, player_id)
);

CREATE TABLE IF NOT EXISTS fpl_entry_transfers (
    entry_id        INTEGER NOT NULL,
    event           INTEGER NOT NULL,
    player_in       INTEGER NOT NULL,
    player_out      INTEGER NOT NULL,
    player_in_cost  INTEGER NOT NULL,
    player_out_cost INTEGER NOT NULL,
    transfer_time   TEXT NOT NULL,
    PRIMARY KEY (entry_id, event, player_in, player_out, transfer_time)
);
"""


def _fetch_json(path: str, timeout: int) -> dict | list:
    response = requests.get(f'{FPL_API_BASE}{path}', timeout=timeout)
    if response.status_code == 404:
        raise FplEntrySyncError('FPL entry was not found or is not yet public.')
    response.raise_for_status()
    return response.json()


def _validate_entry_id(entry_id: int | str) -> int:
    try:
        value = int(entry_id)
    except (TypeError, ValueError) as exc:
        raise FplEntrySyncError('FPL entry ID must be a positive integer.') from exc
    if value <= 0:
        raise FplEntrySyncError('FPL entry ID must be a positive integer.')
    return value


def _estimate_next_free_transfers(gameweeks: list[dict[str, Any]]) -> int:
    available = 1
    for row in sorted(gameweeks, key=lambda item: int(item['event'])):
        chip = row.get('active_chip')
        used = int(row.get('event_transfers', 0) or 0)
        if chip in ('wildcard', 'freehit', 'free_hit'):
            continue
        if used == 0:
            available = min(5, available + 1)
        else:
            available = max(1, available - used + 1)
    return available


def sync_public_entry(
    entry_id: int | str,
    database: str | Path = DB_FILE,
    timeout: int = 20,
) -> dict[str, Any]:
    """Persist all publicly visible completed-Gameweek state for an FPL entry."""
    entry = _validate_entry_id(entry_id)
    try:
        profile = _fetch_json(f'/entry/{entry}/', timeout)
        history = _fetch_json(f'/entry/{entry}/history/', timeout)
        transfers = _fetch_json(f'/entry/{entry}/transfers/', timeout)
        current = list(history.get('current', []))
        picks_by_event = {
            int(row['event']): _fetch_json(
                f"/entry/{entry}/event/{int(row['event'])}/picks/", timeout
            )
            for row in current
        }
    except (requests.RequestException, KeyError, TypeError, ValueError) as exc:
        raise FplEntrySyncError(f'Unable to synchronize FPL entry {entry}: {exc}') from exc

    synced_at = datetime.now(timezone.utc).isoformat(timespec='seconds')
    gameweek_rows = []
    pick_rows = []
    for history_row in current:
        event = int(history_row['event'])
        picks_payload = picks_by_event[event]
        entry_history = picks_payload.get('entry_history', history_row)
        active_chip = picks_payload.get('active_chip')
        gameweek_rows.append((
            entry,
            event,
            int(entry_history.get('points', 0) or 0),
            int(entry_history.get('total_points', 0) or 0),
            entry_history.get('rank'),
            entry_history.get('overall_rank'),
            int(entry_history.get('bank', 0) or 0),
            int(entry_history.get('value', 0) or 0),
            int(entry_history.get('event_transfers', 0) or 0),
            int(entry_history.get('event_transfers_cost', 0) or 0),
            int(entry_history.get('points_on_bench', 0) or 0),
            active_chip,
            synced_at,
        ))
        for pick in picks_payload.get('picks', []):
            pick_rows.append((
                entry,
                event,
                int(pick['element']),
                int(pick['position']),
                int(pick.get('multiplier', 0) or 0),
                int(bool(pick.get('is_captain'))),
                int(bool(pick.get('is_vice_captain'))),
                int(pick.get('element_type', 0) or 0),
            ))

    transfer_rows = [(
        entry,
        int(row['event']),
        int(row['element_in']),
        int(row['element_out']),
        int(row.get('element_in_cost', 0) or 0),
        int(row.get('element_out_cost', 0) or 0),
        str(row.get('time') or ''),
    ) for row in transfers]

    with closing(sqlite3.connect(str(database))) as connection:
        connection.executescript(_DDL)
        connection.execute(
            """
            INSERT INTO fpl_entry_sync (entry_id, started_event, current_event, synced_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(entry_id) DO UPDATE SET
                started_event=excluded.started_event,
                current_event=excluded.current_event,
                synced_at=excluded.synced_at
            """,
            (
                entry,
                profile.get('started_event'),
                profile.get('current_event'),
                synced_at,
            ),
        )
        for table in ('fpl_entry_gameweeks', 'fpl_entry_picks', 'fpl_entry_transfers'):
            connection.execute(f'DELETE FROM {table} WHERE entry_id = ?', (entry,))
        connection.executemany(
            """
            INSERT INTO fpl_entry_gameweeks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            gameweek_rows,
        )
        connection.executemany(
            "INSERT INTO fpl_entry_picks VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            pick_rows,
        )
        connection.executemany(
            "INSERT INTO fpl_entry_transfers VALUES (?, ?, ?, ?, ?, ?, ?)",
            transfer_rows,
        )
        connection.commit()

    return load_synced_entry(entry, database)


def load_synced_entry(
    entry_id: int | str,
    database: str | Path = DB_FILE,
) -> dict[str, Any]:
    """Load the latest permanent public squad and its acquisition metadata."""
    entry = _validate_entry_id(entry_id)
    with closing(sqlite3.connect(str(database))) as connection:
        connection.row_factory = sqlite3.Row
        table_exists = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='fpl_entry_sync'"
        ).fetchone()
        if not table_exists:
            raise FplEntrySyncError('No synchronized FPL entry exists locally.')
        sync_row = connection.execute(
            'SELECT * FROM fpl_entry_sync WHERE entry_id = ?', (entry,)
        ).fetchone()
        if sync_row is None:
            raise FplEntrySyncError(f'FPL entry {entry} has not been synchronized.')
        gameweeks = [dict(row) for row in connection.execute(
            'SELECT * FROM fpl_entry_gameweeks WHERE entry_id = ? ORDER BY event',
            (entry,),
        ).fetchall()]
        permanent = [
            row for row in gameweeks
            if row.get('active_chip') not in ('freehit', 'free_hit')
        ]
        squad_event = int(permanent[-1]['event']) if permanent else None
        latest_event = int(gameweeks[-1]['event']) if gameweeks else None
        picks = [dict(row) for row in connection.execute(
            """
            SELECT * FROM fpl_entry_picks
            WHERE entry_id = ? AND event = ? ORDER BY pick_position
            """,
            (entry, squad_event),
        ).fetchall()] if squad_event is not None else []
        transfers = [dict(row) for row in connection.execute(
            'SELECT * FROM fpl_entry_transfers WHERE entry_id = ? ORDER BY event, transfer_time',
            (entry,),
        ).fetchall()]

    acquisition_costs: dict[int, int] = {}
    if squad_event is not None:
        first_event = int(gameweeks[0]['event'])
        first_picks = [dict(row) for row in sqlite_rows(
            database,
            'SELECT * FROM fpl_entry_picks WHERE entry_id = ? AND event = ?',
            (entry, first_event),
        )]
        initial_ids = [int(row['player_id']) for row in first_picks]
        acquisition_costs.update(
            load_initial_purchase_prices(database, initial_ids, first_event)
        )
        for transfer in transfers:
            if int(transfer['event']) > squad_event:
                continue
            acquisition_costs.pop(int(transfer['player_out']), None)
            acquisition_costs[int(transfer['player_in'])] = int(transfer['player_in_cost'])

    latest_gameweek = permanent[-1] if permanent else (gameweeks[-1] if gameweeks else {})
    return {
        'entry_id': entry,
        'started_event': sync_row['started_event'],
        'current_event': sync_row['current_event'],
        'synced_at': sync_row['synced_at'],
        'latest_event': latest_event,
        'squad_event': squad_event,
        'gameweeks': gameweeks,
        'picks': picks,
        'transfers': transfers,
        'acquisition_costs': acquisition_costs,
        'bank': int(latest_gameweek.get('bank', 0) or 0),
        'squad_value': int(latest_gameweek.get('squad_value', 0) or 0),
        'next_free_transfers': _estimate_next_free_transfers(gameweeks),
    }


def sqlite_rows(
    database: str | Path,
    query: str,
    parameters: tuple,
) -> list[sqlite3.Row]:
    with closing(sqlite3.connect(str(database))) as connection:
        connection.row_factory = sqlite3.Row
        return connection.execute(query, parameters).fetchall()


def load_initial_purchase_prices(
    database: str | Path,
    player_ids: list[int],
    event: int,
) -> dict[int, int]:
    if not player_ids:
        return {}
    with closing(sqlite3.connect(str(database))) as connection:
        table_exists = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='player_gw'"
        ).fetchone()
        if not table_exists:
            return {}
        metadata_exists = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='app_metadata'"
        ).fetchone()
        metadata = dict(connection.execute(
            'SELECT key, value FROM app_metadata'
        ).fetchall()) if metadata_exists else {}
        live_start = int(metadata.get('live_start_index', 0))
        internal_gameweek = live_start + int(event) - 1
        placeholders = ','.join('?' for _ in player_ids)
        rows = connection.execute(
            f"""
            SELECT player_id, MAX(value)
            FROM player_gw
            WHERE Game_Week = ? AND player_id IN ({placeholders})
            GROUP BY player_id
            """,
            (internal_gameweek, *player_ids),
        ).fetchall()
    return {
        int(player_id): int(value)
        for player_id, value in rows
        if value is not None
    }
