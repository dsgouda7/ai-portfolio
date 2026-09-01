from __future__ import annotations

import json
import os
import sqlite3
import tempfile
from collections import Counter
from contextlib import closing
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from utils import DB_FILE, MAX_PLAYERS_PER_TEAM, MAX_SPEND, pick_starting_xi

SQUAD_SIZE = 15
POSITION_LIMITS = {'GK': 2, 'DEF': 5, 'MID': 5, 'FWD': 3}
STARTER_LIMITS = {
    'GK': (1, 1),
    'DEF': (3, 5),
    'MID': (2, 5),
    'FWD': (1, 3),
}
CHIPS = ('wildcard', 'free_hit', 'bench_boost', 'triple_captain')
MAX_FREE_TRANSFERS = 5


class SquadValidationError(ValueError):
    pass


def default_squad_dir() -> Path:
    configured = os.environ.get('FPL_SQUAD_DIR')
    return Path(configured) if configured else Path(__file__).parent / 'squads'


def selling_price(purchase_price: int, current_price: int) -> int:
    """Return the official FPL sale value, in tenths of a million."""
    purchase_price = int(purchase_price)
    current_price = int(current_price)
    if current_price <= purchase_price:
        return current_price
    return purchase_price + (current_price - purchase_price) // 2


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f'.{path.name}.', suffix='.tmp'
    )
    try:
        with os.fdopen(handle, 'w', encoding='utf-8') as stream:
            json.dump(data, stream, indent=2, ensure_ascii=True)
            stream.write('\n')
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open(encoding='utf-8') as stream:
        return json.load(stream)


def state_paths(directory: Path | None = None) -> dict[str, Path]:
    root = directory or default_squad_dir()
    return {
        'root': root,
        'current': root / 'current_squad.json',
        'draft': root / 'draft_squad.json',
        'history': root / 'history',
    }


def _database_path(database: str | Path | None) -> str:
    return str(database or DB_FILE)


def _ensure_state_schema(connection: sqlite3.Connection) -> None:
    connection.executescript("""
        CREATE TABLE IF NOT EXISTS squad_versions (
            version_id            INTEGER PRIMARY KEY AUTOINCREMENT,
            season                TEXT NOT NULL,
            game_week             INTEGER NOT NULL,
            revision              INTEGER NOT NULL,
            status                TEXT NOT NULL CHECK (status IN ('draft', 'committed')),
            source                TEXT NOT NULL,
            created_at            TEXT NOT NULL,
            updated_at            TEXT NOT NULL,
            saved_at              TEXT NOT NULL,
            bank                  INTEGER NOT NULL,
            free_transfers        INTEGER NOT NULL,
            transfer_points_cost  INTEGER NOT NULL,
            active_chip           TEXT,
            state_json            TEXT NOT NULL,
            UNIQUE (season, game_week, revision)
        );

        CREATE TABLE IF NOT EXISTS squad_version_players (
            version_id        INTEGER NOT NULL,
            player_id         INTEGER NOT NULL,
            squad_order       INTEGER NOT NULL,
            lineup_role       TEXT NOT NULL CHECK (lineup_role IN ('starter', 'bench')),
            bench_order       INTEGER,
            is_captain        INTEGER NOT NULL,
            is_vice_captain   INTEGER NOT NULL,
            first_name        TEXT NOT NULL,
            second_name       TEXT NOT NULL,
            position          TEXT NOT NULL,
            team              INTEGER NOT NULL,
            purchase_price    INTEGER NOT NULL,
            current_price     INTEGER NOT NULL,
            selling_price     INTEGER NOT NULL,
            predicted_points  REAL NOT NULL,
            status            TEXT NOT NULL,
            news              TEXT NOT NULL,
            PRIMARY KEY (version_id, player_id),
            FOREIGN KEY (version_id) REFERENCES squad_versions(version_id)
        );

        CREATE INDEX IF NOT EXISTS idx_squad_versions_latest
            ON squad_versions (status, season, game_week, version_id DESC);
    """)


def _save_version(
    state: dict[str, Any],
    status: str,
    database: str | Path | None,
) -> dict[str, Any]:
    stored = deepcopy(state)
    stored['status'] = status
    stored['updated_at'] = _now()
    validate_state(stored)
    saved_at = _now()

    with closing(sqlite3.connect(_database_path(database))) as connection:
        connection.execute('PRAGMA foreign_keys = ON')
        _ensure_state_schema(connection)
        row = connection.execute(
            """
            SELECT COALESCE(MAX(revision), 0) + 1
            FROM squad_versions
            WHERE season = ? AND game_week = ?
            """,
            (stored['season'], int(stored['game_week'])),
        ).fetchone()
        stored['revision'] = int(row[0])
        cursor = connection.execute(
            """
            INSERT INTO squad_versions (
                season, game_week, revision, status, source, created_at,
                updated_at, saved_at, bank, free_transfers,
                transfer_points_cost, active_chip, state_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                stored['season'], int(stored['game_week']), stored['revision'],
                status, stored.get('source', 'unknown'), stored['created_at'],
                stored['updated_at'], saved_at, int(stored['bank']),
                int(stored['free_transfers']), int(stored['transfer_points_cost']),
                stored.get('active_chip'), '{}',
            ),
        )
        stored['version_id'] = int(cursor.lastrowid)
        connection.execute(
            'UPDATE squad_versions SET state_json = ? WHERE version_id = ?',
            (json.dumps(stored, ensure_ascii=True, sort_keys=True), stored['version_id']),
        )

        starter_ids = set(int(player_id) for player_id in stored['lineup']['starters'])
        bench_order = {
            int(player_id): index
            for index, player_id in enumerate(stored['lineup']['bench'])
        }
        player_rows = []
        for squad_order, player in enumerate(stored['players']):
            player_id = int(player['id'])
            player_rows.append((
                stored['version_id'], player_id, squad_order,
                'starter' if player_id in starter_ids else 'bench',
                bench_order.get(player_id),
                int(player_id == int(stored['lineup']['captain'])),
                int(player_id == int(stored['lineup']['vice_captain'])),
                player['first_name'], player['second_name'], player['position'],
                int(player['team']), int(player['purchase_price']),
                int(player['current_price']), int(player['selling_price']),
                float(player['predicted_points']), player['status'], player['news'],
            ))
        connection.executemany(
            """
            INSERT INTO squad_version_players (
                version_id, player_id, squad_order, lineup_role, bench_order,
                is_captain, is_vice_captain, first_name, second_name, position,
                team, purchase_price, current_price, selling_price,
                predicted_points, status, news
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            player_rows,
        )
        connection.commit()
    return stored


def _load_latest(status: str, database: str | Path | None) -> dict[str, Any] | None:
    with closing(sqlite3.connect(_database_path(database))) as connection:
        _ensure_state_schema(connection)
        row = connection.execute(
            """
            SELECT state_json
            FROM squad_versions
            WHERE status = ?
            ORDER BY version_id DESC
            LIMIT 1
            """,
            (status,),
        ).fetchone()
    return json.loads(row[0]) if row else None


def list_versions(
    database: str | Path | None = None,
    season: str | None = None,
) -> list[dict[str, Any]]:
    query = """
        SELECT version_id, season, game_week, revision, status, source,
               saved_at, bank, free_transfers, transfer_points_cost, active_chip
        FROM squad_versions
    """
    parameters: tuple[Any, ...] = ()
    if season:
        query += ' WHERE season = ?'
        parameters = (season,)
    query += ' ORDER BY version_id DESC'
    with closing(sqlite3.connect(_database_path(database))) as connection:
        connection.row_factory = sqlite3.Row
        _ensure_state_schema(connection)
        return [dict(row) for row in connection.execute(query, parameters).fetchall()]


def _player_record(row: dict[str, Any]) -> dict[str, Any]:
    current_price = int(row.get('value', row.get('current_price', 0)))
    purchase_price = int(row.get('purchase_price', current_price))
    return {
        'id': int(row['id']),
        'first_name': str(row.get('first_name', '')),
        'second_name': str(row.get('second_name', '')),
        'position': str(row.get('element_type', row.get('position', ''))),
        'team': int(row.get('team', 0)),
        'purchase_price': purchase_price,
        'current_price': current_price,
        'selling_price': selling_price(purchase_price, current_price),
        'predicted_points': round(float(row.get('predicted_points', 0)), 4),
        'status': str(row.get('elig_status', row.get('status', 'a'))),
        'news': str(row.get('news', '') or ''),
    }


def _default_lineup(squad: pd.DataFrame) -> dict[str, Any]:
    starters, bench, _ = pick_starting_xi(squad)
    starter_ids = [int(player_id) for player_id in starters['id']]
    bench_ids = [int(player_id) for player_id in bench['id']]
    ranked = starters.sort_values('predicted_points', ascending=False)
    return {
        'starters': starter_ids,
        'bench': bench_ids,
        'captain': int(ranked.iloc[0]['id']),
        'vice_captain': int(ranked.iloc[1]['id']),
    }


def create_state(
    squad: pd.DataFrame,
    game_week: int,
    season: str,
    previous: dict[str, Any] | None = None,
    source: str = 'generated',
    generation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    now = _now()
    prior_chips = (previous or {}).get('chips') or {
        chip: {'remaining': 2, 'used_gameweeks': []} for chip in CHIPS
    }
    state = {
        'schema_version': 1,
        'status': 'draft',
        'source': source,
        'season': str(season),
        'game_week': int(game_week),
        'created_at': now,
        'updated_at': now,
        'bank': max(0, MAX_SPEND - int(squad['value'].sum())),
        'free_transfers': int((previous or {}).get('free_transfers', 1)),
        'transfer_points_cost': 0,
        'transfers': [],
        'active_chip': None,
        'chips': deepcopy(prior_chips),
        'players': [_player_record(row) for row in squad.to_dict(orient='records')],
        'lineup': _default_lineup(squad),
    }
    if generation is not None:
        state['generation'] = deepcopy(generation)
    validate_state(state)
    return state


def validate_state(state: dict[str, Any]) -> None:
    players = state.get('players') or []
    if len(players) != SQUAD_SIZE:
        raise SquadValidationError(f'A squad must contain exactly {SQUAD_SIZE} players.')

    ids = [int(player['id']) for player in players]
    if len(set(ids)) != SQUAD_SIZE:
        raise SquadValidationError('A squad cannot contain duplicate players.')

    positions = Counter(player['position'] for player in players)
    if positions != Counter(POSITION_LIMITS):
        expected = ', '.join(f'{count} {position}' for position, count in POSITION_LIMITS.items())
        raise SquadValidationError(f'Squad positions must be {expected}.')

    clubs = Counter(int(player['team']) for player in players)
    if clubs and max(clubs.values()) > MAX_PLAYERS_PER_TEAM:
        raise SquadValidationError(
            f'A squad may contain at most {MAX_PLAYERS_PER_TEAM} players from one club.'
        )

    if int(state.get('bank', 0)) < 0:
        raise SquadValidationError('The squad exceeds the available budget.')
    free_transfers = int(state.get('free_transfers', 1))
    if free_transfers < 0 or free_transfers > MAX_FREE_TRANSFERS:
        raise SquadValidationError(
            f'Free transfers must be between 0 and {MAX_FREE_TRANSFERS}.'
        )
    if len(state.get('transfers', [])) > 20 and state.get('active_chip') not in (
        'wildcard', 'free_hit'
    ):
        raise SquadValidationError(
            'A maximum of 20 transfers is allowed without Wildcard or Free Hit.'
        )

    lineup = state.get('lineup') or {}
    starters = [int(player_id) for player_id in lineup.get('starters', [])]
    bench = [int(player_id) for player_id in lineup.get('bench', [])]
    if len(starters) != 11 or len(bench) != 4:
        raise SquadValidationError('The lineup must contain 11 starters and four substitutes.')
    if set(starters).intersection(bench) or set(starters + bench) != set(ids):
        raise SquadValidationError('Every squad player must appear once in the lineup or bench.')

    by_id = {int(player['id']): player for player in players}
    starter_positions = Counter(by_id[player_id]['position'] for player_id in starters)
    for position, (minimum, maximum) in STARTER_LIMITS.items():
        count = starter_positions[position]
        if count < minimum or count > maximum:
            raise SquadValidationError(
                f'The starting XI requires {minimum}-{maximum} {position} players.'
            )

    captain = int(lineup.get('captain', 0))
    vice_captain = int(lineup.get('vice_captain', 0))
    if captain not in starters or vice_captain not in starters:
        raise SquadValidationError('Captain and vice-captain must both be in the starting XI.')
    if captain == vice_captain:
        raise SquadValidationError('Captain and vice-captain must be different players.')

    active_chip = state.get('active_chip')
    if active_chip is not None and active_chip not in CHIPS:
        raise SquadValidationError(f'Unknown chip: {active_chip}.')
    if active_chip is not None:
        chip_state = (state.get('chips') or {}).get(active_chip, {})
        if int(chip_state.get('remaining', 0)) < 1:
            raise SquadValidationError(
                f'No {active_chip.replace("_", " ")} chips remain.'
            )


def save_draft(
    state: dict[str, Any],
    database: str | Path | None = None,
    export_directory: Path | None = None,
) -> dict[str, Any]:
    stored = _save_version(state, 'draft', database)
    _write_json(state_paths(export_directory)['draft'], stored)
    return stored


def commit_draft(
    state: dict[str, Any],
    database: str | Path | None = None,
    export_directory: Path | None = None,
) -> dict[str, Any]:
    state = deepcopy(state)
    state['status'] = 'committed'
    state['updated_at'] = _now()
    active_chip = state.get('active_chip')
    if active_chip:
        chip_state = state['chips'][active_chip]
        game_week = int(state['game_week'])
        used_gameweeks = [int(gw) for gw in chip_state.get('used_gameweeks', [])]
        if game_week not in used_gameweeks:
            chip_state['used_gameweeks'] = used_gameweeks + [game_week]
            chip_state['remaining'] = int(chip_state.get('remaining', 0)) - 1
    stored = _save_version(state, 'committed', database)
    paths = state_paths(export_directory)
    _write_json(paths['current'], stored)
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    history_path = paths['history'] / (
        f"gw{int(stored['game_week']):02d}-v{int(stored['revision']):03d}-"
        f"{timestamp}.json"
    )
    _write_json(history_path, stored)
    _write_json(paths['draft'], stored)
    return stored


def load_current(database: str | Path | None = None) -> dict[str, Any] | None:
    return _load_latest('committed', database)


def load_draft(database: str | Path | None = None) -> dict[str, Any] | None:
    return _load_latest('draft', database)


def load_working_state(database: str | Path | None = None) -> dict[str, Any] | None:
    """Return the newest draft or commit without creating a new revision."""
    current = load_current(database)
    draft = load_draft(database)
    if current is None:
        return draft
    if draft is None:
        return current
    return draft if int(draft['version_id']) > int(current['version_id']) else current


def set_lineup(
    state: dict[str, Any],
    starters: list[int],
    bench: list[int],
    captain: int,
    vice_captain: int,
) -> dict[str, Any]:
    updated = deepcopy(state)
    updated['lineup'] = {
        'starters': [int(player_id) for player_id in starters],
        'bench': [int(player_id) for player_id in bench],
        'captain': int(captain),
        'vice_captain': int(vice_captain),
    }
    updated['updated_at'] = _now()
    validate_state(updated)
    return updated


def set_chip(state: dict[str, Any], chip: str | None) -> dict[str, Any]:
    updated = deepcopy(state)
    if chip is not None:
        if chip not in CHIPS:
            raise SquadValidationError(f'Unknown chip: {chip}.')
        if int(updated['chips'][chip].get('remaining', 0)) < 1:
            raise SquadValidationError(f'No {chip.replace("_", " ")} chips remain.')
    updated['active_chip'] = chip
    transfer_count = len(updated.get('transfers', []))
    free_transfers = int(updated.get('free_transfers', 1))
    updated['transfer_points_cost'] = (
        0 if chip in ('wildcard', 'free_hit')
        else max(0, transfer_count - free_transfers) * 4
    )
    validate_state(updated)
    return updated


def apply_transfer(
    state: dict[str, Any],
    incoming: dict[str, Any],
    outgoing_id: int,
) -> dict[str, Any]:
    updated = deepcopy(state)
    outgoing_id = int(outgoing_id)
    by_id = {int(player['id']): player for player in updated['players']}
    if outgoing_id not in by_id:
        raise SquadValidationError('The outgoing player is not in the squad.')
    if int(incoming['id']) in by_id and int(incoming['id']) != outgoing_id:
        raise SquadValidationError('The incoming player is already in the squad.')

    outgoing = by_id[outgoing_id]
    incoming_record = _player_record(incoming)
    if incoming_record['position'] != outgoing['position']:
        raise SquadValidationError('Transfers must replace a player in the same position.')

    available = int(updated.get('bank', 0)) + int(outgoing['selling_price'])
    if incoming_record['current_price'] > available:
        raise SquadValidationError('The incoming player is outside the available transfer budget.')

    incoming_record['purchase_price'] = incoming_record['current_price']
    incoming_record['selling_price'] = incoming_record['current_price']
    updated['players'] = [
        incoming_record if int(player['id']) == outgoing_id else player
        for player in updated['players']
    ]
    updated['bank'] = available - incoming_record['current_price']
    updated.setdefault('transfers', []).append({
        'out': {'id': outgoing_id, 'name': f"{outgoing['first_name']} {outgoing['second_name']}"},
        'in': {
            'id': incoming_record['id'],
            'name': f"{incoming_record['first_name']} {incoming_record['second_name']}",
        },
        'made_at': _now(),
    })

    lineup = updated['lineup']
    lineup['starters'] = [
        incoming_record['id'] if player_id == outgoing_id else player_id
        for player_id in lineup['starters']
    ]
    lineup['bench'] = [
        incoming_record['id'] if player_id == outgoing_id else player_id
        for player_id in lineup['bench']
    ]
    if lineup['captain'] == outgoing_id:
        lineup['captain'] = incoming_record['id']
    if lineup['vice_captain'] == outgoing_id:
        lineup['vice_captain'] = incoming_record['id']

    transfer_count = len(updated['transfers'])
    free_transfers = int(updated.get('free_transfers', 1))
    updated['transfer_points_cost'] = (
        0 if updated.get('active_chip') in ('wildcard', 'free_hit')
        else max(0, transfer_count - free_transfers) * 4
    )
    updated['updated_at'] = _now()
    validate_state(updated)
    return updated


def refresh_player_data(
    state: dict[str, Any], pool: pd.DataFrame, eligibility: dict | None = None
) -> dict[str, Any]:
    updated = deepcopy(state)
    pool_by_id = {int(row['id']): row for row in pool.to_dict(orient='records')}
    eligibility = eligibility or {}
    for player in updated['players']:
        current = pool_by_id.get(int(player['id']))
        if current:
            player['current_price'] = int(current.get('value', player['current_price']))
            player['selling_price'] = selling_price(
                player['purchase_price'], player['current_price']
            )
            player['predicted_points'] = round(float(current.get('predicted_points', 0)), 4)
            key = (
                str(current.get('first_name', '')).lower(),
                str(current.get('second_name', '')).lower(),
            )
            info = eligibility.get(key)
            if info:
                player['status'] = info.status
                player['news'] = info.news
    updated['updated_at'] = _now()
    validate_state(updated)
    return updated


def roll_to_game_week(
    state: dict[str, Any],
    game_week: int,
    season: str,
) -> dict[str, Any]:
    game_week = int(game_week)
    if str(state.get('season')) == str(season) and int(state.get('game_week', 0)) == game_week:
        return deepcopy(state)

    updated = deepcopy(state)
    previous_transfers = len(updated.get('transfers', []))
    previous_free = int(updated.get('free_transfers', 1))
    previous_chip = updated.get('active_chip')
    if previous_chip in ('wildcard', 'free_hit'):
        next_free = previous_free
    elif previous_transfers == 0:
        next_free = min(MAX_FREE_TRANSFERS, previous_free + 1)
    else:
        next_free = max(1, previous_free - previous_transfers + 1)

    updated.pop('version_id', None)
    updated.pop('revision', None)
    updated['status'] = 'draft'
    updated['source'] = 'gameweek_rollover'
    updated['season'] = str(season)
    updated['game_week'] = game_week
    updated['created_at'] = _now()
    updated['updated_at'] = updated['created_at']
    updated['free_transfers'] = next_free
    updated['transfer_points_cost'] = 0
    updated['transfers'] = []
    updated['active_chip'] = None
    validate_state(updated)
    return updated
