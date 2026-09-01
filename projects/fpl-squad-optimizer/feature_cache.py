"""Persistent SQLite cache for model-ready temporal player features."""
from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import pandas as pd

FEATURE_CACHE_TABLE = 'model_feature_cache'
FEATURE_CACHE_METADATA_TABLE = 'model_feature_cache_metadata'
RNN_SEQUENCE_INDEX_TABLE = 'rnn_sequence_index'
FEATURE_CACHE_SCHEMA_VERSION = 2


def canonical_player_gameweeks(rows: pd.DataFrame) -> pd.DataFrame:
    """Choose one deterministic, target-bearing row per player/Gameweek."""
    canonical = rows.copy()
    canonical['_target_available'] = canonical['target'].notna().astype(int)
    order = ['id', 'Game_Week', '_target_available']
    ascending = [True, True, False]
    if 'kickoff_time' in canonical.columns:
        order.append('kickoff_time')
        ascending.append(True)
    return (
        canonical.sort_values(order, ascending=ascending)
        .drop_duplicates(['id', 'Game_Week'], keep='first')
        .drop(columns='_target_available')
    )


def _table_exists(connection: sqlite3.Connection, table: str) -> bool:
    return connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchone() is not None


def _table_content_digest(
    connection: sqlite3.Connection,
    table: str,
) -> str | None:
    if not _table_exists(connection, table):
        return None
    digest = hashlib.sha256()
    cursor = connection.execute(f'SELECT * FROM "{table}" ORDER BY rowid')
    digest.update(json.dumps([column[0] for column in cursor.description]).encode('utf-8'))
    for row in cursor:
        digest.update(json.dumps(row, default=str, separators=(',', ':')).encode('utf-8'))
        digest.update(b'\n')
    return digest.hexdigest()


def source_fingerprint(connection: sqlite3.Connection) -> str:
    """Hash every persisted source whose change invalidates engineered rows."""
    source_state = {
        table: _table_content_digest(connection, table)
        for table in (
            'app_metadata',
            'player_gw',
            'players_raw',
            'player_history',
            'player_transfer_value_history',
            'player_external_appearances',
            'player_attributes',
        )
    }
    source_state['feature_pipeline_version'] = FEATURE_CACHE_SCHEMA_VERSION
    encoded = json.dumps(source_state, sort_keys=True, default=str).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def persist_feature_cache(db_file: str | Path, rows: pd.DataFrame) -> dict:
    """Replace the local model-ready cache and return its audit metadata."""
    with closing(sqlite3.connect(str(db_file))) as connection:
        fingerprint = source_fingerprint(connection)
        cached = rows.copy()
        cached.insert(0, '_cache_row_id', rows.index.astype(int))
        cached.to_sql(FEATURE_CACHE_TABLE, connection, if_exists='replace', index=False)
        connection.execute(
            f'CREATE UNIQUE INDEX IF NOT EXISTS idx_{FEATURE_CACHE_TABLE}_row '
            f'ON {FEATURE_CACHE_TABLE} (_cache_row_id)'
        )
        connection.execute(
            f'CREATE INDEX IF NOT EXISTS idx_{FEATURE_CACHE_TABLE}_lookup '
            f'ON {FEATURE_CACHE_TABLE} (Game_Week, element_type, id)'
        )
        metadata = {
            'schema_version': FEATURE_CACHE_SCHEMA_VERSION,
            'source_fingerprint': fingerprint,
            'created_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
            'row_count': len(rows),
            'column_count': len(rows.columns),
            'columns_json': json.dumps(list(rows.columns)),
            'min_game_week': int(rows['Game_Week'].min()),
            'max_game_week': int(rows['Game_Week'].max()),
        }
        pd.DataFrame([metadata]).to_sql(
            FEATURE_CACHE_METADATA_TABLE, connection, if_exists='replace', index=False
        )
        connection.commit()
    return metadata


def load_feature_cache(db_file: str | Path) -> pd.DataFrame | None:
    """Return cached features only when their complete source fingerprint matches."""
    with closing(sqlite3.connect(str(db_file))) as connection:
        if not (
            _table_exists(connection, FEATURE_CACHE_TABLE)
            and _table_exists(connection, FEATURE_CACHE_METADATA_TABLE)
        ):
            return None
        metadata_row = connection.execute(
            f'SELECT schema_version, source_fingerprint, row_count, columns_json '
            f'FROM {FEATURE_CACHE_METADATA_TABLE} LIMIT 1'
        ).fetchone()
        if metadata_row is None:
            return None
        schema_version, fingerprint, row_count, columns_json = metadata_row
        if int(schema_version) != FEATURE_CACHE_SCHEMA_VERSION:
            return None
        if fingerprint != source_fingerprint(connection):
            return None
        cached = pd.read_sql(
            f'SELECT * FROM {FEATURE_CACHE_TABLE} ORDER BY _cache_row_id', connection
        )

    if len(cached) != int(row_count):
        return None
    columns = json.loads(columns_json)
    if [column for column in cached.columns if column != '_cache_row_id'] != columns:
        return None
    return cached.set_index('_cache_row_id').rename_axis(None)


def feature_cache_metadata(db_file: str | Path) -> dict | None:
    """Return audit metadata for the valid current cache, or ``None``."""
    with closing(sqlite3.connect(str(db_file))) as connection:
        if not _table_exists(connection, FEATURE_CACHE_METADATA_TABLE):
            return None
        columns = [
            row[1]
            for row in connection.execute(
                f'PRAGMA table_info({FEATURE_CACHE_METADATA_TABLE})'
            ).fetchall()
        ]
        row = connection.execute(
            f'SELECT * FROM {FEATURE_CACHE_METADATA_TABLE} LIMIT 1'
        ).fetchone()
        if row is None:
            return None
        metadata = dict(zip(columns, row))
        if metadata.get('source_fingerprint') != source_fingerprint(connection):
            return None
        metadata.pop('columns_json', None)
        return metadata


def load_or_build_feature_cache(
    db_file: str | Path,
    builder: Callable[[str | Path], pd.DataFrame],
) -> tuple[pd.DataFrame, bool]:
    """Load a valid cache or rebuild and persist it; returns ``(rows, cache_hit)``."""
    cached = load_feature_cache(db_file)
    if cached is not None:
        return cached, True
    rows = builder(db_file)
    persist_feature_cache(db_file, rows)
    return rows, False


def persist_rnn_sequence_index(
    db_file: str | Path,
    rows: pd.DataFrame,
    completed_internal_index: int,
) -> int:
    """Persist the exact all-history span used by every GRU training sample."""
    eligible = canonical_player_gameweeks(
        rows[rows['Game_Week'] <= int(completed_internal_index)]
    )
    eligible = eligible[eligible['target'].notna()].copy()
    eligible['sample_row_id'] = eligible.index.astype(int)
    eligible['sequence_start_game_week'] = eligible.groupby('id')['Game_Week'].transform('min')
    eligible['sequence_length'] = eligible.groupby('id').cumcount() + 1
    eligible['distinct_gameweeks'] = eligible['sequence_length']
    index_rows = eligible[[
        'sample_row_id', 'id', 'element_type', 'Game_Week',
        'sequence_start_game_week', 'sequence_length', 'distinct_gameweeks'
    ]].rename(columns={
        'id': 'player_id',
        'Game_Week': 'sequence_end_game_week',
    })
    index_rows['completed_internal_index'] = int(completed_internal_index)
    index_rows['sequence_policy'] = 'all_available'
    index_rows['cached_at'] = datetime.now(timezone.utc).isoformat(timespec='seconds')

    with closing(sqlite3.connect(str(db_file))) as connection:
        index_rows.to_sql(
            RNN_SEQUENCE_INDEX_TABLE, connection, if_exists='replace', index=False
        )
        connection.execute(
            f'CREATE UNIQUE INDEX IF NOT EXISTS idx_{RNN_SEQUENCE_INDEX_TABLE}_sample '
            f'ON {RNN_SEQUENCE_INDEX_TABLE} (sample_row_id)'
        )
        connection.execute(
            f'CREATE INDEX IF NOT EXISTS idx_{RNN_SEQUENCE_INDEX_TABLE}_lookup '
            f'ON {RNN_SEQUENCE_INDEX_TABLE} (player_id, sequence_end_game_week)'
        )
        connection.commit()
    return len(index_rows)
