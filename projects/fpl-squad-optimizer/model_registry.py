from __future__ import annotations

import os
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from feature_cache import feature_cache_metadata
from utils import (
    DB_FILE,
    MODELS_FILE,
    POS_FEATURES,
    apply_market_value_weighting,
    get_runtime_context,
    normalize_pool_scores,
)

MODEL_TYPES = ('xgboost', 'rnn')


def model_artifact_path(
    model_type: str,
    configured_path: str | Path | None = None,
) -> Path:
    if model_type not in MODEL_TYPES:
        raise ValueError(f'Unsupported model type: {model_type}')
    if configured_path:
        return Path(configured_path)
    if model_type == 'rnn' and os.environ.get('FPL_RNN_MODELS_FILE'):
        return Path(os.environ['FPL_RNN_MODELS_FILE'])
    base = Path(MODELS_FILE)
    if model_type == 'xgboost':
        return base
    return base.with_name(f'{base.stem}_rnn{base.suffix}')


def available_model_artifacts() -> dict[str, Path]:
    return {
        model_type: path
        for model_type in MODEL_TYPES
        if (path := model_artifact_path(model_type)).exists()
    }


def build_training_manifest(
    all_data: pd.DataFrame,
    model_type: str,
    completed_internal_index: int,
    db_file: str | Path = DB_FILE,
    sequence_length: int | None = None,
) -> dict[str, Any]:
    """Describe exactly which temporal observations a checkpoint may use."""
    cutoff = int(completed_internal_index)
    training_rows = all_data[
        (all_data['Game_Week'] <= cutoff) & all_data['target'].notna()
    ].copy()
    if training_rows.empty:
        raise ValueError(f'No completed training rows exist at cutoff {cutoff}.')
    if int(training_rows['Game_Week'].max()) > cutoff:
        raise ValueError('Training rows exceed the completed-data cutoff.')

    runtime = get_runtime_context(str(db_file))
    kickoff = pd.to_datetime(
        training_rows.get('kickoff_time'), errors='coerce', utc=True
    )
    fpl_cutoff = kickoff.max().isoformat() if kickoff.notna().any() else None
    fpl_price_rows = int(training_rows.get('value', pd.Series(dtype=float)).notna().sum())

    tm_manifest = {
        'available': False,
        'observation_count': 0,
        'first_observed_at': None,
        'last_observed_at': None,
        'sources': [],
        'historical_backfill_available': False,
        'imputation_allowed': False,
        'policy': (
            'as-of observations only; current Transfermarkt values are never '
            'backfilled into earlier gameweeks'
        ),
    }
    try:
        with closing(sqlite3.connect(str(db_file))) as connection:
            row = connection.execute(
                """
                SELECT COUNT(*), MIN(observed_at), MAX(observed_at)
                FROM player_transfer_value_history
                                WHERE datetime(observed_at) <= datetime(?)
                                    AND tm_value_imputed = 0
                """
                , (fpl_cutoff,)
            ).fetchone()
            sources = connection.execute(
                """
                SELECT DISTINCT source
                FROM player_transfer_value_history
                WHERE datetime(observed_at) <= datetime(?)
                                    AND tm_value_imputed = 0
                ORDER BY source
                """,
                (fpl_cutoff,),
            ).fetchall()
        tm_manifest.update({
            'available': bool(row and row[0]),
            'observation_count': int(row[0] or 0),
            'first_observed_at': row[1],
            'last_observed_at': row[2],
            'sources': [source[0] for source in sources],
            'historical_backfill_available': any(
                source[0] == 'transfermarkt_historical_cc0' for source in sources
            ),
        })
    except sqlite3.Error:
        pass

    return {
        'schema_version': 1,
        'model_type': model_type,
        'trained_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'season': runtime['season'],
        'latest_completed_game_week': runtime['latest_completed_game_week'],
        'target_game_week': runtime['target_game_week'],
        'completed_internal_index': cutoff,
        'snapshot_internal_index': runtime['snapshot_game_week'],
        'max_training_internal_index': int(training_rows['Game_Week'].max()),
        'min_training_internal_index': int(training_rows['Game_Week'].min()),
        'fpl_data_cutoff': fpl_cutoff,
        'training_rows': len(training_rows),
        'features': {
            position: list(features) for position, features in POS_FEATURES.items()
        },
        'temporal_fpl_price_rows': fpl_price_rows,
        'fpl_price_source': 'official per-player gameweek value field',
        'transfermarkt': tm_manifest,
        'feature_cache': feature_cache_metadata(db_file),
        'sequence_length': sequence_length,
    }


def validate_checkpoint_cutoff(checkpoint: dict[str, Any]) -> None:
    manifest = checkpoint.get('training_manifest') or {}
    cutoff = manifest.get('completed_internal_index')
    maximum = manifest.get('max_training_internal_index')
    if cutoff is None or maximum is None:
        raise ValueError('Checkpoint does not contain a temporal cutoff manifest.')
    if int(maximum) > int(cutoff):
        raise ValueError('Checkpoint training data extends beyond its completed cutoff.')


def build_checkpoint(
    model_type: str,
    models: dict[str, Any],
    metrics: dict[str, Any],
    manifest: dict[str, Any],
    epl_members: Any = None,
) -> dict[str, Any]:
    checkpoint = {
        'model_type': model_type,
        'models': models,
        'metrics': metrics,
        'epl_members': epl_members,
        'pos_features': {
            position: list(features) for position, features in POS_FEATURES.items()
        },
        'training_manifest': manifest,
        'model_names': {
            position: metric.get('model_name', position)
            for position, metric in metrics.items()
        },
    }
    validate_checkpoint_cutoff(checkpoint)
    return checkpoint


def score_checkpoint_snapshot(
    all_data: pd.DataFrame,
    checkpoint: dict[str, Any],
    snapshot_game_week: int,
) -> pd.DataFrame:
    validate_checkpoint_cutoff(checkpoint)
    model_type = checkpoint.get('model_type')
    if model_type == 'rnn':
        from train.rnn_trainer import score_rnn_snapshot
        scored = score_rnn_snapshot(all_data, checkpoint, snapshot_game_week)
    elif model_type == 'xgboost':
        snapshot = all_data[all_data['Game_Week'] == snapshot_game_week].copy()
        parts = []
        for position, model in checkpoint['models'].items():
            players = snapshot[snapshot['element_type'] == position].copy()
            if players.empty:
                continue
            features = checkpoint['pos_features'][position]
            inputs = players[features].apply(
                pd.to_numeric, errors='coerce'
            )
            players['predicted_points'] = model.predict(inputs)
            parts.append(players)
        if not parts:
            raise ValueError(
                f'No XGBoost prediction rows exist for snapshot {snapshot_game_week}.'
            )
        scored = pd.concat(parts, ignore_index=True)
    else:
        raise ValueError(f'Checkpoint has unsupported model type: {model_type!r}')

    return normalize_pool_scores(apply_market_value_weighting(scored))
