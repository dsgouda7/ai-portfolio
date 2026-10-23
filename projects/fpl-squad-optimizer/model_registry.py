from __future__ import annotations

import os
import json
import hashlib
import logging
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from feature_cache import feature_cache_metadata
from opposition import (
    OPPOSITION_VERSION,
    OppositionAdjustment,
    apply_opposition_adjustment,
    evaluation_gate,
)
from utils import (
    DB_FILE,
    MODELS_FILE,
    POS_FEATURES,
    apply_market_value_weighting,
    get_runtime_context,
    normalize_pool_scores,
)

MODEL_TYPES = ('xgboost', 'catboost', 'lambdarank', 'rnn', 'ensemble')


def model_artifact_path(
    model_type: str,
    configured_path: str | Path | None = None,
) -> Path:
    if model_type not in MODEL_TYPES:
        raise ValueError(f'Unsupported model type: {model_type}')
    if configured_path:
        return Path(configured_path)
    env_name = f'FPL_{model_type.upper()}_MODELS_FILE'
    if os.environ.get(env_name):
        return Path(os.environ[env_name])
    base = Path(MODELS_FILE)
    if model_type == 'xgboost':
        return base
    return base.with_name(f'{base.stem}_{model_type}{base.suffix}')


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


def _aggregate_gameweek_predictions(scored: pd.DataFrame) -> pd.DataFrame:
    fixture_rows = scored.copy()
    if 'opponent_team' in fixture_rows:
        has_fixture = pd.to_numeric(
            fixture_rows['opponent_team'], errors='coerce'
        ).fillna(0).gt(0)
    else:
        has_fixture = pd.Series(True, index=fixture_rows.index)
    fixture_rows['_has_fixture'] = has_fixture
    fixture_rows['_fixture_prediction'] = (
        fixture_rows['predicted_points'].astype(float) * has_fixture.astype(float)
    )
    sort_columns = ['id']
    if 'kickoff_time' in fixture_rows:
        sort_columns.append('kickoff_time')
    if 'scheduled_fixture' in fixture_rows:
        sort_columns.append('scheduled_fixture')
    fixture_rows = fixture_rows.sort_values(sort_columns)

    predictions = fixture_rows.groupby('id')['_fixture_prediction'].sum()
    fixture_counts = fixture_rows.groupby('id')['_has_fixture'].sum().astype(int)
    aggregated = fixture_rows.drop_duplicates('id', keep='first').copy()
    aggregated['predicted_points'] = aggregated['id'].map(predictions)
    aggregated['fixture_count'] = aggregated['id'].map(fixture_counts).astype(int)
    if 'model_predicted_points' in fixture_rows:
        raw_predictions = (
            fixture_rows['model_predicted_points'].astype(float)
            * fixture_rows['_has_fixture'].astype(float)
        ).groupby(fixture_rows['id']).sum()
        aggregated['model_predicted_points'] = aggregated['id'].map(raw_predictions)
    if 'opposition_base_points' in fixture_rows:
        base = (
            fixture_rows['opposition_base_points'] * fixture_rows['_has_fixture']
        ).groupby(fixture_rows['id']).sum()
        aggregated['opposition_base_points'] = aggregated['id'].map(base)
        aggregated['opposition_factor'] = (
            aggregated['predicted_points'].div(aggregated['opposition_base_points'])
            .where(aggregated['opposition_base_points'].gt(0), 1.0)
        )
        details = {}
        for player_id, group in fixture_rows.groupby('id'):
            details[player_id] = [
                {
                    'opponent_team': int(row['opponent_team']),
                    'fixture': next((
                        int(float(row[column]))
                        for column in ('scheduled_fixture', 'fixture')
                        if pd.notna(row.get(column)) and float(row[column]) > 0
                    ), 0),
                    'base_points': float(row['opposition_base_points']),
                    'pressure': float(row['opposition_pressure']),
                    'factor': float(row['opposition_factor']),
                    'adjusted_points': float(row['predicted_points']),
                }
                for row in group[group['_has_fixture']].to_dict(orient='records')
            ]
        aggregated['opposition_fixtures'] = aggregated['id'].map(details)
    return aggregated.drop(
        columns=['_has_fixture', '_fixture_prediction'], errors='ignore'
    ).reset_index(drop=True)


def score_checkpoint_fixture_rows(
    all_data: pd.DataFrame,
    checkpoint: dict[str, Any],
    snapshot_game_week: int,
) -> pd.DataFrame:
    validate_checkpoint_cutoff(checkpoint)
    model_type = checkpoint.get('model_type')
    if model_type == 'ensemble':
        import joblib
        component_names = sorted(
            checkpoint['component_artifacts'], key=lambda name: name != 'rnn'
        )
        component_pools = []
        for name in component_names:
            component_path = Path(checkpoint['component_artifacts'][name])
            if not component_path.is_absolute():
                component_path = model_artifact_path('ensemble').parent / component_path
            component = joblib.load(component_path)
            component_manifest = component.get('training_manifest') or {}
            if (
                component_manifest.get('completed_internal_index')
                != checkpoint['training_manifest'].get('completed_internal_index')
            ):
                raise ValueError(f'Ensemble component cutoff mismatch: {name}')
            component_pools.append(
                score_checkpoint_fixture_rows(all_data, component, snapshot_game_week)
            )
        keys = ['id'] + [
            column for column in ('scheduled_fixture', 'fixture', 'kickoff_time')
            if all(column in pool for pool in component_pools)
        ]
        if any(pool.duplicated(keys).any() for pool in component_pools):
            raise ValueError('Ensemble components contain duplicate fixture predictions.')
        scored = component_pools[0].set_index(keys).copy()
        predictions = pd.concat(
            [pool.set_index(keys)['predicted_points'] for pool in component_pools],
            axis=1,
            join='inner',
        )
        expected_players = len(component_pools[0])
        if len(predictions) != expected_players or any(
            len(pool) != expected_players for pool in component_pools
        ):
            raise ValueError('Ensemble components do not cover the same player pool.')
        # Component scores are already on the points scale and receive the
        # market-value adjustment exactly once inside their own scorer.
        predictions.columns = component_names
        positions = scored['element_type']
        blended = pd.Series(index=predictions.index, dtype=float)
        for position in positions.unique():
            player_ids = positions[positions == position].index.intersection(predictions.index)
            raw_weights = checkpoint['ensemble_weights'][position]
            weights = pd.Series(raw_weights, dtype=float).reindex(component_names)
            weights = weights / weights.sum()
            blended.loc[player_ids] = predictions.loc[player_ids].mul(weights, axis=1).sum(axis=1)
        scored['predicted_points'] = blended
        return scored.reset_index()
    if model_type == 'rnn':
        from train.rnn_trainer import score_rnn_snapshot
        scored = score_rnn_snapshot(all_data, checkpoint, snapshot_game_week)
    elif model_type in ('xgboost', 'catboost', 'lambdarank'):
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

    return apply_market_value_weighting(scored)


def checkpoint_fingerprint(checkpoint: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    path = model_artifact_path(checkpoint['model_type'])
    digest.update(checkpoint['model_type'].encode('utf-8'))
    digest.update(path.read_bytes())
    if checkpoint.get('model_type') == 'ensemble':
        for name, filename in sorted(checkpoint['component_artifacts'].items()):
            path = Path(filename)
            if not path.is_absolute():
                path = model_artifact_path('ensemble').parent / path
            digest.update(name.encode('utf-8'))
            digest.update(path.read_bytes())
    return digest.hexdigest()


def opposition_policy_path(model_type: str) -> Path:
    path = model_artifact_path(model_type)
    return path.with_name(f'{path.stem}_opposition.json')


def score_checkpoint_snapshot(
    all_data: pd.DataFrame,
    checkpoint: dict[str, Any],
    snapshot_game_week: int,
) -> pd.DataFrame:
    """Apply a validated opposition policy once, before summing fixture scores."""
    fixtures = score_checkpoint_fixture_rows(all_data, checkpoint, snapshot_game_week)
    status = {'status': 'disabled', 'reason': 'No validated opposition policy.'}
    path = opposition_policy_path(checkpoint['model_type'])
    if path.exists():
        report = json.loads(path.read_text(encoding='utf-8'))
        if report.get('opposition_version') != OPPOSITION_VERSION:
            raise ValueError('Unsupported opposition policy version.')
        if report.get('model_type') != checkpoint['model_type']:
            raise ValueError('Opposition policy belongs to a different model family.')
        adjustment = OppositionAdjustment(**report['adjustment'])
        if not evaluation_gate(report['gameweeks'])['passed']:
            raise ValueError('Opposition policy has not passed its rollout gate.')
        if report['checkpoint_fingerprint'] != checkpoint_fingerprint(checkpoint):
            status = {
                'status': 'stale',
                'reason': 'Opposition policy belongs to an older checkpoint; re-evaluate it.',
            }
            logging.getLogger(__name__).warning(status['reason'])
        else:
            fixtures = apply_opposition_adjustment(fixtures, adjustment)
            status = {
                'status': 'active',
                'reason': 'Frozen opponent forecasts; validated final-stage adjustment.',
                'version': OPPOSITION_VERSION,
                'strength': adjustment.strength,
                'max_reduction': adjustment.max_reduction,
            }
    result = normalize_pool_scores(_aggregate_gameweek_predictions(fixtures))
    result.attrs['opposition_adjustment'] = status
    return result
