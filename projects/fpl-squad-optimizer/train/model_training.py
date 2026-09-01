"""Shared model training and artifact persistence used locally and in pipelines."""
from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from feature_cache import persist_rnn_sequence_index
from model_registry import (
    MODEL_TYPES,
    build_checkpoint,
    build_training_manifest,
    model_artifact_path,
)
from train.rnn_trainer import train_rnn_models
from train.trainer import train_models
from train.advanced_models import train_catboost_models, train_lambdarank_models
from utils import (
    DB_FILE,
    MARKET_VALUE_WEIGHT_NO_HISTORY,
    MARKET_VALUE_WEIGHT_PL_HISTORY,
)


def requested_model_types(selection: str) -> tuple[str, ...]:
    if selection == 'all':
        return MODEL_TYPES
    if selection not in MODEL_TYPES:
        raise ValueError(f'Unsupported model selection: {selection}')
    return (selection,)


def train_and_save_models(
    all_data: pd.DataFrame,
    completed_internal_index: int,
    selection: str,
    epl_members=None,
    db_file: str | Path = DB_FILE,
) -> dict[str, Path]:
    saved: dict[str, Path] = {}
    trained_checkpoints = {}
    requested = requested_model_types(selection)
    base_types = tuple(model_type for model_type in requested if model_type != 'ensemble')
    if 'ensemble' in requested:
        base_types = tuple(dict.fromkeys((*base_types, 'xgboost', 'catboost', 'lambdarank', 'rnn')))
    for model_type in base_types:
        print(f'Training {model_type} position models...')
        sequence_length = None
        if model_type == 'xgboost':
            models, metrics = train_models(all_data, completed_internal_index)
        elif model_type == 'catboost':
            models, metrics = train_catboost_models(all_data, completed_internal_index)
        elif model_type == 'lambdarank':
            models, metrics = train_lambdarank_models(all_data, completed_internal_index)
        elif model_type == 'rnn':
            sequence_rows = persist_rnn_sequence_index(
                db_file, all_data, completed_internal_index
            )
            print(f'Persisted {sequence_rows:,} RNN sequence spans in SQLite.')
            models, metrics = train_rnn_models(
                all_data, completed_internal_index, sequence_length
            )

        manifest = build_training_manifest(
            all_data,
            model_type,
            completed_internal_index,
            db_file,
            sequence_length,
        )
        checkpoint = build_checkpoint(
            model_type, models, metrics, manifest, epl_members
        )
        if model_type == 'rnn':
            trained_samples = sum(metric['n'] for metric in metrics.values())
            if trained_samples != sequence_rows:
                raise ValueError(
                    'RNN SQLite sequence index does not match training samples: '
                    f'{sequence_rows} persisted versus {trained_samples} trained.'
                )
            checkpoint['training_manifest']['sequence_policy'] = 'all_available'
            checkpoint['training_manifest']['sequence_samples'] = sequence_rows
            checkpoint['training_manifest']['max_sequence_lengths'] = {
                position: spec['max_training_sequence_length']
                for position, spec in models.items()
            }
        checkpoint['market_value_weights'] = {
            'pl_history': MARKET_VALUE_WEIGHT_PL_HISTORY,
            'no_pl_history': MARKET_VALUE_WEIGHT_NO_HISTORY,
            'imputed': 0.0,
        }
        path = model_artifact_path(model_type)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(checkpoint, path)
        trained_checkpoints[model_type] = checkpoint
        saved[model_type] = path
        print(
            f'Saved {model_type} checkpoint to {path} '
            f'(cutoff={completed_internal_index})'
        )
    if 'ensemble' in requested:
        component_types = ('xgboost', 'catboost', 'lambdarank', 'rnn')
        manifest = build_training_manifest(
            all_data, 'ensemble', completed_internal_index, db_file
        )
        component_metrics = {
            position: {
                'r2': round(float(pd.Series([
                    trained_checkpoints[model]['metrics'][position]['r2']
                    for model in component_types
                ]).mean()), 4),
                'rmse': round(float(pd.Series([
                    trained_checkpoints[model]['metrics'][position]['rmse']
                    for model in component_types
                ]).mean()), 4),
                'n': min(
                    trained_checkpoints[model]['metrics'][position]['n']
                    for model in component_types
                ),
                'top_feature': 'inverse-RMSE calibrated blend',
                'model_name': f'{position} ensemble',
                'n_features': len(trained_checkpoints['xgboost']['pos_features'][position]),
            }
            for position in ('GK', 'DEF', 'MID', 'FWD')
        }
        checkpoint = build_checkpoint('ensemble', {}, component_metrics, manifest, epl_members)
        checkpoint['component_artifacts'] = {
            model_type: saved[model_type].name
            for model_type in component_types
        }
        checkpoint['ensemble_weights'] = {
            position: {
                model_type: (
                    1 / max(
                        float(trained_checkpoints[model_type]['metrics'][position]['rmse']),
                        1e-6,
                    ) ** 2
                )
                for model_type in component_types
            }
            for position in ('GK', 'DEF', 'MID', 'FWD')
        }
        path = model_artifact_path('ensemble')
        joblib.dump(checkpoint, path)
        saved['ensemble'] = path
        print(f'Saved ensemble checkpoint to {path}')
    return saved
