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
    for model_type in requested_model_types(selection):
        print(f'Training {model_type} position models...')
        sequence_length = None
        if model_type == 'xgboost':
            models, metrics = train_models(all_data, completed_internal_index)
        else:
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
        saved[model_type] = path
        print(
            f'Saved {model_type} checkpoint to {path} '
            f'(cutoff={completed_internal_index})'
        )
    return saved
