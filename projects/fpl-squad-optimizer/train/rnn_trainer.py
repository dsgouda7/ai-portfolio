"""Leakage-safe recurrent training and inference for player-GW sequences."""
from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

from feature_cache import canonical_player_gameweeks
from utils import MODEL_NAMES, POS_FEATURES

DEFAULT_EPOCHS = int(os.environ.get('FPL_RNN_EPOCHS', '35'))


class PositionGRU(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 32) -> None:
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, sequences: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        packed = pack_padded_sequence(
            sequences, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, hidden = self.gru(packed)
        return self.head(hidden[-1]).squeeze(-1)


def _numeric_features(
    rows: pd.DataFrame,
    features: list[str],
    fill_missing: bool = True,
) -> np.ndarray:
    numeric = rows[features].apply(pd.to_numeric, errors='coerce')
    if fill_missing:
        numeric = numeric.fillna(0.0)
    return numeric.to_numpy(dtype=np.float32)


def build_position_sequences(
    rows: pd.DataFrame,
    position: str,
    cutoff: int,
    sequence_length: int | None = None,
    scaler: StandardScaler | None = None,
    target_game_week: int | None = None,
    features: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.Index, StandardScaler]:
    """Build sequences from every available player row through ``cutoff``.

    ``sequence_length`` remains an optional compatibility cap. The production
    trainer passes ``None``, so each sample starts at that player's earliest
    persisted Gameweek and ends at the sample Gameweek.
    """
    features = list(features or POS_FEATURES[position])
    position_rows = canonical_player_gameweeks(rows[
        (rows['element_type'] == position) & (rows['Game_Week'] <= cutoff)
    ])
    # One canonical pre-fixture feature row per player/Gameweek. This prevents
    # duplicate source rows or multiple fixtures in one FPL round from leaking
    # same-Gameweek information into a recurrent history.
    candidates = position_rows
    if target_game_week is not None:
        candidates = candidates[candidates['Game_Week'] == target_game_week]
    else:
        candidates = candidates[candidates['target'].notna()]
    if candidates.empty:
        raise ValueError(f'No {position} rows are available through cutoff {cutoff}.')

    if scaler is None:
        scaler = StandardScaler().fit(
            _numeric_features(position_rows, features, fill_missing=False)
        )

    histories: list[np.ndarray] = []
    lengths: list[int] = []
    targets: list[float] = []
    indexes: list[Any] = []
    by_player = {player_id: group for player_id, group in position_rows.groupby('id')}
    for index, row in candidates.iterrows():
        player_rows = by_player[row['id']]
        history = player_rows[player_rows['Game_Week'] < row['Game_Week']]
        history = pd.concat([history, row.to_frame().T], ignore_index=True)
        if sequence_length is not None:
            history = history.tail(sequence_length)
        raw_values = _numeric_features(history, features, fill_missing=False)
        values = scaler.transform(raw_values).astype(np.float32)
        # Missing Transfermarkt values remain explicitly unavailable via
        # tm_value_available=0; zero here is only the neutral standardized
        # placeholder required by dense PyTorch tensors.
        values = np.nan_to_num(values, nan=0.0)
        histories.append(values)
        lengths.append(len(values))
        targets.append(float(row.get('target', 0.0)))
        indexes.append(index)

    padded_length = max(lengths)
    sequences = np.zeros(
        (len(histories), padded_length, len(features)), dtype=np.float32
    )
    for sample_index, values in enumerate(histories):
        sequences[sample_index, :len(values)] = values

    return (
        sequences,
        np.asarray(lengths, dtype=np.int64),
        np.asarray(targets, dtype=np.float32),
        pd.Index(indexes),
        scaler,
    )


def _state_to_numpy(model: nn.Module) -> dict[str, np.ndarray]:
    return {name: tensor.detach().cpu().numpy() for name, tensor in model.state_dict().items()}


def _load_model(spec: dict[str, Any]) -> PositionGRU:
    model = PositionGRU(spec['input_size'], spec['hidden_size'])
    model.load_state_dict({
        name: torch.as_tensor(value) for name, value in spec['state_dict'].items()
    })
    model.eval()
    return model


def train_rnn_models(
    df: pd.DataFrame,
    completed_internal_index: int,
    sequence_length: int | None = None,
    epochs: int = DEFAULT_EPOCHS,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    torch.manual_seed(42)
    np.random.seed(42)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    model_specs: dict[str, dict[str, Any]] = {}
    metrics: dict[str, dict[str, Any]] = {}

    for position in ('GK', 'DEF', 'MID', 'FWD'):
        features = list(POS_FEATURES[position])
        sequences, lengths, targets, _, scaler = build_position_sequences(
            df, position, completed_internal_index, sequence_length
        )
        sequence_tensor = torch.from_numpy(sequences)
        length_tensor = torch.from_numpy(lengths)
        target_tensor = torch.from_numpy(targets)
        hidden_size = 32
        model = PositionGRU(len(features), hidden_size)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
        loss_function = nn.SmoothL1Loss()

        model.train()
        for _ in range(epochs):
            optimizer.zero_grad()
            loss = loss_function(model(sequence_tensor, length_tensor), target_tensor)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            predictions = model(sequence_tensor, length_tensor).numpy()
        model_specs[position] = {
            'architecture': 'gru',
            'input_size': len(features),
            'hidden_size': hidden_size,
            'state_dict': _state_to_numpy(model),
            'scaler_mean': scaler.mean_.astype(float).tolist(),
            'scaler_scale': scaler.scale_.astype(float).tolist(),
            'features': features,
            'sequence_length': sequence_length,
            'sequence_policy': (
                'all_available' if sequence_length is None else 'fixed_window'
            ),
            'max_training_sequence_length': int(lengths.max()),
        }
        metrics[position] = {
            'r2': round(float(r2_score(targets, predictions)), 4),
            'rmse': round(float(root_mean_squared_error(targets, predictions)), 4),
            'n': len(targets),
            'top_feature': 'ordered player history',
            'model_name': f'{MODEL_NAMES[position]} (GRU)',
            'n_features': len(features),
        }
        metric = metrics[position]
        print(
            f"  [{metric['model_name']}] n={metric['n']:,} "
            f"features={metric['n_features']} RMSE={metric['rmse']:.4f}"
        )

    return model_specs, metrics


def score_rnn_snapshot(
    df: pd.DataFrame,
    checkpoint: dict[str, Any],
    snapshot_game_week: int,
) -> pd.DataFrame:
    manifest = checkpoint.get('training_manifest') or {}
    if manifest.get('sequence_policy') != 'all_available':
        raise ValueError('RNN checkpoint does not declare all-available history.')
    parts = []
    for position, spec in checkpoint['models'].items():
        features = spec.get('features') or []
        if spec.get('sequence_policy') != 'all_available' or spec.get('sequence_length') is not None:
            raise ValueError(
                f'RNN checkpoint has an inconsistent sequence policy for {position}.'
            )
        if int(spec.get('input_size', -1)) != len(features):
            raise ValueError(f'RNN checkpoint feature count is invalid for {position}.')
        if not (
            len(spec.get('scaler_mean', [])) == len(features)
            and len(spec.get('scaler_scale', [])) == len(features)
        ):
            raise ValueError(f'RNN checkpoint scaler state is invalid for {position}.')
        scaler = StandardScaler()
        scaler.mean_ = np.asarray(spec['scaler_mean'], dtype=float)
        scaler.scale_ = np.asarray(spec['scaler_scale'], dtype=float)
        scaler.var_ = scaler.scale_ ** 2
        scaler.n_features_in_ = len(features)
        sequences, lengths, _, indexes, _ = build_position_sequences(
            df,
            position,
            snapshot_game_week,
            spec.get('sequence_length'),
            scaler=scaler,
            target_game_week=snapshot_game_week,
            features=features,
        )
        if sequences.shape[2] != len(features):
            raise ValueError(f'RNN inference feature width is invalid for {position}.')
        model = _load_model(spec)
        with torch.no_grad():
            predictions = model(
                torch.from_numpy(sequences), torch.from_numpy(lengths)
            ).numpy()
        position_rows = df.loc[indexes].copy()
        position_rows['predicted_points'] = predictions
        parts.append(position_rows)
    if not parts:
        raise ValueError(f'No RNN prediction rows exist for snapshot {snapshot_game_week}.')
    return pd.concat(parts, ignore_index=True)
