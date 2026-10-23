"""One-pass opposition adjustment of frozen, fixture-level player forecasts."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any

import numpy as np
import pandas as pd

OPPOSITION_VERSION = 1
MIN_EVALUATION_GAMEWEEKS = 5
SCORE_AUDIT_FIELDS = (
    'opposition_base_points', 'opposition_factor', 'opposition_fixtures',
    'opposition_version',
)


@dataclass(frozen=True)
class OppositionAdjustment:
    strength: float = 0.5
    max_reduction: float = 0.25

    def __post_init__(self) -> None:
        if not np.isfinite(self.strength) or not 0 < self.strength <= 1:
            raise ValueError('Opposition strength must be finite and in (0, 1].')
        if not np.isfinite(self.max_reduction) or not 0 < self.max_reduction < 1:
            raise ValueError('Maximum opposition reduction must be in (0, 1).')


def projected_club_xi(players: pd.DataFrame) -> pd.DataFrame:
    """Choose a legal club XI by baseline points, without using actual minutes."""
    ordered = players.sort_values(
        ['predicted_points', 'id'], ascending=[False, True]
    )
    positions = {
        position: ordered[ordered['element_type'] == position]
        for position in ('GK', 'DEF', 'MID', 'FWD')
    }
    best: pd.DataFrame | None = None
    best_score = -np.inf
    for defenders, midfielders, forwards in product(range(3, 6), range(2, 6), range(1, 4)):
        if defenders + midfielders + forwards != 10:
            continue
        counts = {'GK': 1, 'DEF': defenders, 'MID': midfielders, 'FWD': forwards}
        if any(len(positions[position]) < count for position, count in counts.items()):
            continue
        candidate = pd.concat([
            positions[position].head(count) for position, count in counts.items()
        ])
        score = float(candidate['predicted_points'].sum())
        if score > best_score:
            best, best_score = candidate, score
    if best is None:
        raise ValueError('Opposition forecasts do not cover a legal projected club XI.')
    return best


def apply_opposition_adjustment(
    fixture_predictions: pd.DataFrame,
    adjustment: OppositionAdjustment,
) -> pd.DataFrame:
    """Dampen positive scores once using the opposing projected XI's forecasts."""
    required = {'id', 'team', 'opponent_team', 'element_type', 'predicted_points'}
    missing = required - set(fixture_predictions.columns)
    if missing:
        raise ValueError(f'Opposition adjustment is missing columns: {sorted(missing)}')
    if 'opposition_base_points' in fixture_predictions:
        raise ValueError('Opposition adjustment must not be applied twice.')
    adjusted = fixture_predictions.copy()
    scores = pd.to_numeric(adjusted['predicted_points'], errors='raise').astype(float)
    if not np.isfinite(scores).all():
        raise ValueError('Opposition forecasts must contain finite point predictions.')
    adjusted['predicted_points'] = scores
    adjusted['opposition_base_points'] = scores
    adjusted['opposition_version'] = OPPOSITION_VERSION
    adjusted['opposition_pressure'] = 1.0
    adjusted['opposition_factor'] = 1.0
    active = pd.to_numeric(adjusted['opponent_team'], errors='raise').gt(0)
    rows = adjusted.loc[active].copy()
    if rows.empty:
        return adjusted

    scheduled = pd.to_numeric(
        rows.get('scheduled_fixture', pd.Series(0, index=rows.index)), errors='raise'
    ).fillna(0)
    played = pd.to_numeric(
        rows.get('fixture', pd.Series(0, index=rows.index)), errors='raise'
    ).fillna(0)
    rows['_opposition_fixture'] = scheduled.where(scheduled.gt(0), played).astype(int)
    if rows['_opposition_fixture'].le(0).any():
        raise ValueError('Opposition adjustment requires a real ID for every fixture.')
    if rows.duplicated(['_opposition_fixture', 'id']).any():
        raise ValueError('Duplicate player forecasts exist within an opposition fixture.')
    if not rows['element_type'].isin(['GK', 'DEF', 'MID', 'FWD']).all():
        raise ValueError('Opposition forecasts contain an unsupported player position.')

    units: dict[tuple[int, int], dict[str, float]] = {}
    selected = []
    for (fixture, team), group in rows.groupby(['_opposition_fixture', 'team'], sort=True):
        xi = projected_club_xi(group)
        selected.append(xi)
        units[(int(fixture), int(team))] = {}
    projected = pd.concat(selected)
    position_means = projected.groupby('element_type')['predicted_points'].agg(
        lambda values: values.clip(lower=0).mean()
    )
    for (fixture, team), group in projected.groupby(['_opposition_fixture', 'team']):
        relative = group['predicted_points'].clip(lower=0).div(
            group['element_type'].map(position_means).replace(0, np.nan)
        ).fillna(1.0)
        attack = group['element_type'].isin(['MID', 'FWD'])
        units[(int(fixture), int(team))] = {
            'attack': float(relative.loc[attack].mean()),
            'defence': float(relative.loc[~attack].mean()),
        }
    reference = {
        role: float(np.mean([unit[role] for unit in units.values()]))
        for role in ('attack', 'defence')
    }
    for fixture, group in rows.groupby('_opposition_fixture'):
        teams = set(group['team'].astype(int))
        if len(teams) != 2:
            raise ValueError(f'Fixture {fixture} does not cover both opposition teams.')
        for index, row in group.iterrows():
            team, opponent = int(row['team']), int(row['opponent_team'])
            if opponent == team or opponent not in teams:
                raise ValueError(f'Fixture {fixture} contains inconsistent opponent IDs.')
            role = 'attack' if row['element_type'] in ('GK', 'DEF') else 'defence'
            pressure = units[(int(fixture), opponent)][role] / reference[role]
            factor = max(
                1.0 - adjustment.max_reduction,
                1.0 / (1.0 + adjustment.strength * max(pressure - 1.0, 0.0)),
            )
            adjusted.at[index, 'opposition_pressure'] = pressure
            adjusted.at[index, 'opposition_factor'] = factor
    adjusted['predicted_points'] = scores.where(
        scores.le(0), scores * adjusted['opposition_factor']
    )
    return adjusted


def evaluation_gate(gameweeks: list[dict[str, Any]]) -> dict[str, Any]:
    """Predeclared gate; a single good Gameweek is not rollout evidence."""
    if len({(row['season'], row['game_week']) for row in gameweeks}) != len(gameweeks):
        raise ValueError('Opposition evaluation contains duplicate season/Gameweeks.')
    prospective = []
    for row in gameweeks:
        values = [
            row[key] for key in (
                'baseline_points', 'adjusted_points',
                'baseline_captain_points', 'adjusted_captain_points',
                'baseline_squared_error', 'adjusted_squared_error', 'player_count',
            )
        ]
        if not np.isfinite(np.asarray(values, dtype=float)).all():
            raise ValueError('Opposition evaluation metrics must be finite.')
        if row['player_count'] <= 0:
            raise ValueError('Opposition evaluation must include scored players.')
        if row['baseline_squared_error'] < 0 or row['adjusted_squared_error'] < 0:
            raise ValueError('Squared prediction errors cannot be negative.')
        deadline = pd.Timestamp(row['deadline'])
        if pd.isna(deadline) or deadline.tzinfo is None:
            raise ValueError('Evaluation requires a timezone-aware FPL deadline.')
        for key in ('trained_at', 'features_created_at'):
            observed = pd.Timestamp(row[key])
            if pd.isna(observed) or observed.tzinfo is None or observed >= deadline:
                raise ValueError(f'{key} must precede the evaluated Gameweek deadline.')
        if not row['finished'] or row['snapshot_index'] <= row['training_cutoff']:
            raise ValueError('Evaluation must use completed, post-training Gameweeks.')
        registered = pd.Timestamp(row['candidate_registered_at'])
        frozen = pd.Timestamp(row['forecasts_frozen_at'])
        if (
            pd.isna(registered) or registered.tzinfo is None
            or pd.isna(frozen) or frozen.tzinfo is None
        ):
            raise ValueError('Evaluation requires a timezone-aware candidate registration.')
        if registered < deadline and frozen < deadline:
            prospective.append(row)
    count = sum(int(row['player_count']) for row in prospective)
    baseline_error = sum(row['baseline_squared_error'] for row in prospective)
    adjusted_error = sum(row['adjusted_squared_error'] for row in prospective)
    points_gain = sum(row['adjusted_points'] - row['baseline_points'] for row in prospective)
    captain_gain = sum(
        row['adjusted_captain_points'] - row['baseline_captain_points']
        for row in prospective
    )
    checks = {
        'enough_gameweeks': len(prospective) >= MIN_EVALUATION_GAMEWEEKS,
        'higher_actual_team_points': points_gain > 0,
        'captaincy_not_worse': captain_gain >= 0,
        'player_rmse_not_worse': adjusted_error <= baseline_error,
    }
    return {
        'passed': all(checks.values()),
        'checks': checks,
        'gameweeks': len(prospective),
        'diagnostic_gameweeks': len(gameweeks) - len(prospective),
        'minimum_gameweeks': MIN_EVALUATION_GAMEWEEKS,
        'points_gain': points_gain,
        'captain_points_gain': captain_gain,
        'baseline_rmse': float(np.sqrt(baseline_error / count)) if count else None,
        'adjusted_rmse': float(np.sqrt(adjusted_error / count)) if count else None,
    }
