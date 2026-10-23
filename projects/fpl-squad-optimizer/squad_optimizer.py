"""Shared full-advisor squad/XI optimization for serving and paired evaluation."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp


def select_best_advisor_side(
    pool: pd.DataFrame,
    max_per_team: int,
    max_spend: int,
    time_limit: int = 20,
) -> tuple[pd.DataFrame, dict]:
    candidates = pool.drop_duplicates(subset='id').reset_index(drop=True)
    player_count = len(candidates)
    constraint_rows = []
    lower_bounds = []
    upper_bounds = []

    def add_constraint(squad_values, starter_values, lower, upper):
        constraint_rows.append(np.concatenate([squad_values, starter_values]))
        lower_bounds.append(lower)
        upper_bounds.append(upper)

    zeroes = np.zeros(player_count)
    for position, count in {'GK': 2, 'DEF': 5, 'MID': 5, 'FWD': 3}.items():
        mask = (candidates['element_type'] == position).astype(float).to_numpy()
        add_constraint(mask, zeroes, count, count)

    add_constraint(candidates['value'].astype(float).to_numpy(), zeroes, 0, max_spend)
    for team in candidates['team'].dropna().unique():
        mask = (candidates['team'] == team).astype(float).to_numpy()
        add_constraint(mask, zeroes, 0, max_per_team)

    add_constraint(zeroes, np.ones(player_count), 11, 11)
    for position, (minimum, maximum) in {
        'GK': (1, 1), 'DEF': (3, 5), 'MID': (2, 5), 'FWD': (1, 3),
    }.items():
        mask = (candidates['element_type'] == position).astype(float).to_numpy()
        add_constraint(zeroes, mask, minimum, maximum)

    for index in range(player_count):
        squad_values, starter_values = np.zeros(player_count), np.zeros(player_count)
        squad_values[index], starter_values[index] = -1, 1
        add_constraint(squad_values, starter_values, -np.inf, 0)

    predictions = candidates['predicted_points'].astype(float).to_numpy()
    result = milp(
        c=-np.concatenate([predictions * 0.001, predictions]),
        integrality=np.ones(player_count * 2),
        bounds=Bounds(0, 1),
        constraints=LinearConstraint(np.vstack(constraint_rows), lower_bounds, upper_bounds),
        options={'time_limit': time_limit},
    )
    if not result.success:
        raise RuntimeError('No legal all-advisor squad and XI could be generated.')
    squad = candidates.loc[result.x[:player_count] > 0.5].copy().reset_index(drop=True)
    starter_ids = candidates.loc[result.x[player_count:] > 0.5, 'id'].astype(int).tolist()
    bench = squad[~squad['id'].astype(int).isin(starter_ids)].copy()
    bench['bench_rank'] = bench['element_type'].ne('GK').astype(int)
    bench = bench.sort_values(['bench_rank', 'predicted_points'], ascending=[True, False])
    ranked = squad[squad['id'].astype(int).isin(starter_ids)].sort_values(
        'predicted_points', ascending=False
    )
    return squad, {
        'starters': starter_ids,
        'bench': bench['id'].astype(int).tolist(),
        'captain': int(ranked.iloc[0]['id']),
        'vice_captain': int(ranked.iloc[1]['id']),
    }
