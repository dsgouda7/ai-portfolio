import joblib
import os
from collections import Counter

import numpy as np
import pandas as pd
from tabulate import tabulate

from utils import DB_FILE, MODELS_FILE, GAME_WEEK, FEATURES, ROLL_COLS, build_features

MAX_PLAYERS_PER_TEAM = 4
MAX_SPEND        = 1000


def build_pool(df, models, game_week):
    """Score every player using their position's model. Returns pool with 'predicted_points'."""
    snapshot = df[df['Game_Week'] == game_week - 1].copy()
    parts = []

    for pos, model in models.items():
        pos_players = snapshot[snapshot['element_type'] == pos].copy()
        if pos_players.empty:
            continue
        X = pos_players[FEATURES].fillna(0)
        pos_players['predicted_points'] = model.predict(X)
        parts.append(pos_players)

    return pd.concat(parts, ignore_index=True)


def select_team(pool, structure, max_per_team, max_spend):
    """
    Greedy selection: fill each position by predicted points, respecting budget
    and max_per_team constraints.

    Also computes selection_margin per player: predicted pts above the next
    available alternative at that position. Margin < 0.5 = coin-flip pick.
    """
    squad = []
    spend = 0
    team_counts = Counter()

    for pos, count in structure.items():
        eligible = pool[pool['element_type'] == pos].sort_values('predicted_points', ascending=False)
        selected = 0
        current_ids = {p['id'] for p in squad}

        for _, player in eligible.iterrows():
            if selected == count:
                break
            if (
                player['id'] not in current_ids
                and team_counts.get(player['team'], 0) < max_per_team
                and spend + player['value'] <= max_spend
            ):
                squad.append(player)
                spend += player['value']
                team_counts[player['team']] += 1
                current_ids.add(player['id'])
                selected += 1

    team = pd.DataFrame(squad)

    # selection margin: predicted_points - best available alternative not picked
    selected_ids = set(team['id'])
    margins = []
    for _, player in team.iterrows():
        pos = player['element_type']
        alternatives = pool[
            (pool['element_type'] == pos) & (~pool['id'].isin(selected_ids))
        ]
        next_best = alternatives['predicted_points'].max() if not alternatives.empty else player['predicted_points']
        margins.append(round(player['predicted_points'] - next_best, 4))
    team['selection_margin'] = margins

    return team


def print_player_profiles(team, pool):
    """
    Print rolling-form stats and within-position percentile ranks for each
    selected player. Percentiles are position-scoped to avoid meaningless
    cross-position comparisons (GKs and FWDs have structurally different stat
    distributions).
    """
    # which roll5 features are most informative per position
    pos_features = {
        'GK':  ['roll5_minutes', 'roll5_saves', 'roll5_clean_sheets', 'roll5_bonus', 'roll5_total_points'],
        'DEF': ['roll5_minutes', 'roll5_clean_sheets', 'roll5_goals_scored', 'roll5_assists', 'roll5_bonus', 'roll5_total_points'],
        'MID': ['roll5_minutes', 'roll5_ict_index', 'roll5_creativity', 'roll5_goals_scored', 'roll5_assists', 'roll5_bonus', 'roll5_total_points'],
        'FWD': ['roll5_minutes', 'roll5_goals_scored', 'roll5_assists', 'roll5_threat', 'roll5_bonus', 'roll5_total_points'],
    }

    print("\n" + "=" * 70)
    print("Player Profiles  (stats = rolling-5GW avg | pct = within-position %ile)")
    print("=" * 70)

    for pos in ['GK', 'DEF', 'MID', 'FWD']:
        pos_players = team[team['element_type'] == pos]
        if pos_players.empty:
            continue

        features = pos_features[pos]
        pos_pool  = pool[pool['element_type'] == pos]

        for _, player in pos_players.iterrows():
            name = f"{player['first_name']} {player['second_name']}"
            print(f"\n  {name}  |  {pos}  |  £{player['value']/10:.1f}M  "
                  f"|  pred={player['predicted_points']:.2f}  margin={player['selection_margin']:.2f}")

            rows = []
            for feat in features:
                if feat not in pool.columns:
                    continue
                val = player[feat] if feat in player.index else np.nan
                # percentile: fraction of position peers with a lower value
                # fillna(0) so players with no data don't inflate the pool
                col_vals = pos_pool[feat].fillna(0)
                pct = int(np.round((col_vals < val).mean() * 100)) if not np.isnan(val) else 0
                label = feat.replace('roll5_', '')
                bar = '#' * (pct // 10) + '.' * (10 - pct // 10)   # 10-char ASCII bar
                rows.append([label, f"{val:.2f}" if not np.isnan(val) else 'n/a', f"{pct:3d}th", bar])

            print(tabulate(rows, headers=['stat', 'avg(5GW)', 'pct', 'distribution'], tablefmt='simple'))


if not os.path.exists(MODELS_FILE):
    raise FileNotFoundError(f"{MODELS_FILE} not found. Run train.py first.")

print(f"Loading models from {MODELS_FILE}...")
checkpoint = joblib.load(MODELS_FILE)
models  = checkpoint['models']
metrics = checkpoint['metrics']

# model quality header -- R² here is in-sample so it's optimistic, but
# anything close to 0 signals the model found no predictive structure for
# that position and the team selections from it should be treated with scepticism
print("\nModel quality (in-sample, optimistic):")
quality_rows = [
    [pos, m['n'], f"{m['r2']:.4f}", f"{m['rmse']:.4f}", m['top_feature']]
    for pos, m in metrics.items()
]
print(tabulate(quality_rows, headers=['pos', 'train_rows', 'R2', 'RMSE', 'top_feature'], tablefmt='simple'))

print("\nBuilding features from DB...")
all_data = build_features(DB_FILE)

print(f"Scoring players for GW {GAME_WEEK}...")
pool = build_pool(all_data, models, GAME_WEEK)

structure = {'GK': 2, 'DEF': 5, 'MID': 5, 'FWD': 3}
best_team = select_team(pool, structure, MAX_PLAYERS_PER_TEAM, MAX_SPEND)

display_cols = ['first_name', 'second_name', 'element_type', 'team', 'value', 'predicted_points', 'selection_margin']
print("\nSelected Team:")
print(tabulate(best_team[display_cols], headers='keys', tablefmt='fancy_grid', floatfmt='.2f'))

# team-level confidence: mean margin across all 15 players.
# above ~1.0 pt average margin = model was decisive across the squad.
# below ~0.3 pt = most picks were coin-flips; consider more training data.
avg_margin = best_team['selection_margin'].mean()
min_margin_row = best_team.loc[best_team['selection_margin'].idxmin()]
print(f"\nTotal Predicted Points : {best_team['predicted_points'].sum():.2f}")
print(f"Total Spend            : {best_team['value'].sum():.0f}  (budget: {MAX_SPEND})")
print(f"Avg Selection Margin   : {avg_margin:.2f} pts")
print(f"Weakest Pick           : {min_margin_row['first_name']} {min_margin_row['second_name']} "
      f"({min_margin_row['element_type']}, margin={min_margin_row['selection_margin']:.2f} pts)")

print_player_profiles(best_team, pool)
