import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import argparse
import joblib
import os
import sqlite3
from collections import Counter

import numpy as np
import pandas as pd
from tabulate import tabulate

from utils import (
    DB_FILE, MODELS_FILE, GAME_WEEK, SEASON, POS_FEATURES,
    build_features, normalize_pool_scores, apply_market_value_weighting,
    save_squad, load_squad, score_squad_from_pool, pick_starting_xi,
    suggest_transfer, find_ineligible_replacements,
)
from eligibility import get_eligibility, _DEFAULT as ELIG_DEFAULT, _ABSENT as ELIG_ABSENT, player_name_key

MAX_PLAYERS_PER_TEAM = 4
MAX_SPEND            = 1000

_parser = argparse.ArgumentParser(description='FPL weekly team picker')
_parser.add_argument(
    '--new-team', action='store_true',
    help='Ignore saved squad and generate a brand-new 15-player team from scratch',
)
_args = _parser.parse_args()


def build_pool(df, models, game_week):
    """
    Score every player using their position-specific model and feature set.
    Adds ``predicted_points_norm``: z-score within each position group,
    enabling cross-position (uber-model) comparison for flex formation slots.
    """
    snapshot = df[df['Game_Week'] == game_week - 1].copy()
    parts = []

    for pos, model in models.items():
        pos_players = snapshot[snapshot['element_type'] == pos].copy()
        if pos_players.empty:
            continue
        X = pos_players[POS_FEATURES[pos]].fillna(0)
        pos_players['predicted_points'] = model.predict(X)
        parts.append(pos_players)

    pool = apply_market_value_weighting(pd.concat(parts, ignore_index=True))
    return normalize_pool_scores(pool)


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
    # curated display stats per position (mirrors POS_STATS in web.py)
    pos_features = {
        'GK':  ['roll5_minutes', 'roll5_starts', 'roll5_saves', 'roll5_clean_sheets',
                'roll5_expected_goals_conceded', 'roll5_bonus', 'roll5_total_points'],
        'DEF': ['roll5_minutes', 'roll5_starts', 'roll5_clean_sheets',
                'roll5_goals_scored', 'roll5_assists',
                'roll5_expected_goals', 'roll5_expected_assists',
                'roll5_bonus', 'roll5_total_points'],
        'MID': ['roll5_minutes', 'roll5_starts', 'roll5_goals_scored', 'roll5_assists',
                'roll5_expected_goals', 'roll5_expected_assists',
                'roll5_creativity', 'roll5_bonus', 'roll5_total_points'],
        'FWD': ['roll5_minutes', 'roll5_starts', 'roll5_goals_scored', 'roll5_assists',
                'roll5_expected_goals', 'roll5_expected_assists',
                'roll5_threat', 'roll5_bonus', 'roll5_total_points'],
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


# ── Main execution ──────────────────────────────────────────────────────────

if not os.path.exists(MODELS_FILE):
    raise FileNotFoundError(f"{MODELS_FILE} not found. Run train/train.py first.")

print(f"Loading models from {MODELS_FILE}...")
checkpoint = joblib.load(MODELS_FILE)
models  = checkpoint['models']
metrics = checkpoint['metrics']

print("\nModel quality (in-sample, optimistic):")
quality_rows = [
    [m.get('model_name', pos), m['n'], m.get('n_features', '?'), f"{m['r2']:.4f}", f"{m['rmse']:.4f}", m['top_feature']]
    for pos, m in metrics.items()
]
print(tabulate(quality_rows, headers=['model', 'train_rows', 'features', 'R2', 'RMSE', 'top_feature'], tablefmt='simple'))

print("\nBuilding features from DB...")
all_data = build_features(DB_FILE)

print(f"Scoring players for GW {GAME_WEEK}...")
full_pool = build_pool(all_data, models, GAME_WEEK)   # ALL players, pre-filter

# --- tier 1: EPL membership filter ---
epl_members = checkpoint.get('epl_members')
pool = full_pool.copy()
if epl_members is not None:
    before = len(pool)
    pool = pool[
        pool.apply(lambda r: player_name_key(r.get('first_name', ''), r.get('second_name', '')) in epl_members, axis=1)
    ].copy()
    print(f"  EPL filter: {before - len(pool)} non-EPL players removed, {len(pool)} remain")

# --- tier 2: live FPL eligibility ---
print("Fetching current eligibility from FPL API...")
try:
    eligibility = get_eligibility()
    n_before = len(pool)
    def _elig(row):
        key = (str(row.get('first_name', '') or '').lower(),
               str(row.get('second_name', '') or '').lower())
        return eligibility.get(key, ELIG_ABSENT).eligible
    eligible_pool = pool[pool.apply(_elig, axis=1)].copy()
    n_llm = sum(1 for v in eligibility.values() if getattr(v, 'method', '') == 'llm')
    print(f"  Eligibility: {n_before} players checked, {n_before - len(eligible_pool)} filtered out, "
          f"{n_llm} resolved by LLM")
except Exception as exc:
    print(f'  Warning: eligibility check failed ({exc}), proceeding without filter')
    eligibility = {}
    eligible_pool = pool.copy()


# ── Check for saved squad ────────────────────────────────────────────────────

if _args.new_team:
    print("\n[--new-team] Clearing saved squad and generating a fresh selection.")
    conn = sqlite3.connect(DB_FILE)
    conn.execute("DROP TABLE IF EXISTS saved_squad")
    conn.commit()
    conn.close()
    saved_squad = None
else:
    saved_squad = load_squad(DB_FILE)


if saved_squad is not None:
    # ── SQUAD ITERATION MODE ────────────────────────────────────────────────
    gw_saved = int(saved_squad['gw_saved'].iloc[0])
    print(f"\n{'='*70}")
    print(f"SQUAD MODE — GW {GAME_WEEK}  (squad saved from GW {gw_saved})")
    print(f"{'='*70}")

    # Score all 15 saved squad members from the full (pre-eligibility) pool
    scored_squad = score_squad_from_pool(saved_squad, full_pool)

    # Detect ineligible squad members (injured/suspended or left EPL)
    eligible_ids  = set(eligible_pool['id'].astype(int))
    ineligible_df = scored_squad[~scored_squad['id'].astype(int).isin(eligible_ids)]
    eligible_df   = scored_squad[scored_squad['id'].astype(int).isin(eligible_ids)]

    # --- Starting XI from eligible squad members ---
    if eligible_df.empty or eligible_df[eligible_df['element_type'] == 'GK'].empty:
        print("\n  ⚠  Not enough eligible players to pick a starting XI.")
        print("     Run with --new-team to regenerate from scratch.")
        starters, bench, formation = pd.DataFrame(), pd.DataFrame(), 'N/A'
    else:
        starters, bench, formation = pick_starting_xi(eligible_df)

    if not starters.empty:
        xi_cols = ['first_name', 'second_name', 'element_type', 'value', 'predicted_points']
        print(f"\nStarting XI  (formation: {formation})")
        print(tabulate(starters[xi_cols], headers='keys', tablefmt='simple', floatfmt='.2f'))
        print(f"\nBench:")
        print(tabulate(bench[xi_cols], headers='keys', tablefmt='simple', floatfmt='.2f'))
        print(f"\nTotal Predicted (XI)   : {starters['predicted_points'].sum():.2f}")
        print(f"Squad Spend            : £{scored_squad['value'].sum()/10:.1f}M / £{MAX_SPEND/10:.0f}M")

    # --- Suggested transfer ---
    print(f"\n{'━'*70}")
    print("SUGGESTED TRANSFER  (1 free transfer — no point deduction)")
    print(f"{'━'*70}")
    transfer = suggest_transfer(scored_squad, eligible_pool, MAX_PLAYERS_PER_TEAM, MAX_SPEND)
    if transfer:
        out_p    = transfer['out']
        in_p     = transfer['in_']
        out_name = f"{out_p['first_name']} {out_p['second_name']}"
        in_name  = f"{in_p['first_name']} {in_p['second_name']}"
        gain_str = f"+{transfer['gain']:.2f}" if transfer['gain'] >= 0 else f"{transfer['gain']:.2f}"
        print(f"  OUT: {out_name:<30} {out_p['element_type']}  "
              f"£{float(out_p['value'])/10:.1f}M  pred={float(out_p['predicted_points']):.2f} pts")
        print(f"   IN: {in_name:<30} {in_p['element_type']}  "
              f"£{float(in_p['value'])/10:.1f}M  pred={float(in_p['predicted_points']):.2f} pts")
        print(f"  Gain: {gain_str} pts  |  New squad value: "
              f"£{transfer['new_spend']/10:.1f}M / £{MAX_SPEND/10:.0f}M")

        # Apply transfer → save updated squad
        new_squad = scored_squad.copy()
        out_idx   = new_squad[new_squad['id'].astype(int) == int(out_p['id'])].index
        new_squad = new_squad.drop(index=out_idx)
        in_row    = eligible_pool[eligible_pool['id'].astype(int) == int(in_p['id'])].iloc[[0]]
        new_squad = pd.concat([new_squad, in_row], ignore_index=True)
    else:
        print("  No beneficial transfer found — squad is already optimal.")
        new_squad = scored_squad

    # --- Ineligible squad members report ---
    if not ineligible_df.empty:
        print(f"\n{'━'*70}")
        print("SQUAD HEALTH — Ineligible players & suggested replacements")
        print(f"{'━'*70}")
        replacements = find_ineligible_replacements(
            ineligible_df, eligible_pool, scored_squad, MAX_PLAYERS_PER_TEAM, MAX_SPEND
        )
        for item in replacements:
            out_p    = item['out']
            out_name = f"{out_p['first_name']} {out_p['second_name']}"
            elig_info = eligibility.get(
                (str(out_p.get('first_name', '') or '').lower(),
                 str(out_p.get('second_name', '') or '').lower()),
                ELIG_ABSENT,
            )
            reason = getattr(elig_info, 'news', 'Not in eligible pool (left EPL?)')
            print(f"\n  ✗ {out_name}  ({out_p['element_type']})  £{float(out_p.get('value', 0))/10:.1f}M")
            if reason:
                print(f"    {reason}")
            if item['replacement']:
                r      = item['replacement']
                r_name = f"{r['first_name']} {r['second_name']}"
                gain   = f"+{item['gain']:.2f}" if item['gain'] >= 0 else f"{item['gain']:.2f}"
                print(f"    → Best replacement: {r_name}  {r['element_type']}  "
                      f"£{float(r['value'])/10:.1f}M  pred={float(r['predicted_points']):.2f} pts  "
                      f"({gain} pts)")
            else:
                print("    → No valid replacement found within budget")
    else:
        print("\n  ✓ All squad members are eligible this week.")

    # Save updated squad
    save_squad(DB_FILE, new_squad, GAME_WEEK, SEASON)
    print(f"\n  Squad saved for GW {GAME_WEEK}.")

    if not starters.empty:
        print_player_profiles(starters, eligible_pool)

else:
    # ── FRESH TEAM MODE ─────────────────────────────────────────────────────
    structure = {'GK': 2, 'DEF': 5, 'MID': 5, 'FWD': 3}
    best_team = select_team(eligible_pool, structure, MAX_PLAYERS_PER_TEAM, MAX_SPEND)

    display_cols = ['first_name', 'second_name', 'element_type', 'team', 'value',
                    'predicted_points', 'predicted_points_norm', 'selection_margin']
    print("\nSelected Team:")
    print(tabulate(best_team[display_cols], headers='keys', tablefmt='fancy_grid', floatfmt='.2f'))

    avg_margin     = best_team['selection_margin'].mean()
    min_margin_row = best_team.loc[best_team['selection_margin'].idxmin()]
    print(f"\nTotal Predicted Points : {best_team['predicted_points'].sum():.2f}")
    print(f"Total Spend            : {best_team['value'].sum():.0f}  (budget: {MAX_SPEND})")
    print(f"Avg Selection Margin   : {avg_margin:.2f} pts")
    print(f"Weakest Pick           : {min_margin_row['first_name']} {min_margin_row['second_name']} "
          f"({min_margin_row['element_type']}, margin={min_margin_row['selection_margin']:.2f} pts)")

    save_squad(DB_FILE, best_team, GAME_WEEK, SEASON)
    print(f"\n  Squad saved to DB (GW {GAME_WEEK}).")
    print(f"  Next run will iterate from this squad. Use --new-team to regenerate.")

    print_player_profiles(best_team, eligible_pool)


print(f"Loading models from {MODELS_FILE}...")
checkpoint = joblib.load(MODELS_FILE)
models  = checkpoint['models']
metrics = checkpoint['metrics']

# model quality header -- R² here is in-sample so it's optimistic, but
# anything close to 0 signals the model found no predictive structure for
# that position and the team selections from it should be treated with scepticism
print("\nModel quality (in-sample, optimistic):")
quality_rows = [
    [m.get('model_name', pos), m['n'], m.get('n_features', '?'), f"{m['r2']:.4f}", f"{m['rmse']:.4f}", m['top_feature']]
    for pos, m in metrics.items()
]
print(tabulate(quality_rows, headers=['model', 'train_rows', 'features', 'R2', 'RMSE', 'top_feature'], tablefmt='simple'))

print("\nBuilding features from DB...")
all_data = build_features(DB_FILE)

print(f"Scoring players for GW {GAME_WEEK}...")
pool = build_pool(all_data, models, GAME_WEEK)

# --- tier 1: training-time EPL membership (removes players who left the league;
#             injured/suspended players are kept here and handled by tier 2) ---
epl_members = checkpoint.get('epl_members')
if epl_members is not None:
    before = len(pool)
    pool = pool[
        pool.apply(lambda r: player_name_key(r.get('first_name', ''), r.get('second_name', '')) in epl_members, axis=1)
    ].copy()
    print(f"  EPL filter: {before - len(pool)} non-EPL players removed, {len(pool)} remain")

# --- tier 2: live FPL API refresh (injury/suspension status) ---
print("Fetching current eligibility from FPL API...")
try:
    eligibility = get_eligibility()
    def _elig(row):
        key = (str(row.get('first_name', '') or '').lower(),
               str(row.get('second_name', '') or '').lower())
        return eligibility.get(key, ELIG_ABSENT).eligible
    pool = pool[pool.apply(_elig, axis=1)].copy()
except Exception as exc:
    print(f'  Warning: eligibility check failed ({exc}), proceeding without filter')

structure = {'GK': 2, 'DEF': 5, 'MID': 5, 'FWD': 3}
best_team = select_team(pool, structure, MAX_PLAYERS_PER_TEAM, MAX_SPEND)

display_cols = ['first_name', 'second_name', 'element_type', 'team', 'value',
                'predicted_points', 'predicted_points_norm', 'selection_margin']
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
