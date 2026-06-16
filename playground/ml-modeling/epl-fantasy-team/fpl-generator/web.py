"""
web.py -- Flask app: render the FPL team selection on an interactive pitch.

Run: python fpl-generator/web.py  (requires models.joblib -- run train/train.py first)
"""
import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import json
import math
import os
from collections import Counter

import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template

from utils import DB_FILE, MODELS_FILE, GAME_WEEK, POS_FEATURES, build_features, normalize_pool_scores
from eligibility import get_eligibility, _DEFAULT as ELIG_DEFAULT, _ABSENT as ELIG_ABSENT, player_name_key


def _elig_key(player_row) -> tuple[str, str]:
    """Return the name key used by the eligibility dict."""
    return player_name_key(
        player_row.get('first_name', ''),
        player_row.get('second_name', ''),
    )

app = Flask(__name__)

# FPL colour scheme (matches official app)
POS_COLORS = {
    'GK':  '#e8fa45',
    'DEF': '#4fd6e8',
    'MID': '#70e85a',
    'FWD': '#ff5757',
}

# Stats surfaced per-player in the hover card — curated per position.
# Removed ict_index (redundant with its components); added xG/xA/xGC and starts.
POS_STATS = {
    'GK':  ['roll5_minutes', 'roll5_starts', 'roll5_saves', 'roll5_clean_sheets',
            'roll5_goals_conceded', 'roll5_expected_goals_conceded',
            'roll5_bonus', 'roll5_total_points'],
    'DEF': ['roll5_minutes', 'roll5_starts', 'roll5_clean_sheets',
            'roll5_goals_conceded', 'roll5_expected_goals_conceded',
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
    pool = pd.concat(parts, ignore_index=True)
    return normalize_pool_scores(pool)


def select_squad(pool, structure, max_per_team, max_spend):
    squad, spend, team_counts = [], 0, Counter()
    for pos, count in structure.items():
        eligible = pool[pool['element_type'] == pos].sort_values(
            'predicted_points', ascending=False
        )
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

    df_squad = pd.DataFrame(squad)

    # selection margin: predicted_pts - next best available alternative not picked
    selected_ids = set(df_squad['id'])
    margins = []
    for _, player in df_squad.iterrows():
        alts = pool[
            (pool['element_type'] == player['element_type'])
            & (~pool['id'].isin(selected_ids))
        ]
        nb = float(alts['predicted_points'].max()) if not alts.empty else float(player['predicted_points'])
        margins.append(round(float(player['predicted_points']) - nb, 4))
    df_squad['selection_margin'] = margins
    return df_squad


def pick_starting_xi(squad):
    """
    Pick the best starting 11 from a 15-player squad using FPL formation rules
    (min 3 DEF / 2 MID / 1 FWD). Fill minimums first, then top up by predicted
    points. Returns (starters, bench, formation) e.g. (df, df, '4-4-2').
    """
    gk_sorted = squad[squad['element_type'] == 'GK'].sort_values(
        'predicted_points', ascending=False
    )
    starter_gk = gk_sorted.iloc[[0]]
    bench_gk   = gk_sorted.iloc[[1]]

    outfield = squad[squad['element_type'] != 'GK']
    mins     = {'DEF': 3, 'MID': 2, 'FWD': 1}
    counts   = {'DEF': 0, 'MID': 0, 'FWD': 0}
    starters_out, used_ids = [], set()

    # pass 1 -- fill minimums
    for pos, min_n in mins.items():
        for _, p in outfield[outfield['element_type'] == pos].sort_values(
            'predicted_points', ascending=False
        ).iterrows():
            if counts[pos] < min_n:
                starters_out.append(p)
                counts[pos] += 1
                used_ids.add(p['id'])

    # pass 2 -- fill remaining 10 - len(starters_out) spots using normalised
    # scores (uber-model): positions compared on even footing so an outstanding
    # DEF beats an average FWD for a flex slot.
    sort_col = 'predicted_points_norm' if 'predicted_points_norm' in outfield.columns else 'predicted_points'
    for _, p in outfield[~outfield['id'].isin(used_ids)].sort_values(
        sort_col, ascending=False
    ).iterrows():
        if len(starters_out) == 10:
            break
        starters_out.append(p)
        counts[p['element_type']] += 1
        used_ids.add(p['id'])

    starters  = pd.concat([starter_gk, pd.DataFrame(starters_out)], ignore_index=True)
    bench_out = outfield[~outfield['id'].isin(used_ids)].sort_values(
        'predicted_points', ascending=False
    )
    bench     = pd.concat([bench_gk, bench_out], ignore_index=True)
    formation = f"{counts['DEF']}-{counts['MID']}-{counts['FWD']}"
    return starters, bench, formation



_ROW_MARGINS = {1: 50, 2: 28, 3: 18, 4: 12, 5: 10}

def _row_xs(n):
    """Even x% positions, spacing tightens for smaller rows so 2 FWDs aren't at the edges."""
    if n == 1:
        return [50.0]
    margin = _ROW_MARGINS.get(n, 10)
    step = (100.0 - 2 * margin) / (n - 1)
    return [round(margin + i * step, 1) for i in range(n)]


# y% from top of pitch div: GK at bottom, FWD at top
_ROW_Y = {'GK': 84, 'DEF': 65, 'MID': 45, 'FWD': 15}


def assign_positions(starters):
    result = []
    for pos in ('GK', 'DEF', 'MID', 'FWD'):
        group = starters[starters['element_type'] == pos].sort_values(
            'predicted_points', ascending=False
        )
        xs = _row_xs(len(group))
        for i, (_, p) in enumerate(group.iterrows()):
            d = p.to_dict()
            d['pitch_x'] = round(xs[i], 1)
            d['pitch_y'] = _ROW_Y[pos]
            result.append(d)
    return result


# ---------------------------------------------------------------------------
# per-player profile stats
# ---------------------------------------------------------------------------

def player_stats(player_dict, pool):
    pos   = player_dict.get('element_type', '')
    feats = POS_STATS.get(pos, [])
    pos_pool = pool[pool['element_type'] == pos]
    stats = []
    for f in feats:
        if f not in pool.columns:
            continue
        try:
            val = float(player_dict.get(f) or 0)
        except (TypeError, ValueError):
            val = 0.0
        col = pos_pool[f].fillna(0)
        pct = int(round((col < val).mean() * 100))
        stats.append({
            'label': f.replace('roll5_', ''),
            'value': round(val, 2),
            'pct':   pct,
        })
    return stats


# ---------------------------------------------------------------------------
# JSON serialisation
# ---------------------------------------------------------------------------

def _safe(obj):
    """Recursively convert numpy scalars and NaN to plain Python types."""
    if isinstance(obj, dict):
        return {k: _safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return 0 if math.isnan(v) else v
    if isinstance(obj, float) and math.isnan(obj):
        return 0
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


# ---------------------------------------------------------------------------
# route
# ---------------------------------------------------------------------------

@app.route('/')
def index():
    if not os.path.exists(MODELS_FILE):
        return (
            f'<h2 style="font-family:sans-serif;padding:2rem">'
            f'{MODELS_FILE} not found — run train/train.py first.</h2>',
            500,
        )

    checkpoint  = joblib.load(MODELS_FILE)
    models_map  = checkpoint['models']
    metrics     = checkpoint['metrics']
    # epl_members is a frozenset of (first_name_lower, second_name_lower) tuples
    # saved at train time; None if the checkpoint predates this feature.
    epl_members: frozenset | None = checkpoint.get('epl_members')

    all_data = build_features(DB_FILE)
    pool     = build_pool(all_data, models_map, GAME_WEEK)

    # --- tier 1: training-time EPL membership (removes players who left the league;
    #             injured/suspended players are kept here and handled by tier 2) ---
    if epl_members is not None:
        pool = pool[pool.apply(lambda r: _elig_key(r) in epl_members, axis=1)].copy()

    # --- tier 2: live FPL API refresh (injury/suspension status) ---
    print('Fetching current eligibility from FPL API...')
    try:
        eligibility = get_eligibility(use_llm=True)
    except Exception as exc:
        print(f'  Warning: eligibility check failed ({exc}), proceeding without filter')
        eligibility = {}

    # build the excluded list *before* filtering the pool so we can show
    # which high-ranking players were dropped and why
    excluded = []
    if eligibility:
        # players absent from the current API response have left the Premier League
        absent = ELIG_ABSENT
        full_pool_sorted = pool.sort_values('predicted_points', ascending=False)
        for _, p in full_pool_sorted.iterrows():
            elig = eligibility.get(_elig_key(p), absent)
            if not elig.eligible:
                excluded.append({
                    'first_name': p.get('first_name', ''),
                    'second_name': p.get('second_name', ''),
                    'element_type': p.get('element_type', ''),
                    'predicted_points': round(float(p['predicted_points']), 2),
                    'news': elig.news,
                    'method': elig.method,
                    'color': POS_COLORS.get(str(p.get('element_type', '')), '#aaa'),
                })
            if len(excluded) >= 10:   # show the top-10 most impactful exclusions
                break

        def _elig(row_like):
            return eligibility.get(_elig_key(row_like) if hasattr(row_like, 'get') else row_like, ELIG_ABSENT).eligible
        pool = pool[pool.apply(_elig, axis=1)].copy()

    squad    = select_squad(pool, {'GK': 2, 'DEF': 5, 'MID': 5, 'FWD': 3},
                            max_per_team=4, max_spend=1000)
    starters, bench, formation = pick_starting_xi(squad)

    positioned = assign_positions(starters)
    for p in positioned:
        p['stats'] = player_stats(p, pool)
        p['color'] = POS_COLORS[p['element_type']]
        elig = eligibility.get(_elig_key(p), ELIG_ABSENT)
        p['news']    = elig.news
        p['ep_next'] = elig.ep_next
        p['elig_status'] = elig.status

    bench_list = []
    for _, p in bench.iterrows():
        b = p.to_dict()
        b['stats'] = player_stats(b, pool)
        b['color'] = POS_COLORS[b['element_type']]
        elig = eligibility.get(_elig_key(b), ELIG_ABSENT)
        b['news']    = elig.news
        b['ep_next'] = elig.ep_next
        b['elig_status'] = elig.status
        bench_list.append(b)

    payload = {
        'game_week':       GAME_WEEK,
        'formation':       formation,
        'total_predicted': round(float(starters['predicted_points'].sum()), 2),
        'total_spend':     round(float(squad['value'].sum()) / 10, 1),
        'players':         positioned,
        'bench':           bench_list,
        'excluded':        excluded,
        'model_quality':   [{'pos': pos, **m} for pos, m in metrics.items()],
    }

    return render_template('index.html', data=json.dumps(_safe(payload)))


if __name__ == '__main__':
    app.run(debug=True, port=5000)
