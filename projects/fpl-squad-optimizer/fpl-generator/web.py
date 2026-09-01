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
import secrets
import time
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request
from scipy.optimize import Bounds, LinearConstraint, milp

from utils import (
    DB_FILE,
    MAX_PLAYERS_PER_TEAM, MAX_SPEND,
    build_features, normalize_pool_scores, apply_market_value_weighting,
    load_squad, score_squad_from_pool, pick_starting_xi,
    suggest_transfer, find_ineligible_replacements, get_runtime_context,
)
from model_registry import (
    available_model_artifacts,
    score_checkpoint_snapshot,
    validate_checkpoint_cutoff,
)
from feature_cache import load_or_build_feature_cache
from eligibility import get_eligibility, _DEFAULT as ELIG_DEFAULT, _ABSENT as ELIG_ABSENT, player_name_key
from squad_state import (
    SquadValidationError,
    commit_draft,
    create_state,
    list_versions,
    load_current,
    load_draft,
    load_working_state,
    refresh_player_data,
    roll_to_game_week,
    save_draft,
    validate_state,
)
from performance_reviewer import build_performance_review
from fpl_account import FplEntrySyncError, sync_public_entry


def _elig_key(player_row) -> tuple[str, str]:
    """Return the name key used by the eligibility dict."""
    return player_name_key(
        player_row.get('first_name', ''),
        player_row.get('second_name', ''),
    )

app = Flask(__name__)
_SYNC_LAST_REQUEST: dict[str, float] = {}
_SYNC_COOLDOWN_SECONDS = 3.0

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


def select_squad(
    pool,
    structure,
    max_per_team,
    max_spend,
    score_column='predicted_points',
    time_limit=10,
):
    candidates = pool.drop_duplicates(subset='id').reset_index(drop=True)
    constraint_rows = []
    lower_bounds = []
    upper_bounds = []

    for position, count in structure.items():
        constraint_rows.append(
            (candidates['element_type'] == position).astype(float).to_numpy()
        )
        lower_bounds.append(count)
        upper_bounds.append(count)

    constraint_rows.append(candidates['value'].astype(float).to_numpy())
    lower_bounds.append(0)
    upper_bounds.append(max_spend)

    for team in candidates['team'].dropna().unique():
        constraint_rows.append((candidates['team'] == team).astype(float).to_numpy())
        lower_bounds.append(0)
        upper_bounds.append(max_per_team)

    result = milp(
        c=-candidates[score_column].astype(float).to_numpy(),
        integrality=np.ones(len(candidates)),
        bounds=Bounds(0, 1),
        constraints=LinearConstraint(
            np.vstack(constraint_rows), lower_bounds, upper_bounds
        ),
        options={'time_limit': time_limit},
    )
    if not result.success:
        raise RuntimeError(
            'No legal 15-player squad could be generated from the eligible player pool.'
        )
    df_squad = candidates.loc[result.x > 0.5].copy().reset_index(drop=True)

    # Baseline quality margin: unperturbed model score versus the best unselected
    # same-position alternative. In stochastic mode it can be negative by design.
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


def _squad_quality(squad: pd.DataFrame) -> tuple[float, float]:
    starters, _, _ = pick_starting_xi(squad)
    return (
        float(squad['predicted_points'].sum()),
        float(starters['predicted_points'].sum()),
    )


def select_squad_variation(
    pool: pd.DataFrame,
    structure: dict[str, int],
    max_per_team: int,
    max_spend: int,
    uncertainty_by_position: dict[str, float],
    seed: int,
    quality_floor: float = 0.95,
    noise_scale: float = 0.10,
    attempts: int = 24,
) -> tuple[pd.DataFrame, dict]:
    """Sample a legal near-optimal squad and retain an auditable quality floor."""
    if not 0 < quality_floor <= 1:
        raise ValueError('quality_floor must be in (0, 1].')
    if attempts < 1:
        raise ValueError('attempts must be positive.')

    deterministic = select_squad(
        pool, structure, max_per_team, max_spend
    )
    optimal_squad_points, optimal_xi_points = _squad_quality(deterministic)
    random = np.random.default_rng(seed)
    accepted: dict[tuple[int, ...], tuple[pd.DataFrame, float, float]] = {}
    failed_attempts = 0

    for _ in range(attempts):
        varied_pool = pool.copy()
        standard_deviation = varied_pool['element_type'].map(
            uncertainty_by_position
        ).fillna(0.0).astype(float)
        varied_pool['_selection_score'] = (
            varied_pool['predicted_points'].astype(float)
            + random.normal(0.0, standard_deviation * noise_scale)
        )
        try:
            candidate = select_squad(
                varied_pool,
                structure,
                max_per_team,
                max_spend,
                score_column='_selection_score',
                time_limit=2,
            ).drop(columns='_selection_score', errors='ignore')
        except RuntimeError:
            failed_attempts += 1
            continue
        squad_points, xi_points = _squad_quality(candidate)
        if (
            squad_points >= optimal_squad_points * quality_floor
            and xi_points >= optimal_xi_points * quality_floor
        ):
            key = tuple(sorted(candidate['id'].astype(int)))
            accepted[key] = (candidate, squad_points, xi_points)

    deterministic_key = tuple(sorted(deterministic['id'].astype(int)))
    alternatives = [key for key in sorted(accepted) if key != deterministic_key]
    if alternatives:
        selected_key = alternatives[int(random.integers(len(alternatives)))]
        selected, selected_squad_points, selected_xi_points = accepted[selected_key]
    else:
        selected = deterministic
        selected_squad_points, selected_xi_points = (
            optimal_squad_points, optimal_xi_points
        )
    fallback_reason = None if alternatives else 'no_qualified_variations'

    metadata = {
        'mode': 'stochastic_near_optimal',
        'seed': int(seed),
        'quality_floor': float(quality_floor),
        'noise_scale': float(noise_scale),
        'attempts': int(attempts),
        'failed_attempts': failed_attempts,
        'accepted_unique_squads': len(accepted),
        'qualified_alternatives': len(alternatives),
        'varied_from_optimum': tuple(sorted(selected['id'].astype(int))) != deterministic_key,
        'fallback_reason': fallback_reason,
        'optimal_squad_predicted': round(optimal_squad_points, 4),
        'selected_squad_predicted': round(selected_squad_points, 4),
        'squad_quality_ratio': round(selected_squad_points / optimal_squad_points, 4),
        'optimal_xi_predicted': round(optimal_xi_points, 4),
        'selected_xi_predicted': round(selected_xi_points, 4),
        'xi_quality_ratio': round(selected_xi_points / optimal_xi_points, 4),
    }
    return selected.reset_index(drop=True), metadata


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


def _state_frame(state: dict) -> pd.DataFrame:
    """Convert readable JSON players into the legacy scoring join shape."""
    return pd.DataFrame([
        {
            'player_id': player['id'],
            'first_name': player['first_name'],
            'second_name': player['second_name'],
            'element_type': player['position'],
            'team': player['team'],
            'value': player['current_price'],
        }
        for player in state['players']
    ])


def _state_from_synced_entry(
    synced: dict,
    scored_pool: pd.DataFrame,
    runtime: dict,
    free_transfers: int | None = None,
    database: str | Path = DB_FILE,
) -> dict:
    picks = synced.get('picks') or []
    if len(picks) != 15:
        raise FplEntrySyncError(
            'No complete public 15-player squad is available yet. Picks become '
            'public after the Gameweek deadline.'
        )
    pool_by_id = scored_pool.drop_duplicates('id').set_index('id')
    missing = [
        int(pick['player_id']) for pick in picks
        if int(pick['player_id']) not in pool_by_id.index
    ]
    missing_players = {}
    if missing:
        import sqlite3
        from contextlib import closing
        placeholders = ','.join('?' for _ in missing)
        with closing(sqlite3.connect(str(database))) as connection:
            rows = connection.execute(
                f"""
                SELECT id, first_name, second_name, element_type, team, now_cost
                FROM players_raw WHERE id IN ({placeholders})
                """,
                missing,
            ).fetchall()
        missing_players = {
            int(player_id): {
                'id': int(player_id),
                'first_name': str(first_name or ''),
                'second_name': str(second_name or ''),
                'element_type': {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}.get(
                    int(element_type), ''
                ),
                'team': int(team or 0),
                'value': int(now_cost or 0),
                'predicted_points': 0.0,
                'predicted_points_norm': 0.0,
                'elig_status': 'u',
                'news': 'Not present in the current model snapshot.',
            }
            for player_id, first_name, second_name, element_type, team, now_cost in rows
        }
        unresolved = sorted(set(missing) - set(missing_players))
        if unresolved:
            raise FplEntrySyncError(
                'Official picks cannot be resolved from current FPL data: '
                + ', '.join(map(str, unresolved))
            )

    acquisition_costs = {
        int(player_id): int(cost)
        for player_id, cost in synced.get('acquisition_costs', {}).items()
    }
    squad_rows = []
    for pick in sorted(picks, key=lambda row: int(row['pick_position'])):
        player_id = int(pick['player_id'])
        row = (
            pool_by_id.loc[player_id].to_dict()
            if player_id in pool_by_id.index
            else missing_players[player_id]
        )
        row['id'] = player_id
        purchase_price = acquisition_costs.get(player_id) or int(row['value'])
        row['purchase_price'] = purchase_price
        squad_rows.append(row)
    squad = pd.DataFrame(squad_rows)
    state = create_state(
        squad,
        runtime['target_game_week'],
        runtime['season'],
        previous=load_current(),
        source='official_fpl_public_sync',
    )
    # create_state selects the next XI, bench order and captaincy from current
    # model scores. The prior official lineup remains in fpl_entry_picks.
    state['bank'] = int(synced.get('bank', 0))
    estimated_free_transfers = int(synced.get('next_free_transfers', 1))
    state['free_transfers'] = (
        int(free_transfers)
        if free_transfers is not None
        else estimated_free_transfers
    )
    state['official_entry'] = {
        'entry_id': int(synced['entry_id']),
        'synced_at': synced['synced_at'],
        'latest_public_event': synced.get('latest_event'),
        'squad_source_event': synced.get('squad_event'),
        'free_transfers_estimated': free_transfers is None,
        'bank_source_event': synced.get('squad_event'),
    }
    chip_names = {
        'wildcard': 'wildcard',
        'freehit': 'free_hit',
        'free_hit': 'free_hit',
        'bboost': 'bench_boost',
        'bench_boost': 'bench_boost',
        '3xc': 'triple_captain',
        'triple_captain': 'triple_captain',
    }
    for gameweek in synced.get('gameweeks', []):
        chip = chip_names.get(str(gameweek.get('active_chip') or '').lower())
        if chip is None:
            continue
        event = int(gameweek['event'])
        used = state['chips'][chip]['used_gameweeks']
        if event not in used:
            used.append(event)
            state['chips'][chip]['remaining'] = max(
                0, int(state['chips'][chip]['remaining']) - 1
            )
    validate_state(state)
    return state


def _lineup_frames(state: dict, squad: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Apply the user's persisted XI and ordered bench to a freshly scored squad."""
    by_id = {int(row['id']): row for row in squad.to_dict(orient='records')}

    def ordered(player_ids):
        return pd.DataFrame([by_id[int(player_id)] for player_id in player_ids])

    starters = ordered(state['lineup']['starters'])
    bench = ordered(state['lineup']['bench'])
    counts = Counter(starters['element_type'])
    formation = f"{counts['DEF']}-{counts['MID']}-{counts['FWD']}"
    return starters, bench, formation


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


def _setup_error(title: str, message: str, fix_cmd: str, status: int = 500) -> tuple:
    """Return a styled setup-error page with the exact command needed to fix the issue."""
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Setup required \u2014 FPL Generator</title>
  <style>
    body {{ font-family: 'Segoe UI', system-ui, sans-serif; background:#0a0f1a; color:#e8eaf0;
           display:flex; align-items:center; justify-content:center; min-height:100vh; margin:0; }}
    .card {{ background:rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.09);
             border-radius:10px; padding:2.5rem 3rem; max-width:560px; width:90%; }}
    h1 {{ font-size:1.1rem; color:#f87171; margin:0 0 0.5rem; }}
    p  {{ font-size:0.85rem; color:#9ba3b8; margin:0.4rem 0 1.2rem; line-height:1.55; }}
    .cmd {{ background:#111827; border:1px solid rgba(255,255,255,0.1); border-radius:6px;
            padding:0.7rem 1rem; font-family:monospace; font-size:0.8rem; color:#38bdf8;
            white-space:pre-wrap; word-break:break-all; }}
    .label {{ font-size:0.68rem; color:#7a8299; text-transform:uppercase;
              letter-spacing:0.05em; margin-bottom:0.3rem; }}
  </style>
</head>
<body>
  <div class="card">
    <h1>\u26a0 {title}</h1>
    <p>{message}</p>
    <div class="label">Run this to fix it:</div>
    <div class="cmd">{fix_cmd}</div>
  </div>
</body>
</html>"""
    return html, status


# ---------------------------------------------------------------------------
# routes
# ---------------------------------------------------------------------------

@app.route('/generate-team')
def index():
    if not os.path.exists(DB_FILE):
        return _setup_error(
            'Database not found',
            'The player database has not been created yet. '
            'Run setup.ps1 to clone the FPL dataset and prepare the environment, '
            'then run train.py which ingests the data automatically.',
            '.\\setup.ps1\n.venv\\Scripts\\python.exe train\\train.py',
        )
    artifacts = available_model_artifacts()
    if not artifacts:
        return _setup_error(
            'Models not trained',
            'No trained models found. models.joblib is created by the training script. '
            'Training takes ~30 seconds on a modern CPU.',
            '.venv\\Scripts\\python.exe train\\train.py',
        )

    requested_model = request.args.get('model', 'xgboost').lower()
    if requested_model not in artifacts:
        requested_model = next(iter(artifacts))
    checkpoint = joblib.load(artifacts[requested_model])
    validate_checkpoint_cutoff(checkpoint)
    metrics     = checkpoint['metrics']
    epl_members: frozenset | None = checkpoint.get('epl_members')

    runtime = get_runtime_context(DB_FILE)
    all_data, _ = load_or_build_feature_cache(DB_FILE, build_features)
    full_pool = score_checkpoint_snapshot(
        all_data, checkpoint, runtime['snapshot_game_week']
    )

    # --- tier 1: EPL membership filter ---
    pool = full_pool.copy()
    if epl_members is not None:
        pool = pool[pool.apply(lambda r: _elig_key(r) in epl_members, axis=1)].copy()

    # --- tier 2: live FPL eligibility ---
    print('Fetching current eligibility from FPL API...')
    try:
        eligibility = get_eligibility()
    except Exception as exc:
        print(f'  Warning: eligibility check failed ({exc}), proceeding without filter')
        eligibility = {}

    # Build excluded list (top-10 most impactful ineligible players from the full pool)
    excluded = []
    if eligibility:
        for _, p in pool.sort_values('predicted_points', ascending=False).iterrows():
            elig = eligibility.get(_elig_key(p), ELIG_ABSENT)
            if not elig.eligible:
                excluded.append({
                    'first_name':       p.get('first_name', ''),
                    'second_name':      p.get('second_name', ''),
                    'element_type':     p.get('element_type', ''),
                    'predicted_points': round(float(p['predicted_points']), 2),
                    'news':             elig.news,
                    'method':           elig.method,
                    'color':            POS_COLORS.get(str(p.get('element_type', '')), '#aaa'),
                })
            if len(excluded) >= 10:
                break

    def _elig(row_like):
        return eligibility.get(
            _elig_key(row_like) if hasattr(row_like, 'get') else row_like, ELIG_ABSENT
        ).eligible

    eligible_pool = pool[pool.apply(_elig, axis=1)].copy() if eligibility else pool.copy()

    # Squad mode vs draft generation. Regeneration never deletes current state.
    force_new   = request.args.get('new_team') == '1'
    current_state = None if force_new else load_working_state()
    if current_state is None and not force_new:
        legacy_squad = load_squad(DB_FILE)
        if legacy_squad is not None:
            legacy_scored = score_squad_from_pool(legacy_squad, full_pool)
            current_state = create_state(
                legacy_scored,
                runtime['target_game_week'],
                runtime['season'],
                source='legacy_sqlite_import',
            )
            current_state = commit_draft(current_state)
    state = None
    squad_mode = current_state is not None
    transfer_payload   = None
    ineligible_payload = []

    if current_state is not None:
        rolled_state = roll_to_game_week(
            current_state, runtime['target_game_week'], runtime['season']
        )
        state = refresh_player_data(rolled_state, full_pool, eligibility)
        if int(current_state['game_week']) != int(state['game_week']):
            state = save_draft(state)
        scored_squad = score_squad_from_pool(_state_frame(state), full_pool)
        eligible_ids = set(eligible_pool['id'].astype(int))

        ineligible_df = scored_squad[~scored_squad['id'].astype(int).isin(eligible_ids)]
        starters, bench, formation = _lineup_frames(state, scored_squad)

        transfer_squad = scored_squad.copy()
        selling_prices = {
            int(player['id']): int(player['selling_price'])
            for player in state['players']
        }
        transfer_squad['value'] = transfer_squad['id'].astype(int).map(selling_prices)
        transfer_budget = int(transfer_squad['value'].sum()) + int(state['bank'])
        t = suggest_transfer(
            transfer_squad, eligible_pool,
            MAX_PLAYERS_PER_TEAM, transfer_budget,
        )
        if t:
            transfer_payload = {
                'out_id': int(t['out']['id']),
                'out_name': f"{t['out']['first_name']} {t['out']['second_name']}",
                'out_pos':  t['out']['element_type'],
                'out_val':  t['out']['value'],
                'out_pts':  t['out']['predicted_points'],
                'in_id': int(t['in_']['id']),
                'in_name':  f"{t['in_']['first_name']} {t['in_']['second_name']}",
                'in_pos':   t['in_']['element_type'],
                'in_val':   t['in_']['value'],
                'in_pts':   t['in_']['predicted_points'],
                'gain':     t['gain'],
                'new_spend': t['new_spend'],
                'bank_after': transfer_budget - t['new_spend'],
            }

        if not ineligible_df.empty:
            for item in find_ineligible_replacements(
                ineligible_df, eligible_pool, scored_squad,
                MAX_PLAYERS_PER_TEAM, MAX_SPEND,
            ):
                out_p = item['out']
                elig_info = eligibility.get(_elig_key(out_p), ELIG_ABSENT)
                ineligible_payload.append({
                    'id': int(out_p['id']),
                    'name': f"{out_p['first_name']} {out_p['second_name']}",
                    'pos': out_p['element_type'],
                    'color': POS_COLORS.get(str(out_p['element_type']), '#aaa'),
                    'news': getattr(elig_info, 'news', ''),
                    'repl_id': int(item['replacement']['id']) if item['replacement'] else None,
                    'repl_name': f"{item['replacement']['first_name']} {item['replacement']['second_name']}"
                                 if item['replacement'] else None,
                    'repl_val': item['replacement']['value'] if item['replacement'] else None,
                    'repl_pts': item['replacement']['predicted_points'] if item['replacement'] else None,
                    'gain': item['gain'],
                })

        squad = scored_squad
        gw_saved = int(state['game_week'])
    else:
        quality_floor = float(os.environ.get('FPL_VARIATION_QUALITY_FLOOR', '0.95'))
        noise_scale = float(os.environ.get('FPL_VARIATION_NOISE_SCALE', '0.10'))
        attempts = int(os.environ.get('FPL_VARIATION_ATTEMPTS', '24'))
        uncertainty = {
            position: float(metric.get('error_p80', metric.get('rmse', 0.0)))
            for position, metric in metrics.items()
        }
        requested_seed = request.args.get('seed')
        try:
            generation_seed = (
                int(requested_seed) if requested_seed is not None
                else secrets.randbits(63)
            )
            if generation_seed < 0 or generation_seed >= 2 ** 63:
                raise ValueError
        except ValueError:
            return _setup_error(
                'Invalid variation seed',
                'The replay seed must be an integer between 0 and 2^63-1.',
                '/generate-team?new_team=1',
                400,
            )
        squad, generation = select_squad_variation(
            eligible_pool,
            {'GK': 2, 'DEF': 5, 'MID': 5, 'FWD': 3},
            max_per_team=MAX_PLAYERS_PER_TEAM,
            max_spend=MAX_SPEND,
            uncertainty_by_position=uncertainty,
            seed=generation_seed,
            quality_floor=quality_floor,
            noise_scale=noise_scale,
            attempts=attempts,
        )
        state = create_state(
            squad, runtime['target_game_week'], runtime['season'],
            previous=load_current(), source='regenerated' if force_new else 'generated',
            generation=generation,
        )
        save_draft(state)
        starters, bench, formation = _lineup_frames(state, squad)
        gw_saved = runtime['target_game_week']

    # ── Pitch layout ─────────────────────────────────────────────────────────
    positioned = assign_positions(starters)
    for p in positioned:
        p['stats'] = player_stats(p, eligible_pool)
        p['color'] = POS_COLORS.get(p['element_type'], '#aaa')
        elig = eligibility.get(_elig_key(p), ELIG_ABSENT)
        p['news']        = elig.news
        p['ep_next']     = elig.ep_next
        p['elig_status'] = elig.status

    bench_list = []
    for _, p in bench.iterrows():
        b = p.to_dict()
        b['stats'] = player_stats(b, eligible_pool)
        b['color'] = POS_COLORS.get(b['element_type'], '#aaa')
        elig = eligibility.get(_elig_key(b), ELIG_ABSENT)
        b['news']        = elig.news
        b['ep_next']     = elig.ep_next
        b['elig_status'] = elig.status
        bench_list.append(b)

    payload = {
        'game_week':        runtime['target_game_week'],
        'model_type':       requested_model,
        'available_models': list(artifacts),
        'training_manifest': checkpoint['training_manifest'],
        'generation':       state.get('generation'),
        'gw_saved':         gw_saved,
        'squad_mode':       squad_mode,
        'formation':        formation,
        'total_predicted':  round(float(starters['predicted_points'].sum()), 2),
        'total_spend':      round(float(squad['value'].sum()) / 10, 1),
        'players':          positioned,
        'bench':            bench_list,
        'excluded':         excluded,
        'model_quality':    [{'pos': pos, **m} for pos, m in metrics.items()],
        'transfer':         transfer_payload,
        'ineligible_squad': ineligible_payload,
        'state':             state,
        'candidates': [
            {
                'id': int(player['id']),
                'first_name': str(player.get('first_name', '')),
                'second_name': str(player.get('second_name', '')),
                'position': str(player.get('element_type', '')),
                'team': int(player.get('team', 0)),
                'current_price': int(player.get('value', 0)),
                'predicted_points': round(float(player.get('predicted_points', 0)), 4),
            }
            for player in eligible_pool.to_dict(orient='records')
        ],
    }

    return render_template('index.html', data=json.dumps(_safe(payload)))


@app.get('/api/squad')
def get_squad_state():
    return jsonify({'current': load_current(), 'draft': load_draft()})


@app.get('/api/squad/history')
def get_squad_history():
    return jsonify({'versions': list_versions(season=request.args.get('season'))})


@app.post('/api/fpl-entry/sync')
def sync_fpl_entry():
    """Import completed public FPL history and latest permanent picks by entry ID."""
    origin = request.headers.get('Origin')
    if origin and origin.rstrip('/') != request.host_url.rstrip('/'):
        return jsonify({'ok': False, 'error': 'Cross-origin sync requests are rejected.'}), 403
    if not request.is_json:
        return jsonify({'ok': False, 'error': 'FPL sync accepts JSON only.'}), 415
    try:
        payload = request.get_json() or {}
        allowed_fields = {'entry_id', 'model', 'free_transfers'}
        unexpected_fields = set(payload) - allowed_fields
        if unexpected_fields:
            raise FplEntrySyncError(
                'Credentials and unknown fields are not accepted. Provide only '
                'entry_id, model, and optional free_transfers.'
            )
        free_transfers = payload.get('free_transfers')
        if free_transfers not in (None, ''):
            free_transfers = int(free_transfers)
            if not 1 <= free_transfers <= 5:
                raise FplEntrySyncError('Free transfers must be between 1 and 5.')
        else:
            free_transfers = None
        requester = request.remote_addr or 'local'
        now = time.monotonic()
        last_request = _SYNC_LAST_REQUEST.get(requester, 0.0)
        if now - last_request < _SYNC_COOLDOWN_SECONDS:
            return jsonify({
                'ok': False,
                'error': 'Please wait a few seconds before synchronizing again.',
            }), 429
        _SYNC_LAST_REQUEST[requester] = now
        model_type = str(payload.get('model') or 'xgboost').lower()
        artifacts = available_model_artifacts()
        if model_type not in artifacts:
            raise FplEntrySyncError(f'Model artifact is unavailable: {model_type}')
        checkpoint = joblib.load(artifacts[model_type])
        validate_checkpoint_cutoff(checkpoint)
        runtime = get_runtime_context(DB_FILE)
        all_data, _ = load_or_build_feature_cache(DB_FILE, build_features)
        scored_pool = score_checkpoint_snapshot(
            all_data, checkpoint, runtime['snapshot_game_week']
        )
        synced = sync_public_entry(payload.get('entry_id'), DB_FILE)
        state = _state_from_synced_entry(
            synced,
            scored_pool,
            runtime,
            free_transfers=free_transfers,
            database=DB_FILE,
        )
        stored = save_draft(state)
        return jsonify({
            'ok': True,
            'state': stored,
            'entry_id': synced['entry_id'],
            'gameweeks_imported': len(synced['gameweeks']),
            'message': (
                f"Imported public picks through GW{synced['latest_event']}. "
                'Bank is from the latest permanent public squad; free transfers '
                f"are {'user-supplied' if free_transfers is not None else 'estimated'}. "
                'The next XI and captaincy were selected from current model scores.'
            ),
        })
    except (FplEntrySyncError, TypeError, KeyError, ValueError) as exc:
        return jsonify({'ok': False, 'error': str(exc)}), 400


@app.post('/api/squad/draft')
def persist_draft():
    try:
        state = request.get_json(force=True)
        validate_state(state)
        stored = save_draft(state)
        return jsonify({'ok': True, 'state': stored})
    except (SquadValidationError, TypeError, KeyError, ValueError) as exc:
        return jsonify({'ok': False, 'error': str(exc)}), 400


@app.post('/api/squad/commit')
def persist_current_squad():
    try:
        state = request.get_json(force=True)
        validate_state(state)
        stored = commit_draft(state)
        return jsonify({'ok': True, 'state': stored})
    except (SquadValidationError, TypeError, KeyError, ValueError) as exc:
        return jsonify({'ok': False, 'error': str(exc)}), 400


# ---------------------------------------------------------------------------
# /  redirect → /generate-team
# ---------------------------------------------------------------------------

@app.route('/')
def root():
    from flask import redirect
    return redirect('/generate-team')


@app.get('/performance-review')
def performance_review():
    """Review persisted squads against actual FPL Gameweek outcomes."""
    season = request.args.get('season')
    payload = build_performance_review(DB_FILE, season)
    return render_template(
        'performance_review.html', data=json.dumps(_safe(payload))
    )


# ---------------------------------------------------------------------------
# /validation-report
# ---------------------------------------------------------------------------

@app.route('/validation-report')
def validation_report():
    """Side-by-side per-GW comparison of our generated team vs oracle optimal."""
    import csv, urllib.request as _req

    sim_csv    = Path(os.environ.get(
                    'FPL_RESULTS_DIR',
                    str(Path(__file__).parent.parent / 'simulations' / 'results'),
                )) / 'simulation_results.csv'
    player_csv = Path(__file__).parent.parent / 'simulations' / 'results' / 'player_rows.csv'

    if not sim_csv.exists():
        return _setup_error(
            'Simulation results not found',
            'The validation report requires simulation data that has not been generated yet. '
            'This runs the model against held-out game weeks and compares it to the '
            'oracle-optimal squad. It takes 1–2 minutes to complete.',
            '.venv\\Scripts\\python.exe simulations\\simulate_season.py '
            '--test-from 15 --test-to 37',
            404,
        )

    gw_df  = pd.read_csv(sim_csv)
    pl_df  = pd.read_csv(player_csv) if player_csv.exists() else pd.DataFrame()

    # ── compute summary metrics ───────────────────────────────────────────
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).parent.parent))
    from simulations.metrics import compute_metrics, compute_per_position_rmse

    summary = compute_metrics(gw_df)
    pos_rmse = compute_per_position_rmse(pl_df).to_dict(orient='records') if not pl_df.empty else []

    # ── per-GW pitch data ─────────────────────────────────────────────────
    POS_ORDER = ['GK', 'DEF', 'MID', 'FWD']

    def _pitch_players(squad_df: pd.DataFrame, pos_colors: dict) -> list:
        """Assign basic pitch positions and return list of dicts."""
        rows = []
        for pos in POS_ORDER:
            sub = squad_df[squad_df['element_type'] == pos].sort_values(
                'actual_points', ascending=False
            )
            for i, (_, p) in enumerate(sub.iterrows()):
                rows.append({
                    'first_name':    str(p.get('first_name', '')),
                    'second_name':   str(p.get('second_name', '')),
                    'element_type':  pos,
                    'actual_points': round(float(p.get('actual_points', 0)), 1),
                    'predicted_points': round(float(p.get('predicted_points', 0)), 3),
                    'value':         int(p.get('value', 0)),
                    'color':         pos_colors.get(pos, '#aaa'),
                    'slot':          i,
                })
        return rows

    pos_colors = {'GK': '#e8fa45', 'DEF': '#4fd6e8', 'MID': '#70e85a', 'FWD': '#ff5757'}

    # ── GW date lookup from DB ────────────────────────────────────────────
    gw_dates: dict[int, str] = {}
    try:
        import sqlite3 as _sqlite3
        with _sqlite3.connect(DB_FILE) as _dbc:
            for _gw, _ts in _dbc.execute(
                "SELECT round, MIN(kickoff_time) FROM player_gw GROUP BY round"
            ).fetchall():
                if _ts:
                    from datetime import datetime as _dt
                    try:
                        _d = _dt.strptime(_ts[:10], '%Y-%m-%d')
                        gw_dates[int(_gw)] = _d.strftime('%b %-d') if hasattr(_d, 'strftime') else _ts[:10]
                    except ValueError:
                        gw_dates[int(_gw)] = _ts[:10]
    except Exception:
        pass

    # ── fallback: strftime '%-d' not on Windows → use %d ─────────────────
    if not gw_dates:
        pass  # stays empty; JS will fall back to just showing GW number
    else:
        # re-format using %d (cross-platform)
        try:
            import sqlite3 as _sqlite3
            from datetime import datetime as _dt2
            gw_dates = {}
            with _sqlite3.connect(DB_FILE) as _dbc2:
                for _gw2, _ts2 in _dbc2.execute(
                    "SELECT round, MIN(kickoff_time) FROM player_gw GROUP BY round"
                ).fetchall():
                    if _ts2:
                        _d2 = _dt2.strptime(_ts2[:10], '%Y-%m-%d')
                        gw_dates[int(_gw2)] = _d2.strftime('%b %d').replace(' 0', ' ')
        except Exception:
            pass

    gw_payloads = []
    for _, row in gw_df.sort_values('gw').iterrows():
        gw = int(row['gw'])
        if not pl_df.empty:
            our_pl    = pl_df[(pl_df['gw'] == gw) & (pl_df['source'] == 'our')].copy()
            oracle_pl = pl_df[(pl_df['gw'] == gw) & (pl_df['source'] == 'oracle')].copy()
        else:
            our_pl = oracle_pl = pd.DataFrame()

        gw_payloads.append({
            'gw':           gw,
            'gw_date':      gw_dates.get(gw, ''),
            'our_actual':   round(float(row['our_actual_xi']), 1),
            'oracle_actual':round(float(row['oracle_xi']), 1),
            'gap':          round(float(row['gap_to_oracle']), 1),
            'pct':          round(float(row['pct_of_oracle']), 1),
            'our_players':  _pitch_players(our_pl, pos_colors) if not our_pl.empty else [],
            'oracle_players': _pitch_players(oracle_pl, pos_colors) if not oracle_pl.empty else [],
        })

    model_info = {
        'algorithm':  'Five selectable predictors, including a two-layer Deep GRU',
        'target':     'fixture-row total_points from pre-fixture features',
        'training':   'All completed Gameweeks through each checkpoint cutoff',
        'eval_metric': 'Spearman ρ (ranking accuracy) + RMSE',
        'features': {
            'GK':  ['saves', 'expected_goals_conceded (roll5)', 'penalties_saved', 'clean_sheets (roll5)',
                    'minutes (roll5)', 'goals_conceded (roll5)', 'was_home', 'value', 'opponent_team'],
            'DEF': ['expected_goals_conceded (roll5)', 'expected_goals (roll5)', 'expected_assists (roll5)',
                    'clean_sheets (roll5)', 'creativity (roll5)', 'goals_scored (roll5)', 'assists (roll5)',
                    'minutes (roll5)', 'starts (roll5)', 'was_home', 'value', 'opponent_team',
                    'tm_market_value', 'form_data_density', 'tm_market_value_x_sparsity'],
            'MID': ['expected_goals (roll5)', 'expected_assists (roll5)', 'creativity (roll5)',
                    'influence (roll5)', 'goals_scored (roll5)', 'assists (roll5)', 'bonus (roll5)',
                    'minutes (roll5)', 'starts (roll5)', 'was_home', 'value', 'opponent_team',
                    'tm_market_value', 'form_data_density', 'tm_market_value_x_sparsity'],
            'FWD': ['expected_goals (roll5)', 'expected_assists (roll5)', 'threat (roll5)',
                    'influence (roll5)', 'goals_scored (roll5)', 'assists (roll5)', 'bonus (roll5)',
                    'minutes (roll5)', 'starts (roll5)', 'was_home', 'value', 'opponent_team',
                    'tm_market_value', 'form_data_density', 'tm_market_value_x_sparsity'],
        },
        'notes': [
            '5-GW rolling windows — always shifted 1 GW before windowing (no leakage)',
            'Deep GRU sequences use every available prior player-Gameweek through the completed-data cutoff',
            'ict_index excluded (linear combination of sub-features already present)',
            'form_data_density = roll5_minutes / 450 — quantifies data sparsity for injury returnees',
            'tm_market_value_x_sparsity = market_value × (1 − density) — quality prior for sparse players',
            'Flex XI slots filled by z-score within position (prevents MID dominance of unconstrained slots)',
        ],
    }

    payload = {
        'summary':    summary,
        'pos_rmse':   pos_rmse,
        'gw_data':    gw_payloads,
        'gw_list':    [int(r['gw']) for r in gw_payloads],
        'model_info': model_info,
    }

    return render_template('validation_report.html', data=json.dumps(_safe(payload)))


if __name__ == '__main__':
    app.run(
        debug=os.environ.get('FLASK_DEBUG', '1') == '1',
        port=int(os.environ.get('FPL_PORT', '5000')),
    )
