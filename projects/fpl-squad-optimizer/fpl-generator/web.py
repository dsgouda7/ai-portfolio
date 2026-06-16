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
import sqlite3
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request

from utils import (
    DB_FILE, MODELS_FILE, GAME_WEEK, SEASON, POS_FEATURES,
    build_features, normalize_pool_scores,
    save_squad, load_squad, score_squad_from_pool, pick_starting_xi,
    suggest_transfer, find_ineligible_replacements,
)
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
    <h1>\u26a0 {{title}}</h1>
    <p>{{message}}</p>
    <div class="label">Run this to fix it:</div>
    <div class="cmd">{{fix_cmd}}</div>
  </div>
</body>
</html>"""
    return html.format(title=title, message=message, fix_cmd=fix_cmd), status


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
    if not os.path.exists(MODELS_FILE):
        return _setup_error(
            'Models not trained',
            'No trained models found. models.joblib is created by the training script. '
            'Training takes ~30 seconds on a modern CPU.',
            '.venv\\Scripts\\python.exe train\\train.py',
        )

    checkpoint  = joblib.load(MODELS_FILE)
    models_map  = checkpoint['models']
    metrics     = checkpoint['metrics']
    epl_members: frozenset | None = checkpoint.get('epl_members')

    all_data  = build_features(DB_FILE)
    full_pool = build_pool(all_data, models_map, GAME_WEEK)

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

    # ── Squad mode vs fresh mode ─────────────────────────────────────────────
    force_new   = request.args.get('new_team') == '1'
    saved_squad = None if force_new else load_squad(DB_FILE)
    squad_mode  = False
    transfer_payload   = None
    ineligible_payload = []

    if force_new:
        conn = sqlite3.connect(DB_FILE)
        conn.execute("DROP TABLE IF EXISTS saved_squad")
        conn.commit()
        conn.close()

    if saved_squad is not None:
        squad_mode   = True
        scored_squad = score_squad_from_pool(saved_squad, full_pool)
        eligible_ids = set(eligible_pool['id'].astype(int))

        ineligible_df = scored_squad[~scored_squad['id'].astype(int).isin(eligible_ids)]
        eligible_df   = scored_squad[scored_squad['id'].astype(int).isin(eligible_ids)]

        if not eligible_df.empty and not eligible_df[eligible_df['element_type'] == 'GK'].empty:
            starters, bench, formation = pick_starting_xi(eligible_df)
        else:
            # Fall back to a fresh selection if squad is unworkable
            squad_mode = False
            saved_squad = None

        if squad_mode:
            # Transfer suggestion
            t = suggest_transfer(scored_squad, eligible_pool, 4, 1000)
            if t:
                transfer_payload = {
                    'out_name': f"{t['out']['first_name']} {t['out']['second_name']}",
                    'out_pos':  t['out']['element_type'],
                    'out_val':  t['out']['value'],
                    'out_pts':  t['out']['predicted_points'],
                    'in_name':  f"{t['in_']['first_name']} {t['in_']['second_name']}",
                    'in_pos':   t['in_']['element_type'],
                    'in_val':   t['in_']['value'],
                    'in_pts':   t['in_']['predicted_points'],
                    'gain':     t['gain'],
                    'new_spend': t['new_spend'],
                }

            # Ineligible replacements
            if not ineligible_df.empty:
                for item in find_ineligible_replacements(
                    ineligible_df, eligible_pool, scored_squad
                ):
                    out_p = item['out']
                    elig_info = eligibility.get(
                        (str(out_p.get('first_name', '') or '').lower(),
                         str(out_p.get('second_name', '') or '').lower()),
                        ELIG_ABSENT,
                    )
                    ineligible_payload.append({
                        'name':      f"{out_p['first_name']} {out_p['second_name']}",
                        'pos':       out_p['element_type'],
                        'color':     POS_COLORS.get(str(out_p['element_type']), '#aaa'),
                        'news':      getattr(elig_info, 'news', ''),
                        'repl_name': f"{item['replacement']['first_name']} {item['replacement']['second_name']}"
                                     if item['replacement'] else None,
                        'repl_val':  item['replacement']['value'] if item['replacement'] else None,
                        'repl_pts':  item['replacement']['predicted_points'] if item['replacement'] else None,
                        'gain':      item['gain'],
                    })

            squad = scored_squad
            gw_saved = int(saved_squad['gw_saved'].iloc[0])

    if not squad_mode:
        squad = select_squad(eligible_pool, {'GK': 2, 'DEF': 5, 'MID': 5, 'FWD': 3},
                             max_per_team=4, max_spend=1000)
        save_squad(DB_FILE, squad, GAME_WEEK, SEASON)
        starters, bench, formation = pick_starting_xi(squad)
        gw_saved = GAME_WEEK

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
        'game_week':        GAME_WEEK,
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
    }

    return render_template('index.html', data=json.dumps(_safe(payload)))


# ---------------------------------------------------------------------------
# /  redirect → /generate-team
# ---------------------------------------------------------------------------

@app.route('/')
def root():
    from flask import redirect
    return redirect('/generate-team')


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
        'algorithm':  'XGBoost — 4 separate XGBRegressors (one per position)',
        'target':     'next-GW total_points per player',
        'training':   '14 GWs of 2025-26 season data (GW 1–14); ~8,400 player-GW rows',
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
    app.run(debug=True, port=5000)
