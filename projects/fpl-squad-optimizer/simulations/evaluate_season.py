"""Strict current-season replay from dated inputs, never from today's checkpoint."""
from __future__ import annotations

import argparse
from contextlib import closing, contextmanager
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
import sqlite3
import sys
import tempfile
from typing import Any

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eligibility import _rule_based
from feature_cache import (
    FEATURE_CACHE_SCHEMA_VERSION, load_or_build_feature_cache, source_fingerprint,
)
from model_registry import MODEL_TYPES, model_artifact_path, score_checkpoint_snapshot
from performance_reviewer import score_saved_squad
from simulations.evaluate_opposition import _write_report as write_report
from squad_state import (
    POSITION_LIMITS, apply_transfer, create_state, refresh_player_data,
    roll_to_game_week, set_lineup,
)
from utils import (
    DB_FILE, FPL_API_BASE, ROLL_COLS, build_features, pick_starting_xi,
    score_squad_from_pool, suggest_transfers,
)

SNAPSHOT_VERSION = 1
PUBLIC_TABLES = (
    'app_metadata', 'players_raw', 'player_gw', 'player_history',
    'player_transfer_value_history', 'player_external_appearances',
    'player_attributes', 'model_feature_cache', 'model_feature_cache_metadata',
)


def snapshot_root() -> Path:
    return Path(os.environ.get('FPL_SNAPSHOT_DIR', str(ROOT / 'simulations' / 'snapshots')))


def evaluation_root() -> Path:
    return Path(os.environ.get(
        'FPL_EVALUATION_DIR', str(ROOT / 'simulations' / 'results' / 'season-evaluations')
    ))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def _timestamp(value: str, name: str) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if pd.isna(stamp) or stamp.tzinfo is None:
        raise ValueError(f'{name} must be a timezone-aware timestamp.')
    return stamp


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def current_context(database: Path) -> dict[str, Any]:
    with closing(sqlite3.connect(database.resolve().as_uri() + '?mode=ro', uri=True)) as conn:
        metadata = dict(conn.execute('SELECT key, value FROM app_metadata'))
    season = metadata['season']
    if len(season) != 7 or not season[:4].isdigit() or season[4] != '-' or not season[5:].isdigit():
        raise ValueError('The persisted season must have YYYY-YY format.')
    completed = int(metadata['latest_completed_game_week'])
    if not 0 <= completed <= 38:
        raise ValueError('Invalid completed Gameweek in runtime metadata.')
    return {
        'season': season, 'latest_completed': completed,
        'target_gameweek': int(metadata['target_game_week']),
        'data_refreshed_at': metadata['fpl_refreshed_at'],
    }


def validate_selection(models: list[str], seeds: int, through: int, completed: int) -> None:
    if (
        not isinstance(models, list) or not models
        or any(not isinstance(model, str) or model not in MODEL_TYPES for model in models)
        or len(set(models)) != len(models)
    ):
        raise ValueError('Select one or more distinct supported models.')
    if type(seeds) is not int or not 1 <= seeds <= 25:
        raise ValueError('Starting squads must be an integer between 1 and 25.')
    if type(through) is not int or not 1 <= through <= completed:
        raise ValueError('Evaluate GW1 through a persisted completed Gameweek only.')


def _validate_source(database: Path, manifest: dict) -> None:
    captured = _timestamp(manifest['captured_at'], 'Capture time')
    deadline = _timestamp(manifest['deadline'], 'Deadline')
    if captured >= deadline:
        raise ValueError('The input bundle was not captured before the deadline.')
    with closing(sqlite3.connect(database.resolve().as_uri() + '?mode=ro', uri=True)) as conn:
        metadata = dict(conn.execute('SELECT key, value FROM app_metadata'))
        if (
            metadata['season'] != manifest['season']
            or int(metadata['target_game_week']) != manifest['game_week']
            or int(metadata['latest_completed_game_week']) != manifest['game_week'] - 1
        ):
            raise ValueError('Snapshot season/Gameweek does not match its source metadata.')
        refreshed = _timestamp(metadata['fpl_refreshed_at'], 'Source refresh')
        if refreshed > captured:
            raise ValueError('Source refresh occurs after the recorded capture.')
        cursor = conn.execute('SELECT * FROM model_feature_cache_metadata')
        row = cursor.fetchone()
        if row is None:
            raise ValueError('The captured feature cache is missing.')
        cache = dict(zip([column[0] for column in cursor.description], row))
        if (
            int(cache['schema_version']) != FEATURE_CACHE_SCHEMA_VERSION
            or cache['source_fingerprint'] != source_fingerprint(conn)
        ):
            raise ValueError('The input feature cache is stale or inconsistent.')
        if _timestamp(cache['created_at'], 'Feature creation') > captured:
            raise ValueError('Feature creation occurs after the recorded capture.')
        if conn.execute('SELECT COUNT(*) FROM model_feature_cache').fetchone()[0] != cache['row_count']:
            raise ValueError('Captured feature-cache row count does not match its metadata.')
        snapshot = int(metadata['snapshot_game_week'])
        cutoff = int(metadata['completed_internal_index'])
        if snapshot != cutoff + 1:
            raise ValueError('Inference snapshot must immediately follow the training cutoff.')
        roster_columns = {row[1] for row in conn.execute('PRAGMA table_info(players_raw)')}
        required_roster = {
            'id', 'code', 'team', 'element_type', 'first_name', 'second_name',
            'birth_date', 'status', 'chance_of_playing_next_round',
        }
        if not required_roster.issubset(roster_columns):
            raise ValueError('Captured player metadata lacks required identity/availability fields.')
        past = pd.read_sql_query(
            'SELECT kickoff_time FROM player_gw WHERE Game_Week <= ?', conn, params=[cutoff]
        )
        kickoffs = pd.to_datetime(past['kickoff_time'], errors='raise', utc=True)
        if past.empty or kickoffs.isna().any() or kickoffs.ge(deadline).any():
            raise ValueError('Training history must be nonempty and strictly before the deadline.')
        if kickoffs.ge(captured).any() or kickoffs.gt(refreshed).any():
            raise ValueError('Training history was not available at the recorded source refresh/capture.')
        future = conn.execute(
            'SELECT COUNT(*) FROM player_gw WHERE Game_Week > ?', (snapshot,)
        ).fetchone()[0]
        columns = {row[1] for row in conn.execute('PRAGMA table_info(player_gw)')}
        outcomes = ['fixture', *[column for column in ROLL_COLS if column in columns]]
        selected_columns = ','.join(f'"{column}"' for column in outcomes)
        target = pd.read_sql_query(
            f'SELECT {selected_columns} FROM player_gw WHERE Game_Week=?',
            conn, params=[snapshot],
        )
        if target.empty:
            raise ValueError('The target-Gameweek inference snapshot is empty.')
        leaked = target.apply(pd.to_numeric, errors='raise').fillna(0).ne(0).to_numpy().any()
        if future or leaked:
            raise ValueError('Future rows or target-Gameweek outcomes exist in prediction inputs.')


def inspect_bundle(root: Path, season: str, gameweek: int) -> dict[str, Any]:
    directory = root / season / f'gw{gameweek:02d}'
    result = {'game_week': gameweek, 'ready': False, 'status': 'missing',
              'reason': 'No dated pre-deadline input bundle.', 'captured_at': None}
    if not directory.exists():
        return result
    try:
        manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
        if (
            manifest['schema_version'] != SNAPSHOT_VERSION
            or manifest['season'] != season or manifest['game_week'] != gameweek
        ):
            raise ValueError('Unsupported or mismatched snapshot manifest.')
        database = directory / 'inputs.sqlite'
        if file_hash(database) != manifest['sha256']:
            raise ValueError('Input database does not match its captured SHA-256 hash.')
        _validate_source(database, manifest)
    except (OSError, ValueError, KeyError, TypeError, sqlite3.Error) as exc:
        result.update(status='invalid', reason=str(exc))
        return result
    result.update(ready=True, status='ready', reason='Verified pre-deadline inputs.',
                  captured_at=manifest['captured_at'])
    return result


def readiness(database: Path, root: Path, through: int | None = None) -> dict[str, Any]:
    context = current_context(database)
    through = context['latest_completed'] if through is None else through
    if type(through) is not int or not 0 <= through <= context['latest_completed']:
        raise ValueError('Requested range exceeds the persisted completed Gameweeks.')
    weeks = [inspect_bundle(root, context['season'], gw) for gw in range(1, through + 1)]
    return {
        **context, 'through': through, 'weeks': weeks,
        'ready': bool(weeks) and all(week['ready'] for week in weeks),
        'missing_gameweeks': [week['game_week'] for week in weeks if not week['ready']],
        'policy': 'strict',
    }


def fetch_bootstrap() -> dict:
    response = requests.get(f'{FPL_API_BASE}/bootstrap-static/', timeout=20)
    response.raise_for_status()
    return response.json()


def capture_inputs(database: Path, root: Path, bootstrap: dict | None = None) -> dict:
    context = current_context(database)
    bootstrap = fetch_bootstrap() if bootstrap is None else bootstrap
    events = {int(event['id']): event for event in bootstrap['events']}
    event = events[context['target_gameweek']]
    if _timestamp(events[1]['deadline_time'], 'Season start').year != int(context['season'][:4]):
        raise ValueError('Official events and the persisted database belong to different seasons.')
    if _timestamp(_now(), 'Current time') >= _timestamp(event['deadline_time'], 'Deadline'):
        raise ValueError(
            f"GW{event['id']} deadline has passed; refusing a retrospective capture. "
            'Refresh after the next completed Gameweek and capture before its target deadline.'
        )
    if any(
        not (events[gw].get('finished') and events[gw].get('data_checked'))
        for gw in range(1, int(event['id']))
    ):
        raise ValueError('Prior Gameweeks must be officially finished and data-checked before capture.')
    destination = root / context['season'] / f"gw{int(event['id']):02d}"
    if destination.exists():
        raise FileExistsError('A frozen bundle already exists; it will not be overwritten.')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.capture-', dir=destination.parent) as temporary:
        stage = Path(temporary) / 'bundle'
        stage.mkdir()
        target = stage / 'inputs.sqlite'
        with closing(sqlite3.connect(target, uri=True)) as conn:
            conn.execute('ATTACH DATABASE ? AS original', (
                database.resolve().as_uri() + '?mode=ro',
            ))
            conn.execute('BEGIN')
            copied = []
            for table in PUBLIC_TABLES:
                source = conn.execute(
                    "SELECT sql FROM original.sqlite_master WHERE type='table' AND name=?",
                    (table,),
                ).fetchone()
                if source is not None:
                    conn.execute(source[0])
                    conn.execute(
                        f'INSERT INTO main."{table}" SELECT * FROM original."{table}" ORDER BY rowid'
                    )
                    copied.append(table)
            conn.commit()
        manifest = {
            'schema_version': SNAPSHOT_VERSION, 'season': context['season'],
            'game_week': int(event['id']), 'deadline': event['deadline_time'],
            'captured_at': _now(), 'sha256': file_hash(target), 'public_tables': copied,
        }
        _validate_source(target, manifest)
        write_report(stage / 'manifest.json', manifest)
        os.rename(stage, destination)
    return manifest


def new_report(context: dict, models: list[str], seeds: int, through: int) -> dict:
    return {
        'schema_version': 1, 'status': 'running', 'created_at': _now(),
        'message': 'Checking strict historical inputs before training.',
        'season': context['season'], 'through': through, 'models': models,
        'seeds': seeds, 'seed_values': list(range(seeds)), 'policy': 'strict',
        'opposition_adjustment': 'disabled',
        'summary': [], 'weekly': [], 'transfers': [], 'initial_squads': {},
        'input_bundles': [],
        'implementation_sha256': {
            name: file_hash(ROOT / name)
            for name in (
                'utils.py', 'model_registry.py', 'squad_state.py',
                'train/trainer.py', 'train/advanced_models.py', 'train/rnn_trainer.py',
                'train/model_training.py', 'simulations/evaluate_season.py',
            )
        },
        'limitations': [
            'Random starting squads can be much weaker than ordinary manager squads.',
            'No chips, wildcards or paid transfers; recommendations use free transfers only.',
            'Ten starting squads share forecasts and are not ten independent model-validation samples.',
            'Retrospective results do not count toward prospective opposition-policy enablement.',
        ],
    }


def random_squad(pool: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ordered = pool.sort_values('id').reset_index(drop=True)
    groups = {
        position: ordered[ordered['element_type'] == position]
        for position in POSITION_LIMITS
    }
    if any(len(groups[position]) < count for position, count in POSITION_LIMITS.items()):
        raise ValueError('The eligible pool cannot form a complete random squad.')
    for _ in range(10000):
        squad = pd.concat([
            group.iloc[rng.choice(len(group), POSITION_LIMITS[position], replace=False)]
            for position, group in groups.items()
        ])
        if squad['value'].sum() <= 1000 and squad['team'].value_counts().max() <= 3:
            return squad.copy().reset_index(drop=True)
    raise ValueError('No legal random squad was sampled within the fixed attempt limit.')


def _owned_frame(state: dict, pool: pd.DataFrame) -> pd.DataFrame:
    saved = pd.DataFrame([{
        'player_id': player['id'], 'first_name': player['first_name'],
        'second_name': player['second_name'], 'element_type': player['position'],
        'team': player['team'], 'value': player['current_price'],
    } for player in state['players']])
    return score_squad_from_pool(saved, pool)


def advance_squad(
    state: dict, pool: pd.DataFrame, eligible: pd.DataFrame,
    gameweek: int, actuals: dict, allow_transfers: bool,
) -> tuple[dict, dict, list[dict]]:
    state = roll_to_game_week(state, gameweek, state['season'])
    state = refresh_player_data(state, pool)
    scored = _owned_frame(state, pool)
    moves = []
    if allow_transfers and gameweek > 1:
        prices = {player['id']: player['selling_price'] for player in state['players']}
        transfer_squad = scored.copy()
        transfer_squad['value'] = transfer_squad['id'].map(prices)
        budget = int(transfer_squad['value'].sum()) + int(state['bank'])
        moves = suggest_transfers(
            transfer_squad, eligible, transfer_limit=int(state['free_transfers']),
            max_per_team=3, max_spend=budget,
        )
        for move in moves:
            state = apply_transfer(state, move['in_'], int(move['out']['id']))
        scored = _owned_frame(state, pool)
    starters, bench, formation = pick_starting_xi(scored)
    ranked = starters.sort_values('predicted_points', ascending=False)
    captain, vice = int(ranked.iloc[0]['id']), int(ranked.iloc[1]['id'])
    state = set_lineup(
        state, starters['id'].astype(int).tolist(), bench['id'].astype(int).tolist(),
        captain, vice,
    )
    score = score_saved_squad(state, actuals)
    forecast = float(starters['predicted_points'].sum() + ranked.iloc[0]['predicted_points'])
    result = {
        'game_week': gameweek, 'predicted_points': forecast,
        'actual_points': score['total_points'], 'captain_bonus': score['captain_bonus'],
        'formation': formation, 'bank': state['bank'],
        'free_transfers': state['free_transfers'], 'transfer_count': len(moves),
        'transfer_cost': state['transfer_points_cost'],
        'captain': captain, 'vice_captain': vice,
        'starters': state['lineup']['starters'], 'bench': state['lineup']['bench'],
        'squad': [player['id'] for player in state['players']],
        'substitutions': score['substitutions'],
    }
    return state, result, moves


@contextmanager
def isolated_model_paths(directory: Path):
    names = [f'FPL_{model.upper()}_MODELS_FILE' for model in MODEL_TYPES]
    previous = {name: os.environ.get(name) for name in names}
    directory.mkdir(parents=True, exist_ok=True)
    try:
        for model, name in zip(MODEL_TYPES, names):
            os.environ[name] = str((directory / f'{model}.joblib').resolve())
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def train_and_score(database: Path, models: list[str], directory: Path) -> tuple[dict, set[int]]:
    import joblib
    from train.model_training import train_and_save_models

    with closing(sqlite3.connect(database)) as conn:
        metadata = dict(conn.execute('SELECT key, value FROM app_metadata'))
        roster = pd.read_sql_query(
            'SELECT id,status,chance_of_playing_next_round FROM players_raw', conn
        )
    rows, _ = load_or_build_feature_cache(database, build_features)
    cutoff = int(metadata['completed_internal_index'])
    if rows.loc[rows['Game_Week'] <= cutoff, 'Game_Week'].max() > cutoff:
        raise ValueError('Training rows exceed the completed cutoff.')
    eligible = {
        int(row.id) for row in roster.itertuples()
        if _rule_based(row.status, row.chance_of_playing_next_round)
    }
    pools = {}
    with isolated_model_paths(directory):
        selections = ['all'] if 'ensemble' in models else models
        for selection in selections:
            train_and_save_models(rows, cutoff, selection, db_file=database)
        for model in models:
            checkpoint = joblib.load(model_artifact_path(model))
            pools[model] = score_checkpoint_snapshot(
                rows, checkpoint, int(metadata['snapshot_game_week'])
            )
            if pools[model].attrs['opposition_adjustment']['status'] != 'disabled':
                raise ValueError('Season replay must not activate the opposition experiment.')
    return pools, eligible


def fetch_outcomes(season: str, through: int) -> tuple[dict, dict]:
    bootstrap = fetch_bootstrap()
    events = {int(event['id']): event for event in bootstrap['events']}
    if _timestamp(events[1]['deadline_time'], 'Season start').year != int(season[:4]):
        raise ValueError('Official outcomes belong to a different season.')
    actuals = {}
    for gameweek in range(1, through + 1):
        if not (events[gameweek].get('finished') and events[gameweek].get('data_checked')):
            raise ValueError(f'Official GW{gameweek} results are not finished and data-checked.')
        response = requests.get(f'{FPL_API_BASE}/event/{gameweek}/live/', timeout=20)
        response.raise_for_status()
        actuals[gameweek] = {}
        for player in response.json()['elements']:
            stats = player['stats']
            minutes = int(stats['minutes'])
            played = stats.get(
                'played', minutes > 0 or stats.get('yellow_cards', 0) > 0 or stats.get('red_cards', 0) > 0
            )
            actuals[gameweek][int(player['id'])] = {
                'points': int(stats['total_points']), 'minutes': minutes, 'played': bool(played),
            }
    return events, actuals


def _summarize(report: dict, predictions: list[dict], events: dict) -> None:
    weekly = pd.DataFrame(report['weekly'])
    player_rows = pd.DataFrame(predictions)
    for model in report['models']:
        rows = weekly[weekly['model'] == model]
        totals = rows.groupby('seed')[['actual_points', 'hold_points']].sum()
        predicted = player_rows[player_rows['model'] == model]
        error = predicted['predicted_points'] - predicted['actual_points']
        report['summary'].append({
            'model': model, 'mean_total_points': float(totals['actual_points'].mean()),
            'min_total_points': int(totals['actual_points'].min()),
            'max_total_points': int(totals['actual_points'].max()),
            'hold_mean_total': float(totals['hold_points'].mean()),
            'mean_transfer_gain': float((totals['actual_points'] - totals['hold_points']).mean()),
            'official_average_total': sum(
                int(events[gw]['average_entry_score']) for gw in range(1, report['through'] + 1)
            ),
            'player_rmse': float(np.sqrt(error.pow(2).mean())),
            'player_mae': float(error.abs().mean()),
            'player_bias': float(error.mean()), 'player_observations': len(predicted),
            'mean_transfers': float(rows.groupby('seed')['transfer_count'].sum().mean()),
        })


def run_evaluation(
    database: Path, root: Path, output: Path, models: list[str], seeds: int,
    through: int | None = None,
) -> dict:
    context = current_context(database)
    through = context['latest_completed'] if through is None else through
    validate_selection(models, seeds, through, context['latest_completed'])
    if output.exists():
        raise FileExistsError('Choose a new evaluation output directory; existing runs are immutable.')
    output.mkdir(parents=True)
    report = new_report(context, models, seeds, through)
    report['readiness'] = readiness(database, root, through)
    if not report['readiness']['ready']:
        report.update(status='blocked', completed_at=_now(),
                      message='Missing or invalid strict pre-deadline input bundles. No models trained.')
        write_report(output / 'report.json', report)
        return report
    report['message'] = 'Fetching checked official outcomes before model training.'
    write_report(output / 'report.json', report)
    try:
        events, outcomes = fetch_outcomes(context['season'], through)
        write_report(output / 'official_outcomes.json', {'events': events, 'actuals': outcomes})
        states: dict[tuple[str, int, str], dict] = {}
        predictions = []
        for gameweek in range(1, through + 1):
            print('SEASON_PROGRESS ' + json.dumps({
                'progress': round((gameweek - 1) / through * 95),
                'phase': f'GW{gameweek}/{through}: training selected models on prior data',
            }), flush=True)
            original = root / context['season'] / f'gw{gameweek:02d}'
            verified = inspect_bundle(root, context['season'], gameweek)
            if not verified['ready']:
                raise ValueError(f'GW{gameweek} bundle changed: {verified["reason"]}')
            manifest = json.loads((original / 'manifest.json').read_text(encoding='utf-8'))
            if manifest['deadline'] != events[gameweek]['deadline_time']:
                raise ValueError(f'GW{gameweek} bundle deadline differs from the official event.')
            report['input_bundles'].append({
                'game_week': gameweek, 'sha256': manifest['sha256'],
                'captured_at': manifest['captured_at'], 'deadline': manifest['deadline'],
            })
            working = output / f'gw{gameweek:02d}'
            working.mkdir()
            fold_database = working / 'inputs.sqlite'
            shutil.copy2(original / 'inputs.sqlite', fold_database)
            if file_hash(fold_database) != manifest['sha256']:
                raise ValueError('Copied replay database differs from its frozen bundle.')
            report['message'] = f'GW{gameweek}/{through}: retraining selected models on prior data.'
            write_report(output / 'report.json', report)
            pools, eligible_ids = train_and_score(fold_database, models, working / 'models')
            if gameweek == 1:
                first = pools[models[0]]
                eligible_first = first[first['id'].astype(int).isin(eligible_ids)]
                for seed in range(seeds):
                    report['initial_squads'][str(seed)] = (
                        random_squad(eligible_first, seed)['id'].astype(int).tolist()
                    )
            for model in models:
                pool = pools[model]
                if not set(pool['id'].astype(int)).issubset(outcomes[gameweek]):
                    raise ValueError('Official actuals do not cover the complete forecast pool.')
                eligible = pool[pool['id'].astype(int).isin(eligible_ids)].copy()
                for player in eligible.to_dict(orient='records'):
                    predictions.append({
                        'model': model, 'game_week': gameweek, 'player_id': int(player['id']),
                        'position': player['element_type'], 'predicted_points': float(player['predicted_points']),
                        'actual_points': outcomes[gameweek][int(player['id'])]['points'],
                    })
                for seed in range(seeds):
                    for strategy in ('advice', 'hold'):
                        key = (model, seed, strategy)
                        if gameweek == 1:
                            ids = report['initial_squads'][str(seed)]
                            squad = pool[pool['id'].astype(int).isin(ids)].copy()
                            if len(squad) != 15:
                                raise ValueError('Selected models do not cover the same initial squad.')
                            states[key] = create_state(squad, 1, context['season'], source='season_replay')
                            states[key]['free_transfers'] = 0
                        states[key], result, moves = advance_squad(
                            states[key], pool, eligible, gameweek, outcomes[gameweek],
                            allow_transfers=strategy == 'advice',
                        )
                        if strategy == 'advice':
                            row = {'model': model, 'seed': seed, **result}
                            for move in moves:
                                outgoing, incoming = int(move['out']['id']), int(move['in_']['id'])
                                report['transfers'].append({
                                    'model': model, 'seed': seed, 'game_week': gameweek,
                                    'out_id': outgoing, 'in_id': incoming,
                                    'predicted_gain': move['gain'],
                                    'actual_player_gain': outcomes[gameweek][incoming]['points']
                                    - outcomes[gameweek][outgoing]['points'],
                                    'bank_after': move['bank_after'],
                                })
                        else:
                            row['hold_points'] = result['actual_points']
                            row['official_average'] = int(events[gameweek]['average_entry_score'])
                            report['weekly'].append(row)
            write_report(output / 'report.json', report)
        _summarize(report, predictions, events)
        pd.DataFrame(predictions).to_csv(output / 'player_predictions.csv', index=False)
        pd.DataFrame(report['weekly']).to_csv(output / 'weekly_results.csv', index=False)
        pd.DataFrame(report['transfers']).to_csv(output / 'transfers.csv', index=False)
        report.update(status='succeeded', completed_at=_now(),
                      message='Completed strict, isolated current-season replay.')
        write_report(output / 'report.json', report)
        return report
    except Exception as exc:
        logging.getLogger(__name__).exception('Season evaluation failed')
        report.update(status='failed', completed_at=_now(), message=str(exc))
        write_report(output / 'report.json', report)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('check', 'capture', 'evaluate'))
    parser.add_argument('--database', type=Path, default=Path(DB_FILE))
    parser.add_argument('--snapshots', type=Path, default=snapshot_root())
    parser.add_argument('--output', type=Path)
    parser.add_argument('--models', nargs='+', choices=MODEL_TYPES, default=['ensemble'])
    parser.add_argument('--seeds', type=int, default=10)
    parser.add_argument('--through', type=int)
    args = parser.parse_args()
    if args.action == 'capture':
        result = capture_inputs(args.database, args.snapshots)
    elif args.action == 'check':
        result = readiness(args.database, args.snapshots, args.through)
    else:
        output = args.output or evaluation_root() / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')
        result = run_evaluation(
            args.database, args.snapshots, output, args.models, args.seeds, args.through
        )
    print(json.dumps(result, indent=2, allow_nan=False))
    return 2 if result.get('status') == 'blocked' else 0


if __name__ == '__main__':
    raise SystemExit(main())
