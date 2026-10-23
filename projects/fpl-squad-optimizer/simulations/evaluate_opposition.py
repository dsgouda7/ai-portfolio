"""Paired, post-cutoff evaluation; never retrains models or refreshes the database."""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import tempfile
from contextlib import closing
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eligibility import _rule_based
from feature_cache import FEATURE_CACHE_SCHEMA_VERSION, source_fingerprint
from model_registry import (
    MODEL_TYPES,
    _aggregate_gameweek_predictions,
    checkpoint_fingerprint,
    model_artifact_path,
    opposition_policy_path,
    score_checkpoint_fixture_rows,
)
from opposition import (
    OPPOSITION_VERSION,
    OppositionAdjustment,
    apply_opposition_adjustment,
    evaluation_gate,
)
from performance_reviewer import score_saved_squad
from squad_optimizer import select_best_advisor_side
from utils import DB_FILE, FPL_API_BASE, normalize_pool_scores


def _read_frozen_inputs(database: Path) -> tuple[pd.DataFrame, dict, dict, set[int]]:
    with closing(sqlite3.connect(database.resolve().as_uri() + '?mode=ro', uri=True)) as conn:
        conn.execute('PRAGMA query_only=ON')
        metadata = dict(conn.execute('SELECT key, value FROM app_metadata'))
        cursor = conn.execute('SELECT * FROM model_feature_cache_metadata')
        cache_row = cursor.fetchone()
        if cache_row is None:
            raise ValueError('A frozen feature cache is required; evaluation never rebuilds it.')
        cache = dict(zip([column[0] for column in cursor.description], cache_row))
        if (
            cache['schema_version'] != FEATURE_CACHE_SCHEMA_VERSION
            or cache['source_fingerprint'] != source_fingerprint(conn)
        ):
            raise ValueError('The feature cache is stale; refusing retrospective reconstruction.')
        rows = pd.read_sql_query(
            'SELECT * FROM model_feature_cache ORDER BY _cache_row_id', conn
        ).set_index('_cache_row_id').rename_axis(None)
        conn.row_factory = sqlite3.Row
        eligible = {
            int(row['id']) for row in conn.execute(
                'SELECT id, status, chance_of_playing_next_round FROM players_raw'
            )
            if _rule_based(row['status'], row['chance_of_playing_next_round'])
        }
    return rows, metadata, cache, eligible


def _actual_team_score(pool: pd.DataFrame, actuals: dict) -> dict[str, Any]:
    squad, lineup = select_best_advisor_side(pool, max_per_team=3, max_spend=1000)
    state = {
        'players': [
            {
                'id': int(row['id']), 'position': row['element_type'],
                'first_name': row.get('first_name', ''),
                'second_name': row.get('second_name', ''),
            }
            for row in squad.to_dict(orient='records')
        ],
        'lineup': lineup,
    }
    score = score_saved_squad(state, actuals)
    return {
        'points': score['total_points'],
        'captain_points': score['captain_bonus'],
        'captain': lineup['captain'],
        'starters': lineup['starters'],
        'bench': lineup['bench'],
        'substitutions': score['substitutions'],
    }


def compare_forecasts(
    fixtures: pd.DataFrame,
    actuals: dict[int, dict[str, Any]],
    eligible_ids: set[int],
    adjustment: OppositionAdjustment,
) -> dict[str, Any]:
    baseline = normalize_pool_scores(_aggregate_gameweek_predictions(fixtures))
    adjusted = normalize_pool_scores(_aggregate_gameweek_predictions(
        apply_opposition_adjustment(fixtures, adjustment)
    ))
    baseline = baseline[baseline['id'].astype(int).isin(eligible_ids)].copy()
    adjusted = adjusted[adjusted['id'].astype(int).isin(eligible_ids)].copy()
    if baseline.empty or not set(baseline['id'].astype(int)).issubset(actuals):
        raise ValueError('Official actuals do not cover the eligible forecast pool.')
    base_score = _actual_team_score(baseline, actuals)
    adjusted_score = _actual_team_score(adjusted, actuals)

    def squared_error(pool: pd.DataFrame) -> float:
        observed = pool['id'].map(lambda player_id: actuals[int(player_id)]['points'])
        return float((pool['predicted_points'] - observed).pow(2).sum())

    return {
        'baseline_points': base_score['points'],
        'adjusted_points': adjusted_score['points'],
        'baseline_captain_points': base_score['captain_points'],
        'adjusted_captain_points': adjusted_score['captain_points'],
        'baseline_squared_error': squared_error(baseline),
        'adjusted_squared_error': squared_error(adjusted),
        'player_count': len(baseline),
        'baseline_team': base_score,
        'adjusted_team': adjusted_score,
        'dampened_players': int(adjusted['opposition_factor'].lt(1).sum()),
        'minimum_factor': float(adjusted['opposition_factor'].min()),
    }


def _write_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, suffix='.tmp')
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8') as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write('\n')
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=MODEL_TYPES, default='ensemble')
    parser.add_argument('--gameweek', type=int, required=True)
    parser.add_argument('--database', type=Path, default=Path(DB_FILE))
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--strength', type=float, default=0.5)
    parser.add_argument('--max-reduction', type=float, default=0.25)
    parser.add_argument('--publish-if-passed', action='store_true')
    parser.add_argument('--register', action='store_true',
                        help='Freeze forecasts before the deadline without reading actuals.')
    parser.add_argument('--forecast', type=Path,
                        help='Frozen forecast JSON: output with --register, input otherwise.')
    args = parser.parse_args()
    adjustment = OppositionAdjustment(args.strength, args.max_reduction)
    frozen = None
    if args.forecast is not None and not args.register:
        frozen = json.loads(args.forecast.read_text(encoding='utf-8'))
        if (
            frozen['model_type'] != args.model
            or frozen['opposition_version'] != OPPOSITION_VERSION
            or frozen['adjustment'] != asdict(adjustment)
            or frozen['provenance']['game_week'] != args.gameweek
        ):
            raise ValueError('Frozen forecast does not match the requested candidate/Gameweek.')
        fingerprint = frozen['checkpoint_fingerprint']
    else:
        checkpoint = joblib.load(model_artifact_path(args.model))
        if checkpoint.get('model_type') != args.model:
            raise ValueError('Configured checkpoint does not match the requested model family.')
        fingerprint = checkpoint_fingerprint(checkpoint)
    previous = []
    registered_at = (
        frozen['candidate_registered_at'] if frozen
        else datetime.now(timezone.utc).isoformat(timespec='seconds')
    )
    if args.report.exists():
        existing = json.loads(args.report.read_text(encoding='utf-8'))
        if (
            existing['checkpoint_fingerprint'] != fingerprint
            or existing['adjustment'] != asdict(adjustment)
            or existing['opposition_version'] != OPPOSITION_VERSION
        ):
            raise ValueError('An evaluation ledger cannot mix checkpoints or candidate settings.')
        previous = existing['gameweeks']
        registered_at = existing['candidate_registered_at']
    response = requests.get(f'{FPL_API_BASE}/bootstrap-static/', timeout=20)
    response.raise_for_status()
    bootstrap = response.json()
    events = {int(event['id']): event for event in bootstrap['events']}
    event = events[args.gameweek]
    season_year = pd.Timestamp(events[1]['deadline_time']).year
    if frozen:
        provenance = dict(frozen['provenance'])
        if provenance['deadline'] != event['deadline_time']:
            raise ValueError('Frozen and official deadlines do not match.')
        forecasts = pd.DataFrame(frozen['fixture_forecasts'])
        eligible = set(frozen['eligible_ids'])
    else:
        rows, metadata, cache, eligible = _read_frozen_inputs(args.database)
        if args.gameweek != int(metadata['target_game_week']):
            raise ValueError('Only the persisted pre-deadline inference snapshot can be evaluated.')
        snapshot = int(metadata['snapshot_game_week'])
        manifest = checkpoint['training_manifest']
        if args.gameweek <= int(manifest['latest_completed_game_week']):
            raise ValueError('The requested Gameweek was already available during model training.')
        provenance = {
            'season': metadata['season'],
            'game_week': args.gameweek,
            'snapshot_index': snapshot,
            'training_cutoff': int(manifest['completed_internal_index']),
            'trained_at': manifest['trained_at'],
            'features_created_at': cache['created_at'],
            'candidate_registered_at': registered_at,
            'forecasts_frozen_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
            'deadline': event['deadline_time'],
        }
        forecasts = score_checkpoint_fixture_rows(rows, checkpoint, snapshot)
        if checkpoint_fingerprint(checkpoint) != fingerprint:
            raise ValueError('Checkpoint artifacts changed while forecasts were being scored.')
        provenance['forecasts_frozen_at'] = datetime.now(timezone.utc).isoformat(timespec='seconds')
    if int(str(provenance['season']).split('-')[0]) != season_year:
        raise ValueError('The live actuals and frozen forecasts belong to different seasons.')
    provenance['finished'] = bool(event.get('finished') and event.get('data_checked'))
    # Reject late or in-sample inputs before spending time on model inference.
    evaluation_gate([{
        **provenance, 'finished': True, 'baseline_points': 0, 'adjusted_points': 0,
        'baseline_captain_points': 0, 'adjusted_captain_points': 0,
        'baseline_squared_error': 0, 'adjusted_squared_error': 0, 'player_count': 1,
    }])
    if args.register:
        if args.forecast is None:
            raise ValueError('--register requires an explicit --forecast output path.')
        if args.forecast.exists():
            raise ValueError('Refusing to overwrite a frozen forecast.')
        if pd.Timestamp(provenance['forecasts_frozen_at']) >= pd.Timestamp(event['deadline_time']):
            raise ValueError('Forecast registration must finish before the FPL deadline.')
        columns = [
            column for column in (
                'id', 'first_name', 'second_name', 'element_type', 'team', 'value',
                'opponent_team', 'fixture', 'scheduled_fixture', 'kickoff_time',
                'predicted_points',
            ) if column in forecasts
        ]
        # Validate coverage now, not only after the Gameweek has finished.
        apply_opposition_adjustment(forecasts, adjustment)
        header = {
            'opposition_version': OPPOSITION_VERSION,
            'model_type': args.model,
            'checkpoint_fingerprint': fingerprint,
            'adjustment': asdict(adjustment),
            'candidate_registered_at': registered_at,
        }
        _write_report(args.forecast, {
            **header, 'provenance': provenance,
            'eligible_ids': sorted(eligible),
            'fixture_forecasts': json.loads(forecasts[columns].to_json(orient='records', double_precision=15)),
        })
        _write_report(args.report, {
            **header, 'gameweeks': previous, 'gate': evaluation_gate(previous),
        })
        print(f'Frozen {len(forecasts)} fixture forecasts before {event["deadline_time"]}.')
        print(f'Forecast: {args.forecast}\nEvaluation ledger: {args.report}')
        print('Opposition remains disabled until completed untouched-gameweek evidence passes.')
        return
    if not (event.get('finished') and event.get('data_checked')):
        raise ValueError('Evaluation requires official finished, data-checked results.')
    response = requests.get(f'{FPL_API_BASE}/event/{args.gameweek}/live/', timeout=20)
    response.raise_for_status()
    actuals = {
        int(row['id']): {
            'points': int(row['stats']['total_points']),
            'minutes': int(row['stats']['minutes']),
            'played': int(row['stats']['minutes']) > 0,
        }
        for row in response.json()['elements']
    }
    result = {**provenance, **compare_forecasts(forecasts, actuals, eligible, adjustment)}
    gameweeks = [*previous, result]
    gate = evaluation_gate(gameweeks)
    report = {
        'opposition_version': OPPOSITION_VERSION,
        'model_type': args.model,
        'checkpoint_fingerprint': fingerprint,
        'adjustment': asdict(adjustment),
        'candidate_registered_at': registered_at,
        'evaluated_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'gameweeks': gameweeks,
        'gate': gate,
    }
    _write_report(args.report, report)
    print(json.dumps({'gameweek': args.gameweek, 'comparison': result, 'gate': gate}, indent=2))
    if args.publish_if_passed and gate['passed']:
        path = opposition_policy_path(args.model)
        _write_report(path, report)
        print(f'Published validated opposition policy: {path}')
    else:
        print('No policy published. Existing production predictions remain unchanged.')


if __name__ == '__main__':
    main()
