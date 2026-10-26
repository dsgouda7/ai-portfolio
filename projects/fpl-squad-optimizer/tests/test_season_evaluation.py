from contextlib import closing
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from feature_cache import persist_feature_cache
from simulations.evaluate_season import (
    capture_inputs, file_hash, inspect_bundle,
    random_squad, run_evaluation, train_and_score, validate_selection,
    write_report,
)
from utils import build_features


def pool_for_test():
    rows = []
    for position, count in {'GK': 4, 'DEF': 10, 'MID': 10, 'FWD': 6}.items():
        for _ in range(count):
            player_id = len(rows) + 1
            rows.append({
                'id': player_id, 'element_type': position, 'team': (player_id - 1) % 10 + 1,
                'first_name': 'Test', 'second_name': str(player_id), 'value': 50,
                'predicted_points': player_id / 5, 'predicted_points_norm': player_id / 5,
            })
    return pd.DataFrame(rows)


def make_source(path: Path, target=1, season='2026-27'):
    pool = pool_for_test()
    first = datetime(int(season[:4]), 8, 21, 17, 30, tzinfo=timezone.utc)
    deadline = first + timedelta(days=(target - 1) * 7)
    refreshed = deadline - timedelta(days=1)
    metadata = {
        'season': season, 'latest_completed_game_week': str(target - 1),
        'target_game_week': str(target), 'live_start_index': '1',
        'completed_internal_index': str(target - 1), 'snapshot_game_week': str(target),
        'fpl_refreshed_at': refreshed.isoformat(),
    }
    raw = pool[['id', 'first_name', 'second_name', 'team']].copy()
    raw['element_type'] = pool['element_type'].map({'GK': 1, 'DEF': 2, 'MID': 3, 'FWD': 4})
    raw['code'] = raw['id'] + 1000
    raw['birth_date'] = '1990-01-01'
    raw['status'] = 'a'
    raw['chance_of_playing_next_round'] = None
    rows = []
    for week in range(target + 1):
        for player in pool.to_dict(orient='records'):
            future = week == target
            kickoff = (
                first + timedelta(days=(week - 1) * 7 + 1)
                if week > 0 else first - timedelta(days=100)
            )
            rows.append({
                'player_id': player['id'], 'element': player['id'], 'Game_Week': week,
                'round': week, 'kickoff_time': kickoff.isoformat(),
                'fixture': 0 if future else week + 10, 'scheduled_fixture': week + 10,
                'minutes': 0 if future else 90,
                'total_points': 0 if future else 2 + player['id'] % 4,
                'opponent_team': 11 - player['team'], 'was_home': 1, 'value': 50,
            })
    with closing(sqlite3.connect(path)) as conn:
        raw.to_sql('players_raw', conn, index=False)
        pd.DataFrame(rows).to_sql('player_gw', conn, index=False)
        pd.DataFrame(list(metadata.items()), columns=['key', 'value']).to_sql('app_metadata', conn, index=False)
    features = build_features(str(path))
    persist_feature_cache(path, features)
    with closing(sqlite3.connect(path)) as conn:
        conn.execute('UPDATE model_feature_cache_metadata SET created_at=?', (refreshed.isoformat(),))
        conn.commit()
    return metadata, deadline, refreshed


def make_bundle(root: Path, gameweek=1):
    directory = root / '2026-27' / f'gw{gameweek:02d}'
    directory.mkdir(parents=True)
    source = directory / 'inputs.sqlite'
    _, deadline, refreshed = make_source(source, target=gameweek)
    manifest = {
        'schema_version': 1, 'season': '2026-27', 'game_week': gameweek,
        'deadline': deadline.isoformat(), 'captured_at': (refreshed + timedelta(hours=1)).isoformat(),
        'sha256': file_hash(source),
    }
    write_report(directory / 'manifest.json', manifest)
    return directory, manifest


def refresh_bundle(directory: Path, manifest: dict):
    database = directory / 'inputs.sqlite'
    persist_feature_cache(database, build_features(str(database)))
    created = pd.Timestamp(manifest['captured_at']) - pd.Timedelta(seconds=1)
    with closing(sqlite3.connect(database)) as conn:
        conn.execute('UPDATE model_feature_cache_metadata SET created_at=?', (created.isoformat(),))
        conn.commit()
    manifest['sha256'] = file_hash(database)
    write_report(directory / 'manifest.json', manifest)


class SeasonEvaluationTests(unittest.TestCase):
    def test_missing_inputs_block_without_training_or_accuracy_metrics(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            database = root / 'live.sqlite'
            make_source(database, target=3)
            before = file_hash(database)
            with patch('simulations.evaluate_season.train_and_score') as trainer:
                report = run_evaluation(database, root / 'snapshots', root / 'run', ['ensemble'], 10)
            trainer.assert_not_called()
            self.assertEqual(report['status'], 'blocked')
            self.assertEqual(report['summary'], [])
            self.assertEqual(report['readiness']['missing_gameweeks'], [1, 2])
            self.assertEqual(file_hash(database), before)
            self.assertTrue((root / 'run' / 'report.json').exists())

    def test_random_squads_are_seeded_legal_and_do_not_use_forecasts(self):
        pool = pool_for_test()
        first = random_squad(pool, 0)
        self.assertEqual(first['id'].tolist(), random_squad(pool, 0)['id'].tolist())
        self.assertNotEqual(first['id'].tolist(), random_squad(pool, 1)['id'].tolist())
        changed = pool.assign(predicted_points=-999, predicted_points_norm=999)
        self.assertEqual(first['id'].tolist(), random_squad(changed, 0)['id'].tolist())
        self.assertEqual(first['element_type'].value_counts().to_dict(),
                         {'DEF': 5, 'MID': 5, 'FWD': 3, 'GK': 2})
        self.assertLessEqual(first['value'].sum(), 1000)
        self.assertLessEqual(first['team'].value_counts().max(), 3)

    def test_invalid_selection_is_rejected(self):
        for models, seeds, through in [
            ([], 10, 1), (['unknown'], 10, 1), (['xgboost', 'xgboost'], 10, 1),
            (['xgboost'], True, 1), (['xgboost'], 26, 1), (['xgboost'], 10, 3),
        ]:
            with self.assertRaises(ValueError):
                validate_selection(models, seeds, through, 2)

    def test_snapshot_hash_late_capture_and_target_outcomes_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory, manifest = make_bundle(root)
            self.assertTrue(inspect_bundle(root, '2026-27', 1)['ready'])
            manifest['captured_at'] = manifest['deadline']
            write_report(directory / 'manifest.json', manifest)
            self.assertIn('before the deadline', inspect_bundle(root, '2026-27', 1)['reason'])
            manifest['captured_at'] = '2026-08-20T18:30:00+00:00'
            with closing(sqlite3.connect(directory / 'inputs.sqlite')) as conn:
                conn.execute('UPDATE player_gw SET total_points=99 WHERE Game_Week=1')
                conn.commit()
            write_report(directory / 'manifest.json', manifest)
            self.assertIn('SHA-256', inspect_bundle(root, '2026-27', 1)['reason'])
            refresh_bundle(directory, manifest)
            self.assertIn('target-Gameweek outcomes', inspect_bundle(root, '2026-27', 1)['reason'])

    def test_numeric_string_outcomes_are_validated_as_numbers(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory, manifest = make_bundle(root)
            with closing(sqlite3.connect(directory / 'inputs.sqlite')) as conn:
                conn.execute('ALTER TABLE player_gw ADD COLUMN expected_goals TEXT')
                conn.execute("UPDATE player_gw SET expected_goals='0.0'")
                conn.commit()
            refresh_bundle(directory, manifest)
            self.assertTrue(inspect_bundle(root, '2026-27', 1)['ready'])
            with closing(sqlite3.connect(directory / 'inputs.sqlite')) as conn:
                conn.execute("UPDATE player_gw SET expected_goals='1.2' WHERE Game_Week=1")
                conn.commit()
            refresh_bundle(directory, manifest)
            self.assertIn('target-Gameweek outcomes', inspect_bundle(root, '2026-27', 1)['reason'])

    def test_capture_excludes_private_tables_and_never_overwrites(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / 'live.sqlite'
            make_source(source, season='2099-00')
            with closing(sqlite3.connect(source)) as conn:
                conn.execute('CREATE TABLE fpl_entry_sync (private_marker TEXT)')
                conn.execute('CREATE TABLE squad_versions (private_marker TEXT)')
                conn.commit()
            bootstrap = {'events': [{'id': 1, 'deadline_time': '2099-08-21T17:30:00Z'}]}
            with patch('simulations.evaluate_season._now', return_value='2099-08-20T19:30:00Z'):
                manifest = capture_inputs(source, root / 'snapshots', bootstrap)
                with self.assertRaises(FileExistsError):
                    capture_inputs(source, root / 'snapshots', bootstrap)
            captured = root / 'snapshots' / '2099-00' / 'gw01' / 'inputs.sqlite'
            with closing(sqlite3.connect(captured)) as conn:
                tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            self.assertNotIn('fpl_entry_sync', tables)
            self.assertNotIn('squad_versions', tables)
            self.assertTrue(inspect_bundle(root / 'snapshots', '2099-00', 1)['ready'])
            self.assertEqual(file_hash(captured), manifest['sha256'])
            with (
                patch('simulations.evaluate_season._now', return_value='2100-01-01T00:00:00Z'),
                self.assertRaisesRegex(ValueError, 'deadline has passed'),
            ):
                capture_inputs(source, root / 'late', bootstrap)

    def test_real_xgboost_training_and_scoring_are_isolated(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            database = root / 'inputs.sqlite'
            make_source(database)
            before = {key: value for key, value in os.environ.items() if key.startswith('FPL_')}
            pools, eligible = train_and_score(database, ['xgboost'], root / 'models')
            after = {key: value for key, value in os.environ.items() if key.startswith('FPL_')}
            self.assertEqual(before, after)
            self.assertEqual(set(pools), {'xgboost'})
            self.assertEqual(len(eligible), 30)
            self.assertEqual(pools['xgboost'].attrs['opposition_adjustment']['status'], 'disabled')
            self.assertTrue((root / 'models' / 'xgboost.joblib').exists())

    def test_ensemble_selection_trains_all_components_in_the_isolated_folder(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            database = root / 'inputs.sqlite'
            make_source(database)
            pools, eligible = train_and_score(
                database, ['xgboost', 'ensemble'], root / 'models'
            )
            self.assertEqual(set(pools), {'xgboost', 'ensemble'})
            self.assertEqual(len(eligible), 30)
            self.assertEqual(len(pools['ensemble']), 30)
            self.assertFalse(pools['ensemble']['predicted_points'].isna().any())
            for model in ('xgboost', 'catboost', 'lambdarank', 'rnn', 'ensemble'):
                self.assertTrue((root / 'models' / f'{model}.joblib').exists())

    def test_valid_replay_uses_same_initial_squads_and_carries_free_transfers(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / 'live.sqlite'
            make_source(source, target=3)
            snapshots = root / 'snapshots'
            _, first = make_bundle(snapshots, 1)
            _, second = make_bundle(snapshots, 2)
            events = {
                gw: {'deadline_time': bundle['deadline'], 'average_entry_score': 50}
                for gw, bundle in [(1, first), (2, second)]
            }
            pool = pool_for_test()
            actuals = {
                gw: {int(player_id): {'points': 2, 'minutes': 90, 'played': True} for player_id in pool['id']}
                for gw in (1, 2)
            }
            def trainer(database, models, directory):
                self.assertNotEqual(database, source)
                self.assertTrue(database.is_relative_to(root / 'run'))
                self.assertTrue(directory.is_relative_to(root / 'run'))
                self.assertEqual(database.parent, directory.parent)
                running = json.loads((root / 'run' / 'report.json').read_text(encoding='utf-8'))
                self.assertEqual(running['status'], 'running')
                self.assertIn('retraining selected models', running['message'])
                self.assertEqual(running['summary'], [])
                return {model: pool.copy() for model in models}, set(pool['id'])
            hashes = {
                path: file_hash(path) for path in [source, *snapshots.glob('*/gw*/inputs.sqlite')]
            }
            with (
                patch('simulations.evaluate_season.train_and_score', side_effect=trainer) as train,
                patch('simulations.evaluate_season.fetch_outcomes', return_value=(events, actuals)),
            ):
                report = run_evaluation(source, snapshots, root / 'run', ['xgboost', 'catboost'], 2)
            self.assertEqual(train.call_count, 2)
            self.assertEqual(report['status'], 'succeeded')
            self.assertEqual(len(report['weekly']), 8)
            self.assertEqual(len(report['summary']), 2)
            self.assertEqual(report['summary'][0]['player_observations'], 60)
            self.assertEqual(len(report['input_bundles']), 2)
            for row in report['weekly']:
                self.assertEqual(row['actual_points'], 24)
                self.assertEqual(row['hold_points'], 24)
                self.assertEqual(row['transfer_cost'], 0)
                if row['game_week'] == 1:
                    self.assertEqual(row['transfer_count'], 0)
                    self.assertEqual(set(row['squad']), set(report['initial_squads'][str(row['seed'])]))
                else:
                    self.assertEqual(row['free_transfers'], 1)
                    self.assertLessEqual(row['transfer_count'], 1)
            for path, digest in hashes.items():
                self.assertEqual(file_hash(path), digest)
            self.assertTrue((root / 'run' / 'player_predictions.csv').exists())


if __name__ == '__main__':
    unittest.main()
