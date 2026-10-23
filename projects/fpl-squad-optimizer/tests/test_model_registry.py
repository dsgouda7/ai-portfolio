import sys
import json
import os
import subprocess
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

import pandas as pd
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from model_registry import (
    build_checkpoint,
    build_training_manifest,
    validate_checkpoint_cutoff,
    model_artifact_path,
    score_checkpoint_snapshot,
    score_checkpoint_fixture_rows,
    checkpoint_fingerprint,
)


class ModelRegistryTests(unittest.TestCase):
    def test_checkpoint_fingerprint_is_stable_across_processes_and_tracks_components(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'ensemble.joblib'
            component = Path(directory) / 'component.joblib'
            path.write_bytes(b'frozen ensemble artifact')
            component.write_bytes(b'frozen component artifact')
            checkpoint = {
                'model_type': 'ensemble',
                'component_artifacts': {'a': component.name},
            }
            environment = {**os.environ, 'FPL_ENSEMBLE_MODELS_FILE': str(path)}
            root = str(Path(__file__).parent.parent)
            script = (
                f'import sys; sys.path.insert(0, {root!r}); '
                'from model_registry import checkpoint_fingerprint; '
                f'print(checkpoint_fingerprint({checkpoint!r}))'
            )
            hashes = [
                subprocess.check_output(
                    [sys.executable, '-c', script],
                    env={**environment, 'PYTHONHASHSEED': seed},
                    text=True,
                ).strip()
                for seed in ('1', '2')
            ]
            self.assertEqual(hashes[0], hashes[1])
            self.assertEqual(len(hashes[0]), 64)
            with patch.dict(os.environ, environment):
                self.assertEqual(checkpoint_fingerprint(checkpoint), hashes[0])
                component.write_bytes(b'changed component artifact')
                self.assertNotEqual(checkpoint_fingerprint(checkpoint), hashes[0])

    def test_ensemble_blends_each_fixture_before_gameweek_aggregation(self):
        base = pd.DataFrame({
            'id': [1, 1, 2], 'element_type': ['MID'] * 3,
            'scheduled_fixture': [10, 11, 10],
            'opponent_team': [2, 3, 2],
            'predicted_points': [0.0] * 3,
        })
        checkpoint = {
            'model_type': 'ensemble',
            'component_artifacts': {'a': 'a.joblib', 'b': 'b.joblib'},
            'ensemble_weights': {'MID': {'a': 0.25, 'b': 0.75}},
            'training_manifest': {
                'completed_internal_index': 1, 'max_training_internal_index': 1,
            },
        }
        component = {'training_manifest': {'completed_internal_index': 1}}
        with (
            patch('joblib.load', side_effect=[component, component]),
            patch('model_registry.score_checkpoint_fixture_rows', side_effect=[
                base.assign(predicted_points=[2.0, 4.0, 6.0]),
                base.assign(predicted_points=[4.0, 6.0, 8.0]),
            ]),
        ):
            result = score_checkpoint_fixture_rows(pd.DataFrame(), checkpoint, 2)
        self.assertEqual(result['predicted_points'].tolist(), [3.5, 5.5, 7.5])

    def test_validated_policy_is_applied_once_and_preserves_fixture_audit(self):
        from test_opposition import fixture_pool, evidence

        rows = fixture_pool()
        rows.loc[
            rows['team'].eq(2) & rows['element_type'].isin(['MID', 'FWD']),
            'predicted_points',
        ] = 12.0
        checkpoint = {'model_type': 'xgboost'}
        report = {
            'opposition_version': 1,
            'model_type': 'xgboost',
            'checkpoint_fingerprint': 'frozen',
            'adjustment': {'strength': 0.5, 'max_reduction': 0.25},
            'gameweeks': [evidence(gw) for gw in range(1, 6)],
        }
        with tempfile.TemporaryDirectory() as directory:
            policy = Path(directory) / 'policy.json'
            policy.write_text(json.dumps(report), encoding='utf-8')
            with (
                patch('model_registry.score_checkpoint_fixture_rows', return_value=rows),
                patch('model_registry.opposition_policy_path', return_value=policy),
                patch('model_registry.checkpoint_fingerprint', return_value='frozen'),
            ):
                result = score_checkpoint_snapshot(pd.DataFrame(), checkpoint, 2)
                self.assertEqual(result.attrs['opposition_adjustment']['status'], 'active')
                defender = result.loc[result['id'].eq(100)].iloc[0]
                self.assertLess(defender['predicted_points'], defender['opposition_base_points'])
                self.assertEqual(len(defender['opposition_fixtures']), 1)
                self.assertAlmostEqual(
                    defender['predicted_points'],
                    defender['opposition_base_points'] * defender['opposition_factor'],
                )
                report['checkpoint_fingerprint'] = 'old'
                policy.write_text(json.dumps(report), encoding='utf-8')
                with self.assertLogs('model_registry', level='WARNING'):
                    stale = score_checkpoint_snapshot(pd.DataFrame(), checkpoint, 2)
                self.assertEqual(stale.attrs['opposition_adjustment']['status'], 'stale')
                self.assertNotIn('opposition_base_points', stale)
                report['gameweeks'] = report['gameweeks'][:1]
                policy.write_text(json.dumps(report), encoding='utf-8')
                with self.assertRaisesRegex(ValueError, 'rollout gate'):
                    score_checkpoint_snapshot(pd.DataFrame(), checkpoint, 2)

    def test_adjusted_double_gameweek_sums_both_fixtures_and_blank_stays_zero(self):
        from model_registry import _aggregate_gameweek_predictions
        from opposition import OppositionAdjustment, apply_opposition_adjustment
        from test_opposition import fixture_pool

        first = fixture_pool()
        second = first.copy()
        second['scheduled_fixture'] += 100
        second.loc[
            second['team'].eq(2) & second['element_type'].isin(['MID', 'FWD']),
            'predicted_points',
        ] = 12.0
        blank = first.iloc[[0]].copy()
        blank['id'], blank['opponent_team'], blank['scheduled_fixture'] = 9999, 0, 0
        adjusted = apply_opposition_adjustment(
            pd.concat([first, second, blank], ignore_index=True),
            OppositionAdjustment(),
        )
        result = _aggregate_gameweek_predictions(adjusted).set_index('id')
        self.assertEqual(result.loc[100, 'fixture_count'], 2)
        self.assertEqual(len(result.loc[100, 'opposition_fixtures']), 2)
        self.assertAlmostEqual(
            result.loc[100, 'predicted_points'],
            adjusted.loc[adjusted['id'].eq(100), 'predicted_points'].sum(),
        )
        self.assertEqual(result.loc[9999, 'predicted_points'], 0.0)
        self.assertEqual(result.loc[9999, 'opposition_fixtures'], [])

    def test_every_model_type_has_a_distinct_artifact(self):
        paths = {
            model_artifact_path(model_type)
            for model_type in ('xgboost', 'catboost', 'lambdarank', 'rnn', 'ensemble')
        }
        self.assertEqual(len(paths), 5)

    def test_ensemble_averages_component_point_predictions(self):
        base = pd.DataFrame({
            'id': [1, 2], 'element_type': ['MID', 'MID'],
            'predicted_points': [0.0, 0.0],
        })
        component_pools = [
            base.assign(predicted_points=[2.0, 4.0]),
            base.assign(predicted_points=[4.0, 6.0]),
        ]
        checkpoint = {
            'model_type': 'ensemble',
            'component_artifacts': {'a': 'a.joblib', 'b': 'b.joblib'},
            'ensemble_weights': {'MID': {'a': 0.25, 'b': 0.75}},
            'training_manifest': {
                'completed_internal_index': 1,
                'max_training_internal_index': 1,
            },
        }
        component_checkpoint = {
            'training_manifest': {'completed_internal_index': 1}
        }
        with (
            patch('joblib.load', side_effect=[component_checkpoint, component_checkpoint]),
            patch('model_registry.score_checkpoint_fixture_rows', side_effect=component_pools),
        ):
            result = score_checkpoint_fixture_rows(pd.DataFrame(), checkpoint, 2)
        self.assertEqual(result['predicted_points'].tolist(), [3.5, 5.5])

    def test_snapshot_sums_double_gameweek_fixture_predictions(self):
        class FixtureModel:
            @staticmethod
            def predict(rows):
                return rows['fixture_score'].to_numpy()

        rows = pd.DataFrame({
            'id': [1, 1, 2, 3],
            'element_type': ['MID'] * 4,
            'Game_Week': [2] * 4,
            'fixture_score': [3.0, 4.0, 5.0, 9.0],
            'opponent_team': [2, 3, 4, 0],
            'kickoff_time': [
                '2026-12-26T12:00:00Z',
                '2026-12-29T19:30:00Z',
                '2026-12-27T14:00:00Z',
                '2026-12-27T14:00:00Z',
            ],
            'tm_market_value': [None] * 4,
        })
        checkpoint = {
            'model_type': 'xgboost',
            'models': {'MID': FixtureModel()},
            'pos_features': {'MID': ['fixture_score']},
            'training_manifest': {
                'completed_internal_index': 1,
                'max_training_internal_index': 1,
            },
        }

        result = score_checkpoint_snapshot(rows, checkpoint, 2).set_index('id')

        self.assertEqual(result.loc[1, 'fixture_count'], 2)
        self.assertEqual(result.loc[1, 'predicted_points'], 7.0)
        self.assertEqual(result.loc[2, 'fixture_count'], 1)
        self.assertEqual(result.loc[2, 'predicted_points'], 5.0)
        self.assertEqual(result.loc[3, 'fixture_count'], 0)
        self.assertEqual(result.loc[3, 'predicted_points'], 0.0)

    def test_manifest_stops_at_completed_cutoff(self):
        rows = pd.DataFrame({
            'Game_Week': [0, 1, 2],
            'target': [3.0, 4.0, 0.0],
            'value': [50, 51, 52],
            'kickoff_time': [
                '2026-08-01T12:00:00Z',
                '2026-08-08T12:00:00Z',
                '2026-08-15T12:00:00Z',
            ],
        })
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / 'manifest.db'
            import sqlite3
            with closing(sqlite3.connect(database)) as connection:
                pd.DataFrame([
                    {'key': 'season', 'value': '2026-27'},
                    {'key': 'latest_completed_game_week', 'value': '2'},
                    {'key': 'target_game_week', 'value': '3'},
                    {'key': 'snapshot_game_week', 'value': '2'},
                    {'key': 'completed_internal_index', 'value': '1'},
                ]).to_sql('app_metadata', connection, index=False)
                connection.commit()
            manifest = build_training_manifest(
                rows, 'xgboost', 1, database
            )

        self.assertEqual(manifest['max_training_internal_index'], 1)
        self.assertEqual(manifest['training_rows'], 2)
        self.assertEqual(manifest['temporal_fpl_price_rows'], 2)

    def test_invalid_checkpoint_cutoff_is_rejected(self):
        checkpoint = {'training_manifest': {
            'completed_internal_index': 3,
            'max_training_internal_index': 4,
        }}
        with self.assertRaisesRegex(ValueError, 'extends beyond'):
            validate_checkpoint_cutoff(checkpoint)

    def test_checkpoint_persists_model_type_and_manifest(self):
        manifest = {
            'completed_internal_index': 3,
            'max_training_internal_index': 3,
        }
        checkpoint = build_checkpoint('rnn', {}, {}, manifest)
        self.assertEqual(checkpoint['model_type'], 'rnn')
        self.assertIs(checkpoint['training_manifest'], manifest)

    def test_manifest_can_distinguish_raw_rows_from_rnn_samples(self):
        manifest = {
            'completed_internal_index': 3,
            'max_training_internal_index': 3,
            'training_rows': 5,
            'sequence_samples': 4,
        }
        checkpoint = build_checkpoint('rnn', {}, {}, manifest)
        self.assertEqual(checkpoint['training_manifest']['training_rows'], 5)
        self.assertEqual(checkpoint['training_manifest']['sequence_samples'], 4)


if __name__ == '__main__':
    unittest.main()
