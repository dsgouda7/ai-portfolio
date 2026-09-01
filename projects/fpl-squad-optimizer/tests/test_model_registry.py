import sys
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
)


class ModelRegistryTests(unittest.TestCase):
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
            patch('model_registry.score_checkpoint_snapshot', side_effect=component_pools),
        ):
            result = score_checkpoint_snapshot(pd.DataFrame(), checkpoint, 2)
        self.assertEqual(result['predicted_points'].tolist(), [3.5, 5.5])

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
