import sys
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from model_registry import (
    build_checkpoint,
    build_training_manifest,
    validate_checkpoint_cutoff,
)


class ModelRegistryTests(unittest.TestCase):
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
