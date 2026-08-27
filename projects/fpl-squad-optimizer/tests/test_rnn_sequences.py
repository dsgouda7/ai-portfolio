import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from train.rnn_trainer import build_position_sequences, score_rnn_snapshot
from utils import POS_FEATURES


class RnnSequenceTests(unittest.TestCase):
    def test_future_rows_do_not_change_cutoff_sequences(self):
        features = POS_FEATURES['FWD']
        rows = []
        for game_week in range(4):
            row = {feature: float(game_week) for feature in features}
            row.update({
                'id': 9,
                'element_type': 'FWD',
                'Game_Week': game_week,
                'target': float(game_week + 1),
            })
            rows.append(row)
        original = pd.DataFrame(rows)
        changed = original.copy()
        changed.loc[changed['Game_Week'] == 3, features] = 9999.0

        first = build_position_sequences(original, 'FWD', cutoff=2)[0]
        second = build_position_sequences(changed, 'FWD', cutoff=2)[0]

        self.assertEqual(first.tolist(), second.tolist())
        self.assertEqual(first.shape, (3, 3, len(features)))

    def test_checkpoint_feature_order_controls_sequences(self):
        rows = pd.DataFrame([{
            'id': 9,
            'element_type': 'FWD',
            'Game_Week': 0,
            'target': 1.0,
            'value': 50.0,
            'team': 3.0,
        }])
        sequences = build_position_sequences(
            rows, 'FWD', cutoff=0, features=['team', 'value']
        )[0]
        self.assertEqual(sequences.shape, (1, 1, 2))

    def test_sequence_uses_all_available_gameweeks(self):
        features = ['value', 'team']
        rows = pd.DataFrame([
            {
                'id': 9,
                'element_type': 'FWD',
                'Game_Week': game_week,
                'target': float(game_week),
                'value': 50.0 + game_week,
                'team': 3.0,
            }
            for game_week in range(12)
        ])

        sequences, lengths, _, _, _ = build_position_sequences(
            rows, 'FWD', cutoff=11, features=features
        )

        self.assertEqual(sequences.shape, (12, 12, 2))
        self.assertEqual(lengths.tolist(), list(range(1, 13)))

    def test_duplicate_rows_in_same_gameweek_are_not_recurrent_history(self):
        rows = pd.DataFrame([
            {
                'id': 9,
                'element_type': 'FWD',
                'Game_Week': game_week,
                'target': float(game_week),
                'value': value,
                'team': 3.0,
            }
            for game_week, value in [(0, 50.0), (1, 51.0), (1, 999.0), (2, 52.0)]
        ])

        sequences, lengths, _, _, _ = build_position_sequences(
            rows, 'FWD', cutoff=2, features=['team', 'value']
        )

        self.assertEqual(sequences.shape, (3, 3, 2))
        self.assertEqual(lengths.tolist(), [1, 2, 3])

    def test_inference_rejects_fixed_window_checkpoint(self):
        checkpoint = {
            'training_manifest': {'sequence_policy': 'fixed_window'},
            'models': {},
        }
        with self.assertRaisesRegex(ValueError, 'all-available'):
            score_rnn_snapshot(pd.DataFrame(), checkpoint, 2)

    def test_missing_market_value_uses_mask_and_finite_scaler(self):
        rows = pd.DataFrame([
            {
                'id': 9,
                'element_type': 'FWD',
                'Game_Week': game_week,
                'target': float(game_week),
                'tm_market_value': market_value,
                'tm_value_available': float(market_value is not None),
            }
            for game_week, market_value in [(0, None), (1, 7.0), (2, 7.3)]
        ])
        sequences, _, _, _, scaler = build_position_sequences(
            rows,
            'FWD',
            cutoff=2,
            features=['tm_market_value', 'tm_value_available'],
        )
        self.assertTrue(np.isfinite(scaler.mean_).all())
        self.assertTrue(np.isfinite(scaler.scale_).all())
        self.assertTrue(np.isfinite(sequences).all())
        self.assertAlmostEqual(
            float(sequences[0, 0, 1]),
            float(scaler.transform([[7.0, 0.0]])[0, 1]),
            places=6,
        )


if __name__ == '__main__':
    unittest.main()
