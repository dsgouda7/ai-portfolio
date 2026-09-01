import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import _add_fixture_strength_features, _add_temporal_context_features


class EnrichedFeatureTests(unittest.TestCase):
    def test_market_momentum_and_price_features_use_current_and_prior_rows(self):
        rows = pd.DataFrame([
            {'id': 1, 'Game_Week': 0, 'kickoff_time': '2026-08-01T12:00:00Z',
             'selected': 100, 'transfers_in': 10, 'transfers_out': 2,
             'transfers_balance': 8, 'value': 50},
            {'id': 1, 'Game_Week': 1, 'kickoff_time': '2026-08-08T12:00:00Z',
             'selected': 120, 'transfers_in': 30, 'transfers_out': 5,
             'transfers_balance': 25, 'value': 51},
        ])
        enriched = _add_temporal_context_features(rows)
        self.assertEqual(enriched.iloc[1]['price_change_1'], 1)
        self.assertGreater(enriched.iloc[1]['ownership_change_log'], 0)
        self.assertEqual(enriched.iloc[1]['rest_days'], 7)
        self.assertEqual(enriched.iloc[1]['matches_previous_14d'], 1)

    def test_fixture_strength_never_uses_current_result(self):
        rows = pd.DataFrame([
            {'id': 1, 'fixture': 10, 'Game_Week': 0, 'kickoff_time': '2026-08-01T12:00:00Z',
             'opponent_team': 2, 'was_home': True, 'team_h_score': 2,
             'team_a_score': 0, 'team': 1},
            {'id': 2, 'fixture': 10, 'Game_Week': 0, 'kickoff_time': '2026-08-01T12:00:00Z',
             'opponent_team': 1, 'was_home': False, 'team_h_score': 2,
             'team_a_score': 0, 'team': 2},
            {'id': 1, 'fixture': 11, 'Game_Week': 1, 'kickoff_time': '2026-08-08T12:00:00Z',
             'opponent_team': 2, 'was_home': True, 'team_h_score': 1,
             'team_a_score': 1, 'team': 1},
            {'id': 2, 'fixture': 11, 'Game_Week': 1, 'kickoff_time': '2026-08-08T12:00:00Z',
             'opponent_team': 1, 'was_home': False, 'team_h_score': 1,
             'team_a_score': 1, 'team': 2},
        ])
        changed = rows.copy()
        changed.loc[changed['fixture'] == 11, 'team_h_score'] = 9
        changed.loc[changed['fixture'] == 11, 'team_a_score'] = 0
        first = _add_fixture_strength_features(rows)
        second = _add_fixture_strength_features(changed)
        columns = ['team_elo_pre', 'opponent_elo_pre', 'elo_difference']
        self.assertEqual(
            first[first['fixture'] == 11][columns].values.tolist(),
            second[second['fixture'] == 11][columns].values.tolist(),
        )


if __name__ == '__main__':
    unittest.main()
