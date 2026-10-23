import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from opposition import (
    OppositionAdjustment,
    apply_opposition_adjustment,
    evaluation_gate,
    projected_club_xi,
)


def fixture_pool():
    rows = []
    for team, opponent in [(1, 2), (2, 1), (3, 4), (4, 3)]:
        for index, position in enumerate(['GK'] + ['DEF'] * 4 + ['MID'] * 4 + ['FWD'] * 2):
            rows.append({
                'id': team * 100 + index,
                'team': team,
                'opponent_team': opponent,
                'scheduled_fixture': 10 if team < 3 else 20,
                'element_type': position,
                'predicted_points': 4.0,
            })
    return pd.DataFrame(rows)


def evidence(gameweek, gain=1):
    return {
        'season': '2026-27', 'game_week': gameweek,
        'baseline_points': 50, 'adjusted_points': 50 + gain,
        'baseline_captain_points': 4, 'adjusted_captain_points': 4,
        'baseline_squared_error': 200, 'adjusted_squared_error': 190,
        'player_count': 44, 'finished': True,
        'snapshot_index': gameweek + 40, 'training_cutoff': 40,
        'trained_at': '2026-08-01T00:00:00Z',
        'features_created_at': '2026-08-02T00:00:00Z',
        'candidate_registered_at': '2026-08-02T00:00:00Z',
        'forecasts_frozen_at': '2026-08-03T00:00:00Z',
        'deadline': '2026-09-01T10:00:00Z',
    }


class OppositionTests(unittest.TestCase):
    def test_equal_opposition_does_not_change_scores(self):
        pool = fixture_pool()
        adjusted = apply_opposition_adjustment(pool, OppositionAdjustment())
        pd.testing.assert_series_equal(adjusted['predicted_points'], pool['predicted_points'])

    def test_strong_attack_dampens_opposing_defence_only(self):
        pool = fixture_pool()
        strong_attack = pool['team'].eq(2) & pool['element_type'].isin(['MID', 'FWD'])
        pool.loc[strong_attack, 'predicted_points'] = 12.0
        adjusted = apply_opposition_adjustment(pool, OppositionAdjustment())
        team = adjusted[adjusted['team'] == 1]
        self.assertTrue(team.loc[team['element_type'].isin(['GK', 'DEF']), 'opposition_factor'].lt(1).all())
        self.assertTrue(team.loc[team['element_type'].isin(['MID', 'FWD']), 'opposition_factor'].eq(1).all())
        pd.testing.assert_series_equal(pool['predicted_points'], adjusted['opposition_base_points'], check_names=False)

    def test_strong_defence_dampens_opposing_attack(self):
        pool = fixture_pool()
        pool.loc[pool['team'].eq(2) & pool['element_type'].isin(['GK', 'DEF']), 'predicted_points'] = 12.0
        adjusted = apply_opposition_adjustment(pool, OppositionAdjustment())
        attack = adjusted['team'].eq(1) & adjusted['element_type'].isin(['MID', 'FWD'])
        self.assertTrue(adjusted.loc[attack, 'opposition_factor'].lt(1).all())

    def test_factor_is_bounded_and_negative_points_are_not_improved(self):
        pool = fixture_pool()
        pool.loc[pool['team'].eq(2), 'predicted_points'] = 1000.0
        pool.loc[pool['id'].eq(100), 'predicted_points'] = -1.0
        adjusted = apply_opposition_adjustment(pool, OppositionAdjustment(1.0, 0.1))
        self.assertGreaterEqual(adjusted['opposition_factor'].min(), 0.9)
        self.assertEqual(adjusted.loc[adjusted['id'].eq(100), 'predicted_points'].iloc[0], -1.0)

    def test_no_circular_or_double_adjustment(self):
        adjusted = apply_opposition_adjustment(fixture_pool(), OppositionAdjustment())
        with self.assertRaisesRegex(ValueError, 'twice'):
            apply_opposition_adjustment(adjusted, OppositionAdjustment())

    def test_does_not_read_actual_points_or_actual_minutes(self):
        pool = fixture_pool()
        changed = pool.assign(total_points=99, minutes=0)
        first = apply_opposition_adjustment(pool, OppositionAdjustment())
        second = apply_opposition_adjustment(changed, OppositionAdjustment())
        pd.testing.assert_series_equal(first['predicted_points'], second['predicted_points'])

    def test_projected_club_xi_respects_formation_limits(self):
        players = fixture_pool().query('team == 1')
        extra = players.iloc[[1]].copy()
        extra['id'] = 999
        extra['predicted_points'] = 100.0
        selected = projected_club_xi(pd.concat([players, extra]))
        self.assertEqual(len(selected), 11)
        self.assertEqual(selected['element_type'].value_counts()['GK'], 1)
        self.assertLessEqual(selected['element_type'].value_counts()['DEF'], 5)
        self.assertIn(999, selected['id'].tolist())

    def test_missing_opposition_is_an_explicit_error(self):
        pool = fixture_pool().query('team != 2')
        with self.assertRaisesRegex(ValueError, 'both opposition'):
            apply_opposition_adjustment(pool, OppositionAdjustment())

    def test_incomplete_club_coverage_is_an_explicit_error(self):
        pool = fixture_pool().query('id != 100')
        with self.assertRaisesRegex(ValueError, 'legal projected club XI'):
            apply_opposition_adjustment(pool, OppositionAdjustment())

    def test_bad_parameters_and_nonfinite_predictions_are_rejected(self):
        for strength, reduction in [(0, 0.25), (float('nan'), 0.25), (0.5, 1)]:
            with self.assertRaises(ValueError):
                OppositionAdjustment(strength, reduction)
        pool = fixture_pool()
        pool.loc[0, 'predicted_points'] = float('nan')
        with self.assertRaisesRegex(ValueError, 'finite'):
            apply_opposition_adjustment(pool, OppositionAdjustment())

    def test_one_good_gameweek_cannot_pass_rollout_gate(self):
        self.assertFalse(evaluation_gate([evidence(1)])['passed'])

    def test_retrospective_candidate_is_diagnostic_not_untouched_evidence(self):
        rows = [evidence(gw) for gw in range(1, 6)]
        for row in rows:
            row['candidate_registered_at'] = '2026-09-02T00:00:00Z'
        gate = evaluation_gate(rows)
        self.assertFalse(gate['passed'])
        self.assertEqual(gate['gameweeks'], 0)
        self.assertEqual(gate['diagnostic_gameweeks'], 5)

    def test_gate_requires_team_gain_and_no_rmse_or_captain_regression(self):
        rows = [evidence(gw) for gw in range(1, 6)]
        self.assertTrue(evaluation_gate(rows)['passed'])
        rows[0]['adjusted_squared_error'] = 10000
        self.assertFalse(evaluation_gate(rows)['passed'])
        rows[0]['adjusted_squared_error'] = 190
        rows[0]['adjusted_captain_points'] = 0
        self.assertFalse(evaluation_gate(rows)['passed'])
        self.assertFalse(evaluation_gate([evidence(gw, 0) for gw in range(1, 6)])['passed'])

    def test_gate_rejects_duplicate_in_sample_or_late_features(self):
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            evaluation_gate([evidence(1), evidence(1)])
        row = evidence(1)
        row['snapshot_index'] = row['training_cutoff']
        with self.assertRaisesRegex(ValueError, 'post-training'):
            evaluation_gate([row])
        row = evidence(1)
        row['features_created_at'] = row['deadline']
        with self.assertRaisesRegex(ValueError, 'precede'):
            evaluation_gate([row])


if __name__ == '__main__':
    unittest.main()
