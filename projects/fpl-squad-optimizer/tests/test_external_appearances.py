import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from external_appearances import attach_external_appearance_features


class ExternalAppearanceTests(unittest.TestCase):
    def test_future_appearance_cannot_leak_into_prior_fpl_row(self):
        rows = pd.DataFrame([
            {'code': 10, 'kickoff_time': '2026-08-10T12:00:00Z'},
            {'code': 10, 'kickoff_time': '2026-09-10T12:00:00Z'},
        ])
        appearances = pd.DataFrame([
            {'fpl_code': 10, 'appearance_date': '2026-08-01',
             'minutes_played': 90, 'goals': 1, 'assists': 0},
            {'fpl_code': 10, 'appearance_date': '2026-08-20',
             'minutes_played': 80, 'goals': 0, 'assists': 1},
        ])
        enriched = attach_external_appearance_features(rows, appearances)
        self.assertEqual(enriched.iloc[0]['external_minutes_90d'], 90)
        self.assertEqual(enriched.iloc[1]['external_minutes_90d'], 170)
        self.assertEqual(enriched.iloc[0]['external_goal_involvements_365d'], 1)
        self.assertEqual(enriched.iloc[1]['external_goal_involvements_365d'], 2)


if __name__ == '__main__':
    unittest.main()
