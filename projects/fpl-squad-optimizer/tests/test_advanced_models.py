import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from train.advanced_models import train_catboost_models, train_lambdarank_models
from utils import POS_FEATURES


class AdvancedModelTests(unittest.TestCase):
    @staticmethod
    def _rows():
        rows = []
        player_id = 1
        for position in ('GK', 'DEF', 'MID', 'FWD'):
            for gameweek in range(3):
                for rank in range(8):
                    row = {
                        feature: float((rank + gameweek) % 5)
                        for feature in POS_FEATURES[position]
                    }
                    row.update({
                        'id': player_id,
                        'element_type': position,
                        'Game_Week': gameweek,
                        'target': float(rank % 6),
                    })
                    rows.append(row)
                    player_id += 1
        return pd.DataFrame(rows)

    def test_catboost_models_fit_and_predict(self):
        rows = self._rows()
        models, metrics = train_catboost_models(rows, 2)
        predictions = models['MID'].predict(rows[rows['element_type'] == 'MID'])
        self.assertEqual(set(models), {'GK', 'DEF', 'MID', 'FWD'})
        self.assertTrue(np.isfinite(predictions).all())
        self.assertEqual(metrics['MID']['model_name'].split()[-1], '(CatBoost)')

    def test_lambdarank_models_fit_and_return_point_scale_predictions(self):
        rows = self._rows()
        models, metrics = train_lambdarank_models(rows, 2)
        predictions = models['DEF'].predict(rows[rows['element_type'] == 'DEF'])
        self.assertEqual(set(models), {'GK', 'DEF', 'MID', 'FWD'})
        self.assertTrue(np.isfinite(predictions).all())
        self.assertGreaterEqual(predictions.min(), rows['target'].min())
        self.assertLessEqual(predictions.max(), rows['target'].max())
        self.assertIn('LambdaRank', metrics['DEF']['model_name'])


if __name__ == '__main__':
    unittest.main()
