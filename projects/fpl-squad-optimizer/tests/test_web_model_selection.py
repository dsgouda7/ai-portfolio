import importlib.util
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

spec = importlib.util.spec_from_file_location(
    'fpl_web_test', ROOT / 'fpl-generator' / 'web.py'
)
web = importlib.util.module_from_spec(spec)
spec.loader.exec_module(web)


class WebModelSelectionTests(unittest.TestCase):
    def test_squad_selection_is_repeatable_for_fixed_unique_scores(self):
        positions = [('GK', 4), ('DEF', 8), ('MID', 8), ('FWD', 5)]
        players = []
        player_id = 1
        for position, count in positions:
            for rank in range(count):
                players.append({
                    'id': player_id,
                    'element_type': position,
                    'team': (player_id % 10) + 1,
                    'value': 45 + (rank % 3),
                    'predicted_points': 20.0 - player_id / 100,
                })
                player_id += 1
        pool = pd.DataFrame(players)
        structure = {'GK': 2, 'DEF': 5, 'MID': 5, 'FWD': 3}

        squads = [
            tuple(sorted(web.select_squad(pool, structure, 3, 1000)['id']))
            for _ in range(5)
        ]

        self.assertEqual(len(set(squads)), 1)

    def test_requested_rnn_checkpoint_is_loaded(self):
        checkpoint = {
            'model_type': 'rnn',
            'models': {},
            'metrics': {},
            'epl_members': None,
            'training_manifest': {
                'completed_internal_index': 4,
                'max_training_internal_index': 4,
                'latest_completed_game_week': 5,
                'fpl_data_cutoff': '2026-09-01T12:00:00+00:00',
            },
        }
        pool = pd.DataFrame([{
            'id': 1,
            'first_name': 'Test',
            'second_name': 'Player',
            'element_type': 'MID',
            'team': 1,
            'value': 50,
            'predicted_points': 4.0,
        }])
        state = {
            'game_week': 6,
            'season': '2026-27',
            'bank': 0,
            'players': [],
            'lineup': {'starters': [], 'bench': []},
        }

        with (
            web.app.test_request_context('/generate-team?model=rnn'),
            patch.object(web.os.path, 'exists', return_value=True),
            patch.object(web, 'available_model_artifacts', return_value={
                'xgboost': Path('models.joblib'),
                'rnn': Path('models_rnn.joblib'),
            }),
            patch.object(web.joblib, 'load', return_value=checkpoint) as load,
            patch.object(web, 'build_features', return_value=pd.DataFrame()),
            patch.object(web, 'score_checkpoint_snapshot', return_value=pool),
            patch.object(web, 'get_runtime_context', return_value={
                'target_game_week': 6,
                'snapshot_game_week': 5,
                'season': '2026-27',
            }),
            patch.object(web, 'get_eligibility', return_value={}),
            patch.object(web, 'load_working_state', return_value=state),
            patch.object(web, 'roll_to_game_week', return_value=state),
            patch.object(web, 'refresh_player_data', return_value=state),
            patch.object(web, '_state_frame', return_value=pd.DataFrame()),
            patch.object(web, 'score_squad_from_pool', return_value=pool.iloc[0:0]),
            patch.object(web, '_lineup_frames', return_value=(
                pool.iloc[0:0], pool.iloc[0:0], '0-0-0'
            )),
            patch.object(web, 'suggest_transfer', return_value=None),
            patch.object(web, 'render_template', side_effect=lambda _, data: data),
        ):
            payload = json.loads(web.index())

        load.assert_called_once_with(Path('models_rnn.joblib'))
        self.assertEqual(payload['model_type'], 'rnn')
        self.assertEqual(payload['available_models'], ['xgboost', 'rnn'])


if __name__ == '__main__':
    unittest.main()
