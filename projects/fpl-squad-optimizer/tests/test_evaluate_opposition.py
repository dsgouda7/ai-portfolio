import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from simulations.evaluate_opposition import main
from test_opposition import fixture_pool


class OppositionRegistrationTests(unittest.TestCase):
    def test_registration_freezes_forecasts_without_reading_results_or_publishing(self):
        with tempfile.TemporaryDirectory() as directory:
            forecast = Path(directory) / 'forecast.json'
            report = Path(directory) / 'report.json'
            argv = [
                'evaluate_opposition.py', '--model', 'ensemble', '--gameweek', '2',
                '--register', '--forecast', str(forecast), '--report', str(report),
            ]
            checkpoint = {
                'model_type': 'ensemble',
                'training_manifest': {
                    'latest_completed_game_week': 1,
                    'completed_internal_index': 1,
                    'trained_at': '2026-01-01T00:00:00Z',
                },
            }
            rows = fixture_pool()
            response = MagicMock()
            response.json.return_value = {'events': [
                {'id': 1, 'deadline_time': '2099-08-01T10:00:00Z'},
                {
                    'id': 2, 'deadline_time': '2099-08-08T10:00:00Z',
                    'finished': False, 'data_checked': False,
                },
            ]}
            with (
                patch.object(sys, 'argv', argv),
                patch('simulations.evaluate_opposition.joblib.load', return_value=checkpoint),
                patch('simulations.evaluate_opposition.checkpoint_fingerprint', return_value='frozen'),
                patch('simulations.evaluate_opposition._read_frozen_inputs', return_value=(
                    rows, {'season': '2099-00', 'target_game_week': '2', 'snapshot_game_week': '2'},
                    {'created_at': '2026-01-02T00:00:00Z'}, set(rows['id']),
                )),
                patch('simulations.evaluate_opposition.score_checkpoint_fixture_rows', return_value=rows),
                patch('simulations.evaluate_opposition.requests.get', return_value=response) as request,
                patch('simulations.evaluate_opposition._actual_team_score') as score,
                patch('simulations.evaluate_opposition.opposition_policy_path') as policy,
                redirect_stdout(io.StringIO()),
            ):
                main()
                saved = json.loads(forecast.read_text(encoding='utf-8'))
                ledger = json.loads(report.read_text(encoding='utf-8'))
                self.assertFalse(saved['provenance']['finished'])
                self.assertEqual(len(saved['fixture_forecasts']), len(rows))
                self.assertEqual(saved['checkpoint_fingerprint'], 'frozen')
                self.assertFalse(ledger['gate']['passed'])
                self.assertEqual(ledger['gameweeks'], [])
                self.assertEqual(request.call_count, 1)
                self.assertNotIn('/live/', request.call_args.args[0])
                score.assert_not_called()
                policy.assert_not_called()
                original = forecast.read_bytes()
                with self.assertRaisesRegex(ValueError, 'overwrite'):
                    main()
                self.assertEqual(forecast.read_bytes(), original)

    def test_past_deadline_registration_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            forecast = Path(directory) / 'forecast.json'
            report = Path(directory) / 'report.json'
            checkpoint = {
                'model_type': 'ensemble',
                'training_manifest': {
                    'latest_completed_game_week': 1, 'completed_internal_index': 1,
                    'trained_at': '1999-01-01T00:00:00Z',
                },
            }
            response = MagicMock()
            response.json.return_value = {'events': [
                {'id': 1, 'deadline_time': '2000-08-01T10:00:00Z'},
                {
                    'id': 2, 'deadline_time': '2000-08-08T10:00:00Z',
                    'finished': True, 'data_checked': True,
                },
            ]}
            rows = fixture_pool()
            with (
                patch.object(sys, 'argv', [
                    'evaluate_opposition.py', '--gameweek', '2', '--register',
                    '--forecast', str(forecast), '--report', str(report),
                ]),
                patch('simulations.evaluate_opposition.joblib.load', return_value=checkpoint),
                patch('simulations.evaluate_opposition.checkpoint_fingerprint', return_value='frozen'),
                patch('simulations.evaluate_opposition._read_frozen_inputs', return_value=(
                    rows, {'season': '2000-01', 'target_game_week': '2', 'snapshot_game_week': '2'},
                    {'created_at': '1999-01-02T00:00:00Z'}, set(rows['id']),
                )),
                patch('simulations.evaluate_opposition.score_checkpoint_fixture_rows', return_value=rows),
                patch('simulations.evaluate_opposition.requests.get', return_value=response),
            ):
                with self.assertRaisesRegex(ValueError, 'before the FPL deadline'):
                    main()
            self.assertFalse(forecast.exists())
            self.assertFalse(report.exists())


if __name__ == '__main__':
    unittest.main()
