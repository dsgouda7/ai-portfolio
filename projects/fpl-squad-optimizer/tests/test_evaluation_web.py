import os
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import MagicMock, patch

from flask import Flask

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation_web import create_evaluation_blueprint
from model_registry import MODEL_TYPES
from simulations.evaluate_season import ROOT, write_report
from test_season_evaluation import make_source


class EvaluationWebTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.database = self.root / 'live.sqlite'
        make_source(self.database, target=3)
        self.environment = patch.dict(os.environ, {
            'FPL_SNAPSHOT_DIR': str(self.root / 'snapshots'),
            'FPL_EVALUATION_DIR': str(self.root / 'results'),
        })
        self.environment.start()
        self.guard = threading.Lock()
        self.app = Flask(__name__, template_folder=str(ROOT / 'fpl-generator' / 'templates'))
        self.app.register_blueprint(create_evaluation_blueprint(
            self.database, self.guard, threading.RLock()
        ))
        self.client = self.app.test_client()

    def tearDown(self):
        if self.guard.locked():
            self.guard.release()
        self.environment.stop()
        self.temporary.cleanup()

    def test_page_and_context_expose_models_and_strict_missing_inputs(self):
        page = self.client.get('/season-evaluation')
        self.assertEqual(page.status_code, 200)
        self.assertIn(b'Run season evaluation', page.data)
        self.assertIn(b'Capture current inputs', page.data)
        self.assertIn(b'id="metrics" class="hidden', page.data)
        data = self.client.get('/api/season-evaluation').get_json()
        self.assertEqual(data['models'], list(MODEL_TYPES))
        self.assertFalse(data['readiness']['ready'])
        self.assertEqual(data['readiness']['missing_gameweeks'], [1, 2])
        self.assertEqual(data['job']['status'], 'idle')

    def test_invalid_settings_and_nonlocal_requests_do_not_launch_jobs(self):
        for payload in [
            {'models': []}, {'models': ['unknown']}, {'seeds': True},
            {'seeds': 26}, {'through': 3}, {'output': 'outside'}, ['ensemble'],
        ]:
            with patch('evaluation_web.threading.Thread') as thread:
                response = self.client.post('/api/season-evaluation/start', json=payload)
            self.assertEqual(response.status_code, 400)
            thread.assert_not_called()
        self.assertEqual(self.client.post('/api/season-evaluation/start', data='{}').status_code, 415)
        malformed = self.client.post(
            '/api/season-evaluation/start', data='{invalid', content_type='application/json'
        )
        self.assertEqual(malformed.status_code, 400)
        self.assertEqual(malformed.get_json()['error'], 'Malformed JSON request.')
        self.assertEqual(self.client.post(
            '/api/season-evaluation/start', json={}, headers={'Origin': 'https://untrusted.example'}
        ).status_code, 403)
        self.assertEqual(self.client.post(
            '/api/season-evaluation/start', json={}, environ_overrides={'REMOTE_ADDR': '192.0.2.1'}
        ).status_code, 403)

    def test_compute_guard_prevents_concurrent_training_and_evaluation(self):
        self.guard.acquire()
        with patch('evaluation_web.threading.Thread') as thread:
            response = self.client.post('/api/season-evaluation/start', json={})
        self.assertEqual(response.status_code, 409)
        thread.assert_not_called()

    def test_worker_uses_selected_models_and_treats_blocked_as_not_success(self):
        thread = MagicMock()
        with patch('evaluation_web.threading.Thread', return_value=thread) as constructor:
            response = self.client.post('/api/season-evaluation/start', json={
                'models': ['xgboost', 'catboost'], 'seeds': 2, 'through': 2,
            })
        self.assertEqual(response.status_code, 202)
        thread.start.assert_called_once()
        worker = constructor.call_args.kwargs['target']
        arguments = constructor.call_args.kwargs['args']
        run_id = response.get_json()['run_id']
        directory = self.root / 'results' / run_id
        write_report(directory / 'report.json', {
            'status': 'blocked', 'message': 'Missing strict inputs.',
            'models': ['xgboost', 'catboost'], 'seeds': 2, 'through': 2,
            'season': '2026-27', 'summary': [], 'weekly': [],
        })
        process = MagicMock()
        process.stdout = iter(['SEASON_PROGRESS {"progress": 0, "phase": "Checking"}\n'])
        process.wait.return_value = 2
        with patch('evaluation_web.subprocess.Popen', return_value=process) as popen:
            worker(*arguments)
        command = popen.call_args.args[0]
        self.assertEqual(command[command.index('--models') + 1:], ['xgboost', 'catboost'])
        self.assertEqual(command[command.index('--database') + 1], str(self.database))
        self.assertEqual(popen.call_args.kwargs['env']['PYTHONIOENCODING'], 'utf-8')
        self.assertFalse(self.guard.locked())
        status = self.client.get('/api/season-evaluation/status').get_json()
        self.assertEqual(status['status'], 'blocked')
        self.assertEqual(status['progress'], 0)
        saved = self.client.get(f'/api/season-evaluation/runs/{run_id}').get_json()
        self.assertEqual(saved['report']['summary'], [])
        self.assertEqual(self.client.get('/api/season-evaluation').get_json()['runs'][0]['status'], 'blocked')

    def test_failed_process_releases_guard_and_persists_explicit_failure(self):
        with patch('evaluation_web.threading.Thread', return_value=MagicMock()) as thread:
            response = self.client.post('/api/season-evaluation/start', json={})
        process = MagicMock()
        process.stdout = iter(['The process failed.\n'])
        process.wait.return_value = 1
        with (
            patch('evaluation_web.subprocess.Popen', return_value=process),
            self.assertLogs('evaluation_web', level='ERROR'),
        ):
            thread.call_args.kwargs['target'](*thread.call_args.kwargs['args'])
        self.assertFalse(self.guard.locked())
        status = self.client.get('/api/season-evaluation/status').get_json()
        self.assertEqual(status['status'], 'failed')
        self.assertIn('without a report', status['message'])
        report = self.client.get(f"/api/season-evaluation/runs/{response.get_json()['run_id']}").get_json()
        self.assertEqual(report['report']['status'], 'failed')

    def test_capture_rejects_late_inputs_and_does_not_accept_output_paths(self):
        with patch('evaluation_web.capture_inputs', side_effect=ValueError('GW8 deadline has passed.')):
            response = self.client.post('/api/season-evaluation/capture', json={})
        self.assertEqual(response.status_code, 400)
        self.assertIn('deadline has passed', response.get_json()['error'])
        self.assertEqual(self.client.post(
            '/api/season-evaluation/capture', json={'output': 'elsewhere'}
        ).status_code, 400)

    def test_report_paths_and_prediction_downloads_are_scoped_to_completed_runs(self):
        self.assertEqual(self.client.get('/api/season-evaluation/runs/..').status_code, 400)
        directory = self.root / 'results' / 'example-run'
        write_report(directory / 'report.json', {
            'status': 'blocked', 'summary': [], 'weekly': [], 'models': ['xgboost'],
        })
        (directory / 'player_predictions.csv').write_text('id,points\n1,2\n', encoding='utf-8')
        url = '/api/season-evaluation/runs/example-run/predictions'
        self.assertEqual(self.client.get(url).status_code, 404)
        write_report(directory / 'report.json', {'status': 'succeeded'})
        response = self.client.get(url)
        self.assertEqual(response.status_code, 200)
        self.assertIn('attachment', response.headers['Content-Disposition'])
        response.close()


if __name__ == '__main__':
    unittest.main()
