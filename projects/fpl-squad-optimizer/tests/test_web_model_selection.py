import importlib.util
import json
import sqlite3
import sys
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

spec = importlib.util.spec_from_file_location(
    'fpl_web_test', ROOT / 'fpl-generator' / 'web.py'
)
web = importlib.util.module_from_spec(spec)
spec.loader.exec_module(web)


class WebModelSelectionTests(unittest.TestCase):
    def test_template_labels_official_fpl_actions_as_read_only_or_local(self):
        template = (ROOT / 'fpl-generator' / 'templates' / 'index.html').read_text(
            encoding='utf-8'
        )

        self.assertIn('Public FPL entry (read-only)', template)
        self.assertIn('Import public picks', template)
        self.assertIn('Save locally', template)
        self.assertIn('Copy FPL plan', template)
        self.assertIn('Official FPL is unchanged.', template)

    def tearDown(self):
        if web._TRAINING_GUARD.locked():
            web._TRAINING_GUARD.release()
        web._update_training_state(
            status='idle',
            phase='Idle',
            progress=0,
            started_at=None,
            completed_at=None,
            message='No refresh is running.',
            log=[],
        )

    @staticmethod
    def _variation_pool():
        positions = [('GK', 6), ('DEF', 12), ('MID', 12), ('FWD', 8)]
        players = []
        player_id = 1
        for position, count in positions:
            for rank in range(count):
                players.append({
                    'id': player_id,
                    'element_type': position,
                    'team': (player_id % 15) + 1,
                    'value': 45 + rank % 4,
                    'predicted_points': 8.0 - rank * 0.08 - player_id / 10000,
                })
                player_id += 1
        return pd.DataFrame(players)

    def test_stochastic_squads_are_reproducible_by_seed_and_keep_quality_floor(self):
        pool = self._variation_pool()
        structure = {'GK': 2, 'DEF': 5, 'MID': 5, 'FWD': 3}
        kwargs = {
            'structure': structure,
            'max_per_team': 3,
            'max_spend': 1000,
            'uncertainty_by_position': {
                'GK': 1.5, 'DEF': 2.0, 'MID': 2.0, 'FWD': 2.2,
            },
            'quality_floor': 0.95,
            'noise_scale': 0.35,
            'attempts': 32,
        }
        first, first_meta = web.select_squad_variation(pool, seed=7, **kwargs)
        replay, replay_meta = web.select_squad_variation(pool, seed=7, **kwargs)

        self.assertEqual(sorted(first['id']), sorted(replay['id']))
        self.assertEqual(first_meta, replay_meta)
        self.assertGreaterEqual(first_meta['squad_quality_ratio'], 0.95)
        self.assertGreaterEqual(first_meta['xi_quality_ratio'], 0.95)
        self.assertEqual(first['element_type'].value_counts().to_dict(), structure)
        self.assertLessEqual(first.groupby('team').size().max(), 3)
        self.assertLessEqual(first['value'].sum(), 1000)

    def test_different_seeds_produce_multiple_near_optimal_squads(self):
        pool = self._variation_pool()
        squads = set()
        for seed in range(8):
            squad, metadata = web.select_squad_variation(
                pool,
                {'GK': 2, 'DEF': 5, 'MID': 5, 'FWD': 3},
                3,
                1000,
                {'GK': 1.5, 'DEF': 2.0, 'MID': 2.0, 'FWD': 2.2},
                seed,
                quality_floor=0.95,
                noise_scale=0.35,
                attempts=32,
            )
            self.assertGreaterEqual(metadata['xi_quality_ratio'], 0.95)
            squads.add(tuple(sorted(squad['id'])))

        self.assertGreater(len(squads), 1)

    def test_all_advisor_optimizer_returns_deterministic_legal_best_xi(self):
        pool = self._variation_pool()

        first_squad, first_lineup = web.select_best_advisor_side(pool, 3, 1000)
        second_squad, second_lineup = web.select_best_advisor_side(pool, 3, 1000)

        self.assertEqual(
            sorted(first_squad['id'].astype(int)),
            sorted(second_squad['id'].astype(int)),
        )
        self.assertEqual(first_lineup, second_lineup)
        self.assertEqual(len(first_squad), 15)
        self.assertEqual(len(first_lineup['starters']), 11)
        self.assertEqual(len(first_lineup['bench']), 4)
        self.assertLessEqual(int(first_squad['value'].sum()), 1000)
        self.assertLessEqual(int(first_squad.groupby('team').size().max()), 3)
        self.assertEqual(
            first_squad['element_type'].value_counts().to_dict(),
            {'DEF': 5, 'MID': 5, 'FWD': 3, 'GK': 2},
        )
        by_id = first_squad.set_index('id')
        starter_positions = by_id.loc[first_lineup['starters'], 'element_type'].value_counts()
        self.assertEqual(int(starter_positions['GK']), 1)
        self.assertGreaterEqual(int(starter_positions['DEF']), 3)
        self.assertGreaterEqual(int(starter_positions['MID']), 2)
        self.assertGreaterEqual(int(starter_positions['FWD']), 1)
        ranked = by_id.loc[first_lineup['starters']].sort_values(
            'predicted_points', ascending=False
        )
        self.assertEqual(first_lineup['captain'], int(ranked.iloc[0].name))
        self.assertEqual(first_lineup['vice_captain'], int(ranked.iloc[1].name))

    def test_variation_falls_back_explicitly_when_floor_allows_no_alternative(self):
        pool = self._variation_pool()
        squad, metadata = web.select_squad_variation(
            pool,
            {'GK': 2, 'DEF': 5, 'MID': 5, 'FWD': 3},
            3,
            1000,
            {'GK': 20.0, 'DEF': 20.0, 'MID': 20.0, 'FWD': 20.0},
            seed=1,
            quality_floor=1.0,
            noise_scale=1.0,
            attempts=1,
        )
        deterministic = web.select_squad(
            pool, {'GK': 2, 'DEF': 5, 'MID': 5, 'FWD': 3}, 3, 1000
        )
        self.assertEqual(sorted(squad['id']), sorted(deterministic['id']))
        self.assertEqual(metadata['fallback_reason'], 'no_qualified_variations')
        self.assertFalse(metadata['varied_from_optimum'])

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

    def test_advisor_run_forces_ensemble_checkpoint(self):
        checkpoint = {
            'model_type': 'ensemble',
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
        pool = self._variation_pool()
        with (
            web.app.test_request_context('/generate-team?advisor_run=1&model=xgboost'),
            patch.object(web.os.path, 'exists', return_value=True),
            patch.object(web, 'available_model_artifacts', return_value={
                'xgboost': Path('models.joblib'),
                'ensemble': Path('models_ensemble.joblib'),
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
            patch.object(web, 'load_current', return_value=None),
            patch.object(web, 'save_draft', side_effect=lambda state: state),
            patch.object(web, 'render_template', side_effect=lambda _, data: data),
        ):
            payload = json.loads(web.index())

        load.assert_called_once_with(Path('models_ensemble.joblib'))
        self.assertEqual(payload['model_type'], 'ensemble')
        self.assertEqual(payload['generation']['strategy'], 'all_model_advisors')
        self.assertTrue(payload['generation']['deterministic'])

    def test_training_start_is_local_only(self):
        response = web.app.test_client().post(
            '/api/training/start',
            json={},
            environ_base={'REMOTE_ADDR': '203.0.113.7'},
        )

        self.assertEqual(response.status_code, 403)
        self.assertFalse(response.get_json()['ok'])

    def test_training_start_rejects_duplicate_job(self):
        web._TRAINING_GUARD.acquire()

        response = web.app.test_client().post('/api/training/start', json={})

        self.assertEqual(response.status_code, 409)
        self.assertIn('already running', response.get_json()['error'])

    def test_training_start_launches_daemon_and_exposes_status(self):
        worker = MagicMock()
        with patch.object(web.threading, 'Thread', return_value=worker) as thread:
            response = web.app.test_client().post('/api/training/start', json={})

        self.assertEqual(response.status_code, 202)
        self.assertEqual(response.get_json()['status'], 'running')
        thread.assert_called_once_with(target=web._run_refresh_and_training, daemon=True)
        worker.start.assert_called_once_with()
        status = web.app.test_client().get('/api/training/status').get_json()
        self.assertEqual(status['status'], 'running')
        self.assertEqual(status['phase'], 'Queued')
        self.assertEqual(status['progress'], 1)
        self.assertIsNotNone(status['started_at'])

    def test_training_log_advances_progress_monotonically(self):
        web._update_training_state(status='running', phase='Queued', progress=1, log=[])

        web._append_training_log('Training catboost position models...')
        web._append_training_log('Feature cache: hit')

        status = web._training_snapshot()
        self.assertEqual(status['progress'], 68)
        self.assertEqual(status['phase'], 'Training CatBoost')
        self.assertEqual(status['log'][-1], 'Feature cache: hit')

    def test_player_history_log_reports_intra_phase_progress(self):
        web._update_training_state(status='running', phase='Queued', progress=10, log=[])

        web._append_training_log('FPL history download: 300/600 players')

        status = web._training_snapshot()
        self.assertEqual(status['progress'], 20)
        self.assertEqual(status['phase'], 'Downloading player histories (300/600)')

    def test_training_blocks_squad_writes(self):
        web._update_training_state(status='running')

        response = web.app.test_client().post('/api/squad/commit', json={})

        self.assertEqual(response.status_code, 409)
        self.assertIn('after it completes', response.get_json()['error'])

    def test_staged_training_publishes_database_and_all_models(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            live = root / 'live'
            stage = root / 'stage'
            live.mkdir()
            stage.mkdir()
            database = live / 'fpl.db'
            staged_database = stage / 'fpl.db'
            with closing(sqlite3.connect(database)) as connection:
                connection.execute('CREATE TABLE marker (value TEXT)')
                connection.execute("INSERT INTO marker VALUES ('old-db')")
                connection.commit()
            with closing(sqlite3.connect(staged_database)) as connection:
                connection.execute('CREATE TABLE marker (value TEXT)')
                connection.execute("INSERT INTO marker VALUES ('new-db')")
                connection.commit()
            targets = {}
            for model_type in web.MODEL_TYPES:
                target = live / f'{model_type}.joblib'
                target.write_text(f'old-{model_type}', encoding='utf-8')
                (stage / target.name).write_text(f'new-{model_type}', encoding='utf-8')
                targets[model_type] = target

            with closing(sqlite3.connect(database)) as active_reader:
                self.assertEqual(
                    active_reader.execute('SELECT value FROM marker').fetchone()[0],
                    'old-db',
                )
                with (
                    patch.object(web, 'DB_FILE', str(database)),
                    patch.object(web, 'model_artifact_path', side_effect=targets.get),
                ):
                    web._publish_staged_training(stage, staged_database)

            with closing(sqlite3.connect(database)) as connection:
                self.assertEqual(
                    connection.execute('SELECT value FROM marker').fetchone()[0],
                    'new-db',
                )
            for model_type, target in targets.items():
                self.assertEqual(
                    target.read_text(encoding='utf-8'), f'new-{model_type}'
                )


if __name__ == '__main__':
    unittest.main()
