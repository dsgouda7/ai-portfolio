import importlib.util
import sqlite3
import sys
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from fpl_account import load_private_team, sync_public_entry

spec = importlib.util.spec_from_file_location(
    'fpl_web_account_test', ROOT / 'fpl-generator' / 'web.py'
)
web = importlib.util.module_from_spec(spec)
spec.loader.exec_module(web)


class FplAccountSyncTests(unittest.TestCase):
    @patch('fpl_account.requests.get')
    def test_private_team_uses_temporary_bearer_token_without_returning_it(self, get):
        get.side_effect = [self._response({'player': {'entry': 123}}), self._response({
            'picks': [
                {
                    'element': player_id,
                    'position': position,
                    'purchase_price': 50 + position,
                    'selling_price': 50,
                    'multiplier': 2 if position == 1 else 1,
                    'is_captain': position == 1,
                    'is_vice_captain': position == 2,
                }
                for position, player_id in enumerate(range(1, 16), start=1)
            ],
            'transfers': {'bank': 12, 'value': 988, 'limit': 2, 'made': 1},
        })]

        private = load_private_team(123, 'temporary-secret-token')

        self.assertEqual(get.call_count, 2)
        for expected_url, call in zip((
            'https://fantasy.premierleague.com/api/me/',
            'https://fantasy.premierleague.com/api/my-team/123/',
        ), get.call_args_list):
            self.assertEqual(call.args[0], expected_url)
            self.assertEqual(call.kwargs, {
                'timeout': 20,
                'headers': {
                    'X-API-Authorization': 'Bearer temporary-secret-token',
                    'X-API-Language': 'en',
                },
            })
        self.assertEqual(len(private['picks']), 15)
        self.assertEqual(private['picks'][0]['player_id'], 1)
        self.assertEqual(private['acquisition_costs'][1], 51)
        self.assertEqual(private['bank'], 12)
        self.assertEqual(private['next_free_transfers'], 1)
        self.assertNotIn('temporary-secret-token', repr(private))

    @patch('fpl_account.requests.get')
    def test_private_team_rejects_token_for_different_entry(self, get):
        get.return_value = self._response({'player': {'entry': 999}})

        with self.assertRaisesRegex(
            web.FplEntrySyncError,
            'belongs to FPL entry 999, not 123',
        ):
            load_private_team(123, 'temporary-secret-token')

        self.assertEqual(get.call_count, 1)

    def test_sync_route_rejects_credentials(self):
        web.app.config['TESTING'] = True
        with web.app.test_client() as client:
            response = client.post('/api/fpl-entry/sync', json={
                'entry_id': 123,
                'password': 'must-not-be-accepted',
            })
        self.assertEqual(response.status_code, 400)
        self.assertIn('Email/password and unknown fields are not accepted', response.get_json()['error'])

    def test_sync_route_rejects_non_json_and_cross_origin_requests(self):
        web.app.config['TESTING'] = True
        with web.app.test_client() as client:
            non_json = client.post('/api/fpl-entry/sync', data='entry_id=123')
            cross_origin = client.post(
                '/api/fpl-entry/sync',
                json={'entry_id': 123},
                headers={'Origin': 'https://example.invalid'},
            )
            remote = client.post(
                '/api/fpl-entry/sync',
                json={'entry_id': 123},
                environ_base={'REMOTE_ADDR': '203.0.113.7'},
            )
        self.assertEqual(non_json.status_code, 415)
        self.assertEqual(cross_origin.status_code, 403)
        self.assertEqual(remote.status_code, 403)
        self.assertIn('localhost', remote.get_json()['error'])

    def test_sync_route_uses_private_token_without_persisting_it(self):
        public = {
            'entry_id': 123,
            'latest_event': 3,
            'squad_event': 3,
            'gameweeks': [{'event': 3}],
            'picks': [],
        }
        private = {
            'picks': [
                {
                    'player_id': player_id,
                    'pick_position': position,
                    'is_captain': position == 1,
                    'is_vice_captain': position == 2,
                }
                for position, player_id in enumerate(range(1, 16), start=1)
            ],
            'acquisition_costs': {},
            'bank': 10,
            'squad_value': 990,
            'next_free_transfers': 1,
        }
        stored = {'official_entry': {'entry_id': 123, 'private': True}}
        captured = {}

        def build_state(synced, *args, **kwargs):
            captured['synced'] = synced
            return stored

        web._SYNC_LAST_REQUEST.clear()
        with (
            patch.object(web, 'available_model_artifacts', return_value={
                'xgboost': Path('models.joblib'),
            }),
            patch.object(web.joblib, 'load', return_value={}),
            patch.object(web, 'validate_checkpoint_cutoff'),
            patch.object(web, 'get_runtime_context', return_value={
                'target_game_week': 4,
                'snapshot_game_week': 40,
                'season': '2026-27',
            }),
            patch.object(web, 'load_or_build_feature_cache', return_value=(pd.DataFrame(), {})),
            patch.object(web, 'score_checkpoint_snapshot', return_value=pd.DataFrame()),
            patch.object(web, 'sync_public_entry', return_value=public),
            patch.object(web, 'load_private_team', return_value=private) as load_private,
            patch.object(web, 'load_event_player_points', return_value={}),
            patch.object(web, '_state_from_synced_entry', side_effect=build_state),
            patch.object(web, 'save_draft', return_value=stored) as save_draft,
        ):
            response = web.app.test_client().post('/api/fpl-entry/sync', json={
                'entry_id': 123,
                'model': 'xgboost',
                'free_transfers': 1,
                'access_token': 'temporary-secret-token',
            })

        self.assertEqual(response.status_code, 200)
        load_private.assert_called_once_with(123, 'temporary-secret-token')
        save_draft.assert_called_once_with(stored)
        self.assertTrue(captured['synced']['private'])
        self.assertNotIn('temporary-secret-token', repr(captured['synced']))
        self.assertNotIn('temporary-secret-token', response.get_data(as_text=True))

    @staticmethod
    def _response(payload):
        response = Mock(status_code=200)
        response.raise_for_status.return_value = None
        response.json.return_value = payload
        return response

    @patch('fpl_account.requests.get')
    def test_public_sync_persists_history_without_personal_profile(self, get):
        event_one = [
            {
                'element': player_id,
                'position': position,
                'multiplier': 2 if position == 1 else 1,
                'is_captain': position == 1,
                'is_vice_captain': position == 2,
                'element_type': 1 if position in (1, 12) else 2,
            }
            for position, player_id in enumerate(range(1, 16), start=1)
        ]
        event_two = [
            {**pick, 'element': pick['element'] + 100}
            for pick in event_one
        ]
        payloads = {
            '/api/entry/123/': {
                'id': 123,
                'player_first_name': 'Private',
                'player_last_name': 'Person',
                'started_event': 1,
                'current_event': 2,
            },
            '/api/entry/123/history/': {
                'current': [
                    {'event': 1, 'points': 55, 'total_points': 55},
                    {'event': 2, 'points': 61, 'total_points': 116},
                ],
            },
            '/api/entry/123/transfers/': [{
                'event': 2,
                'element_in': 101,
                'element_out': 1,
                'element_in_cost': 55,
                'element_out_cost': 50,
                'time': '2026-08-22T10:00:00Z',
            }],
            '/api/entry/123/event/1/picks/': {
                'active_chip': None,
                'entry_history': {
                    'event': 1, 'points': 55, 'total_points': 55,
                    'rank': 100, 'overall_rank': 200, 'bank': 15,
                    'value': 995, 'event_transfers': 0,
                    'event_transfers_cost': 0, 'points_on_bench': 4,
                },
                'picks': event_one,
            },
            '/api/entry/123/event/2/picks/': {
                'active_chip': 'freehit',
                'entry_history': {
                    'event': 2, 'points': 61, 'total_points': 116,
                    'rank': 90, 'overall_rank': 150, 'bank': 0,
                    'value': 1000, 'event_transfers': 5,
                    'event_transfers_cost': 0, 'points_on_bench': 3,
                },
                'picks': event_two,
            },
        }

        def response_for(url, timeout):
            path = url.split('fantasy.premierleague.com', 1)[1]
            return self._response(payloads[path])

        get.side_effect = response_for
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / 'entry.db'
            with closing(sqlite3.connect(database)) as connection:
                connection.execute('CREATE TABLE app_metadata (key TEXT, value TEXT)')
                connection.execute(
                    "INSERT INTO app_metadata VALUES ('live_start_index', '0')"
                )
                connection.execute(
                    'CREATE TABLE player_gw '
                    '(player_id INTEGER, Game_Week INTEGER, total_points INTEGER, value INTEGER)'
                )
                connection.executemany(
                    'INSERT INTO player_gw VALUES (?, 1, ?, 50)',
                    [(player_id, player_id % 8) for player_id in range(1, 16)],
                )
                connection.commit()
            synced = sync_public_entry(123, database)
            with closing(sqlite3.connect(database)) as connection:
                tables = {
                    row[0] for row in connection.execute(
                        "SELECT name FROM sqlite_master WHERE type='table'"
                    )
                }
                dump = '\n'.join(connection.iterdump())
                counts = {
                    'gameweeks': connection.execute(
                        'SELECT COUNT(*) FROM fpl_entry_gameweeks'
                    ).fetchone()[0],
                    'picks': connection.execute(
                        'SELECT COUNT(*) FROM fpl_entry_picks'
                    ).fetchone()[0],
                    'transfers': connection.execute(
                        'SELECT COUNT(*) FROM fpl_entry_transfers'
                    ).fetchone()[0],
                }

        self.assertEqual(synced['latest_event'], 2)
        self.assertEqual(synced['squad_event'], 1)
        self.assertEqual([pick['player_id'] for pick in synced['picks']], list(range(1, 16)))
        self.assertEqual(synced['bank'], 15)
        self.assertEqual(synced['next_free_transfers'], 2)
        self.assertEqual(synced['last_event_points'][1], 1)
        self.assertEqual(synced['last_event_points'][8], 0)
        self.assertEqual(counts, {'gameweeks': 2, 'picks': 30, 'transfers': 1})
        self.assertIn('fpl_entry_sync', tables)
        self.assertNotIn('Private', dump)
        self.assertNotIn('Person', dump)

    def test_synced_entry_becomes_model_rescored_local_draft(self):
        positions = [
            'GK', 'DEF', 'DEF', 'DEF', 'DEF', 'DEF',
            'MID', 'MID', 'MID', 'MID', 'FWD',
            'GK', 'MID', 'FWD', 'FWD',
        ]
        pool = pd.DataFrame([
            {
                'id': index,
                'first_name': f'Player{index}',
                'second_name': 'Test',
                'element_type': position,
                'team': (index % 10) + 1,
                'value': 50,
                'predicted_points': 5.0 - index / 100,
            }
            for index, position in enumerate(positions, start=1)
        ])
        picks = [
            {
                'player_id': index,
                'pick_position': index,
                'is_captain': index == 3,
                'is_vice_captain': index == 4,
            }
            for index in range(1, 16)
        ]
        synced = {
            'entry_id': 123,
            'synced_at': '2026-08-26T12:00:00+00:00',
            'latest_event': 2,
            'squad_event': 1,
            'picks': picks,
            'last_event_points': {1: 6, 2: 2},
            'acquisition_costs': {},
            'bank': 25,
            'next_free_transfers': 2,
            'private': True,
            'private_matches_public': True,
        }
        runtime = {'target_game_week': 3, 'season': '2026-27'}

        with patch.object(web, 'load_current', return_value=None):
            state = web._state_from_synced_entry(
                synced, pool, runtime, free_transfers=4
            )

        self.assertEqual(state['source'], 'official_fpl_private_sync')
        self.assertEqual(state['bank'], 25)
        self.assertEqual(state['free_transfers'], 4)
        self.assertEqual(set(state['lineup']['starters']), set(range(1, 12)))
        self.assertEqual(set(state['lineup']['bench']), set(range(12, 16)))
        self.assertEqual(state['lineup']['captain'], 1)
        self.assertEqual(state['lineup']['vice_captain'], 2)
        self.assertFalse(state['official_entry']['free_transfers_estimated'])
        self.assertTrue(state['official_entry']['private'])
        self.assertTrue(state['official_entry']['private_matches_public'])
        self.assertEqual(len(state['official_squad']), 15)
        self.assertEqual(state['official_squad'][0]['pick_position'], 1)
        self.assertEqual(state['official_squad'][0]['last_gameweek_points'], 6)
        self.assertEqual(state['official_squad'][1]['last_gameweek_points'], 2)
        self.assertIsNone(state['official_squad'][2]['last_gameweek_points'])
        self.assertTrue(state['official_squad'][2]['is_captain'])
        self.assertEqual(state['official_squad'][11]['lineup_role'], 'bench')

    def test_new_entry_explains_pre_deadline_public_sync_limit(self):
        synced = {
            'entry_id': 10378874,
            'started_event': 3,
            'current_event': 2,
            'gameweeks': [],
            'picks': [],
        }

        with self.assertRaisesRegex(
            web.FplEntrySyncError,
            'starts in GW3.*private before the GW3 deadline.*ends at GW2',
        ):
            web._state_from_synced_entry(
                synced,
                pd.DataFrame(),
                {'target_game_week': 3, 'season': '2026-27'},
            )



if __name__ == '__main__':
    unittest.main()
