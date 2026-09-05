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

from fpl_account import sync_public_entry

spec = importlib.util.spec_from_file_location(
    'fpl_web_account_test', ROOT / 'fpl-generator' / 'web.py'
)
web = importlib.util.module_from_spec(spec)
spec.loader.exec_module(web)


class FplAccountSyncTests(unittest.TestCase):
    def test_sync_route_rejects_credentials(self):
        web.app.config['TESTING'] = True
        with web.app.test_client() as client:
            response = client.post('/api/fpl-entry/sync', json={
                'entry_id': 123,
                'password': 'must-not-be-accepted',
            })
        self.assertEqual(response.status_code, 400)
        self.assertIn('Credentials and unknown fields are not accepted', response.get_json()['error'])

    def test_sync_route_rejects_non_json_and_cross_origin_requests(self):
        web.app.config['TESTING'] = True
        with web.app.test_client() as client:
            non_json = client.post('/api/fpl-entry/sync', data='entry_id=123')
            cross_origin = client.post(
                '/api/fpl-entry/sync',
                json={'entry_id': 123},
                headers={'Origin': 'https://example.invalid'},
            )
        self.assertEqual(non_json.status_code, 415)
        self.assertEqual(cross_origin.status_code, 403)

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
            'acquisition_costs': {},
            'bank': 25,
            'next_free_transfers': 2,
        }
        runtime = {'target_game_week': 3, 'season': '2026-27'}

        with patch.object(web, 'load_current', return_value=None):
            state = web._state_from_synced_entry(
                synced, pool, runtime, free_transfers=4
            )

        self.assertEqual(state['source'], 'official_fpl_public_sync')
        self.assertEqual(state['bank'], 25)
        self.assertEqual(state['free_transfers'], 4)
        self.assertEqual(set(state['lineup']['starters']), set(range(1, 12)))
        self.assertEqual(set(state['lineup']['bench']), set(range(12, 16)))
        self.assertEqual(state['lineup']['captain'], 1)
        self.assertEqual(state['lineup']['vice_captain'], 2)
        self.assertFalse(state['official_entry']['free_transfers_estimated'])

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
