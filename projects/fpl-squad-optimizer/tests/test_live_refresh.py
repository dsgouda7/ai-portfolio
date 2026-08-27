import sys
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import get_runtime_context, refresh_live_season_data


class LiveRefreshTests(unittest.TestCase):
    @patch('utils.requests.get')
    def test_refresh_separates_completed_cutoff_from_future_snapshot(self, get):
        bootstrap = {
            'events': [
                {'id': 1, 'finished': True, 'data_checked': True},
                {'id': 2, 'finished': False, 'data_checked': False,
                 'deadline_time': '2026-08-28T17:30:00Z'},
            ],
            'elements': [{
                'id': 7, 'code': 700, 'team': 1, 'status': 'a', 'removed': False,
                'element_type': 3, 'first_name': 'Test', 'second_name': 'Player',
                'birth_date': '2000-01-01', 'now_cost': 55,
            }],
        }
        history_response = Mock()
        history_response.raise_for_status.return_value = None
        history_response.json.return_value = {'history': [{
            'element': 7, 'round': 1, 'total_points': 8, 'minutes': 90,
            'kickoff_time': '2026-08-21T17:30:00Z', 'value': 50,
        }]}
        fixtures_response = Mock()
        fixtures_response.raise_for_status.return_value = None
        fixtures_response.json.return_value = [{
            'team_h': 1, 'team_a': 2, 'kickoff_time': '2026-08-29T12:30:00Z'
        }]
        get.side_effect = [history_response, fixtures_response]

        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / 'refresh.db'
            old_raw = pd.DataFrame([{
                'id': 70, 'code': 700, 'team': 1, 'element_type': 3,
                'first_name': 'Test', 'second_name': 'Player',
                'birth_date': '2000-01-01',
            }])
            old_gw = pd.DataFrame([{
                'element': 70, 'player_id': 70, 'Game_Week': 0, 'round': 38,
                'total_points': 2, 'minutes': 90,
                'kickoff_time': '2026-05-20T12:00:00Z', 'value': 50,
                'team': 1, 'opponent_team': 2, 'was_home': True,
            }])
            import sqlite3
            with closing(sqlite3.connect(database)) as connection:
                old_raw.to_sql('players_raw', connection, index=False)
                old_gw.to_sql('player_gw', connection, index=False)
                pd.DataFrame([
                    {'key': 'snapshot_game_week', 'value': '1'},
                    {'key': 'live_start_index', 'value': '1'},
                ]).to_sql('app_metadata', connection, index=False)
                connection.commit()

            result = refresh_live_season_data(database, bootstrap)
            context = get_runtime_context(database)
            with closing(sqlite3.connect(database)) as connection:
                rows = pd.read_sql(
                    'SELECT Game_Week, round, total_points FROM player_gw '
                    'ORDER BY Game_Week', connection
                )

        self.assertEqual(result['latest_completed_game_week'], 1)
        self.assertEqual(context['completed_internal_index'], 1)
        self.assertEqual(context['snapshot_game_week'], 2)
        self.assertEqual(rows['Game_Week'].tolist(), [0, 1, 2])
        self.assertEqual(rows.iloc[1]['total_points'], 8)
        self.assertEqual(rows.iloc[2]['total_points'], 0)

    @patch('utils.requests.get')
    def test_refresh_rejects_incomplete_player_history(self, get):
        bootstrap = {
            'events': [
                {'id': 1, 'finished': True, 'data_checked': True},
                {'id': 2, 'finished': False, 'data_checked': False},
            ],
            'elements': [{
                'id': 7, 'code': 700, 'team': 1, 'status': 'a',
                'removed': False, 'element_type': 3, 'first_name': 'Test',
                'second_name': 'Player', 'birth_date': '2000-01-01',
                'now_cost': 55,
            }],
        }
        get.side_effect = requests.ConnectionError('offline')
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / 'refresh.db'
            with closing(__import__('sqlite3').connect(database)) as connection:
                pd.DataFrame([{
                    'id': 70, 'code': 700, 'team': 1, 'element_type': 3,
                    'first_name': 'Test', 'second_name': 'Player',
                    'birth_date': '2000-01-01',
                }]).to_sql('players_raw', connection, index=False)
                pd.DataFrame([{
                    'player_id': 70, 'Game_Week': 0, 'round': 38,
                    'total_points': 2,
                }]).to_sql('player_gw', connection, index=False)
                connection.commit()

            with self.assertRaisesRegex(RuntimeError, 'Incomplete FPL history'):
                refresh_live_season_data(database, bootstrap)

        self.assertEqual(get.call_count, 3)

    @patch('utils.SEASON', '2026-27')
    @patch('utils.requests.get')
    def test_fresh_same_season_archive_rows_are_replaced(self, get):
        bootstrap = {
            'events': [
                {'id': 1, 'finished': True, 'data_checked': True,
                 'deadline_time': '2026-08-20T17:30:00Z'},
                {'id': 2, 'finished': False, 'data_checked': False,
                 'deadline_time': '2026-08-27T17:30:00Z'},
            ],
            'elements': [{
                'id': 7, 'code': 700, 'team': 1, 'status': 'a',
                'removed': False, 'element_type': 3, 'first_name': 'Test',
                'second_name': 'Player', 'birth_date': '2000-01-01',
                'now_cost': 55,
            }],
        }
        history_response = Mock()
        history_response.raise_for_status.return_value = None
        history_response.json.return_value = {'history': [{
            'element': 7, 'round': 1, 'total_points': 8, 'minutes': 90,
            'kickoff_time': '2026-08-21T17:30:00Z', 'value': 50,
        }]}
        fixtures_response = Mock()
        fixtures_response.raise_for_status.return_value = None
        fixtures_response.json.return_value = [{
            'team_h': 1, 'team_a': 2, 'kickoff_time': '2026-08-29T12:30:00Z'
        }]
        get.side_effect = [history_response, fixtures_response]

        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / 'same-season.db'
            with closing(__import__('sqlite3').connect(database)) as connection:
                pd.DataFrame([{
                    'id': 7, 'code': 700, 'team': 1, 'element_type': 3,
                    'first_name': 'Test', 'second_name': 'Player',
                    'birth_date': '2000-01-01',
                }]).to_sql('players_raw', connection, index=False)
                pd.DataFrame([{
                    'player_id': 7, 'Game_Week': 0, 'round': 1,
                    'total_points': 99, 'minutes': 90,
                }]).to_sql('player_gw', connection, index=False)
                connection.commit()

            result = refresh_live_season_data(database, bootstrap)
            with closing(__import__('sqlite3').connect(database)) as connection:
                rows = pd.read_sql(
                    'SELECT Game_Week, total_points FROM player_gw ORDER BY Game_Week',
                    connection,
                )

        self.assertEqual(result['season'], '2026-27')
        self.assertEqual(rows['Game_Week'].tolist(), [0, 1])
        self.assertEqual(rows.iloc[0]['total_points'], 8)

    @patch('utils.SEASON', '2026-27')
    @patch('utils.requests.get')
    def test_departed_player_history_is_trained_but_not_snapshotted(self, get):
        bootstrap = {
            'events': [
                {'id': 1, 'finished': True, 'data_checked': True,
                 'deadline_time': '2026-08-20T17:30:00Z'},
                {'id': 2, 'finished': False, 'data_checked': False,
                 'deadline_time': '2026-08-27T17:30:00Z'},
            ],
            'elements': [{
                'id': 7, 'code': 700, 'team': 1, 'status': 'u',
                'removed': True, 'element_type': 3, 'first_name': 'Gone',
                'second_name': 'Player', 'birth_date': '2000-01-01',
                'now_cost': 55,
            }],
        }
        history_response = Mock()
        history_response.raise_for_status.return_value = None
        history_response.json.return_value = {'history': [{
            'element': 7, 'round': 1, 'total_points': 8, 'minutes': 90,
            'kickoff_time': '2026-08-21T17:30:00Z', 'value': 50,
        }]}
        fixtures_response = Mock()
        fixtures_response.raise_for_status.return_value = None
        fixtures_response.json.return_value = []
        get.side_effect = [history_response, fixtures_response]

        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / 'departed.db'
            with closing(__import__('sqlite3').connect(database)) as connection:
                pd.DataFrame([{
                    'id': 7, 'code': 700, 'team': 1, 'element_type': 3,
                    'first_name': 'Gone', 'second_name': 'Player',
                    'birth_date': '2000-01-01',
                }]).to_sql('players_raw', connection, index=False)
                pd.DataFrame([{
                    'player_id': 7, 'Game_Week': 0, 'round': 1,
                    'total_points': 2,
                }]).to_sql('player_gw', connection, index=False)
                connection.commit()

            result = refresh_live_season_data(database, bootstrap)
            with closing(__import__('sqlite3').connect(database)) as connection:
                rows = pd.read_sql('SELECT * FROM player_gw', connection)

        self.assertEqual(result['live_rows'], 1)
        self.assertEqual(result['snapshot_players'], 0)
        self.assertEqual(rows['Game_Week'].tolist(), [0])


if __name__ == '__main__':
    unittest.main()
