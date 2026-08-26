import sys
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from performance_reviewer import (
    load_actual_points,
    load_committed_gameweek_states,
    load_sqlite_actual_points,
    score_saved_squad,
)
from squad_state import create_state


def make_state():
    rows = []
    player_id = 1
    for position, count in {'GK': 2, 'DEF': 5, 'MID': 5, 'FWD': 3}.items():
        for index in range(count):
            rows.append({
                'id': player_id,
                'first_name': position,
                'second_name': str(index + 1),
                'element_type': position,
                'team': ((player_id - 1) % 8) + 1,
                'value': 50,
                'predicted_points': 20 - player_id,
                'predicted_points_norm': 20 - player_id,
            })
            player_id += 1
    return create_state(pd.DataFrame(rows), 1, '2026-27')


class PerformanceReviewerTests(unittest.TestCase):
    def test_missing_squad_tables_return_empty_history(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / 'empty.db'
            self.assertEqual(load_committed_gameweek_states(database), [])

    def test_scores_captain_and_transfer_cost(self):
        state = make_state()
        actuals = {
            player['id']: {'points': 2, 'minutes': 90}
            for player in state['players']
        }
        captain = state['lineup']['captain']
        actuals[captain]['points'] = 8
        state['transfer_points_cost'] = 4

        result = score_saved_squad(state, actuals)

        self.assertEqual(result['total_points'], 32)
        self.assertEqual(result['captain_bonus'], 8)
        self.assertEqual(result['transfer_cost'], 4)

    def test_vice_captain_takes_over_and_bench_sub_is_legal(self):
        state = make_state()
        actuals = {
            player['id']: {'points': 2, 'minutes': 90}
            for player in state['players']
        }
        captain = state['lineup']['captain']
        vice = state['lineup']['vice_captain']
        actuals[captain] = {'points': 0, 'minutes': 0}
        actuals[vice] = {'points': 5, 'minutes': 90}

        result = score_saved_squad(state, actuals)

        self.assertEqual(result['effective_captain'], vice)
        self.assertEqual(len(result['substitutions']), 1)
        self.assertEqual(result['total_points'], 30)

    def test_bench_boost_counts_all_fifteen(self):
        state = make_state()
        state['active_chip'] = 'bench_boost'
        actuals = {
            player['id']: {'points': 2, 'minutes': 90}
            for player in state['players']
        }
        result = score_saved_squad(state, actuals)
        self.assertEqual(result['total_points'], 32)

    def test_actual_results_are_filtered_by_season_and_aggregate_dgw(self):
        import sqlite3
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / 'review.db'
            with closing(sqlite3.connect(database)) as connection:
                connection.execute(
                    'CREATE TABLE player_gw (element INTEGER, round INTEGER, '
                    'total_points INTEGER, minutes INTEGER, kickoff_time TEXT)'
                )
                connection.executemany(
                    'INSERT INTO player_gw VALUES (?, ?, ?, ?, ?)',
                    [
                        (7, 1, 12, 90, '2025-08-20T12:00:00Z'),
                        (7, 1, 3, 60, '2026-08-20T12:00:00Z'),
                        (7, 1, 5, 30, '2026-08-24T12:00:00Z'),
                    ],
                )
                connection.commit()
            actuals = load_sqlite_actual_points(1, '2026-27', database)
            self.assertEqual(
                actuals[7], {'points': 8, 'minutes': 90, 'played': True}
            )

    @patch('performance_reviewer.fetch_live_actual_points')
    def test_official_live_results_take_precedence(self, fetch_live):
        fetch_live.return_value = {7: {'points': 12, 'minutes': 90}}
        self.assertEqual(
            load_actual_points(1, '2026-27', 'unused.db')[7]['points'], 12
        )

    def test_zero_minute_card_appearance_prevents_captain_fallback(self):
        state = make_state()
        captain = state['lineup']['captain']
        vice = state['lineup']['vice_captain']
        actuals = {
            player['id']: {'points': 2, 'minutes': 90, 'played': True}
            for player in state['players']
        }
        actuals[captain] = {'points': -1, 'minutes': 0, 'played': True}
        actuals[vice] = {'points': 8, 'minutes': 90, 'played': True}

        result = score_saved_squad(state, actuals)

        self.assertEqual(result['effective_captain'], captain)
        self.assertEqual(result['captain_bonus'], -1)


if __name__ == '__main__':
    unittest.main()
