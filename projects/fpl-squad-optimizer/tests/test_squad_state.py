import sys
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from squad_state import (
    SquadValidationError,
    apply_transfer,
    commit_draft,
    create_state,
    list_versions,
    load_current,
    load_working_state,
    refresh_player_data,
    roll_to_game_week,
    save_draft,
    selling_price,
    set_lineup,
)
from utils import suggest_replacements, suggest_transfers


def make_squad() -> pd.DataFrame:
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
    return pd.DataFrame(rows)


class SquadStateTests(unittest.TestCase):
    def test_opposition_audit_persists_and_is_cleared_when_layer_is_disabled(self):
        pool = make_squad()
        pool['opposition_base_points'] = pool['predicted_points'] / 0.9
        pool['opposition_factor'] = 0.9
        pool['opposition_fixtures'] = [
            [{'opponent_team': 2, 'factor': 0.9}] for _ in range(len(pool))
        ]
        state = create_state(pool, 4, '2026-27')
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / 'squad.db'
            exports = Path(directory) / 'exports'
            save_draft(state, database, exports)
            restored = load_working_state(database)
        self.assertEqual(restored['players'][0]['opposition_factor'], 0.9)
        self.assertEqual(
            restored['players'][0]['opposition_fixtures'],
            [{'opponent_team': 2, 'factor': 0.9}],
        )
        disabled = refresh_player_data(restored, make_squad().query('id != 1'))
        self.assertNotIn('opposition_factor', disabled['players'][0])
        self.assertNotIn('opposition_factor', disabled['players'][1])
        self.assertIn('opposition_factor', restored['players'][0])

    def test_replacement_shortlist_respects_budget_club_cap_and_ranking(self):
        squad = pd.DataFrame([
            {
                'id': 1, 'element_type': 'MID', 'team': 1, 'value': 50,
                'predicted_points': 4.0,
            },
            {
                'id': 2, 'element_type': 'DEF', 'team': 4, 'value': 50,
                'predicted_points': 2.0,
            },
            {
                'id': 3, 'element_type': 'DEF', 'team': 4, 'value': 50,
                'predicted_points': 2.0,
            },
            {
                'id': 4, 'element_type': 'FWD', 'team': 4, 'value': 50,
                'predicted_points': 2.0,
            },
        ])
        pool = pd.DataFrame([
            {
                'id': 5, 'element_type': 'MID', 'team': 4, 'value': 50,
                'predicted_points': 10.0,
            },
            {
                'id': 6, 'element_type': 'MID', 'team': 5, 'value': 61,
                'predicted_points': 9.0,
            },
            {
                'id': 7, 'element_type': 'MID', 'team': 5, 'value': 60,
                'predicted_points': 8.0,
            },
            {
                'id': 8, 'element_type': 'MID', 'team': 6, 'value': 55,
                'predicted_points': 7.0,
            },
            {
                'id': 9, 'element_type': 'MID', 'team': 7, 'value': 50,
                'predicted_points': 3.0,
            },
        ])

        replacements = suggest_replacements(
            squad,
            pool,
            outgoing_id=1,
            limit=3,
            max_per_team=3,
            max_spend=210,
        )

        self.assertEqual(
            [int(item['in_']['id']) for item in replacements],
            [7, 8, 9],
        )
        self.assertEqual(
            [item['gain'] for item in replacements],
            [4.0, 3.0, -1.0],
        )
        self.assertEqual(
            [item['bank_after'] for item in replacements],
            [0, 5, 10],
        )

    def test_transfer_plan_uses_available_free_transfers(self):
        squad = pd.DataFrame([
            {
                'id': 1, 'element_type': 'MID', 'team': 1, 'value': 50,
                'predicted_points': 1.0,
            },
            {
                'id': 2, 'element_type': 'MID', 'team': 2, 'value': 50,
                'predicted_points': 2.0,
            },
        ])
        pool = pd.DataFrame([
            {
                'id': 3, 'element_type': 'MID', 'team': 3, 'value': 50,
                'predicted_points': 6.0,
            },
            {
                'id': 4, 'element_type': 'MID', 'team': 4, 'value': 50,
                'predicted_points': 5.0,
            },
        ])

        recommendations = suggest_transfers(
            squad, pool, transfer_limit=2, max_spend=100
        )

        self.assertEqual(len(recommendations), 2)
        self.assertEqual(
            [(item['out']['id'], item['in_']['id']) for item in recommendations],
            [(1, 3), (2, 4)],
        )
        self.assertEqual([item['gain'] for item in recommendations], [5.0, 3.0])

    def test_transfer_plan_stops_when_no_move_improves_points(self):
        squad = pd.DataFrame([{
            'id': 1, 'element_type': 'MID', 'team': 1, 'value': 50,
            'predicted_points': 5.0,
        }])
        pool = pd.DataFrame([{
            'id': 2, 'element_type': 'MID', 'team': 2, 'value': 50,
            'predicted_points': 4.0,
        }])

        recommendations = suggest_transfers(
            squad, pool, transfer_limit=2, max_spend=50
        )

        self.assertEqual(recommendations, [])

    def test_selling_price_uses_half_of_profit_rounded_down(self):
        self.assertEqual(selling_price(50, 53), 51)
        self.assertEqual(selling_price(50, 48), 48)

    def test_draft_and_commit_are_readable_json(self):
        state = create_state(make_squad(), 4, '2026-27')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database = root / 'test.db'
            draft = save_draft(state, database, root / 'exports')
            current = commit_draft(state, database, root / 'exports')

            self.assertEqual(draft['revision'], 1)
            self.assertEqual(current['revision'], 2)
            self.assertEqual(load_current(database)['game_week'], 4)
            self.assertEqual(load_current(database)['status'], 'committed')
            self.assertEqual(
                len(list((root / 'exports' / 'history').glob('*.json'))), 1
            )

            import sqlite3
            with closing(sqlite3.connect(database)) as connection:
                versions = connection.execute(
                    'SELECT status, revision FROM squad_versions ORDER BY version_id'
                ).fetchall()
                player_versions = connection.execute(
                    'SELECT COUNT(*) FROM squad_version_players'
                ).fetchone()[0]
            self.assertEqual(versions, [('draft', 1), ('committed', 2)])
            self.assertEqual(player_versions, 30)
            history = list_versions(database, season='2026-27')
            self.assertEqual([row['revision'] for row in history], [2, 1])
            self.assertEqual(load_working_state(database)['status'], 'committed')

            newer_draft = save_draft(current, database, root / 'exports')
            self.assertEqual(load_working_state(database)['version_id'], newer_draft['version_id'])

    def test_transfer_updates_bank_lineup_and_points_cost(self):
        state = create_state(make_squad(), 4, '2026-27')
        outgoing = state['players'][2]
        incoming = {
            'id': 99,
            'first_name': 'New',
            'second_name': 'Defender',
            'element_type': 'DEF',
            'team': 9,
            'value': 51,
            'predicted_points': 8,
        }
        state['bank'] = 10
        updated = apply_transfer(state, incoming, outgoing['id'])

        self.assertEqual(updated['bank'], 9)
        self.assertIn(99, updated['lineup']['starters'] + updated['lineup']['bench'])
        self.assertEqual(updated['transfer_points_cost'], 0)

    def test_gameweek_rollover_carries_free_transfers_and_resets_moves(self):
        state = create_state(make_squad(), 4, '2026-27')
        state['free_transfers'] = 2
        rolled = roll_to_game_week(state, 5, '2026-27')

        self.assertEqual(rolled['game_week'], 5)
        self.assertEqual(rolled['free_transfers'], 3)
        self.assertEqual(rolled['transfers'], [])
        self.assertIsNone(rolled['active_chip'])
        self.assertEqual(rolled['source'], 'gameweek_rollover')

    def test_gameweek_rollover_retains_saved_transfers_after_wildcard(self):
        state = create_state(make_squad(), 4, '2026-27')
        state['free_transfers'] = 4
        state['active_chip'] = 'wildcard'
        state['transfers'] = [{'out': {'id': 1}, 'in': {'id': 99}}]
        rolled = roll_to_game_week(state, 5, '2026-27')

        self.assertEqual(rolled['free_transfers'], 4)

    def test_refresh_updates_imported_prediction_but_preserves_last_actual(self):
        state = create_state(make_squad(), 4, '2026-27')
        official_player = dict(state['players'][0])
        official_player.update({
            'pick_position': 1,
            'lineup_role': 'starter',
            'is_captain': False,
            'is_vice_captain': False,
            'last_gameweek_points': 7,
        })
        state['official_squad'] = [official_player]
        refreshed_pool = make_squad()
        refreshed_pool['predicted_points'] = refreshed_pool['predicted_points'].astype(float)
        refreshed_pool.loc[refreshed_pool['id'] == 1, 'predicted_points'] = 9.5
        refreshed_pool.loc[refreshed_pool['id'] == 1, 'fixture_count'] = 2

        updated = refresh_player_data(state, refreshed_pool)

        self.assertEqual(updated['players'][0]['predicted_points'], 9.5)
        self.assertEqual(updated['players'][0]['fixture_count'], 2)
        self.assertEqual(updated['official_squad'][0]['predicted_points'], 9.5)
        self.assertEqual(updated['official_squad'][0]['last_gameweek_points'], 7)

    def test_invalid_formation_is_rejected(self):
        state = create_state(make_squad(), 4, '2026-27')
        goalkeepers = [player['id'] for player in state['players'] if player['position'] == 'GK']
        starters = state['lineup']['starters'][:]
        bench = state['lineup']['bench'][:]
        starter_outfield = next(player_id for player_id in starters if player_id not in goalkeepers)
        bench_goalkeeper = next(player_id for player_id in bench if player_id in goalkeepers)
        starters[starters.index(starter_outfield)] = bench_goalkeeper
        bench[bench.index(bench_goalkeeper)] = starter_outfield

        with self.assertRaisesRegex(SquadValidationError, '1-1 GK'):
            set_lineup(
                state,
                starters,
                bench,
                state['lineup']['captain'],
                state['lineup']['vice_captain'],
            )

    def test_four_players_from_one_club_are_rejected(self):
        squad = make_squad()
        squad.loc[:3, 'team'] = 20
        with self.assertRaisesRegex(SquadValidationError, 'at most 3'):
            create_state(squad, 4, '2026-27')

    def test_exhausted_chip_is_rejected(self):
        state = create_state(make_squad(), 4, '2026-27')
        state['chips']['wildcard']['remaining'] = 0
        state['active_chip'] = 'wildcard'
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(SquadValidationError, 'No wildcard chips remain'):
                save_draft(
                    state,
                    Path(directory) / 'test.db',
                    Path(directory) / 'exports',
                )


if __name__ == '__main__':
    unittest.main()
