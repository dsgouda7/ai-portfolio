import sqlite3
import sys
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal

sys.path.insert(0, str(Path(__file__).parent.parent))

from feature_cache import (
    canonical_player_gameweeks,
    feature_cache_metadata,
    load_feature_cache,
    persist_feature_cache,
    persist_rnn_sequence_index,
)


class FeatureCacheTests(unittest.TestCase):
    def test_canonical_rows_prefer_a_target_bearing_duplicate(self):
        rows = pd.DataFrame({
            'id': [7, 7],
            'Game_Week': [1, 1],
            'target': [None, 5.0],
            'value': [50, 51],
        })
        canonical = canonical_player_gameweeks(rows)
        self.assertEqual(len(canonical), 1)
        self.assertEqual(canonical.iloc[0]['target'], 5.0)

    def test_cache_round_trips_and_invalidates_when_temporal_source_changes(self):
        rows = pd.DataFrame({
            'id': [7, 7],
            'element_type': ['MID', 'MID'],
            'Game_Week': [0, 1],
            'target': [3.0, 5.0],
            'value': [50, 51],
        })
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / 'cache.db'
            with closing(sqlite3.connect(database)) as connection:
                pd.DataFrame({
                    'player_id': [7, 7],
                    'Game_Week': [0, 1],
                    'kickoff_time': ['2026-08-01', '2026-08-08'],
                }).to_sql('player_gw', connection, index=False)
                pd.DataFrame({'id': [7]}).to_sql('players_raw', connection, index=False)
                connection.commit()

            metadata = persist_feature_cache(database, rows)
            loaded = load_feature_cache(database)
            assert_frame_equal(loaded, rows, check_dtype=False)
            self.assertEqual(metadata['row_count'], 2)
            self.assertEqual(feature_cache_metadata(database)['column_count'], 5)

            with closing(sqlite3.connect(database)) as connection:
                connection.execute(
                    'UPDATE player_gw SET kickoff_time = ? WHERE Game_Week = ?',
                    ('2026-08-09', 1),
                )
                connection.commit()

            self.assertIsNone(load_feature_cache(database))
            self.assertIsNone(feature_cache_metadata(database))

    def test_rnn_sequence_index_persists_all_available_spans(self):
        rows = pd.DataFrame({
            'id': [7, 7, 7, 7, 9],
            'element_type': ['MID', 'MID', 'MID', 'MID', 'FWD'],
            'Game_Week': [0, 1, 2, 2, 2],
            'target': [3.0, 5.0, 2.0, 6.0, 4.0],
        })
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / 'sequences.db'
            count = persist_rnn_sequence_index(database, rows, 2)
            with closing(sqlite3.connect(database)) as connection:
                spans = pd.read_sql(
                    'SELECT player_id, sequence_start_game_week, '
                    'sequence_end_game_week, sequence_length, sequence_policy '
                    'FROM rnn_sequence_index ORDER BY player_id, sequence_end_game_week',
                    connection,
                )

        self.assertEqual(count, 4)
        self.assertEqual(spans['sequence_length'].tolist(), [1, 2, 3, 1])
        self.assertEqual(spans['sequence_start_game_week'].tolist(), [0, 0, 0, 2])
        self.assertEqual(set(spans['sequence_policy']), {'all_available'})


if __name__ == '__main__':
    unittest.main()
