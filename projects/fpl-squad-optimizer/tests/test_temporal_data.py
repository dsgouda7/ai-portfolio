import sys
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from transfer_values import (
    attach_transfer_values_asof,
    ensure_table,
    purge_imputed_transfer_values,
    refresh_transfer_value_history_from_bulk,
)


class TemporalMarketValueTests(unittest.TestCase):
    def test_future_transfermarkt_value_cannot_leak_into_old_gameweek(self):
        rows = pd.DataFrame([
            {'code': 10, 'kickoff_time': '2025-08-01T12:00:00Z'},
            {'code': 10, 'kickoff_time': '2026-08-25T12:00:00Z'},
        ])
        history = pd.DataFrame([{
            'fpl_code': 10,
            'observed_at': '2026-08-20T00:00:00Z',
            'tm_market_value': 7.5,
            'tm_value_imputed': 0,
            'tm_value_source': 'transfermarkt_bulk',
        }])

        result = attach_transfer_values_asof(rows, history)

        self.assertTrue(pd.isna(result.iloc[0]['tm_market_value']))
        self.assertEqual(result.iloc[0]['tm_value_source'], 'unavailable_at_cutoff')
        self.assertEqual(result.iloc[0]['tm_value_imputed'], 0)
        self.assertEqual(result.iloc[1]['tm_market_value'], 7.5)
        self.assertEqual(result.iloc[1]['tm_value_source'], 'transfermarkt_bulk')

    def test_imputed_values_are_purged_not_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / 'tm.db'
            with closing(sqlite3.connect(database)) as connection:
                ensure_table(connection)
                connection.execute(
                    """
                    INSERT INTO player_transfer_values (
                        fpl_id, fpl_code, tm_id, tm_value_eur, tm_value_imputed
                    ) VALUES (1, 10, '100', 25000000, 1)
                    """
                )
                connection.execute(
                    """
                    INSERT INTO player_transfer_value_history (
                        fpl_code, tm_id, observed_at, tm_value_eur,
                        tm_value_imputed, source
                    ) VALUES (10, '100', '2025-01-01T00:00:00Z', 25000000, 1, 'old')
                    """
                )
                purge_imputed_transfer_values(connection)
                current = connection.execute(
                    'SELECT tm_value_eur, tm_value_imputed, tm_retry '
                    'FROM player_transfer_values'
                ).fetchone()
                history_count = connection.execute(
                    'SELECT COUNT(*) FROM player_transfer_value_history'
                ).fetchone()[0]
            self.assertEqual(current, (None, 0, 1))
            self.assertEqual(history_count, 0)

    @patch('transfer_values.requests.get')
    def test_historical_bulk_ingest_uses_real_dated_values(self, get):
        csv_bytes = (
            'player_id,date,market_value_in_eur\n'
            '100,2025-08-01,10000000\n'
            '100,2025-09-01,12000000\n'
        ).encode()
        import gzip
        response = Mock(content=gzip.compress(csv_bytes))
        response.raise_for_status.return_value = None
        get.return_value = response
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / 'tm.db'
            with closing(sqlite3.connect(database)) as connection:
                ensure_table(connection)
                connection.execute(
                    'INSERT INTO player_transfer_values (fpl_id, fpl_code, tm_id) '
                    "VALUES (1, 10, '100')"
                )
                count = refresh_transfer_value_history_from_bulk(connection)
                rows = connection.execute(
                    'SELECT observed_at, tm_value_eur, tm_value_imputed, source '
                    'FROM player_transfer_value_history ORDER BY observed_at'
                ).fetchall()
        self.assertEqual(count, 2)
        self.assertEqual(rows[0][0], '2025-08-01T23:59:59Z')
        self.assertEqual(rows[1][1], 12000000)
        self.assertTrue(all(row[2] == 0 for row in rows))
        self.assertTrue(all(row[3] == 'transfermarkt_historical_cc0' for row in rows))


if __name__ == '__main__':
    unittest.main()
