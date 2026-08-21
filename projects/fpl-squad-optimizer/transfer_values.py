"""Transfermarkt market value pipeline.

Resolves FPL player codes to Transfermarkt IDs via the Reep cross-provider
register, then fetches current market values and stores them in SQLite.

Key design decisions
--------------------
* **ID bridge**: FPL ``player.code`` (stable across seasons) → Reep
  ``key_opta_numeric`` → ``key_transfermarkt``.  Reep's ``data/people.csv`` is
  downloaded once and cached locally; refreshed when the cache is > 7 days old.

* **Conditional update**: a stored value is only overwritten when the new
  Transfermarkt ``tm_value_date`` is strictly newer than what is already stored.
  This avoids accidentally replacing good data with a stale snapshot.

* **Imputed fallback**: if Transfermarkt returns no value for a player, the
  mean of all fetched values is stored and ``tm_retry = 1`` is set.  Every
  subsequent training run will retry that player until real data is obtained.

* **Force flag**: ``--force`` on the CLI (or ``force=True`` in
  ``refresh_transfer_values``) re-fetches every player regardless of cache age.

CLI
---
    python transfer_values.py               # fetch only missing / retry players
    python transfer_values.py --force       # re-fetch all players
    python transfer_values.py --rebuild-ids # reload Reep CSV and remap all IDs
    python transfer_values.py --status      # print row counts and exit
"""

from __future__ import annotations

import argparse
import io
import math
import os
import pathlib
import re
import sqlite3
import time
import unicodedata
from datetime import date, datetime, timedelta

import pandas as pd
import requests

from transfermarkt_client import get_transfermarkt_data

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_ROOT = pathlib.Path(__file__).parent
_DB_PATH = str(_ROOT / "fantasy_football.db")
_REEP_CSV_URL = (
    "https://raw.githubusercontent.com/withqwerty/reep/main/data/people.csv"
)
_REEP_CACHE_PATH = str(_ROOT / ".reep_people_cache.csv")
_REEP_CACHE_TTL_DAYS = 7     # refresh local CSV cache every 7 days
_TM_SLEEP_SECONDS = 1.0      # polite crawl rate between player fetches
_BULK_PLAYERS_URL = (
    "https://pub-e682421888d945d684bcae8890b0ec20.r2.dev/data/players.csv.gz"
)

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}

# ---------------------------------------------------------------------------
# DDL
# ---------------------------------------------------------------------------

_DDL = """
CREATE TABLE IF NOT EXISTS player_transfer_values (
    fpl_id            INTEGER PRIMARY KEY,
    fpl_code          INTEGER,
    tm_id             TEXT,
    tm_value_eur      REAL,
    tm_value_date     TEXT,
    tm_fetched_at     TEXT,
    tm_value_imputed  INTEGER DEFAULT 0,
    tm_retry          INTEGER DEFAULT 0
);
"""

# ---------------------------------------------------------------------------
# Reep CSV helpers
# ---------------------------------------------------------------------------


def _reep_cache_is_fresh() -> bool:
    """Return True if the local Reep cache exists and is < TTL days old."""
    if not os.path.exists(_REEP_CACHE_PATH):
        return False
    mtime = datetime.fromtimestamp(os.path.getmtime(_REEP_CACHE_PATH))
    return datetime.now() - mtime < timedelta(days=_REEP_CACHE_TTL_DAYS)


def _download_reep_csv() -> pd.DataFrame:
    """Download Reep people.csv from GitHub and return as a DataFrame."""
    print("[transfer_values] Downloading Reep people.csv from GitHub...")
    resp = requests.get(_REEP_CSV_URL, headers=_HEADERS, timeout=60)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text), low_memory=False)
    df.to_csv(_REEP_CACHE_PATH, index=False)
    print(f"[transfer_values] Reep CSV cached ({len(df):,} rows) → {_REEP_CACHE_PATH}")
    return df


def _load_reep_df(force_download: bool = False) -> pd.DataFrame:
    """Return the Reep people DataFrame, downloading if cache is stale."""
    if not force_download and _reep_cache_is_fresh():
        return pd.read_csv(_REEP_CACHE_PATH, low_memory=False)
    return _download_reep_csv()


def _build_code_to_tm_id(reep_df: pd.DataFrame) -> dict[int, str]:
    """
    Return a ``{fpl_code: tm_id}`` dict from the Reep DataFrame.

    ``key_opta_numeric`` is the Opta legacy numeric ID, which is identical to
    the FPL ``code`` field (stable across seasons, unlike ``id``).
    ``key_transfermarkt`` is the Transfermarkt player ID.
    """
    col_opta = "key_opta_numeric"
    col_tm = "key_transfermarkt"
    if col_opta not in reep_df.columns or col_tm not in reep_df.columns:
        print(
            f"[transfer_values] Warning: Reep CSV missing expected columns "
            f"({col_opta}, {col_tm}).  Columns present: {list(reep_df.columns[:10])}"
        )
        return {}

    subset = reep_df[reep_df[col_opta].notna() & reep_df[col_tm].notna()][[col_opta, col_tm]]
    # opta_numeric can be stored as float in CSV (e.g. "244851.0") — normalise to int
    result: dict[int, str] = {}
    for _, row in subset.iterrows():
        try:
            code = int(float(row[col_opta]))
            tm_id = str(row[col_tm]).strip()
            if tm_id:
                result[code] = tm_id
        except (ValueError, TypeError):
            continue
    return result


# ---------------------------------------------------------------------------
# Table management
# ---------------------------------------------------------------------------


def ensure_table(conn: sqlite3.Connection) -> None:
    conn.executescript(_DDL)
    conn.commit()


# ---------------------------------------------------------------------------
# Step 1 — populate TM IDs from Reep
# ---------------------------------------------------------------------------


def ensure_tm_ids(conn: sqlite3.Connection, force_download: bool = False) -> None:
    """
    Populate ``tm_id`` for all players in ``players_raw`` that don't have one.

    Uses Reep ``data/people.csv`` to map ``fpl_code`` → ``tm_id``.  Players
    already in ``player_transfer_values`` with a non-null ``tm_id`` are skipped
    unless ``force_download`` is set (which also refreshes the Reep cache).
    """
    ensure_table(conn)

    try:
        raw_df = pd.read_sql(
            "SELECT id AS fpl_id, code AS fpl_code FROM players_raw", conn
        )
    except Exception as exc:
        print(f"[transfer_values] Could not read players_raw: {exc}")
        return

    if raw_df.empty:
        print("[transfer_values] players_raw is empty — nothing to map.")
        return

    # Find players already mapped
    try:
        existing = pd.read_sql(
            "SELECT fpl_id FROM player_transfer_values WHERE tm_id IS NOT NULL",
            conn,
        )
        mapped_ids = set(existing["fpl_id"])
    except Exception:
        mapped_ids = set()

    to_map = raw_df[~raw_df["fpl_id"].isin(mapped_ids)]
    if to_map.empty and not force_download:
        print(
            f"[transfer_values] All {len(mapped_ids)} players already have TM IDs — "
            "skipping Reep download."
        )
        return

    reep_df = _load_reep_df(force_download=force_download)
    code_to_tm = _build_code_to_tm_id(reep_df)

    rows = []
    for _, player in raw_df.iterrows():
        fpl_id = int(player["fpl_id"])
        fpl_code = int(player["fpl_code"]) if not pd.isna(player["fpl_code"]) else None
        tm_id = code_to_tm.get(fpl_code) if fpl_code is not None else None
        rows.append({"fpl_id": fpl_id, "fpl_code": fpl_code, "tm_id": tm_id})

    mapped = sum(1 for r in rows if r["tm_id"] is not None)
    print(
        f"[transfer_values] Reep mapping: {mapped}/{len(rows)} players "
        f"({mapped/len(rows)*100:.1f}%) resolved to Transfermarkt IDs"
    )

    # Upsert — preserve any existing tm_value_* columns
    for row in rows:
        conn.execute(
            """
            INSERT INTO player_transfer_values (fpl_id, fpl_code, tm_id)
            VALUES (:fpl_id, :fpl_code, :tm_id)
            ON CONFLICT(fpl_id) DO UPDATE SET
                fpl_code = excluded.fpl_code,
                tm_id = COALESCE(excluded.tm_id, player_transfer_values.tm_id)
            """,
            row,
        )
    conn.commit()


# ---------------------------------------------------------------------------
# Step 2 — fetch values from Transfermarkt
# ---------------------------------------------------------------------------


def refresh_transfer_values(
    conn: sqlite3.Connection, force: bool = False
) -> None:
    """
    Fetch current market values from Transfermarkt for eligible players.

    Eligibility:
    - ``force=True``: all players with a non-null ``tm_id``
    - ``force=False``: players where ``tm_fetched_at IS NULL OR tm_retry = 1``

    Conditional update:
    - Stored value is only replaced if the new ``tm_value_date`` is strictly
      newer than what is already stored (or if no value was stored yet).

    Fallback:
    - Players whose Transfermarkt page returns no value receive the mean of
      all successfully fetched values, and ``tm_retry = 1`` is set so every
      subsequent training run retries them.
    """
    ensure_table(conn)

    if force:
        query = (
            "SELECT fpl_id, tm_id, tm_value_date FROM player_transfer_values "
            "WHERE tm_id IS NOT NULL"
        )
    else:
        query = (
            "SELECT fpl_id, tm_id, tm_value_date FROM player_transfer_values "
            "WHERE tm_id IS NOT NULL AND (tm_fetched_at IS NULL OR tm_retry = 1)"
        )

    try:
        to_fetch = pd.read_sql(query, conn)
    except Exception as exc:
        print(f"[transfer_values] Could not query player_transfer_values: {exc}")
        return

    if to_fetch.empty:
        print("[transfer_values] No players need Transfermarkt value refresh.")
        return

    print(
        f"[transfer_values] Fetching Transfermarkt values for "
        f"{len(to_fetch)} player(s)..."
    )

    fetched_values: list[float] = []
    results: list[dict] = []

    for _, row in to_fetch.iterrows():
        fpl_id = int(row["fpl_id"])
        tm_id = str(row["tm_id"])
        stored_date = row.get("tm_value_date")

        try:
            data = get_transfermarkt_data(tm_id)
            new_value: float | None = data.get("transfer_value_eur")
            new_date: str | None = data.get("last_updated")
        except Exception as exc:
            print(
                f"[transfer_values]   fpl_id={fpl_id} tm_id={tm_id}: "
                f"fetch error — {exc}"
            )
            new_value = None
            new_date = None

        now = datetime.utcnow().isoformat(timespec="seconds") + "Z"

        if new_value is not None:
            # Only update if new date is strictly newer (or we have nothing stored)
            should_update = (
                stored_date is None
                or new_date is None
                or new_date > stored_date
            )
            if should_update:
                conn.execute(
                    """
                    UPDATE player_transfer_values SET
                        tm_value_eur     = ?,
                        tm_value_date    = ?,
                        tm_fetched_at    = ?,
                        tm_value_imputed = 0,
                        tm_retry         = 0
                    WHERE fpl_id = ?
                    """,
                    (new_value, new_date, now, fpl_id),
                )
                fetched_values.append(new_value)
                results.append(
                    {"fpl_id": fpl_id, "status": "updated", "value": new_value}
                )
            else:
                # Data exists but isn't newer — mark as fetched, clear retry
                conn.execute(
                    "UPDATE player_transfer_values SET tm_fetched_at = ?, tm_retry = 0 "
                    "WHERE fpl_id = ?",
                    (now, fpl_id),
                )
                results.append({"fpl_id": fpl_id, "status": "skipped_stale"})
        else:
            # No value found — will apply mean fallback after the loop
            results.append({"fpl_id": fpl_id, "status": "missing"})
            conn.execute(
                "UPDATE player_transfer_values SET tm_fetched_at = ?, tm_retry = 1 "
                "WHERE fpl_id = ?",
                (now, fpl_id),
            )

        conn.commit()
        time.sleep(_TM_SLEEP_SECONDS)

    # Apply mean fallback to players with no value
    missing_ids = [r["fpl_id"] for r in results if r["status"] == "missing"]
    if missing_ids:
        if fetched_values:
            mean_val = sum(fetched_values) / len(fetched_values)
        else:
            # Fall back to mean of all stored non-imputed values in the table
            cur = conn.execute(
                "SELECT AVG(tm_value_eur) FROM player_transfer_values "
                "WHERE tm_value_eur IS NOT NULL AND tm_value_imputed = 0"
            )
            mean_val = cur.fetchone()[0] or 0.0

        now = datetime.utcnow().isoformat(timespec="seconds") + "Z"
        for fpl_id in missing_ids:
            conn.execute(
                """
                UPDATE player_transfer_values SET
                    tm_value_eur     = ?,
                    tm_value_date    = NULL,
                    tm_fetched_at    = ?,
                    tm_value_imputed = 1,
                    tm_retry         = 1
                WHERE fpl_id = ?
                """,
                (mean_val, now, fpl_id),
            )
        conn.commit()
        print(
            f"[transfer_values]   {len(missing_ids)} players had no TM data — "
            f"assigned mean value (€{mean_val:,.0f}), tm_retry=1 set."
        )

    updated = sum(1 for r in results if r["status"] == "updated")
    skipped = sum(1 for r in results if r["status"] == "skipped_stale")
    print(
        f"[transfer_values] Done: {updated} updated, "
        f"{skipped} skipped (data not newer), {len(missing_ids)} imputed."
    )


# ---------------------------------------------------------------------------
# train.py entry point
# ---------------------------------------------------------------------------


def ensure_transfer_values(conn: sqlite3.Connection) -> None:
    """
    Called by ``train.py`` after ``ensure_new_players()``.

    Ensures TM IDs are resolved for all current players and fetches values
    for any player that hasn't been fetched yet or has ``tm_retry = 1``.
    """
    ensure_table(conn)
    ensure_tm_ids(conn)
    if not refresh_transfer_values_from_bulk(conn):
        refresh_transfer_values(conn, force=False)


def refresh_transfer_values_from_bulk(conn: sqlite3.Connection) -> bool:
    """Refresh values from the weekly CC0 Transfermarkt bulk dataset."""
    try:
        response = requests.get(_BULK_PLAYERS_URL, headers=_HEADERS, timeout=120)
        response.raise_for_status()
        players = pd.read_csv(
            io.BytesIO(response.content),
            compression="gzip",
            usecols=[
                "player_id", "name", "first_name", "last_name",
                "date_of_birth", "market_value_in_eur",
            ],
        ).dropna(subset=["player_id", "date_of_birth", "market_value_in_eur"])
    except Exception as exc:
        print(f"[transfer_values] Bulk refresh failed ({exc}); using profile fetches.")
        return False

    players["tm_id"] = players["player_id"].astype(int).astype(str)
    values = dict(zip(players["tm_id"], players["market_value_in_eur"].astype(float)))
    rows = conn.execute(
        "SELECT fpl_id, tm_id FROM player_transfer_values WHERE tm_id IS NOT NULL"
    ).fetchall()
    now = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    updated = 0
    for fpl_id, tm_id in rows:
        value = values.get(str(tm_id))
        if value is None:
            continue
        conn.execute(
            """
            UPDATE player_transfer_values SET
                tm_value_eur = ?, tm_fetched_at = ?,
                tm_value_imputed = 0, tm_retry = 0
            WHERE fpl_id = ?
            """,
            (value, now, int(fpl_id)),
        )
        updated += 1

    players["birth_date"] = pd.to_datetime(
        players["date_of_birth"], errors="coerce"
    ).dt.strftime("%Y-%m-%d")
    players["normalized_names"] = players.apply(
        lambda player: {
            _normalize_name(player["name"]),
            _normalize_name(f"{player['first_name']} {player['last_name']}"),
        },
        axis=1,
    )
    missing = pd.read_sql(
        """
        SELECT p.id AS fpl_id, p.first_name, p.second_name, p.web_name,
               p.known_name, p.birth_date
        FROM players_raw p
        JOIN player_transfer_values v ON v.fpl_id = p.id
        WHERE v.tm_value_eur IS NULL
        """,
        conn,
    )
    fallback_matches = 0
    for player in missing.itertuples():
        fpl_names = {
            _normalize_name(f"{player.first_name} {player.second_name}"),
            _normalize_name(player.web_name),
            _normalize_name(player.known_name),
        } - {""}
        candidates = players[players["birth_date"] == str(player.birth_date)]
        exact = candidates[
            candidates["normalized_names"].map(
                lambda names: not fpl_names.isdisjoint(names)
            )
        ]
        if len(exact) != 1:
            continue
        match = exact.iloc[0]
        conn.execute(
            """
            UPDATE player_transfer_values SET
                tm_id = ?, tm_value_eur = ?, tm_fetched_at = ?,
                tm_value_imputed = 0, tm_retry = 0
            WHERE fpl_id = ?
            """,
            (
                str(int(match["player_id"])),
                float(match["market_value_in_eur"]),
                now,
                int(player.fpl_id),
            ),
        )
        fallback_matches += 1

    mean_value = conn.execute(
        """
        SELECT AVG(tm_value_eur) FROM player_transfer_values
        WHERE tm_value_eur IS NOT NULL AND tm_value_imputed = 0
        """
    ).fetchone()[0]
    if mean_value:
        conn.execute(
            """
            UPDATE player_transfer_values SET
                tm_value_eur = ?, tm_fetched_at = ?,
                tm_value_imputed = 1, tm_retry = 1
            WHERE tm_value_eur IS NULL
            """,
            (float(mean_value), now),
        )
    conn.commit()
    imputed = conn.execute(
        "SELECT COUNT(*) FROM player_transfer_values WHERE tm_value_imputed = 1"
    ).fetchone()[0]
    print(
        f"[transfer_values] Bulk dataset updated {updated} mapped players, "
        f"matched {fallback_matches} by DOB/name, imputed {imputed}."
    )
    return updated + fallback_matches > 0


def _normalize_name(value) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    ascii_text = text.encode("ascii", "ignore").decode("ascii").lower()
    return re.sub(r"[^a-z0-9]", "", ascii_text)


# ---------------------------------------------------------------------------
# Feature accessor (used by build_features in utils.py)
# ---------------------------------------------------------------------------


def load_transfer_values(conn: sqlite3.Connection) -> pd.DataFrame:
    """
    Return a DataFrame with columns ``[fpl_id, tm_market_value]``.

    ``tm_market_value`` is the log10-scaled EUR value, which compresses the
    extreme spread between fringe players (~€500K) and elite players (~€150M)
    into a range the model handles well.  Players with NULL values (TM IDs
    not yet resolved or fetch pending) receive 0.0.
    """
    try:
        df = pd.read_sql(
            "SELECT fpl_id, tm_value_eur, tm_value_imputed FROM player_transfer_values "
            "WHERE tm_value_eur IS NOT NULL",
            conn,
        )
    except Exception:
        return pd.DataFrame(columns=[
            "fpl_id", "tm_market_value", "tm_value_imputed"
        ])

    if df.empty:
        return pd.DataFrame(columns=[
            "fpl_id", "tm_market_value", "tm_value_imputed"
        ])

    df["tm_market_value"] = df["tm_value_eur"].apply(
        lambda v: round(math.log10(max(v, 1.0)), 4) if v and v > 0 else 0.0
    )
    return df[["fpl_id", "tm_market_value", "tm_value_imputed"]]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Transfermarkt market value pipeline"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-fetch all players ignoring cached values",
    )
    parser.add_argument(
        "--rebuild-ids",
        action="store_true",
        help="Force re-download of Reep CSV and remap all FPL player codes",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Print row counts and exit",
    )
    args = parser.parse_args()

    conn = sqlite3.connect(_DB_PATH)
    ensure_table(conn)

    if args.status:
        n_total = conn.execute(
            "SELECT COUNT(*) FROM player_transfer_values"
        ).fetchone()[0]
        n_with_tm = conn.execute(
            "SELECT COUNT(*) FROM player_transfer_values WHERE tm_id IS NOT NULL"
        ).fetchone()[0]
        n_with_val = conn.execute(
            "SELECT COUNT(*) FROM player_transfer_values WHERE tm_value_eur IS NOT NULL"
        ).fetchone()[0]
        n_imputed = conn.execute(
            "SELECT COUNT(*) FROM player_transfer_values WHERE tm_value_imputed = 1"
        ).fetchone()[0]
        n_retry = conn.execute(
            "SELECT COUNT(*) FROM player_transfer_values WHERE tm_retry = 1"
        ).fetchone()[0]
        n_raw = conn.execute("SELECT COUNT(*) FROM players_raw").fetchone()[0]
        print(f"players_raw:            {n_raw} rows")
        print(f"player_transfer_values: {n_total} rows")
        print(f"  → with tm_id:         {n_with_tm}")
        print(f"  → with value:         {n_with_val}")
        print(f"  → imputed (mean):     {n_imputed}")
        print(f"  → retry pending:      {n_retry}")
    elif args.rebuild_ids:
        ensure_tm_ids(conn, force_download=True)
        refresh_transfer_values(conn, force=args.force)
    else:
        ensure_tm_ids(conn)
        refresh_transfer_values(conn, force=args.force)

    conn.close()
