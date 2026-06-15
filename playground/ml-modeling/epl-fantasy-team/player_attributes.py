"""EA FC (SOFIFA) season-level player quality features via sofifa.com.

Replaces the previous FBref / soccerdata integration.  SOFIFA (sofifa.com)
serves EPL player EA FC ratings as plain HTML -- no Cloudflare, no Selenium,
no Chrome required.  Ratings are 0-99 integers and cover every player:

  ea_overall      overall rating                         -- general quality
  ea_pace         PAC  (outfield) / DIV (GK)             -- pace / diving
  ea_shooting     SHO  (outfield) / HAN (GK)             -- shooting / handling
  ea_passing      PAS  (outfield) / KIC (GK)             -- passing / kicking
  ea_dribbling    DRI  (outfield) / REF (GK)             -- dribbling / reflexes
  ea_defending    DEF  (outfield) / SPD (GK)             -- defending / speed
  ea_physicality  PHY  (outfield) / POS (GK)             -- physicality / positioning

Cache: 6-month TTL per season.  Ratings are stable mid-season; at season
rollover the cache expires and is re-fetched automatically on the next
training run via ensure_new_players() called from train.py.

Data source: https://sofifa.com  -- plain requests + BeautifulSoup4, no auth.
Note: sofifa.com currently returns 403 (Cloudflare-protected).  The table
schema and registry are maintained for when a data source becomes available.
Player attributes default to zero-fill in the model if not populated.

CLI:
    python player_attributes.py            # fetch new / stale players (default)
    python player_attributes.py --force    # re-fetch all regardless of TTL
    python player_attributes.py --status   # print row counts and quit
"""

from __future__ import annotations

import argparse
import re
import sqlite3
import time
import unicodedata
from datetime import date, timedelta
from difflib import get_close_matches

import pandas as pd
import requests
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------

_DB_PATH = "fantasy_football.db"
_SOFIFA_BASE = "https://sofifa.com"
_EPL_LEAGUE_ID = 13          # EPL league filter on sofifa.com
_PLAYERS_PER_PAGE = 60       # sofifa serves 60 rows per page
_STALE_DAYS = 180            # 6-month TTL

# If this column is missing the schema is old (fb_* / us_*); DROP + recreate.
_REQUIRED_COLUMN = "ea_overall"

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

_DDL_ATTRIBUTES = """
CREATE TABLE IF NOT EXISTS player_attributes (
    fpl_id         INTEGER PRIMARY KEY,
    ea_name        TEXT,
    ea_team        TEXT,
    ea_position    TEXT,
    ea_overall     INTEGER,
    ea_pace        INTEGER,
    ea_shooting    INTEGER,
    ea_passing     INTEGER,
    ea_dribbling   INTEGER,
    ea_defending   INTEGER,
    ea_physicality INTEGER,
    season         TEXT,
    last_updated   TEXT
);
"""

_DDL_REGISTRY = """
CREATE TABLE IF NOT EXISTS player_id_registry (
    fpl_id   INTEGER PRIMARY KEY,
    fpl_name TEXT,
    ea_name  TEXT,
    ea_slug  TEXT,
    verified INTEGER DEFAULT 0,
    notes    TEXT
);
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _slugify(name: str) -> str:
    """Produce a stable ASCII slug: 'Rúben Dias' -> 'ruben-dias'."""
    nfkd = unicodedata.normalize("NFKD", name)
    ascii_name = nfkd.encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^\w\s-]", "", ascii_name).strip().lower()
    return re.sub(r"[\s_-]+", "-", slug)


# ---------------------------------------------------------------------------
# SOFIFA scraper
# ---------------------------------------------------------------------------


def _fetch_sofifa_stats() -> pd.DataFrame:
    """Paginate sofifa.com EPL players and return a DataFrame with ea_* columns.

    Returns columns:
        ea_slug, ea_name, ea_team, ea_position,
        ea_overall, ea_pace, ea_shooting, ea_passing,
        ea_dribbling, ea_defending, ea_physicality

    Note: sofifa.com is currently Cloudflare-protected and returns 403.
    This function will return an empty DataFrame if blocked.
    """
    records: list[dict] = []
    offset = 0

    while True:
        url = (
            f"{_SOFIFA_BASE}/players"
            f"?type=all&lg[]={_EPL_LEAGUE_ID}&offset={offset}"
        )
        try:
            resp = requests.get(url, headers=_HEADERS, timeout=20)
            resp.raise_for_status()
        except requests.RequestException as exc:
            print(f"[player_attributes] SOFIFA request failed (offset={offset}): {exc}")
            break

        soup = BeautifulSoup(resp.text, "html.parser")
        rows = soup.select("table.table tbody tr")
        if not rows:
            break  # no more pages

        for row in rows:
            try:
                # Name + team
                name_td = row.select_one("td.col-name")
                if name_td is None:
                    continue
                name_a = name_td.select_one("a.tooltip")
                if name_a is None:
                    continue
                player_name = name_a.get_text(strip=True)

                team_a = name_td.select_one("small > a")
                team_name = team_a.get_text(strip=True) if team_a else ""

                # Position
                pos_span = name_td.select_one("span[class*=pos]")
                position = pos_span.get_text(strip=True) if pos_span else ""

                def _int(selector: str) -> int:
                    td = row.select_one(selector)
                    if td is None:
                        return 0
                    txt = td.get_text(strip=True)
                    return int(txt) if txt.isdigit() else 0

                records.append(
                    {
                        "ea_slug": _slugify(player_name),
                        "ea_name": player_name,
                        "ea_team": team_name,
                        "ea_position": position,
                        "ea_overall": _int("td.col-ov"),
                        "ea_pace": _int("td.col-pac"),
                        "ea_shooting": _int("td.col-sho"),
                        "ea_passing": _int("td.col-pas"),
                        "ea_dribbling": _int("td.col-dri"),
                        "ea_defending": _int("td.col-def"),
                        "ea_physicality": _int("td.col-phy"),
                    }
                )
            except Exception as exc:  # noqa: BLE001
                print(f"[player_attributes] Row parse error: {exc}")
                continue

        if len(rows) < _PLAYERS_PER_PAGE:
            break  # last page

        offset += _PLAYERS_PER_PAGE
        time.sleep(0.8)  # polite crawl rate

    print(f"[player_attributes] SOFIFA: fetched {len(records)} EPL player rows")
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Table management
# ---------------------------------------------------------------------------


def ensure_table(conn: sqlite3.Connection) -> None:
    """Create player_attributes if absent; DROP + recreate if schema is stale."""
    cur = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='player_attributes'"
    )
    if cur.fetchone() is not None:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(player_attributes)")}
        if _REQUIRED_COLUMN not in cols:
            print(
                "[player_attributes] Old schema detected (no ea_* columns) -- "
                "dropping and recreating player_attributes table."
            )
            conn.execute("DROP TABLE player_attributes")
            conn.commit()
    conn.executescript(_DDL_ATTRIBUTES)
    conn.commit()


def ensure_registry(conn: sqlite3.Connection) -> None:
    """Create player_id_registry; migrate old fb_name/fb_slug schema if needed."""
    cur = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='player_id_registry'"
    )
    if cur.fetchone() is not None:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(player_id_registry)")}
        if "ea_name" not in cols:
            print(
                "[player_attributes] Old registry schema (fb_name/fb_slug) detected -- "
                "dropping and recreating player_id_registry."
            )
            conn.execute("DROP TABLE player_id_registry")
            conn.commit()
    conn.executescript(_DDL_REGISTRY)
    conn.commit()


# ---------------------------------------------------------------------------
# ID registry builder
# ---------------------------------------------------------------------------


def build_id_registry(conn: sqlite3.Connection) -> None:
    """Fuzzy-match FPL players to SOFIFA names and populate player_id_registry."""
    ensure_registry(conn)

    raw_df = pd.read_sql(
        "SELECT id AS fpl_id, web_name, second_name, first_name FROM players_raw", conn
    )
    ea_df = _fetch_sofifa_stats()

    if ea_df.empty:
        print("[player_attributes] SOFIFA returned no data -- registry not built.")
        return

    ea_slugs = ea_df["ea_slug"].tolist()
    ea_lookup = ea_df.set_index("ea_slug")[["ea_name"]].to_dict("index")

    rows = []
    for _, player in raw_df.iterrows():
        fpl_id = int(player["fpl_id"])
        fpl_name = str(player.get("web_name") or "")
        full_name = (
            f"{player.get('first_name', '')} {player.get('second_name', '')}".strip()
        )

        slug_candidates = [_slugify(fpl_name), _slugify(full_name)]
        matched_slug = None
        verified = 0

        for candidate in slug_candidates:
            if candidate in ea_lookup:
                matched_slug = candidate
                verified = 1
                break

        if matched_slug is None:
            # Fuzzy fallback
            for candidate in slug_candidates:
                matches = get_close_matches(candidate, ea_slugs, n=1, cutoff=0.82)
                if matches:
                    matched_slug = matches[0]
                    verified = 0
                    break

        ea_name = ea_lookup[matched_slug]["ea_name"] if matched_slug else None
        rows.append(
            {
                "fpl_id": fpl_id,
                "fpl_name": fpl_name,
                "ea_name": ea_name,
                "ea_slug": matched_slug,
                "verified": verified,
                "notes": None,
            }
        )

    if rows:
        reg_df = pd.DataFrame(rows)
        reg_df.to_sql("player_id_registry", conn, if_exists="replace", index=False)
        conn.commit()
        matched = reg_df["ea_slug"].notna().sum()
        print(
            f"[player_attributes] Registry: {len(rows)} FPL players, "
            f"{matched} matched to SOFIFA ({matched/len(rows)*100:.1f}%)"
        )


# ---------------------------------------------------------------------------
# Attribute refresh
# ---------------------------------------------------------------------------


def refresh_attributes(conn: sqlite3.Connection, force: bool = False) -> None:
    """Fetch SOFIFA data and upsert all EPL players into player_attributes."""
    ensure_table(conn)
    ensure_registry(conn)

    season = str(date.today().year)
    stale_cutoff = (date.today() - timedelta(days=_STALE_DAYS)).isoformat()

    if not force:
        cur = conn.execute(
            "SELECT COUNT(*) FROM player_attributes WHERE last_updated >= ?",
            (stale_cutoff,),
        )
        fresh_count = cur.fetchone()[0]
        if fresh_count > 0:
            print(
                f"[player_attributes] {fresh_count} fresh rows found -- "
                "skipping fetch (use --force to override)."
            )
            return

    ea_df = _fetch_sofifa_stats()
    if ea_df.empty:
        print("[player_attributes] No SOFIFA data returned -- attributes not updated.")
        return

    # Load registry to get fpl_id mapping
    try:
        reg_df = pd.read_sql(
            "SELECT fpl_id, ea_slug FROM player_id_registry WHERE ea_slug IS NOT NULL",
            conn,
        )
    except Exception:
        reg_df = pd.DataFrame(columns=["fpl_id", "ea_slug"])

    if reg_df.empty:
        print(
            "[player_attributes] Registry is empty -- run build_id_registry() first."
        )
        return

    ea_indexed = ea_df.set_index("ea_slug")
    now = date.today().isoformat()
    upserted = 0

    for _, reg_row in reg_df.iterrows():
        slug = reg_row["ea_slug"]
        if slug not in ea_indexed.index:
            continue
        r = ea_indexed.loc[slug]
        conn.execute(
            """INSERT OR REPLACE INTO player_attributes
               (fpl_id, ea_name, ea_team, ea_position,
                ea_overall, ea_pace, ea_shooting, ea_passing,
                ea_dribbling, ea_defending, ea_physicality,
                season, last_updated)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                int(reg_row["fpl_id"]),
                str(r.get("ea_name", "")),
                str(r.get("ea_team", "")),
                str(r.get("ea_position", "")),
                int(r.get("ea_overall", 0)),
                int(r.get("ea_pace", 0)),
                int(r.get("ea_shooting", 0)),
                int(r.get("ea_passing", 0)),
                int(r.get("ea_dribbling", 0)),
                int(r.get("ea_defending", 0)),
                int(r.get("ea_physicality", 0)),
                season,
                now,
            ),
        )
        upserted += 1

    conn.commit()
    print(f"[player_attributes] Upserted {upserted} player attribute rows.")


# ---------------------------------------------------------------------------
# train.py entry point
# ---------------------------------------------------------------------------


def ensure_new_players(conn: sqlite3.Connection) -> None:
    """Called by train.py after ingest.  Populates attributes for any player
    not yet in player_attributes (or rebuilds if table schema is stale).
    """
    ensure_table(conn)
    ensure_registry(conn)

    # Check whether registry is populated
    cur = conn.execute("SELECT COUNT(*) FROM player_id_registry")
    registry_rows = cur.fetchone()[0]
    if registry_rows == 0:
        print(
            "[player_attributes] Registry empty -- "
            "building registry + fetching attributes."
        )
        build_id_registry(conn)
        refresh_attributes(conn, force=True)
        return

    # Check for stale / missing attributes
    stale_cutoff = (date.today() - timedelta(days=_STALE_DAYS)).isoformat()
    cur = conn.execute(
        "SELECT COUNT(*) FROM player_attributes WHERE last_updated >= ?",
        (stale_cutoff,),
    )
    fresh_count = cur.fetchone()[0]

    cur2 = conn.execute("SELECT COUNT(*) FROM players_raw")
    total_players = cur2.fetchone()[0]

    if fresh_count < total_players * 0.5:
        print(
            f"[player_attributes] Only {fresh_count}/{total_players} players "
            "have fresh attributes -- refreshing."
        )
        refresh_attributes(conn, force=True)
    else:
        print(
            f"[player_attributes] {fresh_count} players have up-to-date "
            "attributes -- skipping fetch."
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SOFIFA player attribute fetcher")
    parser.add_argument(
        "--force", action="store_true", help="Re-fetch all players ignoring TTL"
    )
    parser.add_argument(
        "--status", action="store_true", help="Print row counts and exit"
    )
    parser.add_argument(
        "--rebuild-registry",
        action="store_true",
        help="Rebuild the ID registry from scratch",
    )
    args = parser.parse_args()

    conn = sqlite3.connect(_DB_PATH)
    ensure_table(conn)
    ensure_registry(conn)

    if args.status:
        n_attr = conn.execute("SELECT COUNT(*) FROM player_attributes").fetchone()[0]
        n_reg = conn.execute("SELECT COUNT(*) FROM player_id_registry").fetchone()[0]
        n_raw = conn.execute("SELECT COUNT(*) FROM players_raw").fetchone()[0]
        print(f"players_raw:        {n_raw} rows")
        print(f"player_id_registry: {n_reg} rows")
        print(f"player_attributes:  {n_attr} rows")
    elif args.rebuild_registry:
        build_id_registry(conn)
        refresh_attributes(conn, force=True)
    else:
        refresh_attributes(conn, force=args.force)

    conn.close()
