from __future__ import annotations

import sqlite3
from collections import Counter
from contextlib import closing
from pathlib import Path
from typing import Any

import requests

from squad_state import STARTER_LIMITS
from utils import DB_FILE, FPL_API_BASE


def _season_start_year(season: str) -> int:
    try:
        return int(str(season).split('-', 1)[0])
    except (TypeError, ValueError) as exc:
        raise ValueError(f'Invalid season: {season!r}') from exc


def load_committed_gameweek_states(
    database: str | Path | None = None,
    season: str | None = None,
) -> list[dict[str, Any]]:
    """Return the final committed revision for every persisted Gameweek."""
    query = """
        SELECT state_json
        FROM squad_versions AS version
        WHERE status = 'committed'
          AND version_id = (
              SELECT MAX(candidate.version_id)
              FROM squad_versions AS candidate
              WHERE candidate.status = 'committed'
                AND candidate.season = version.season
                AND candidate.game_week = version.game_week
          )
    """
    parameters: tuple[Any, ...] = ()
    if season:
        query += ' AND season = ?'
        parameters = (season,)
    query += ' ORDER BY season, game_week'

    import json
    with closing(sqlite3.connect(str(database or DB_FILE))) as connection:
        table_exists = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'squad_versions'"
        ).fetchone()
        if not table_exists:
            return []
        rows = connection.execute(query, parameters).fetchall()
    return [json.loads(row[0]) for row in rows]


def load_sqlite_actual_points(
    game_week: int,
    season: str,
    database: str | Path | None = None,
) -> dict[int, dict[str, Any]]:
    """Load current-season GW totals, aggregating double Gameweek fixtures."""
    season_year = _season_start_year(season)
    cutoff = f'{season_year}-07-01'
    with closing(sqlite3.connect(str(database or DB_FILE))) as connection:
        rows = connection.execute(
            """
            SELECT element, SUM(total_points), SUM(minutes)
            FROM player_gw
            WHERE round = ?
              AND kickoff_time >= ?
              AND element IS NOT NULL
            GROUP BY element
            """,
            (int(game_week), cutoff),
        ).fetchall()
    return {
        int(player_id): {
            'points': int(points or 0),
            'minutes': int(minutes or 0),
            'played': int(minutes or 0) > 0,
        }
        for player_id, points, minutes in rows
    }


def fetch_live_actual_points(
    game_week: int,
    timeout: int = 15,
) -> dict[int, dict[str, Any]]:
    """Load authoritative current-season totals from FPL's event endpoint."""
    response = requests.get(
        f'{FPL_API_BASE}/event/{int(game_week)}/live/', timeout=timeout
    )
    response.raise_for_status()
    return {
        int(row['id']): {
            'points': int(row.get('stats', {}).get('total_points', 0) or 0),
            'minutes': int(row.get('stats', {}).get('minutes', 0) or 0),
            'played': bool(row.get('stats', {}).get('played')),
        }
        for row in response.json().get('elements', [])
    }


def load_actual_points(
    game_week: int,
    season: str,
    database: str | Path | None = None,
) -> dict[int, dict[str, Any]]:
    try:
        return fetch_live_actual_points(game_week)
    except requests.RequestException:
        return load_sqlite_actual_points(game_week, season, database)


def _formation_is_legal(player_ids: list[int], players: dict[int, dict[str, Any]]) -> bool:
    counts = Counter(players[player_id]['position'] for player_id in player_ids)
    return all(
        minimum <= counts[position] <= maximum
        for position, (minimum, maximum) in STARTER_LIMITS.items()
    )


def score_saved_squad(
    state: dict[str, Any],
    actuals: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    """Score the persisted lineup using official substitution and chip rules."""
    players = {int(player['id']): player for player in state['players']}
    original_starters = [int(player_id) for player_id in state['lineup']['starters']]
    bench = [int(player_id) for player_id in state['lineup']['bench']]
    starters = original_starters[:]
    substitutions: list[dict[str, Any]] = []

    def played(player_id: int) -> bool:
        result = actuals.get(player_id, {})
        return bool(result.get('played', int(result.get('minutes', 0)) > 0))

    missing_goalkeeper = next(
        (
            player_id for player_id in starters
            if players[player_id]['position'] == 'GK' and not played(player_id)
        ),
        None,
    )
    bench_goalkeeper = next(
        (player_id for player_id in bench if players[player_id]['position'] == 'GK'),
        None,
    )
    if missing_goalkeeper and bench_goalkeeper and played(bench_goalkeeper):
        starters[starters.index(missing_goalkeeper)] = bench_goalkeeper
        substitutions.append({'out': missing_goalkeeper, 'in': bench_goalkeeper})

    outfield_bench = [
        player_id for player_id in bench if players[player_id]['position'] != 'GK'
    ]
    for missing_player in original_starters:
        if players[missing_player]['position'] == 'GK' or played(missing_player):
            continue
        if missing_player not in starters:
            continue
        for substitute in outfield_bench:
            if not played(substitute) or substitute in starters:
                continue
            proposed = starters[:]
            proposed[proposed.index(missing_player)] = substitute
            if _formation_is_legal(proposed, players):
                starters = proposed
                substitutions.append({'out': missing_player, 'in': substitute})
                break

    captain = int(state['lineup']['captain'])
    vice_captain = int(state['lineup']['vice_captain'])
    effective_captain = captain if played(captain) else (vice_captain if played(vice_captain) else None)
    captain_multiplier = 3 if state.get('active_chip') == 'triple_captain' else 2

    scoring_ids = set(starters)
    if state.get('active_chip') == 'bench_boost':
        scoring_ids.update(bench)
    gross_points = sum(int(actuals.get(player_id, {}).get('points', 0)) for player_id in scoring_ids)
    captain_bonus = (
        (captain_multiplier - 1) * int(actuals.get(effective_captain, {}).get('points', 0))
        if effective_captain else 0
    )
    transfer_cost = int(state.get('transfer_points_cost', 0))
    total_points = gross_points + captain_bonus - transfer_cost

    player_rows = []
    for player_id, player in players.items():
        player_rows.append({
            'id': player_id,
            'name': f"{player['first_name']} {player['second_name']}".strip(),
            'position': player['position'],
            'points': int(actuals.get(player_id, {}).get('points', 0)),
            'minutes': int(actuals.get(player_id, {}).get('minutes', 0)),
            'started': player_id in original_starters,
            'counted': player_id in scoring_ids,
            'captain': player_id == effective_captain,
        })

    return {
        'total_points': total_points,
        'gross_points': gross_points,
        'captain_bonus': captain_bonus,
        'transfer_cost': transfer_cost,
        'effective_captain': effective_captain,
        'final_starters': starters,
        'substitutions': substitutions,
        'players': player_rows,
    }


def fetch_official_benchmarks(timeout: int = 15) -> dict[int, dict[str, Any]]:
    response = requests.get(f'{FPL_API_BASE}/bootstrap-static/', timeout=timeout)
    response.raise_for_status()
    return {
        int(event['id']): {
            'finished': bool(event.get('finished') and event.get('data_checked')),
            'average_points': int(event.get('average_entry_score') or 0),
            'highest_score': int(event.get('highest_score') or 0),
            'deadline': event.get('deadline_time'),
        }
        for event in response.json().get('events', [])
    }


def fetch_dream_team(game_week: int, timeout: int = 10) -> dict[str, Any]:
    response = requests.get(
        f'{FPL_API_BASE}/dream-team/{int(game_week)}/', timeout=timeout
    )
    response.raise_for_status()
    payload = response.json()
    team = payload.get('team', payload.get('dream_team', []))
    players = [
        {
            'id': int(row.get('element', row.get('id'))),
            'points': int(row.get('points', 0)),
            'position': int(row.get('position', index + 1)),
        }
        for index, row in enumerate(team)
    ]
    return {
        'points': sum(player['points'] for player in players),
        'players': players,
        'top_player': payload.get('top_player'),
    }


def build_performance_review(
    database: str | Path | None = None,
    season: str | None = None,
) -> dict[str, Any]:
    states = load_committed_gameweek_states(database, season)
    if not states:
        return {'season': season, 'gameweeks': [], 'summary': {}}

    selected_season = season or states[-1]['season']
    states = [state for state in states if state['season'] == selected_season]
    try:
        benchmarks = fetch_official_benchmarks()
    except requests.RequestException:
        benchmarks = {}

    reviews = []
    for state in states:
        game_week = int(state['game_week'])
        benchmark = benchmarks.get(game_week, {})
        actuals = load_actual_points(game_week, selected_season, database)
        finished = bool(benchmark.get('finished'))
        if not finished:
            reviews.append({
                'game_week': game_week,
                'revision': int(state.get('revision', 0)),
                'finished': False,
            })
            continue

        score = score_saved_squad(state, actuals)
        try:
            dream_team = fetch_dream_team(game_week)
        except requests.RequestException:
            dream_team = {'points': 0, 'players': [], 'top_player': None}
        average = int(benchmark.get('average_points', 0))
        dream_points = int(dream_team['points'])
        score.update({
            'game_week': game_week,
            'revision': int(state.get('revision', 0)),
            'finished': True,
            'active_chip': state.get('active_chip'),
            'average_points': average,
            'dream_team_points': dream_points,
            'highest_manager_score': int(benchmark.get('highest_score', 0)),
            'vs_average': score['total_points'] - average,
            'vs_dream_team': score['total_points'] - dream_points,
            'dream_team': dream_team,
        })
        reviews.append(score)

    completed = [review for review in reviews if review.get('finished')]
    summary = {
        'completed_gameweeks': len(completed),
        'total_points': sum(review['total_points'] for review in completed),
        'average_points': round(
            sum(review['total_points'] for review in completed) / len(completed), 1
        ) if completed else 0,
        'average_benchmark': round(
            sum(review['average_points'] for review in completed) / len(completed), 1
        ) if completed else 0,
        'points_vs_average': sum(review['vs_average'] for review in completed),
        'gameweeks_above_average': sum(review['vs_average'] > 0 for review in completed),
        'dream_team_capture': round(
            100 * sum(review['total_points'] for review in completed)
            / sum(review['dream_team_points'] for review in completed),
            1,
        ) if completed and sum(review['dream_team_points'] for review in completed) else 0,
    }
    return {'season': selected_season, 'gameweeks': reviews, 'summary': summary}
