"""
eligibility.py -- filter the player pool by current FPL availability.

Pure rule-based approach using the FPL API's own status and
chance_of_playing_next_round fields.  These cover ~95%+ of eligibility
decisions deterministically (loaned out, long-term injured, suspended,
clearly doubtful).  The remaining genuinely ambiguous players (doubtful
with no clear chance figure) are treated as available -- the conservative
choice that avoids dropping players who are likely to start.

Usage:
    from eligibility import get_eligibility
    eligibility = get_eligibility()  # {(first_name, second_name): EligibilityResult}
    pool = pool[pool['id'].map(lambda i: eligibility.get(i, _DEFAULT).eligible)]
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import requests

from utils import FPL_API_BASE

_DEFINITE_OUT = {'u', 'n'}   # u=unavailable/loaned/sold, n=not in squad

_RULE_PLAY_THRESHOLD  = 50   # chance >= this → available
_RULE_BENCH_THRESHOLD = 25   # chance < this  → unavailable
# chance 25-49 or no figure given → treat as available (conservative)


@dataclass
class EligibilityResult:
    eligible:  bool
    status:    str          # raw FPL status letter
    chance:    Optional[int]
    news:      str
    ep_next:   float        # FPL's own expected points for next round
    method:    str          # 'rule' | 'llm' | 'default'
    reason:    str          # human-readable explanation


_DEFAULT = EligibilityResult(
    eligible=True, status='a', chance=None, news='',
    ep_next=0.0, method='default', reason='not in current FPL API response'
)

# used when we have a valid API response but the player is simply absent --
# they have left the Premier League entirely (sold abroad, retired, relegated club)
_ABSENT = EligibilityResult(
    eligible=False, status='u', chance=0, news='',
    ep_next=0.0, method='rule', reason='not in current FPL season'
)


def player_name_key(first_name: str, second_name: str) -> tuple[str, str]:
    """Canonical (first_name_lower, second_name_lower) key used by the eligibility dict."""
    return (str(first_name or '').lower(), str(second_name or '').lower())


def get_epl_members() -> frozenset[tuple[str, str]]:
    """
    Return a frozenset of player_name_key tuples for every player currently
    registered at an EPL club, regardless of fitness or availability.

    Players with status 'u' (sold abroad, loaned out of the PL, retired) are
    excluded.  Players who are injured, suspended, or doubtful are included --
    they are still EPL assets and their historical data is valid for training.
    """
    resp = requests.get(f'{FPL_API_BASE}/bootstrap-static/', timeout=20)
    resp.raise_for_status()
    members = frozenset(
        player_name_key(p['first_name'], p['second_name'])
        for p in resp.json()['elements']
        if p.get('status', 'u') != 'u'
    )
    return members


def _rule_based(status: str, chance: Optional[int]) -> bool:
    """Return True if the player is selectable, False otherwise."""
    if status in _DEFINITE_OUT:
        return False
    if status in ('i', 's'):
        return False
    if chance is not None:
        if chance >= _RULE_PLAY_THRESHOLD:
            return True
        if chance < _RULE_BENCH_THRESHOLD:
            return False
    # available status with no chance figure, or doubtful with chance 25-49
    # → conservative: treat as available
    return True


EligibilityKey = tuple[str, str]  # (first_name.lower(), second_name.lower())

def get_eligibility() -> dict[EligibilityKey, EligibilityResult]:
    """
    Pull the current player list from the FPL API and return a rule-based
    eligibility map keyed by (first_name.lower(), second_name.lower()).

    Keying by name rather than player ID avoids mismatches caused by FPL
    reassigning player IDs between seasons.  Players absent from the current
    API have left the Premier League; callers should use _ABSENT for those.
    """
    resp = requests.get(f'{FPL_API_BASE}/bootstrap-static/', timeout=20)
    resp.raise_for_status()
    players = resp.json()['elements']

    results: dict[EligibilityKey, EligibilityResult] = {}

    for p in players:
        key    = (p['first_name'].lower(), p['second_name'].lower())
        status = p.get('status', 'a')
        chance = p.get('chance_of_playing_next_round')
        news   = p.get('news') or ''
        ep_raw = p.get('ep_next') or '0'
        ep     = float(ep_raw) if ep_raw else 0.0

        eligible = _rule_based(status, chance)
        results[key] = EligibilityResult(
            eligible=eligible,
            status=status,
            chance=chance,
            news=news,
            ep_next=ep,
            method='rule',
            reason=(
                'available per FPL status' if eligible
                else (news[:120] if news else f'status={status}')
            ),
        )

    flagged = sum(1 for r in results.values() if not r.eligible)
    print(f'  Eligibility: {len(results)} players checked, {flagged} filtered out')
    return results
