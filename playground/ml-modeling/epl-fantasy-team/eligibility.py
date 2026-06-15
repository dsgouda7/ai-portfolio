"""
eligibility.py -- filter the player pool by current FPL availability.

Two-tier approach:
  1. Rule-based: FPL's own status/chance_of_playing fields handle ~95% of cases
     without any LLM. Deterministic outcomes (loaned out, long-term injured,
     suspended) are resolved instantly.

  2. LLM-based: for the remaining ~5% -- genuinely doubtful players where
     chance_of_playing is ambiguous and the news string carries meaningful
     signal -- we pass the text to a local Ollama model for a binary call.
     This keeps the LLM entirely off the hot path and only fires on the handful
     of players that rule logic can't resolve confidently.

     Requires Ollama running locally (http://localhost:11434).
     Falls back gracefully to rule-based if Ollama is unavailable.

Usage:
    from eligibility import get_eligibility
    eligibility = get_eligibility(use_llm=True)  # {player_id: EligibilityResult}
    pool = pool[pool['id'].map(lambda i: eligibility.get(i, _DEFAULT).eligible)]
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

import requests

from utils import FPL_API_BASE

OLLAMA_URL    = os.environ.get('OLLAMA_URL', 'http://localhost:11434')
OLLAMA_MODEL  = os.environ.get('OLLAMA_MODEL', 'qwen2.5-coder:7b')

# players with these statuses are never going to play this season regardless
# of what the news says -- no point wasting an LLM call on them
_DEFINITE_OUT = {'u', 'n'}   # u=unavailable/loaned/sold, n=not in squad

# chance thresholds for rule-based decisions
_RULE_PLAY_THRESHOLD  = 50   # >= this → available without LLM
_RULE_BENCH_THRESHOLD = 25   # < this → unavailable without LLM
# between 25-49: ambiguous → LLM if use_llm else assume doubtful-available


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


def _rule_based(status: str, chance: Optional[int]) -> Optional[bool]:
    """
    Return True/False if the rule can decide confidently, None if ambiguous.
    None means: hand off to LLM or conservative default.
    """
    if status in _DEFINITE_OUT:
        return False
    if status == 'i':
        return False
    if status == 's':
        return False
    if chance is not None:
        if chance >= _RULE_PLAY_THRESHOLD:
            return True
        if chance < _RULE_BENCH_THRESHOLD:
            return False
    if status == 'a' and chance is None:
        return True
    return None  # ambiguous: doubtful with chance in 25-49 range, or no chance given


def _llm_call(name: str, pos: str, status: str, chance: Optional[int], news: str) -> tuple[bool, str]:
    """
    Ask Ollama to make an availability call for a genuinely ambiguous player.
    Returns (eligible, one_line_reason). Raises on connection error.
    """
    pos_names = {'1': 'GK', '2': 'DEF', '3': 'MID', '4': 'FWD'}
    pos_label  = pos_names.get(str(pos), str(pos))
    chance_str = f'{chance}%' if chance is not None else 'not stated'

    prompt = (
        f"Premier League fantasy football eligibility check.\n\n"
        f"Player: {name} ({pos_label})\n"
        f"FPL status: {status} | Chance of playing next round: {chance_str}\n"
        f"FPL news: \"{news}\"\n\n"
        f"Will this player be available to play in their next Premier League match?\n"
        f"Reply with exactly AVAILABLE or UNAVAILABLE on the first line, "
        f"then one sentence of reasoning."
    )
    r = requests.post(
        f'{OLLAMA_URL}/api/generate',
        json={'model': OLLAMA_MODEL, 'prompt': prompt, 'stream': False},
        timeout=8,
    )
    r.raise_for_status()
    text = r.json()['response'].strip()
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    eligible = lines[0].upper().startswith('AVAILABLE') if lines else True
    reason = lines[1] if len(lines) > 1 else lines[0] if lines else 'no response'
    return eligible, reason[:200]


# Name key: (first_name.lower(), second_name.lower()) -- survives season ID reassignment
EligibilityKey = tuple[str, str]

def get_eligibility(use_llm: bool = True) -> dict[EligibilityKey, EligibilityResult]:
    """
    Pull the current player list from the FPL API and return an eligibility
    map keyed by (first_name.lower(), second_name.lower()).

    Keying by name rather than player ID avoids mismatches caused by FPL
    reassigning player IDs between seasons.  Players whose name is absent
    from the current API have left the Premier League; callers should use
    _ABSENT for those.
    """
    resp = requests.get(f'{FPL_API_BASE}/bootstrap-static/', timeout=20)
    resp.raise_for_status()
    players = resp.json()['elements']

    results: dict[EligibilityKey, EligibilityResult] = {}
    llm_available = use_llm

    for p in players:
        key    = (p['first_name'].lower(), p['second_name'].lower())
        status = p.get('status', 'a')
        chance = p.get('chance_of_playing_next_round')
        news   = p.get('news') or ''
        ep_raw = p.get('ep_next') or '0'
        ep     = float(ep_raw) if ep_raw else 0.0
        name   = f"{p['first_name']} {p['second_name']}"
        pos    = p.get('element_type', 0)

        rule_result = _rule_based(status, chance)

        if rule_result is True:
            results[key] = EligibilityResult(
                eligible=True, status=status, chance=chance, news=news,
                ep_next=ep, method='rule', reason='available per FPL status'
            )
        elif rule_result is False:
            results[key] = EligibilityResult(
                eligible=False, status=status, chance=chance, news=news,
                ep_next=ep, method='rule',
                reason=news[:120] if news else f'status={status}'
            )
        else:
            # ambiguous -- try LLM, fall back to conservative rule
            if llm_available:
                try:
                    eligible, reason = _llm_call(name, pos, status, chance, news)
                    results[key] = EligibilityResult(
                        eligible=eligible, status=status, chance=chance, news=news,
                        ep_next=ep, method='llm', reason=reason
                    )
                except Exception as exc:
                    llm_available = False  # don't retry if Ollama is down
                    print(f'  Ollama unavailable ({exc}), falling back to rule-based for ambiguous players')
                    # conservative: treat doubtful with no clear signal as available
                    results[key] = EligibilityResult(
                        eligible=True, status=status, chance=chance, news=news,
                        ep_next=ep, method='rule',
                        reason=f'LLM unavailable; treated as available (doubtful, chance={chance})'
                    )
            else:
                results[key] = EligibilityResult(
                    eligible=True, status=status, chance=chance, news=news,
                    ep_next=ep, method='rule',
                    reason=f'doubtful; treated as available (chance={chance})'
                )

    flagged  = sum(1 for r in results.values() if not r.eligible)
    llm_used = sum(1 for r in results.values() if r.method == 'llm')
    print(f'  Eligibility: {len(results)} players checked, '
          f'{flagged} filtered out, {llm_used} resolved by LLM')

    return results
