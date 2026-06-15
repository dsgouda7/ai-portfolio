from __future__ import annotations

import datetime as dt
import re
import time
from typing import Any, Iterable

import requests
from bs4 import BeautifulSoup

_BASE_PROFILE_URL = "https://www.transfermarkt.com/-/profil/spieler/{player_id}"
_DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
}


def _safe_text(value: str | None) -> str | None:
    if value is None:
        return None
    text = re.sub(r"\s+", " ", value).strip()
    return text or None


def _clean_player_name(value: str | None) -> str | None:
    text = _safe_text(value)
    if not text:
        return text
    return re.sub(r"^#\d+\s+", "", text)


def _extract_slug_and_identifier(profile_url: str) -> tuple[str | None, str | None]:
    parts = [p for p in profile_url.strip("/").split("/") if p]
    if len(parts) >= 4 and parts[-2] == "spieler":
        return parts[-4], parts[-1]
    return None, None


def _extract_name(soup: BeautifulSoup) -> str | None:
    node = soup.select_one("h1.data-header__headline-wrapper")
    if node is None:
        node = soup.select_one("h1")
    return _safe_text(node.get_text(" ", strip=True) if node else None)


def _extract_market_value_text(soup: BeautifulSoup) -> str | None:
    node = soup.select_one("a.data-header__market-value-wrapper")
    if node is None:
        # Fallback for markup drift.
        node = soup.select_one("[class*='market-value']")
    return _safe_text(node.get_text(" ", strip=True) if node else None)


def _parse_value_amount_eur(market_value_text: str | None) -> float | None:
    if not market_value_text:
        return None

    match = re.search(r"€\s*([\d.,]+)\s*([mkbn]|th\.)?", market_value_text, flags=re.IGNORECASE)
    if not match:
        return None

    raw_number = match.group(1)
    suffix = (match.group(2) or "").lower()

    normalized = raw_number
    if "." in raw_number and "," in raw_number:
        decimal_separator = "." if raw_number.rfind(".") > raw_number.rfind(",") else ","
        thousands_separator = "," if decimal_separator == "." else "."
        normalized = normalized.replace(thousands_separator, "")
        normalized = normalized.replace(decimal_separator, ".")
    elif "," in raw_number:
        parts = raw_number.split(",")
        if len(parts[-1]) in {1, 2}:
            normalized = raw_number.replace(".", "").replace(",", ".")
        else:
            normalized = raw_number.replace(",", "")
    elif "." in raw_number:
        parts = raw_number.split(".")
        if len(parts[-1]) in {1, 2}:
            normalized = raw_number.replace(",", "")
        else:
            normalized = raw_number.replace(".", "")

    try:
        amount = float(normalized)
    except ValueError:
        return None

    if suffix in {"m"}:
        amount *= 1_000_000
    elif suffix in {"k", "th."}:
        amount *= 1_000
    elif suffix in {"bn"}:
        amount *= 1_000_000_000

    return amount


def _parse_last_updated_date(market_value_text: str | None) -> str | None:
    if not market_value_text:
        return None

    # Common Transfermarkt format: "Last update: dd/mm/yyyy".
    match = re.search(r"last\s*update\s*:\s*(\d{1,2}/\d{1,2}/\d{4})", market_value_text, flags=re.IGNORECASE)
    if match:
        raw_date = match.group(1)
        try:
            return dt.datetime.strptime(raw_date, "%d/%m/%Y").date().isoformat()
        except ValueError:
            return raw_date

    # Fallback to any dd/mm/yyyy date that appears in the value text.
    any_date = re.search(r"(\d{1,2}/\d{1,2}/\d{4})", market_value_text)
    if any_date:
        raw_date = any_date.group(1)
        try:
            return dt.datetime.strptime(raw_date, "%d/%m/%Y").date().isoformat()
        except ValueError:
            return raw_date

    return None


def get_player_market_value(
    player_id: str | int,
    *,
    timeout: int = 20,
    retries: int = 3,
    pause_seconds: float = 1.0,
    session: requests.Session | None = None,
) -> dict[str, Any]:
    """Fetch Transfermarkt player market value payload using a Transfermarkt player id."""
    pid = str(player_id).strip()
    if not pid:
        raise ValueError("player_id is required")

    url = _BASE_PROFILE_URL.format(player_id=pid)
    client = session or requests.Session()

    last_error: Exception | None = None
    response: requests.Response | None = None

    for attempt in range(1, retries + 1):
        try:
            response = client.get(url, headers=_DEFAULT_HEADERS, timeout=timeout, allow_redirects=True)
            response.raise_for_status()
            break
        except Exception as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(pause_seconds)

    if response is None:
        if last_error is not None:
            raise last_error
        raise RuntimeError("Request failed without response")

    soup = BeautifulSoup(response.text, "html.parser")
    resolved_url = response.url
    slug, resolved_id = _extract_slug_and_identifier(resolved_url)

    market_value_text = _extract_market_value_text(soup)
    market_value_eur = _parse_value_amount_eur(market_value_text)
    market_value_last_updated = _parse_last_updated_date(market_value_text)

    return {
        "player_id": resolved_id or pid,
        "slug": slug,
        "name": _clean_player_name(_extract_name(soup)),
        "market_value_text": market_value_text,
        "market_value_eur": market_value_eur,
        "market_value_last_updated": market_value_last_updated,
        "profile_url": resolved_url,
    }


def get_transfermarkt_data(
    player_id: str | int,
    *,
    timeout: int = 20,
    retries: int = 3,
    pause_seconds: float = 1.0,
) -> dict[str, Any]:
    """Return transfer value and last updated date for a Transfermarkt player id."""
    payload = get_player_market_value(
        player_id,
        timeout=timeout,
        retries=retries,
        pause_seconds=pause_seconds,
    )
    return {
        "player_id": payload.get("player_id") or str(player_id),
        "transfer_value_eur": payload.get("market_value_eur"),
        "last_updated": payload.get("market_value_last_updated"),
    }


def get_players_market_values(
    player_ids: Iterable[str | int],
    *,
    timeout: int = 20,
    retries: int = 3,
    pause_seconds: float = 1.0,
) -> list[dict[str, Any]]:
    """Fetch market value payloads for multiple Transfermarkt player ids."""
    with requests.Session() as session:
        return [
            get_player_market_value(
                player_id,
                timeout=timeout,
                retries=retries,
                pause_seconds=pause_seconds,
                session=session,
            )
            for player_id in player_ids
        ]
