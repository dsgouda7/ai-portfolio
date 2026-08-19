"""Official iNaturalist recent-observation feed client."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import requests
import yaml

from wildscope.contracts import ObservationPhoto, WildlifeFeed

API_ROOT = "https://api.inaturalist.org/v1"


class InaturalistClient:
    def __init__(self, *, session: requests.Session | None = None) -> None:
        self._session = session or requests.Session()
        self._session.headers.update(
            {"User-Agent": "WildScope/0.1 personal research portfolio"}
        )

    def fetch_recent(
        self,
        feed: WildlifeFeed,
        *,
        hours: int = 24,
        per_page: int = 100,
        max_pages: int = 10,
        now: datetime | None = None,
    ) -> tuple[ObservationPhoto, ...]:
        if not 1 <= hours <= 168:
            raise ValueError("hours must be between 1 and 168")
        if not 1 <= per_page <= 200:
            raise ValueError("per_page must be between 1 and 200")
        until = (now or datetime.now(UTC)).astimezone(UTC)
        since = until - timedelta(hours=hours)
        return self._fetch_window(
            feed, since=since, until=until, per_page=per_page, max_pages=max_pages
        )

    def fetch_since(
        self,
        feed: WildlifeFeed,
        *,
        since: str | datetime,
        per_page: int = 100,
        max_pages: int = 20,
        now: datetime | None = None,
    ) -> tuple[ObservationPhoto, ...]:
        parsed_since = (
            datetime.fromisoformat(since.replace("Z", "+00:00"))
            if isinstance(since, str)
            else since
        )
        if parsed_since.tzinfo is None:
            raise ValueError("since must include a timezone")
        until = (now or datetime.now(UTC)).astimezone(UTC)
        parsed_since = parsed_since.astimezone(UTC)
        if parsed_since > until:
            raise ValueError("since must not be later than now")
        return self._fetch_window(
            feed,
            since=parsed_since,
            until=until,
            per_page=per_page,
            max_pages=max_pages,
        )

    def resolve_taxon(self, taxon_id: int) -> dict[str, Any]:
        response = self._session.get(
            f"{API_ROOT}/taxa/{int(taxon_id)}",
            params={},
            timeout=(3.0, 20.0),
        )
        response.raise_for_status()
        payload = response.json()
        results = payload.get("results", []) if isinstance(payload, dict) else []
        if not results:
            raise LookupError("iNaturalist taxon was not found")
        taxon = results[0]
        return {
            "taxon_id": int(taxon["id"]),
            "scientific_name": str(taxon.get("name") or "unknown"),
            "common_name": (
                str(taxon["preferred_common_name"])
                if taxon.get("preferred_common_name")
                else None
            ),
        }

    def resolve_taxon_name(self, scientific_name: str) -> dict[str, Any]:
        response = self._session.get(
            f"{API_ROOT}/taxa",
            params={"q": str(scientific_name), "per_page": 10},
            timeout=(3.0, 20.0),
        )
        response.raise_for_status()
        payload = response.json()
        results = payload.get("results", []) if isinstance(payload, dict) else []
        normalized = str(scientific_name).casefold()
        taxon = next(
            (
                row
                for row in results
                if str(row.get("name") or "").casefold() == normalized
            ),
            None,
        )
        if taxon is None:
            raise LookupError("iNaturalist taxon was not found")
        return {
            "taxon_id": int(taxon["id"]),
            "scientific_name": str(taxon.get("name") or scientific_name),
            "common_name": (
                str(taxon["preferred_common_name"])
                if taxon.get("preferred_common_name")
                else None
            ),
        }

    def _fetch_window(
        self,
        feed: WildlifeFeed,
        *,
        since: datetime,
        until: datetime,
        per_page: int,
        max_pages: int,
    ) -> tuple[ObservationPhoto, ...]:
        photos: list[ObservationPhoto] = []
        seen_photo_ids: set[int] = set()
        for page in range(1, max_pages + 1):
            response = self._session.get(
                f"{API_ROOT}/observations",
                params={
                    "place_id": feed.place_id,
                    "taxon_id": 1,
                    "photos": "true",
                    "created_d1": since.isoformat(),
                    "created_d2": until.isoformat(),
                    "order_by": "created_at",
                    "order": "desc",
                    "page": page,
                    "per_page": per_page,
                },
                timeout=(3.0, 20.0),
            )
            response.raise_for_status()
            payload = response.json()
            results = payload.get("results", []) if isinstance(payload, dict) else []
            if not results:
                break
            for observation in results:
                photos.extend(_observation_photos(observation, seen_photo_ids))
            if len(results) < per_page:
                break
        return tuple(photos)

    def close(self) -> None:
        self._session.close()


def load_feeds(path: Path) -> tuple[WildlifeFeed, ...]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    rows = payload.get("feeds")
    if not isinstance(rows, list) or not rows:
        raise ValueError("wildlife feed config requires a non-empty feeds list")
    feeds = tuple(
        WildlifeFeed(
            feed_id=str(row["id"]),
            name=str(row["name"]),
            place_id=int(row["place_id"]),
            country=str(row["country"]),
            habitat=str(row["habitat"]),
        )
        for row in rows
    )
    if len({feed.feed_id for feed in feeds}) != len(feeds):
        raise ValueError("wildlife feed IDs must be unique")
    return feeds


def _observation_photos(
    observation: dict[str, Any], seen_photo_ids: set[int]
) -> list[ObservationPhoto]:
    taxon = observation.get("taxon") or {}
    geojson = observation.get("geojson") or {}
    coordinates = geojson.get("coordinates") or []
    longitude = float(coordinates[0]) if len(coordinates) >= 2 else None
    latitude = float(coordinates[1]) if len(coordinates) >= 2 else None
    observation_id = int(observation["id"])
    result = []
    for photo in observation.get("photos") or []:
        photo_id = int(photo["id"])
        if photo_id in seen_photo_ids:
            continue
        seen_photo_ids.add(photo_id)
        url = str(photo.get("url") or "").replace("square.", "original.")
        if not url.startswith("https://"):
            continue
        dimensions = photo.get("original_dimensions") or {}
        result.append(
            ObservationPhoto(
                observation_id=observation_id,
                photo_id=photo_id,
                observed_at=str(observation.get("observed_on") or ""),
                created_at=str(observation.get("created_at") or ""),
                taxon_id=int(taxon["id"]) if taxon.get("id") is not None else None,
                scientific_name=str(taxon.get("name") or "unknown"),
                common_name=(
                    str(taxon["preferred_common_name"])
                    if taxon.get("preferred_common_name")
                    else None
                ),
                photo_url=url,
                license_code=photo.get("license_code"),
                attribution=photo.get("attribution"),
                quality_grade=str(observation.get("quality_grade") or "unknown"),
                latitude=latitude,
                longitude=longitude,
                positional_accuracy=(
                    float(observation["positional_accuracy"])
                    if observation.get("positional_accuracy") is not None
                    else None
                ),
                coordinates_obscured=bool(observation.get("obscured")),
                original_width=(
                    int(dimensions["width"]) if dimensions.get("width") is not None else None
                ),
                original_height=(
                    int(dimensions["height"]) if dimensions.get("height") is not None else None
                ),
                taxon_group=(
                    str(taxon["iconic_taxon_name"])
                    if taxon.get("iconic_taxon_name")
                    else None
                ),
            )
        )
    return result
