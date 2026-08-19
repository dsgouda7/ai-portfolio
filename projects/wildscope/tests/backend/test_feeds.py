from datetime import UTC, datetime
from pathlib import Path

from wildscope.feeds import InaturalistClient, load_feeds


class Response:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self.payload


class Session:
    def __init__(self, pages):
        self.pages = pages
        self.calls = []
        self.headers = {}

    def get(self, url, *, params, timeout):
        self.calls.append((url, params, timeout))
        return Response(self.pages.pop(0))

    def close(self):
        pass


def test_feed_config_contains_ten_unique_tropical_places() -> None:
    root = Path(__file__).resolve().parents[2]

    feeds = load_feeds(root / "configs" / "feeds.yaml")

    assert len(feeds) == len({feed.feed_id for feed in feeds}) == 10
    assert all(feed.place_id > 0 and feed.country and feed.habitat for feed in feeds)


def test_recent_feed_paginates_and_deduplicates_photo_ids() -> None:
    observation = {
        "id": 10,
        "observed_on": "2026-08-18",
        "created_at": "2026-08-18T10:00:00Z",
        "quality_grade": "research",
        "geojson": {"type": "Point", "coordinates": [-78.3001, -0.7002]},
        "positional_accuracy": 18,
        "obscured": True,
        "taxon": {
            "id": 42,
            "name": "Panthera onca",
            "preferred_common_name": "Jaguar",
            "iconic_taxon_name": "Mammalia",
        },
        "photos": [
            {
                "id": 100,
                "url": "https://static.inaturalist.org/photos/100/square.jpg",
                "original_dimensions": {"width": 4096, "height": 2731},
                "license_code": "cc-by",
                "attribution": "Observer",
            }
        ],
    }
    session = Session(
        [
            {"results": [observation] * 100},
            {"results": [{**observation, "id": 11}]},
        ]
    )
    feed = load_feeds(Path(__file__).resolve().parents[2] / "configs" / "feeds.yaml")[0]

    photos = InaturalistClient(session=session).fetch_recent(
        feed,
        per_page=100,
        max_pages=3,
        now=datetime(2026, 8, 18, 12, tzinfo=UTC),
    )

    assert len(photos) == 1
    assert photos[0].photo_url.endswith("/original.jpg")
    assert photos[0].scientific_name == "Panthera onca"
    assert photos[0].taxon_group == "Mammalia"
    assert photos[0].latitude == -0.7002
    assert photos[0].longitude == -78.3001
    assert photos[0].positional_accuracy == 18
    assert photos[0].coordinates_obscured is True
    assert (photos[0].original_width, photos[0].original_height) == (4096, 2731)
    assert len(session.calls) == 2
    assert session.calls[0][1]["created_d1"].startswith("2026-08-17T12:00:00")
