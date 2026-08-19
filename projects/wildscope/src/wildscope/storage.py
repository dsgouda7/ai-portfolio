"""SQLite-backed observation, prediction, and adaptive-model cache."""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from wildscope.contracts import ModelPrediction, ObservationPhoto


class WildlifeStore:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._artifact_root = path.parent.resolve()
        self._connection = sqlite3.connect(path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        self._create_schema()
        self._rebase_cached_paths()

    def upsert_observation(
        self,
        feed_id: str,
        photo: ObservationPhoto,
        *,
        sha256: str | None = None,
        cached_path: str | None = None,
    ) -> bool:
        with self._lock:
            existing = self._connection.execute(
                "SELECT 1 FROM observations WHERE photo_id = ?", (photo.photo_id,)
            ).fetchone()
            self._connection.execute(
                """
                INSERT INTO observations (
                    photo_id, feed_id, observation_id, observed_at, created_at,
                    taxon_id, scientific_name, common_name, photo_url, license_code,
                    attribution, quality_grade, sha256, cached_path, synced_at,
                    latitude, longitude, positional_accuracy, coordinates_obscured,
                    original_width, original_height, taxon_group
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(photo_id) DO UPDATE SET
                    feed_id=excluded.feed_id, scientific_name=excluded.scientific_name,
                    common_name=excluded.common_name, photo_url=excluded.photo_url,
                    license_code=excluded.license_code, attribution=excluded.attribution,
                    quality_grade=excluded.quality_grade,
                    sha256=COALESCE(excluded.sha256, observations.sha256),
                    cached_path=COALESCE(excluded.cached_path, observations.cached_path),
                    latitude=excluded.latitude, longitude=excluded.longitude,
                    positional_accuracy=excluded.positional_accuracy,
                    coordinates_obscured=excluded.coordinates_obscured,
                    original_width=excluded.original_width,
                    original_height=excluded.original_height,
                    taxon_group=excluded.taxon_group,
                    synced_at=excluded.synced_at
                """,
                (
                    photo.photo_id,
                    feed_id,
                    photo.observation_id,
                    photo.observed_at,
                    photo.created_at,
                    photo.taxon_id,
                    photo.scientific_name,
                    photo.common_name,
                    photo.photo_url,
                    photo.license_code,
                    photo.attribution,
                    photo.quality_grade,
                    sha256,
                    cached_path,
                    datetime.now(UTC).isoformat(),
                    photo.latitude,
                    photo.longitude,
                    photo.positional_accuracy,
                    photo.coordinates_obscured,
                    photo.original_width,
                    photo.original_height,
                    photo.taxon_group,
                ),
            )
            self._connection.commit()
            return existing is None

    def save_prediction(
        self, photo_id: int, kind: str, prediction: ModelPrediction
    ) -> None:
        with self._lock:
            self._connection.execute(
                """
                INSERT INTO predictions (
                    photo_id, model_kind, label, confidence, model_version, trained_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(photo_id, model_kind) DO UPDATE SET
                    label=excluded.label, confidence=excluded.confidence,
                    model_version=excluded.model_version, trained_at=excluded.trained_at
                """,
                (
                    photo_id,
                    kind,
                    prediction.label,
                    prediction.confidence,
                    prediction.model_version,
                    prediction.trained_at,
                ),
            )
            self._connection.commit()

    def cache_photo(
        self,
        photo_id: int,
        sha256: str,
        cached_path: str,
        *,
        normalized_path: str | None = None,
        enhanced_path: str | None = None,
        model_input_path: str | None = None,
        cached_width: int | None = None,
        cached_height: int | None = None,
        enhancement_method: str | None = None,
        enhancement_applied: bool = False,
    ) -> bool:
        with self._lock:
            duplicate = self._connection.execute(
                "SELECT photo_id FROM observations WHERE sha256=? AND photo_id<>?",
                (sha256, photo_id),
            ).fetchone()
            if duplicate is not None:
                self._connection.execute(
                    "DELETE FROM observations WHERE photo_id=?", (photo_id,)
                )
                self._connection.commit()
                return False
            self._connection.execute(
                """
                UPDATE observations SET
                    sha256=?, cached_path=?, normalized_path=?, enhanced_path=?,
                    model_input_path=?, cached_width=?, cached_height=?,
                    enhancement_method=?, enhancement_applied=?
                WHERE photo_id=?
                """,
                (
                    sha256,
                    cached_path,
                    normalized_path,
                    enhanced_path,
                    model_input_path,
                    cached_width,
                    cached_height,
                    enhancement_method,
                    enhancement_applied,
                    photo_id,
                ),
            )
            self._connection.commit()
            return True

    def save_adaptive_model(
        self, feed_id: str, model_id: str, trained_at: str, watermark: str, payload: dict[str, Any]
    ) -> None:
        with self._lock:
            self._connection.execute(
                """
                INSERT INTO adaptive_models (feed_id, model_id, trained_at, watermark, payload_json)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(feed_id) DO UPDATE SET
                    model_id=excluded.model_id, trained_at=excluded.trained_at,
                    watermark=excluded.watermark, payload_json=excluded.payload_json
                """,
                (feed_id, model_id, trained_at, watermark, json.dumps(payload, sort_keys=True)),
            )
            self._connection.commit()

    def adaptive_model(self, feed_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM adaptive_models WHERE feed_id = ?", (feed_id,)
            ).fetchone()
        if row is None:
            return None
        return {
            "feed_id": row["feed_id"],
            "model_id": row["model_id"],
            "trained_at": row["trained_at"],
            "watermark": row["watermark"],
            "payload": json.loads(row["payload_json"]),
        }

    def save_training_run(
        self,
        feed_id: str,
        run_id: str,
        started_at: str,
        finished_at: str,
        details: dict[str, Any],
    ) -> None:
        with self._lock:
            self._connection.execute(
                """
                INSERT INTO training_runs (
                    run_id, feed_id, started_at, finished_at, details_json
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    finished_at=excluded.finished_at, details_json=excluded.details_json
                """,
                (run_id, feed_id, started_at, finished_at, json.dumps(details, sort_keys=True)),
            )
            self._connection.commit()

    def training_history(self, feed_id: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT run_id, feed_id, started_at, finished_at, details_json
                FROM training_runs WHERE feed_id=? ORDER BY finished_at DESC
                """,
                (feed_id,),
            ).fetchall()
        return [
            {
                "run_id": row["run_id"],
                "feed_id": row["feed_id"],
                "started_at": row["started_at"],
                "finished_at": row["finished_at"],
                "details": json.loads(row["details_json"]),
            }
            for row in rows
        ]

    def locations(self, feed_id: str) -> list[dict[str, Any]]:
        animal_filter = self._animal_filter("p")
        with self._lock:
            rows = self._connection.execute(
                f"""
                WITH points AS (
                    SELECT MIN(o.photo_id) AS anchor_photo_id,
                           o.latitude, o.longitude,
                           MAX(o.positional_accuracy) AS positional_accuracy,
                           MAX(o.coordinates_obscured) AS coordinates_obscured,
                           COUNT(*) AS photo_count,
                           MAX(o.created_at) AS latest_created_at
                    FROM observations o
                    JOIN predictions p
                      ON p.photo_id=o.photo_id AND p.model_kind='static'
                    WHERE o.feed_id=? AND o.latitude IS NOT NULL
                      AND o.longitude IS NOT NULL AND {animal_filter}
                    GROUP BY o.latitude, o.longitude
                )
                SELECT points.*,
                       COALESCE(o.common_name, o.scientific_name) AS common_name
                FROM points
                JOIN observations o ON o.photo_id=points.anchor_photo_id
                ORDER BY points.latest_created_at DESC
                """,
                (feed_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def frame_detail(self, photo_id: int) -> dict[str, Any] | None:
        with self._lock:
            row = self._connection.execute(
                """
                SELECT o.*,
                       sp.label AS static_label, sp.confidence AS static_confidence,
                       sp.model_version AS static_model_version,
                       ap.label AS adaptive_label, ap.confidence AS adaptive_confidence,
                       ap.model_version AS adaptive_model_version,
                       ap.trained_at AS adaptive_trained_at
                FROM observations o
                LEFT JOIN predictions sp
                  ON sp.photo_id=o.photo_id AND sp.model_kind='static'
                LEFT JOIN predictions ap
                  ON ap.photo_id=o.photo_id AND ap.model_kind='adaptive'
                WHERE o.photo_id=?
                """,
                (photo_id,),
            ).fetchone()
        return dict(row) if row else None

    def location_frames(
        self, feed_id: str, anchor_photo_id: int, *, limit: int = 50
    ) -> list[dict[str, Any]]:
        animal_filter = self._animal_filter("sp")
        with self._lock:
            anchor = self._connection.execute(
                """
                SELECT latitude, longitude FROM observations
                WHERE photo_id=? AND feed_id=?
                """,
                (anchor_photo_id, feed_id),
            ).fetchone()
            if anchor is None or anchor["latitude"] is None or anchor["longitude"] is None:
                return []
            rows = self._connection.execute(
                f"""
                SELECT o.*,
                       sp.label AS static_label, sp.confidence AS static_confidence,
                       sp.model_version AS static_model_version,
                       ap.label AS adaptive_label, ap.confidence AS adaptive_confidence,
                       ap.model_version AS adaptive_model_version,
                       ap.trained_at AS adaptive_trained_at
                FROM observations o
                JOIN predictions sp
                  ON sp.photo_id=o.photo_id AND sp.model_kind='static'
                LEFT JOIN predictions ap
                  ON ap.photo_id=o.photo_id AND ap.model_kind='adaptive'
                WHERE o.feed_id=? AND o.latitude=? AND o.longitude=?
                  AND {animal_filter}
                ORDER BY o.created_at DESC, o.photo_id DESC LIMIT ?
                """,
                (
                    feed_id,
                    anchor["latitude"],
                    anchor["longitude"],
                    limit,
                ),
            ).fetchall()
        return [dict(row) for row in rows]

    def frames(self, feed_id: str, *, page: int, per_page: int = 10) -> dict[str, Any]:
        if page < 1 or per_page != 10:
            raise ValueError("page must be positive and per_page is fixed at 10")
        offset = (page - 1) * per_page
        animal_filter = self._animal_filter("sp")
        with self._lock:
            total = int(
                self._connection.execute(
                    f"""
                    SELECT COUNT(*) FROM observations o
                    JOIN predictions sp
                      ON sp.photo_id=o.photo_id AND sp.model_kind='static'
                    WHERE o.feed_id = ? AND {animal_filter}
                    """,
                    (feed_id,),
                ).fetchone()[0]
            )
            rows = self._connection.execute(
                f"""
                SELECT o.*,
                    sp.label AS static_label, sp.confidence AS static_confidence,
                    sp.model_version AS static_model_version,
                    ap.label AS adaptive_label, ap.confidence AS adaptive_confidence,
                    ap.model_version AS adaptive_model_version, ap.trained_at AS adaptive_trained_at
                FROM observations o
                JOIN predictions sp ON sp.photo_id=o.photo_id AND sp.model_kind='static'
                LEFT JOIN predictions ap ON ap.photo_id=o.photo_id AND ap.model_kind='adaptive'
                WHERE o.feed_id = ? AND {animal_filter}
                ORDER BY o.created_at DESC, o.photo_id DESC
                LIMIT ? OFFSET ?
                """,
                (feed_id, per_page, offset),
            ).fetchall()
        pages = max(1, (total + per_page - 1) // per_page)
        return {"items": [dict(row) for row in rows], "page": page, "pages": pages, "total": total}

    def training_rows(self, feed_id: str, watermark: str | None) -> list[dict[str, Any]]:
        query = """
                 SELECT o.photo_id, o.observation_id, o.scientific_name,
                     o.common_name, o.created_at,
                 o.quality_grade, p.label AS static_label,
                 p.confidence AS static_confidence,
                  p.model_version AS static_model_version,
                  ap.label AS deployed_label,
                  ap.confidence AS deployed_confidence,
                  ap.model_version AS deployed_model_version
            FROM observations o
            JOIN predictions p ON p.photo_id=o.photo_id AND p.model_kind='static'
              LEFT JOIN predictions ap
                ON ap.photo_id=o.photo_id AND ap.model_kind='adaptive'
             WHERE o.feed_id=? AND o.license_code IS NOT NULL
            AND o.quality_grade='research'
        """
        params: list[Any] = [feed_id]
        if watermark:
            query += " AND o.created_at > ?"
            params.append(watermark)
        query += " ORDER BY o.created_at"
        with self._lock:
            return [dict(row) for row in self._connection.execute(query, params).fetchall()]

    def static_rows(self, feed_id: str) -> list[dict[str, Any]]:
        with self._lock:
            return [
                dict(row)
                for row in self._connection.execute(
                    """
                          SELECT o.photo_id,
                              COALESCE(o.model_input_path, o.cached_path) AS cached_path,
                              o.created_at, o.scientific_name,
                           p.label AS static_label, p.confidence AS static_confidence,
                           p.model_version AS static_model_version
                    FROM observations o
                    JOIN predictions p ON p.photo_id=o.photo_id AND p.model_kind='static'
                    WHERE o.feed_id=?
                    ORDER BY o.created_at DESC
                    """,
                    (feed_id,),
                ).fetchall()
            ]

    def cached_images_without_static(self, feed_id: str) -> dict[int, Path]:
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT o.photo_id, COALESCE(o.model_input_path, o.cached_path) AS cached_path
                FROM observations o
                LEFT JOIN predictions p ON p.photo_id=o.photo_id AND p.model_kind='static'
                WHERE o.feed_id=? AND o.cached_path IS NOT NULL AND p.photo_id IS NULL
                """,
                (feed_id,),
            ).fetchall()
        return {int(row["photo_id"]): Path(row["cached_path"]) for row in rows}

    def confidence_summary(self, feed_id: str) -> dict[str, float | int | None]:
        animal_filter = self._animal_filter("sp")
        with self._lock:
            row = self._connection.execute(
                f"""
                SELECT COUNT(*) AS sample_count,
                       AVG(sp.confidence) AS baseline_mean_confidence,
                       AVG(ap.confidence) AS adaptive_mean_confidence
                FROM observations o
                JOIN predictions sp
                  ON sp.photo_id=o.photo_id AND sp.model_kind='static'
                LEFT JOIN predictions ap
                  ON ap.photo_id=o.photo_id AND ap.model_kind='adaptive'
                WHERE o.feed_id=? AND {animal_filter}
                """,
                (feed_id,),
            ).fetchone()
        baseline = float(row["baseline_mean_confidence"]) if row[1] is not None else None
        adaptive = float(row["adaptive_mean_confidence"]) if row[2] is not None else None
        return {
            "sample_count": int(row["sample_count"]),
            "baseline_mean_confidence": baseline,
            "adaptive_mean_confidence": adaptive,
            "confidence_delta": (
                adaptive - baseline if adaptive is not None and baseline is not None else None
            ),
        }

    def cached_path(self, photo_id: int) -> Path | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT cached_path FROM observations WHERE photo_id=?", (photo_id,)
            ).fetchone()
        return Path(row["cached_path"]) if row and row["cached_path"] else None

    @staticmethod
    def _animal_filter(alias: str) -> str:
        return f"""
            LOWER({alias}.label) NOT LIKE '%blank%'
            AND LOWER({alias}.label) NOT LIKE '%human%'
            AND LOWER({alias}.label) NOT LIKE '%vehicle%'
            AND LOWER({alias}.label) NOT LIKE '%unknown%'
            AND COALESCE(LOWER(o.taxon_group), '')
                NOT IN ('insecta', 'arachnida', 'reptilia')
        """

    def close(self) -> None:
        self._connection.close()

    def _create_schema(self) -> None:
        with self._lock:
            self._connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS observations (
                    photo_id INTEGER PRIMARY KEY,
                    feed_id TEXT NOT NULL,
                    observation_id INTEGER NOT NULL,
                    observed_at TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    taxon_id INTEGER,
                    scientific_name TEXT NOT NULL,
                    common_name TEXT,
                    photo_url TEXT NOT NULL,
                    license_code TEXT,
                    attribution TEXT,
                    quality_grade TEXT NOT NULL,
                    sha256 TEXT,
                    cached_path TEXT,
                    synced_at TEXT NOT NULL,
                    latitude REAL,
                    longitude REAL,
                    positional_accuracy REAL,
                    coordinates_obscured INTEGER NOT NULL DEFAULT 0,
                    original_width INTEGER,
                    original_height INTEGER,
                    normalized_path TEXT,
                    enhanced_path TEXT,
                    model_input_path TEXT,
                    cached_width INTEGER,
                    cached_height INTEGER,
                    enhancement_method TEXT,
                    enhancement_applied INTEGER NOT NULL DEFAULT 0,
                    taxon_group TEXT
                );
                CREATE UNIQUE INDEX IF NOT EXISTS observations_sha256
                    ON observations(sha256) WHERE sha256 IS NOT NULL;
                CREATE INDEX IF NOT EXISTS observations_feed_created
                    ON observations(feed_id, created_at DESC);
                CREATE TABLE IF NOT EXISTS predictions (
                    photo_id INTEGER NOT NULL,
                    model_kind TEXT NOT NULL,
                    label TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    model_version TEXT NOT NULL,
                    trained_at TEXT,
                    PRIMARY KEY(photo_id, model_kind)
                );
                CREATE TABLE IF NOT EXISTS adaptive_models (
                    feed_id TEXT PRIMARY KEY,
                    model_id TEXT NOT NULL,
                    trained_at TEXT NOT NULL,
                    watermark TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS training_runs (
                    run_id TEXT PRIMARY KEY,
                    feed_id TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    finished_at TEXT NOT NULL,
                    details_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS training_runs_feed_finished
                    ON training_runs(feed_id, finished_at DESC);
                """
            )
            columns = {
                row[1] for row in self._connection.execute("PRAGMA table_info(observations)")
            }
            additions = {
                "latitude": "REAL",
                "longitude": "REAL",
                "positional_accuracy": "REAL",
                "coordinates_obscured": "INTEGER NOT NULL DEFAULT 0",
                "original_width": "INTEGER",
                "original_height": "INTEGER",
                "normalized_path": "TEXT",
                "enhanced_path": "TEXT",
                "model_input_path": "TEXT",
                "cached_width": "INTEGER",
                "cached_height": "INTEGER",
                "enhancement_method": "TEXT",
                "enhancement_applied": "INTEGER NOT NULL DEFAULT 0",
                "taxon_group": "TEXT",
            }
            for name, definition in additions.items():
                if name not in columns:
                    self._connection.execute(
                        f"ALTER TABLE observations ADD COLUMN {name} {definition}"
                    )
            self._connection.commit()

    def _rebase_cached_paths(self) -> None:
        columns = (
            "cached_path",
            "normalized_path",
            "enhanced_path",
            "model_input_path",
        )
        with self._lock:
            rows = self._connection.execute(
                f"SELECT photo_id, {', '.join(columns)} FROM observations"
            ).fetchall()
            for row in rows:
                updates = {}
                for column in columns:
                    value = row[column]
                    if not value or Path(value).is_file():
                        continue
                    parts = Path(value).parts
                    try:
                        marker = parts.index("artifacts")
                    except ValueError:
                        continue
                    relative_parts = parts[marker + 1 :]
                    if relative_parts and relative_parts[0] in {"wildscope", "runtime"}:
                        relative_parts = relative_parts[1:]
                    candidate = self._artifact_root.joinpath(*relative_parts)
                    if candidate.is_file():
                        updates[column] = str(candidate)
                if updates:
                    assignments = ", ".join(f"{column}=?" for column in updates)
                    self._connection.execute(
                        f"UPDATE observations SET {assignments} WHERE photo_id=?",
                        (*updates.values(), row["photo_id"]),
                    )
            self._connection.commit()
