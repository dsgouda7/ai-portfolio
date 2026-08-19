"""Feed synchronization, inference, and test-then-train model versioning."""

from __future__ import annotations

import hashlib
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import requests

from wildscope.contracts import ModelPrediction, WildlifeFeed
from wildscope.feeds import InaturalistClient
from wildscope.inference import (
    SUPERVISED_PROTOCOL_VERSION,
    SpeciesNetRunner,
    StaticWildlifeModel,
    apply_adaptive_corrector,
    evaluate_adaptive_corrector,
    evaluate_identification_rows,
    model_timestamp,
    train_adaptive_corrector,
)
from wildscope.preprocessing import prepare_image
from wildscope.storage import WildlifeStore

ALLOWED_IMAGE_HOSTS = {
    "static.inaturalist.org",
    "inaturalist-open-data.s3.amazonaws.com",
}


@dataclass(slots=True)
class FeedJob:
    job_id: str
    feed_id: str
    kind: str
    state: str = "pending"
    started_at: str | None = None
    finished_at: str | None = None
    processed: int = 0
    total: int = 0
    error: str | None = None
    details: dict[str, object] = field(default_factory=dict)

    def public_dict(self) -> dict[str, object]:
        return {
            "job_id": self.job_id,
            "feed_id": self.feed_id,
            "kind": self.kind,
            "state": self.state,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "processed": self.processed,
            "total": self.total,
            "error": self.error,
            "details": dict(self.details),
        }


class WildlifeService:
    def __init__(
        self,
        feeds: tuple[WildlifeFeed, ...],
        store: WildlifeStore,
        cache_root: Path,
        *,
        client: InaturalistClient | None = None,
        static_model: StaticWildlifeModel | None = None,
    ) -> None:
        self.feeds = {feed.feed_id: feed for feed in feeds}
        self.store = store
        self.cache_root = cache_root
        self.client = client or InaturalistClient()
        self.static_model = static_model or SpeciesNetRunner(cache_root / "jobs")
        self._jobs: dict[str, FeedJob] = {}
        self._active_by_feed: dict[str, str] = {}
        self._lock = threading.RLock()
        self._http = requests.Session()
        self._http.trust_env = False
        self._http.headers.update({"User-Agent": "WildScope/0.1 personal research portfolio"})

    def list_feeds(self) -> list[dict[str, object]]:
        result = []
        for feed in self.feeds.values():
            model = self.store.adaptive_model(feed.feed_id)
            history = self.store.training_history(feed.feed_id)
            latest_run = history[0]["details"] if history else None
            result.append(
                {
                    **feed.public_dict(),
                    "adaptive_model": (
                        {
                            "model_id": model["model_id"],
                            "trained_at": model["trained_at"],
                            "watermark": model["watermark"],
                            "sample_count": model["payload"].get(
                                "labeled_sample_count",
                                model["payload"].get("sample_count", 0),
                            ),
                            "latest_metrics": latest_run,
                        }
                        if model
                        else None
                    ),
                }
            )
        return result

    def start_sync(self, feed_id: str, *, hours: int = 24) -> FeedJob:
        feed = self._feed(feed_id)
        with self._lock:
            if feed_id in self._active_by_feed:
                raise RuntimeError("a feed job is already active")
            job = FeedJob(f"sync-{uuid.uuid4().hex}", feed_id, "sync")
            self._jobs[job.job_id] = job
            self._active_by_feed[feed_id] = job.job_id
        threading.Thread(
            target=self._sync_worker,
            args=(job, feed, hours),
            name=f"wildscope-{job.job_id}",
            daemon=True,
        ).start()
        return job

    def start_training(self, feed_id: str) -> FeedJob:
        feed = self._feed(feed_id)
        with self._lock:
            if feed_id in self._active_by_feed:
                raise RuntimeError("a feed job is already active")
            job = FeedJob(f"train-{uuid.uuid4().hex}", feed_id, "training")
            self._jobs[job.job_id] = job
            self._active_by_feed[feed_id] = job.job_id
        threading.Thread(
            target=self._training_worker,
            args=(job, feed),
            name=f"wildscope-{job.job_id}",
            daemon=True,
        ).start()
        return job

    def job(self, job_id: str) -> FeedJob:
        with self._lock:
            try:
                return self._jobs[job_id]
            except KeyError as error:
                raise LookupError("unknown feed job") from error

    def frames(self, feed_id: str, page: int) -> dict[str, Any]:
        self._feed(feed_id)
        return self.store.frames(feed_id, page=page)

    def locations(self, feed_id: str) -> list[dict[str, Any]]:
        self._feed(feed_id)
        return self.store.locations(feed_id)

    def location_frames(self, feed_id: str, anchor_photo_id: int) -> list[dict[str, Any]]:
        self._feed(feed_id)
        return self.store.location_frames(feed_id, anchor_photo_id)

    def frame_detail(self, photo_id: int) -> dict[str, Any]:
        detail = self.store.frame_detail(photo_id)
        if detail is None:
            raise LookupError("unknown frame")
        return detail

    def training_dashboard(self, feed_id: str) -> dict[str, Any]:
        self._feed(feed_id)
        model = self.store.adaptive_model(feed_id)
        history = self.store.training_history(feed_id)
        batch_rows, bootstrap_migration = self._pending_training_rows(
            feed_id, model
        )
        live_batch = self._batch_snapshot(
            batch_rows,
            None if bootstrap_migration else model,
            bootstrap_migration=bootstrap_migration,
        )
        return {
            "model": (
                {
                    "model_id": model["model_id"],
                    "trained_at": model["trained_at"],
                    "watermark": model["watermark"],
                    "sample_count": model["payload"].get(
                        "labeled_sample_count",
                        model["payload"].get("sample_count", 0),
                    ),
                    "training_samples": model["payload"].get("sample_count", 0),
                    "protocol_version": model["payload"].get("protocol_version"),
                }
                if model
                else None
            ),
            "live_batch": live_batch,
            "confidence": self.store.confidence_summary(feed_id),
            "runs": history,
        }

    def image_path(self, photo_id: int, stage: str = "source") -> Path:
        detail = self.store.frame_detail(photo_id)
        if detail is None:
            raise LookupError("image is not cached")
        field = {
            "source": "cached_path",
            "normalized": "normalized_path",
            "enhanced": "enhanced_path",
            "model-input": "model_input_path",
        }.get(stage)
        if field is None:
            raise ValueError("unknown image stage")
        value = detail.get(field) or detail.get("cached_path")
        path = Path(str(value)) if value else None
        if path is None or not path.is_file():
            raise LookupError("image is not cached")
        return path

    def _sync_worker(self, job: FeedJob, feed: WildlifeFeed, hours: int) -> None:
        self._start(job)
        try:
            photos = self.client.fetch_recent(feed, hours=hours)
            job.total = len(photos)
            stats = self._ingest_photos(job, feed, photos)
            self._apply_adaptive(feed.feed_id)
            job.details = {
                "observations": len(photos),
                **stats,
            }
            self._finish(job, "completed")
        except Exception as error:
            job.error = f"{type(error).__name__}: {str(error)[:500]}"
            self._finish(job, "failed")

    def _training_worker(self, job: FeedJob, feed: WildlifeFeed) -> None:
        self._start(job)
        started = time.monotonic()
        try:
            existing = self.store.adaptive_model(job.feed_id)
            watermark = str(existing["watermark"]) if existing else None
            photos = (
                self.client.fetch_since(feed, since=watermark)
                if watermark
                else self.client.fetch_recent(feed, hours=24)
            )
            job.total = len(photos)
            ingest = self._ingest_photos(job, feed, photos)
            if existing:
                self._apply_adaptive(job.feed_id)
            post_watermark_rows = self.store.training_rows(job.feed_id, watermark)
            new_rows, bootstrap_migration = self._pending_training_rows(
                job.feed_id, existing
            )
            if not new_rows:
                raise ValueError("no new labeled, licensed predictions are available")
            baseline_evaluation = evaluate_identification_rows(
                new_rows, label_field="static_label"
            )
            deployed_evaluation = (
                evaluate_identification_rows(new_rows, label_field="deployed_label")
                if existing and not bootstrap_migration
                else {"samples": len(new_rows), "correct": None, "accuracy": None}
            )
            payload = train_adaptive_corrector(new_rows)
            trained_at = model_timestamp()
            newest = max(str(row["created_at"]) for row in new_rows)
            training_evaluation = evaluate_adaptive_corrector(
                new_rows, payload, trained_at=trained_at
            )
            payload.update(
                {
                    "labeled_sample_count": len(new_rows),
                    "target_source": "iNaturalist community identification",
                    "target_quality": "research",
                    "training_strategy": "all-eligible-data-since-previous-watermark",
                    "batch_window": {"from": watermark, "to": newest},
                }
            )
            model_id = f"adaptive-{job.feed_id}-{trained_at.replace(':', '').replace('+', '')}"
            self.store.save_adaptive_model(
                job.feed_id, model_id, trained_at, newest, payload
            )
            self._apply_adaptive(job.feed_id)
            job.processed = job.total = len(new_rows)
            job.details = {
                "model_id": model_id,
                "evaluated_model_id": (
                    str(existing["model_id"])
                    if existing and not bootstrap_migration
                    else "SpeciesNet baseline"
                ),
                "trained_model_id": model_id,
                "trained_at": trained_at,
                "watermark": newest,
                "new_samples": len(post_watermark_rows),
                "total_samples": len(new_rows),
                "training_samples": len(new_rows),
                "bootstrap_migration": bootstrap_migration,
                "duration_seconds": round(time.monotonic() - started, 3),
                "watermark_from": watermark,
                "watermark_to": newest,
                "target_source": payload["target_source"],
                "target_quality": payload["target_quality"],
                "training_strategy": payload["training_strategy"],
                "baseline_correct": baseline_evaluation["correct"],
                "baseline_accuracy": baseline_evaluation["accuracy"],
                "deployed_correct": deployed_evaluation["correct"],
                "deployed_accuracy": deployed_evaluation["accuracy"],
                "training_agreement": training_evaluation["adaptive_accuracy"],
                "samples": self._batch_samples(new_rows),
                **ingest,
            }
            self._finish(job, "completed")
            self.store.save_training_run(
                job.feed_id,
                job.job_id,
                str(job.started_at),
                str(job.finished_at),
                job.details,
            )
        except Exception as error:
            job.error = f"{type(error).__name__}: {str(error)[:500]}"
            self._finish(job, "failed")

    def _batch_snapshot(
        self,
        rows: list[dict[str, Any]],
        model: dict[str, Any] | None,
        *,
        bootstrap_migration: bool = False,
    ) -> dict[str, Any]:
        baseline = evaluate_identification_rows(rows, label_field="static_label")
        deployed = (
            evaluate_identification_rows(rows, label_field="deployed_label")
            if model
            else {"samples": len(rows), "correct": None, "accuracy": None}
        )
        return {
            "status": (
                "bootstrap-ready"
                if rows and bootstrap_migration
                else "ready" if rows else "awaiting-data"
            ),
            "evaluated_model_id": str(model["model_id"]) if model else "SpeciesNet baseline",
            "window_from": (
                str(model["watermark"])
                if model and not bootstrap_migration
                else None
            ),
            "window_to": max((str(row["created_at"]) for row in rows), default=None),
            "eligible_samples": len(rows),
            "bootstrap_migration": bootstrap_migration,
            "baseline": baseline,
            "deployed": deployed,
            "samples": self._batch_samples(rows),
        }

    def _pending_training_rows(
        self, feed_id: str, model: dict[str, Any] | None
    ) -> tuple[list[dict[str, Any]], bool]:
        watermark = str(model["watermark"]) if model else None
        rows = self.store.training_rows(feed_id, watermark)
        bootstrap_migration = bool(model) and (
            model["payload"].get("protocol_version") != SUPERVISED_PROTOCOL_VERSION
        )
        if not rows and bootstrap_migration:
            return self.store.training_rows(feed_id, None), True
        return rows, False

    @staticmethod
    def _batch_samples(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "photo_id": int(row["photo_id"]),
                "observation_id": int(row["observation_id"]),
                "created_at": str(row["created_at"]),
                "obtained_scientific_name": str(row["scientific_name"]),
                "obtained_common_name": row.get("common_name"),
                "quality_grade": str(row["quality_grade"]),
                "baseline_label": str(row["static_label"]),
                "baseline_confidence": float(row["static_confidence"]),
                "deployed_label": row.get("deployed_label"),
                "deployed_confidence": row.get("deployed_confidence"),
            }
            for row in rows
        ]

    def _ingest_photos(
        self,
        job: FeedJob,
        feed: WildlifeFeed,
        photos: tuple[Any, ...],
        *,
        batch_size: int = 25,
    ) -> dict[str, int]:
        downloaded = 0
        duplicates = 0
        for photo in photos:
            self.store.upsert_observation(feed.feed_id, photo)
            existing = self.store.frame_detail(photo.photo_id)
            stage_ready = bool(
                existing
                and existing.get("model_input_path")
                and Path(str(existing["model_input_path"])).is_file()
            )
            if stage_ready:
                job.processed += 1
                continue
            try:
                path, digest = self._download_photo(
                    feed.feed_id, photo.photo_id, photo.photo_url
                )
                stages = prepare_image(
                    path,
                    self.cache_root / "images" / feed.feed_id / str(photo.photo_id) / "stages",
                )
            except Exception:
                job.processed += 1
                continue
            if self.store.cache_photo(
                photo.photo_id,
                digest,
                str(path),
                normalized_path=str(stages.normalized_path),
                enhanced_path=str(stages.enhanced_path),
                model_input_path=str(stages.model_input_path),
                cached_width=stages.source_size[0],
                cached_height=stages.source_size[1],
                enhancement_method=stages.enhancement_method,
                enhancement_applied=stages.enhancement_applied,
            ):
                downloaded += 1
            else:
                path.unlink(missing_ok=True)
                stages.normalized_path.unlink(missing_ok=True)
                stages.enhanced_path.unlink(missing_ok=True)
                duplicates += 1
            job.processed += 1

        images = self.store.cached_images_without_static(feed.feed_id)
        predictions_count = 0
        batch_count = 0
        image_items = list(images.items())
        for offset in range(0, len(image_items), batch_size):
            batch = dict(image_items[offset : offset + batch_size])
            predictions = self.static_model.predict(batch, country=feed.country)
            for photo_id, prediction in predictions.items():
                self.store.save_prediction(photo_id, "static", prediction)
            predictions_count += len(predictions)
            batch_count += 1
        return {
            "downloaded": downloaded,
            "duplicates_skipped": duplicates,
            "static_predictions": predictions_count,
            "batch_count": batch_count,
        }

    def _apply_adaptive(self, feed_id: str) -> None:
        model = self.store.adaptive_model(feed_id)
        if model is None:
            return
        for row in self.store.static_rows(feed_id):
            static = ModelPrediction(
                str(row["static_label"]),
                float(row["static_confidence"]),
                str(row["static_model_version"]),
            )
            prediction = apply_adaptive_corrector(
                static, model["payload"], trained_at=str(model["trained_at"])
            )
            self.store.save_prediction(int(row["photo_id"]), "adaptive", prediction)

    def _download_photo(self, feed_id: str, photo_id: int, url: str) -> tuple[Path, str]:
        candidates = [url]
        if "/original." in url:
            candidates.append(url.replace("/original.", "/large."))
        payload = None
        last_error: Exception | None = None
        for candidate in candidates:
            try:
                payload = self._download_image_payload(candidate)
                break
            except Exception as error:
                last_error = error
        if payload is None:
            raise last_error or RuntimeError("photo download failed")
        digest = hashlib.sha256(payload).hexdigest()
        path = self.cache_root / "images" / feed_id / str(photo_id) / "source.jpg"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        return path, digest

    def _download_image_payload(self, url: str) -> bytes:
        host = (urlsplit(url).hostname or "").lower()
        if host not in ALLOWED_IMAGE_HOSTS:
            raise ValueError("photo host is not allowlisted")
        with self._http.get(url, timeout=(3.0, 30.0), stream=True) as response:
            response.raise_for_status()
            content_type = response.headers.get("Content-Type", "").lower()
            if not content_type.startswith("image/"):
                raise ValueError("photo response is not an image")
            chunks = []
            size = 0
            for chunk in response.iter_content(65_536):
                size += len(chunk)
                if size > 15_000_000:
                    raise ValueError("photo exceeds byte limit")
                chunks.append(chunk)
        return b"".join(chunks)

    def _feed(self, feed_id: str) -> WildlifeFeed:
        try:
            return self.feeds[feed_id]
        except KeyError as error:
            raise LookupError("unknown wildlife feed") from error

    def _start(self, job: FeedJob) -> None:
        job.state = "running"
        job.started_at = datetime.now(UTC).isoformat()

    def _finish(self, job: FeedJob, state: str) -> None:
        job.state = state
        job.finished_at = datetime.now(UTC).isoformat()
        with self._lock:
            self._active_by_feed.pop(job.feed_id, None)
