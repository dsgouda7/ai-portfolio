"""Feed synchronization, inference, and test-then-train model versioning."""

from __future__ import annotations

import hashlib
import threading
import time
import uuid
from collections import Counter
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
    BioClipSpeciesClassifier,
    SpeciesNetRunner,
    StaticWildlifeModel,
    VisualSpeciesModel,
    apply_adaptive_corrector,
    canonical_model_label,
    describe_adaptive_prediction,
    describe_visual_prediction,
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

TRAINING_PIPELINE = (
    ("fetch", "Fetch provider metadata"),
    ("download", "Download uncached images"),
    ("preprocess", "Normalize and enhance images"),
    ("baseline-inference", "Run SpeciesNet baseline"),
    ("evaluate", "Score deployed model"),
    ("train", "Build next selective species model"),
    ("persist", "Persist and deploy version"),
)


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
        visual_model: VisualSpeciesModel | None = None,
    ) -> None:
        self.feeds = {feed.feed_id: feed for feed in feeds}
        self.store = store
        self.cache_root = cache_root
        self.client = client or InaturalistClient()
        self.static_model = static_model or SpeciesNetRunner(cache_root / "jobs")
        self.visual_model = visual_model or BioClipSpeciesClassifier()
        self._jobs: dict[str, FeedJob] = {}
        self._active_by_feed: dict[str, str] = {}
        self._lock = threading.RLock()
        self._http = requests.Session()
        self._http.trust_env = False
        self._http.headers.update({"User-Agent": "WildScope/0.1 personal research portfolio"})
        self._taxon_cache: dict[int, dict[str, Any]] = {}
        self._taxon_name_cache: dict[str, dict[str, Any]] = {}

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
            job.details = {
                "pipeline": [
                    {
                        "id": stage_id,
                        "label": label,
                        "state": "pending",
                        "processed": 0,
                        "total": None,
                        "detail": None,
                    }
                    for stage_id, label in TRAINING_PIPELINE
                ],
                "current_stage": None,
            }
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
        result = self.store.frames(feed_id, page=page)
        result["items"] = [self._enrich_frame(row) for row in result["items"]]
        return result

    def locations(self, feed_id: str) -> list[dict[str, Any]]:
        self._feed(feed_id)
        return self.store.locations(feed_id)

    def location_frames(self, feed_id: str, anchor_photo_id: int) -> list[dict[str, Any]]:
        self._feed(feed_id)
        return [
            self._enrich_frame(row)
            for row in self.store.location_frames(feed_id, anchor_photo_id)
        ]

    def frame_detail(self, photo_id: int) -> dict[str, Any]:
        detail = self.store.frame_detail(photo_id)
        if detail is None:
            raise LookupError("unknown frame")
        return self._enrich_frame(detail)

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
        all_training_rows = self.store.training_rows(feed_id, None)
        dataset = self.store.dataset_summary(feed_id)
        dataset["target_distribution"] = dict(
            sorted(Counter(str(row["scientific_name"]) for row in all_training_rows).items())
        )
        return {
            "baseline_model": {
                "name": "SpeciesNet",
                "engine_version": str(
                    getattr(self.static_model, "model_version", "speciesnet-5.0.5")
                ),
                "prediction_versions": dataset["baseline_prediction_versions"],
            },
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
            "dataset": dataset,
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
            self._set_stage(
                job,
                "fetch",
                "running",
                detail="Querying labels after the deployed watermark.",
            )
            photos = (
                self.client.fetch_since(feed, since=watermark)
                if watermark
                else self.client.fetch_recent(feed, hours=24)
            )
            job.total = len(photos)
            self._set_stage(
                job,
                "fetch",
                "completed",
                processed=len(photos),
                total=len(photos),
                detail=f"{len(photos)} provider photos returned.",
            )
            ingest = self._ingest_photos(job, feed, photos)
            if existing:
                self._apply_adaptive(job.feed_id)
            post_watermark_rows = self.store.training_rows(job.feed_id, watermark)
            new_rows, bootstrap_migration = self._pending_training_rows(
                job.feed_id, existing
            )
            if not new_rows:
                raise ValueError("no new labeled, licensed predictions are available")
            self._set_stage(
                job,
                "evaluate",
                "running",
                total=len(new_rows),
                detail="Comparing generated labels with obtained identifications.",
            )
            baseline_evaluation = self._evaluation_metrics(
                new_rows,
                label_field="static_label",
                confidence_field="static_confidence",
            )
            deployed_evaluation = (
                self._evaluation_metrics(
                    new_rows,
                    label_field="deployed_label",
                    confidence_field="deployed_confidence",
                )
                if existing and not bootstrap_migration
                else self._unavailable_evaluation(len(new_rows))
            )
            self._set_stage(
                job,
                "evaluate",
                "completed",
                processed=len(new_rows),
                total=len(new_rows),
                detail=f"Scored {len(new_rows)} eligible labels before training.",
            )
            self._set_stage(
                job,
                "train",
                "running",
                total=len(new_rows),
                detail="Building the feed candidate catalog for selective BioCLIP inference.",
            )
            existing_payload = (
                existing["payload"]
                if existing
                and existing["payload"].get("protocol_version")
                == SUPERVISED_PROTOCOL_VERSION
                else None
            )
            payload = train_adaptive_corrector(new_rows, existing_payload)
            trained_at = model_timestamp()
            newest = max(str(row["created_at"]) for row in new_rows)
            self._set_stage(
                job,
                "train",
                "completed",
                processed=len(new_rows),
                total=len(new_rows),
                detail=f"Consumed all {len(new_rows)} displayed training labels.",
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
            self._set_stage(
                job,
                "persist",
                "running",
                detail="Writing model payload, watermark, predictions, and lineage.",
            )
            self.store.save_adaptive_model(
                job.feed_id, model_id, trained_at, newest, payload
            )
            self._apply_adaptive(job.feed_id)
            consumed_photo_ids = {int(row["photo_id"]) for row in new_rows}
            deployed_rows = [
                row
                for row in self.store.training_rows(job.feed_id, None)
                if int(row["photo_id"]) in consumed_photo_ids
            ]
            training_evaluation = self._evaluation_metrics(
                deployed_rows,
                label_field="deployed_label",
                confidence_field="deployed_confidence",
            )
            self._set_stage(
                job,
                "persist",
                "completed",
                processed=1,
                total=1,
                detail=f"Deployed {model_id}.",
            )
            job.processed = job.total = len(new_rows)
            job.details = {
                **job.details,
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
                "baseline_errors": baseline_evaluation["errors"],
                "baseline_coverage": baseline_evaluation["coverage"],
                "baseline_mean_confidence": baseline_evaluation["mean_confidence"],
                "deployed_correct": deployed_evaluation["correct"],
                "deployed_accuracy": deployed_evaluation["accuracy"],
                "deployed_errors": deployed_evaluation["errors"],
                "deployed_coverage": deployed_evaluation["coverage"],
                "deployed_mean_confidence": deployed_evaluation["mean_confidence"],
                "target_count": baseline_evaluation["target_count"],
                "training_agreement": training_evaluation["accuracy"],
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
            self._fail_active_stage(job, job.error)
            self._finish(job, "failed")

    def _batch_snapshot(
        self,
        rows: list[dict[str, Any]],
        model: dict[str, Any] | None,
        *,
        bootstrap_migration: bool = False,
    ) -> dict[str, Any]:
        baseline = self._evaluation_metrics(
            rows,
            label_field="static_label",
            confidence_field="static_confidence",
        )
        deployed = (
            self._evaluation_metrics(
                rows,
                label_field="deployed_label",
                confidence_field="deployed_confidence",
            )
            if model
            else self._unavailable_evaluation(len(rows))
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
        if bootstrap_migration:
            return self.store.training_rows(feed_id, None), True
        return rows, False

    @staticmethod
    def _batch_samples(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        samples = []
        for row in rows:
            deployed_version = str(row.get("deployed_model_version") or "")
            deployed_value = row.get("deployed_confidence")
            is_visual = deployed_version.startswith("bioclip-")
            samples.append(
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
                "deployed_confidence": None if is_visual else deployed_value,
                "deployed_margin": deployed_value if is_visual else None,
                "deployed_model_version": deployed_version or None,
            }
            )
        return samples

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
        downloaded_items: list[tuple[Any, Path, str]] = []
        self._set_stage(
            job,
            "download",
            "running",
            total=len(photos),
            detail="Downloading original images with large-image fallback.",
        )
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
            except Exception:
                job.processed += 1
                continue
            downloaded_items.append((photo, path, digest))
            job.processed += 1
            self._set_stage(
                job,
                "download",
                "running",
                processed=job.processed,
                total=len(photos),
            )
        self._set_stage(
            job,
            "download",
            "completed",
            processed=len(downloaded_items),
            total=len(photos),
            detail=f"Downloaded {len(downloaded_items)} uncached images.",
        )
        self._set_stage(
            job,
            "preprocess",
            "running",
            total=len(downloaded_items),
            detail="Applying EXIF/RGB normalization and low-resolution enhancement.",
        )
        for index, (photo, path, digest) in enumerate(downloaded_items, start=1):
            try:
                stages = prepare_image(
                    path,
                    self.cache_root / "images" / feed.feed_id / str(photo.photo_id) / "stages",
                )
            except Exception:
                path.unlink(missing_ok=True)
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
            self._set_stage(
                job,
                "preprocess",
                "running",
                processed=index,
                total=len(downloaded_items),
            )
        self._set_stage(
            job,
            "preprocess",
            "completed",
            processed=len(downloaded_items),
            total=len(downloaded_items),
            detail=f"Prepared {downloaded + duplicates} images; {duplicates} duplicates removed.",
        )

        images = self.store.cached_images_without_static(feed.feed_id)
        predictions_count = 0
        batch_count = 0
        image_items = list(images.items())
        self._set_stage(
            job,
            "baseline-inference",
            "running",
            total=len(image_items),
            detail="Running SpeciesNet in bounded image batches.",
        )
        for offset in range(0, len(image_items), batch_size):
            batch = dict(image_items[offset : offset + batch_size])
            predictions = self.static_model.predict(batch, country=feed.country)
            for photo_id, prediction in predictions.items():
                self.store.save_prediction(photo_id, "static", prediction)
            predictions_count += len(predictions)
            batch_count += 1
            self._set_stage(
                job,
                "baseline-inference",
                "running",
                processed=predictions_count,
                total=len(image_items),
            )
        self._set_stage(
            job,
            "baseline-inference",
            "completed",
            processed=predictions_count,
            total=len(image_items),
            detail=f"Stored {predictions_count} new baseline predictions.",
        )
        return {
            "downloaded": downloaded,
            "duplicates_skipped": duplicates,
            "static_predictions": predictions_count,
            "batch_count": batch_count,
        }

    @staticmethod
    def _evaluation_metrics(
        rows: list[dict[str, Any]], *, label_field: str, confidence_field: str
    ) -> dict[str, int | float | None]:
        result = evaluate_identification_rows(rows, label_field=label_field)
        predicted_rows = [row for row in rows if row.get(label_field)]
        confidences = [
            float(row[confidence_field])
            for row in predicted_rows
            if row.get(confidence_field) is not None
        ]
        samples = int(result["samples"] or 0)
        correct = int(result["correct"] or 0)
        return {
            **result,
            "errors": samples - correct,
            "coverage": len(predicted_rows) / samples if samples else None,
            "mean_confidence": (
                sum(confidences) / len(confidences) if confidences else None
            ),
            "target_count": len({str(row["scientific_name"]) for row in rows}),
        }

    @staticmethod
    def _unavailable_evaluation(samples: int) -> dict[str, int | float | None]:
        return {
            "samples": samples,
            "correct": None,
            "accuracy": None,
            "errors": None,
            "coverage": None,
            "mean_confidence": None,
            "target_count": 0,
        }

    @staticmethod
    def _set_stage(
        job: FeedJob,
        stage_id: str,
        state: str,
        *,
        processed: int | None = None,
        total: int | None = None,
        detail: str | None = None,
    ) -> None:
        pipeline = job.details.get("pipeline", [])
        for stage in pipeline:
            if stage["id"] != stage_id:
                continue
            stage["state"] = state
            if processed is not None:
                stage["processed"] = processed
            if total is not None:
                stage["total"] = total
            if detail is not None:
                stage["detail"] = detail
            break
        job.details["current_stage"] = stage_id if state == "running" else None

    @staticmethod
    def _fail_active_stage(job: FeedJob, detail: str) -> None:
        current = job.details.get("current_stage")
        if current:
            WildlifeService._set_stage(job, str(current), "failed", detail=detail)

    def _apply_adaptive(self, feed_id: str) -> None:
        model = self.store.adaptive_model(feed_id)
        if model is None:
            return
        payload = model["payload"]
        candidates = tuple(sorted(payload.get("target_catalog", {})))
        use_visual_model = bool(payload.get("visual_model")) and len(candidates) >= 2
        for row in self.store.static_rows(feed_id):
            static = ModelPrediction(
                str(row["static_label"]),
                float(row["static_confidence"]),
                str(row["static_model_version"]),
            )
            image_path = Path(str(row.get("cached_path") or ""))
            if use_visual_model and image_path.is_file():
                try:
                    prediction = self.visual_model.predict(
                        image_path, candidates, trained_at=str(model["trained_at"])
                    )
                except (ImportError, OSError, RuntimeError, ValueError):
                    prediction = apply_adaptive_corrector(
                        static, payload, trained_at=str(model["trained_at"])
                    )
            else:
                prediction = apply_adaptive_corrector(
                    static, payload, trained_at=str(model["trained_at"])
                )
            self.store.save_prediction(int(row["photo_id"]), "adaptive", prediction)

    def _static_identification(self, row: dict[str, Any]) -> dict[str, Any]:
        label = str(row.get("static_label") or "unknown")
        leaf = canonical_model_label(label).split(";")[-1]
        return {
            "scientific_name": None,
            "common_name": leaf,
            "source_label": canonical_model_label(label),
            "taxon_id": None,
            "candidate_count": None,
            "ambiguous": False,
            "rank": "model taxon label",
        }

    def _adaptive_identification(self, row: dict[str, Any]) -> dict[str, Any] | None:
        label = row.get("adaptive_label")
        if not label:
            return None
        model = self.store.adaptive_model(str(row["feed_id"]))
        payload = model["payload"] if model else {}
        prediction = ModelPrediction(
            str(label),
            float(row.get("adaptive_confidence") or 0.0),
            str(row.get("adaptive_model_version") or "adaptive-corrector"),
            row.get("adaptive_trained_at"),
        )
        if prediction.model_version.startswith("bioclip-"):
            description = describe_visual_prediction(prediction, payload)
        else:
            description = describe_adaptive_prediction(
                prediction,
                payload,
                source_label=str(row.get("static_label") or "unknown"),
            )
        if not description.get("common_name"):
            description = self._enrich_taxon_description(description, row)
        return description | {
            "rank": (
                "selective visual species prediction"
                if prediction.model_version.startswith("bioclip-")
                else "species correction"
            )
        }

    def _enrich_frame(self, row: dict[str, Any]) -> dict[str, Any]:
        enriched = dict(row)
        enriched["static_identification"] = self._static_identification(row)
        enriched["adaptive_identification"] = self._adaptive_identification(row)
        adaptive_identification = enriched["adaptive_identification"]
        if row.get("adaptive_model_version", "").startswith("bioclip-"):
            enriched["adaptive_margin"] = row.get("adaptive_confidence")
            enriched["adaptive_confidence"] = None
        if adaptive_identification and adaptive_identification.get("abstained"):
            enriched["adaptive_raw_label"] = row.get("adaptive_label")
            enriched["adaptive_raw_confidence"] = row.get("adaptive_confidence")
            enriched["adaptive_label"] = "unidentified"
            enriched["adaptive_confidence"] = None
        return enriched

    def _enrich_taxon_description(
        self, description: dict[str, Any], row: dict[str, Any]
    ) -> dict[str, Any]:
        if description.get("scientific_name") == row.get("scientific_name"):
            return description | {
                "taxon_id": row.get("taxon_id"),
                "common_name": row.get("common_name"),
            }
        taxon_id = description.get("taxon_id")
        if taxon_id is None:
            scientific_name = str(description.get("scientific_name") or "")
            if not scientific_name:
                return description
            if scientific_name not in self._taxon_name_cache:
                try:
                    resolver = self.client.resolve_taxon_name
                    self._taxon_name_cache[scientific_name] = resolver(scientific_name)
                except (
                    AttributeError,
                    LookupError,
                    requests.RequestException,
                    ValueError,
                ):
                    return description
            return description | self._taxon_name_cache[scientific_name]
        taxon_key = int(taxon_id)
        if taxon_key not in self._taxon_cache:
            try:
                resolver = self.client.resolve_taxon
                self._taxon_cache[taxon_key] = resolver(taxon_key)
            except (AttributeError, LookupError, requests.RequestException, ValueError):
                return description
        return description | self._taxon_cache[taxon_key]

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
