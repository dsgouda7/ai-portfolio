"""WildScope Flask API and operational wildlife review surface."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, render_template, request, send_file

from wildscope.feeds import InaturalistClient, load_feeds
from wildscope.inference import identification_matches
from wildscope.service import WildlifeService
from wildscope.storage import WildlifeStore

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def create_app(
    *,
    service: WildlifeService | None = None,
    project_root: Path = PROJECT_ROOT,
) -> Flask:
    feeds = load_feeds(project_root / "configs" / "feeds.yaml")
    store = None
    if service is None:
        cache_root = project_root / "artifacts" / "runtime"
        store = WildlifeStore(cache_root / "state.sqlite3")
        service = WildlifeService(feeds, store, cache_root, client=InaturalistClient())
    app = Flask(
        __name__,
        template_folder=str(project_root / "templates"),
        static_folder=str(project_root / "static"),
    )
    app.json.sort_keys = False
    app.extensions["wildscope"] = {"service": service, "store": store}

    @app.get("/")
    def index():
        return render_template("observations.html")

    @app.get("/training")
    def training():
        return render_template("training.html")

    @app.get("/api/status")
    def status():
        return jsonify(
            {
                "service": "WildScope",
                "ready": True,
                "static_model": "SpeciesNet 5.0.5",
                "feed_count": len(service.feeds),
                "page_size": 10,
            }
        )

    @app.get("/api/feeds")
    def list_feed_records():
        return jsonify({"feeds": service.list_feeds()})

    @app.post("/api/feeds/<feed_id>/sync")
    def sync_feed(feed_id: str):
        payload = request.get_json(silent=True) or {}
        unknown = set(payload) - {"hours"}
        if unknown:
            return _error("INVALID_REQUEST", "Unsupported sync fields.", 400)
        try:
            job = service.start_sync(feed_id, hours=int(payload.get("hours", 24)))
        except LookupError:
            return _error("FEED_NOT_FOUND", "Feed was not found.", 404)
        except (RuntimeError, ValueError) as error:
            return _error("SYNC_NOT_STARTED", str(error), 409)
        return jsonify({"job": job.public_dict()}), 202

    @app.post("/api/feeds/<feed_id>/train")
    def train_feed(feed_id: str):
        try:
            job = service.start_training(feed_id)
        except LookupError:
            return _error("FEED_NOT_FOUND", "Feed was not found.", 404)
        except RuntimeError as error:
            return _error("TRAINING_NOT_STARTED", str(error), 409)
        return jsonify({"job": job.public_dict()}), 202

    @app.get("/api/jobs/<job_id>")
    def get_job(job_id: str):
        try:
            job = service.job(job_id)
        except LookupError:
            return _error("JOB_NOT_FOUND", "Job was not found.", 404)
        return jsonify({"job": job.public_dict()})

    @app.get("/api/feeds/<feed_id>/frames")
    def frames(feed_id: str):
        try:
            page = int(request.args.get("page", "1"))
            result = service.frames(feed_id, page)
        except LookupError:
            return _error("FEED_NOT_FOUND", "Feed was not found.", 404)
        except ValueError as error:
            return _error("INVALID_PAGE", str(error), 400)
        result["items"] = [_public_frame(item) for item in result["items"]]
        return jsonify(result)

    @app.get("/api/feeds/<feed_id>/locations")
    def locations(feed_id: str):
        try:
            rows = service.locations(feed_id)
        except LookupError:
            return _error("FEED_NOT_FOUND", "Feed was not found.", 404)
        for row in rows:
            row["thumbnail_url"] = f"/api/images/{row['anchor_photo_id']}"
        return jsonify({"locations": rows})

    @app.get("/api/feeds/<feed_id>/locations/<int:anchor_photo_id>/frames")
    def location_frames(feed_id: str, anchor_photo_id: int):
        try:
            rows = service.location_frames(feed_id, anchor_photo_id)
        except LookupError:
            return _error("FEED_NOT_FOUND", "Feed was not found.", 404)
        return jsonify({"items": [_public_frame(item) for item in rows]})

    @app.get("/api/frames/<int:photo_id>")
    def frame_detail(photo_id: int):
        try:
            item = service.frame_detail(photo_id)
        except LookupError:
            return _error("FRAME_NOT_FOUND", "Frame was not found.", 404)
        public = _public_frame(item)
        width = item.get("cached_width") or item.get("original_width")
        height = item.get("cached_height") or item.get("original_height")
        enhanced_width = width * 2 if width and item.get("enhancement_applied") else width
        enhanced_height = height * 2 if height and item.get("enhancement_applied") else height
        public["stages"] = [
            {
                "id": "source",
                "name": "Original capture",
                "processor": "iNaturalist original asset",
                "image_url": f"/api/images/{photo_id}?stage=source",
                "dimensions": [width, height],
            },
            {
                "id": "normalized",
                "name": "Clean and normalize",
                "processor": "Pillow EXIF transpose + RGB normalization",
                "image_url": f"/api/images/{photo_id}?stage=normalized",
                "dimensions": [width, height],
            },
            {
                "id": "enhanced",
                "name": "Model input",
                "processor": item.get("enhancement_method") or "legacy image passthrough",
                "applied": bool(item.get("enhancement_applied")),
                "image_url": f"/api/images/{photo_id}?stage=enhanced",
                "dimensions": [enhanced_width, enhanced_height],
            },
            {
                "id": "classification",
                "name": "Identify wildlife",
                "processor": item.get("static_model_version") or "SpeciesNet",
                "obtained": public["obtained_identification"],
                "static": {
                    "label": item.get("static_label"),
                    "confidence": item.get("static_confidence"),
                    "matches_obtained": public["static_match"],
                },
                "adaptive": {
                    "label": item.get("adaptive_label"),
                    "confidence": item.get("adaptive_confidence"),
                    "model_version": item.get("adaptive_model_version"),
                    "trained_at": item.get("adaptive_trained_at"),
                    "matches_obtained": public["adaptive_match"],
                },
            },
        ]
        return jsonify(public)

    @app.get("/api/feeds/<feed_id>/training")
    def training_dashboard(feed_id: str):
        try:
            return jsonify(service.training_dashboard(feed_id))
        except LookupError:
            return _error("FEED_NOT_FOUND", "Feed was not found.", 404)

    @app.get("/api/images/<int:photo_id>")
    def image(photo_id: int):
        try:
            path = service.image_path(photo_id, request.args.get("stage", "source"))
        except LookupError:
            return _error("IMAGE_NOT_FOUND", "Image was not found.", 404)
        except ValueError as error:
            return _error("INVALID_IMAGE_STAGE", str(error), 400)
        return send_file(path, mimetype="image/jpeg", max_age=300)

    return app


def _error(code: str, message: str, status: int) -> tuple[Any, int]:
    return jsonify({"error": {"code": code, "message": message[:300]}}), status


def _public_frame(item: dict[str, Any]) -> dict[str, Any]:
    public = dict(item)
    public["image_url"] = f"/api/images/{item['photo_id']}"
    scientific_name = item.get("scientific_name")
    common_name = item.get("common_name")
    quality_grade = item.get("quality_grade")
    public["obtained_identification"] = {
        "source": "iNaturalist community identification",
        "scientific_name": scientific_name,
        "common_name": common_name,
        "quality_grade": quality_grade,
        "research_grade": quality_grade == "research",
    }
    public["static_match"] = identification_matches(
        item.get("static_label"), scientific_name, common_name
    )
    public["adaptive_match"] = identification_matches(
        item.get("adaptive_label"), scientific_name, common_name
    )
    for field in (
        "cached_path",
        "normalized_path",
        "enhanced_path",
        "model_input_path",
        "sha256",
    ):
        public.pop(field, None)
    return public


if __name__ == "__main__":
    create_app().run(
        host=os.getenv("WILDSCOPE_HOST", "127.0.0.1"),
        port=int(os.getenv("WILDSCOPE_PORT", "5000")),
        threaded=True,
    )
