from dataclasses import replace

from wildscope.contracts import ModelPrediction, ObservationPhoto
from wildscope.storage import WildlifeStore


def photo(
    photo_id: int,
    species: str,
    *,
    license_code: str | None = "cc-by",
    taxon_group: str = "Mammalia",
) -> ObservationPhoto:
    return ObservationPhoto(
        observation_id=photo_id,
        photo_id=photo_id,
        observed_at="2026-08-18",
        created_at=f"2026-08-18T10:00:{photo_id:02d}Z",
        taxon_id=photo_id,
        scientific_name=species,
        common_name=species,
        photo_url=f"https://static.inaturalist.org/photos/{photo_id}/large.jpg",
        license_code=license_code,
        attribution="Observer",
        quality_grade="research",
        latitude=-0.7002,
        longitude=-78.3001,
        positional_accuracy=18,
        coordinates_obscured=True,
        original_width=4096,
        original_height=2731,
        taxon_group=taxon_group,
    )


def test_store_pages_only_animal_predictions_and_tracks_adaptive_watermark(tmp_path) -> None:
    store = WildlifeStore(tmp_path / "wildscope.sqlite3")
    store.upsert_observation("yasuni", photo(1, "Panthera onca"))
    store.upsert_observation("yasuni", photo(2, "blank"))
    store.cache_photo(1, "a" * 64, str(tmp_path / "1.jpg"))
    store.cache_photo(2, "b" * 64, str(tmp_path / "2.jpg"))
    store.save_prediction(1, "static", ModelPrediction("felidae;Panthera onca", 0.82, "static"))
    store.save_prediction(2, "static", ModelPrediction("blank", 0.99, "static"))
    store.save_adaptive_model(
        "yasuni",
        "adaptive-yasuni-1",
        "2026-08-18T12:00:00Z",
        "2026-08-18T10:00:01Z",
        {"counts": {"felidae;Panthera onca": {"Panthera onca": 2}}, "sample_count": 2},
    )
    store.save_prediction(
        1,
        "adaptive",
        ModelPrediction("Panthera onca", 0.91, "adaptive-yasuni-1", "2026-08-18T12:00:00Z"),
    )

    result = store.frames("yasuni", page=1)
    model = store.adaptive_model("yasuni")

    assert result["total"] == 1
    assert result["items"][0]["photo_id"] == 1
    assert result["items"][0]["adaptive_confidence"] == 0.91
    assert model["trained_at"] == "2026-08-18T12:00:00Z"
    assert model["watermark"] == "2026-08-18T10:00:01Z"


def test_sha256_duplicate_removes_second_observation(tmp_path) -> None:
    store = WildlifeStore(tmp_path / "wildscope.sqlite3")
    store.upsert_observation("feed", photo(1, "Jaguar"))
    store.upsert_observation("feed", photo(2, "Jaguar"))

    assert store.cache_photo(1, "c" * 64, str(tmp_path / "1.jpg")) is True
    assert store.cache_photo(2, "c" * 64, str(tmp_path / "2.jpg")) is False


def test_store_exposes_location_stages_and_training_history(tmp_path) -> None:
    store = WildlifeStore(tmp_path / "wildscope.sqlite3")
    source = tmp_path / "1-source.jpg"
    normalized = tmp_path / "1-normalized.jpg"
    enhanced = tmp_path / "1-enhanced.jpg"
    for path in (source, normalized, enhanced):
        path.write_bytes(b"jpeg")
    store.upsert_observation("yasuni", photo(1, "Panthera onca"))
    store.cache_photo(
        1,
        "d" * 64,
        str(source),
        normalized_path=str(normalized),
        enhanced_path=str(enhanced),
        model_input_path=str(enhanced),
        cached_width=4096,
        cached_height=2731,
        enhancement_method="original-resolution-passthrough",
        enhancement_applied=False,
    )
    store.save_prediction(1, "static", ModelPrediction("felidae", 0.72, "speciesnet"))
    store.save_prediction(
        1,
        "adaptive",
        ModelPrediction("Panthera onca", 0.84, "adaptive", "2026-08-18T12:00:00Z"),
    )
    store.save_training_run(
        "yasuni",
        "train-1",
        "2026-08-18T11:59:50Z",
        "2026-08-18T12:00:00Z",
        {
            "duration_seconds": 10.0,
            "new_samples": 12,
            "total_samples": 30,
            "baseline_mean_confidence": 0.72,
            "adaptive_mean_confidence": 0.84,
            "confidence_delta": 0.12,
            "watermark_from": "2026-08-17T12:00:00Z",
            "watermark_to": "2026-08-18T10:00:00Z",
        },
    )

    locations = store.locations("yasuni")
    detail = store.frame_detail(1)
    history = store.training_history("yasuni")

    assert locations == [
        {
            "anchor_photo_id": 1,
            "latitude": -0.7002,
            "longitude": -78.3001,
            "positional_accuracy": 18.0,
            "coordinates_obscured": 1,
            "photo_count": 1,
            "latest_created_at": "2026-08-18T10:00:01Z",
            "common_name": "Panthera onca",
        }
    ]
    assert detail["model_input_path"] == str(enhanced)
    assert detail["enhancement_applied"] == 0
    assert history[0]["details"]["confidence_delta"] == 0.12


def test_results_exclude_insects_and_reptiles_from_all_review_surfaces(tmp_path) -> None:
    store = WildlifeStore(tmp_path / "wildscope.sqlite3")
    rows = (
        photo(1, "Elephas maximus", taxon_group="Mammalia"),
        photo(2, "Calotes thailandensis", taxon_group="Reptilia"),
        photo(3, "Epeus glorius", taxon_group="Insecta"),
        photo(4, "Heteropoda venatoria", taxon_group="Arachnida"),
    )
    for observation in rows:
        store.upsert_observation("khao-yai", observation)
        store.save_prediction(
            observation.photo_id,
            "static",
            ModelPrediction("animal", 0.8, "speciesnet"),
        )

    frames = store.frames("khao-yai", page=1)
    locations = store.locations("khao-yai")
    location_frames = store.location_frames("khao-yai", 1)

    assert [item["photo_id"] for item in frames["items"]] == [1]
    assert frames["total"] == 1
    assert locations[0]["photo_count"] == 1
    assert [item["photo_id"] for item in location_frames] == [1]


def test_supervised_targets_require_research_grade_and_license(tmp_path) -> None:
    store = WildlifeStore(tmp_path / "wildscope.sqlite3")
    research = photo(1, "Panthera onca")
    needs_id = replace(
        photo(2, "Leopardus pardalis"),
        quality_grade="needs_id",
    )
    unlicensed = photo(3, "Puma concolor", license_code=None)
    for observation in (research, needs_id, unlicensed):
        store.upsert_observation("yasuni", observation)
        store.save_prediction(
            observation.photo_id,
            "static",
            ModelPrediction("felidae", 0.8, "speciesnet"),
        )

    assert [row["photo_id"] for row in store.training_rows("yasuni", None)] == [1]


def test_store_rebases_absolute_cache_paths_after_project_move(tmp_path) -> None:
    old_project = tmp_path / "old-project"
    old_cache = old_project / "artifacts" / "wildscope"
    old_image = old_cache / "images" / "1.jpg"
    old_image.parent.mkdir(parents=True)
    old_image.write_bytes(b"jpeg")
    store = WildlifeStore(old_cache / "wildscope.sqlite3")
    store.upsert_observation("yasuni", photo(1, "Panthera onca"))
    store.cache_photo(1, "e" * 64, str(old_image))
    store.close()

    new_project = tmp_path / "wildscope"
    old_project.rename(new_project)
    reopened = WildlifeStore(
        new_project / "artifacts" / "wildscope" / "wildscope.sqlite3"
    )

    assert reopened.cached_path(1) == new_project / "artifacts" / "wildscope" / "images" / "1.jpg"
