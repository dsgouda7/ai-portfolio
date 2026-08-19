from wildscope.contracts import ModelPrediction
from wildscope.inference import (
    apply_adaptive_corrector,
    evaluate_identification_rows,
    train_adaptive_corrector,
)


def test_adaptive_corrector_learns_expert_mapping_and_updates_confidence() -> None:
    rows = [
        {
            "static_label": "felidae",
            "scientific_name": "Panthera onca",
            "created_at": "2026-08-18T10:00:00Z",
        },
        {
            "static_label": "felidae",
            "scientific_name": "Panthera onca",
            "created_at": "2026-08-18T11:00:00Z",
        },
        {
            "static_label": "felidae",
            "scientific_name": "Leopardus pardalis",
            "created_at": "2026-08-18T12:00:00Z",
        },
    ]

    payload = train_adaptive_corrector(rows)
    prediction = apply_adaptive_corrector(
        ModelPrediction("felidae", 0.6, "speciesnet"),
        payload,
        trained_at="2026-08-18T13:00:00Z",
    )

    assert payload["sample_count"] == 3
    assert prediction.label == "Panthera onca"
    assert 0.6 < prediction.confidence < 1.0
    assert prediction.trained_at == "2026-08-18T13:00:00Z"


def test_unknown_static_label_falls_back_without_inventing_species() -> None:
    static = ModelPrediction("aves", 0.72, "speciesnet")

    prediction = apply_adaptive_corrector(static, {"counts": {}}, trained_at="now")

    assert prediction.label == "aves"
    assert prediction.confidence == 0.72


def test_identification_rows_compare_stored_predictions_with_obtained_targets() -> None:
    rows = [
        {
            "photo_id": 1,
            "static_label": "Panthera onca",
            "static_confidence": 0.7,
            "scientific_name": "Panthera onca",
            "common_name": "Jaguar",
        },
        {
            "photo_id": 2,
            "static_label": "felidae",
            "static_confidence": 0.8,
            "scientific_name": "Leopardus pardalis",
            "common_name": "Ocelot",
        },
    ]

    evaluation = evaluate_identification_rows(rows, label_field="static_label")

    assert evaluation == {
        "samples": 2,
        "correct": 1,
        "accuracy": 0.5,
    }
