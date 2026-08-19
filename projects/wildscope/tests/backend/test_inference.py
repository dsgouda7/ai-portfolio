from wildscope.contracts import ModelPrediction
from wildscope.inference import (
    BIOCLIP_MARGIN_THRESHOLD,
    apply_adaptive_corrector,
    describe_adaptive_prediction,
    describe_visual_prediction,
    evaluate_identification_rows,
    train_adaptive_corrector,
)


def test_adaptive_corrector_abstains_when_one_source_maps_to_multiple_species() -> None:
    rows = [
        {
            "photo_id": 1,
            "taxon_id": 41970,
            "static_label": "felidae",
            "scientific_name": "Panthera onca",
            "common_name": "Jaguar",
            "created_at": "2026-08-18T10:00:00Z",
        },
        {
            "photo_id": 2,
            "taxon_id": 41970,
            "static_label": "felidae",
            "scientific_name": "Panthera onca",
            "common_name": "Jaguar",
            "created_at": "2026-08-18T11:00:00Z",
        },
        {
            "photo_id": 3,
            "taxon_id": 41977,
            "static_label": "felidae",
            "scientific_name": "Leopardus pardalis",
            "common_name": "Ocelot",
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
    assert payload["visual_model"] == {
        "name": "hf-hub:imageomics/bioclip",
        "decision_metric": "top-1 minus top-2 cosine similarity",
        "margin_threshold": BIOCLIP_MARGIN_THRESHOLD,
        "candidate_scope": "feed training target catalog",
    }
    assert prediction.label == "felidae"
    assert prediction.confidence == 0.6
    assert prediction.trained_at == "2026-08-18T13:00:00Z"
    assert payload["target_catalog"]["Panthera onca"] == {
        "taxon_id": 41970,
        "scientific_name": "Panthera onca",
        "common_name": "Jaguar",
    }
    description = describe_adaptive_prediction(prediction, payload, source_label="felidae")
    assert description == {
        "scientific_name": None,
        "common_name": "Unidentified",
        "taxon_id": None,
        "source_label": "felidae",
        "candidate_count": 2,
        "candidate_scientific_names": ["Leopardus pardalis", "Panthera onca"],
        "ambiguous": True,
        "abstained": True,
    }


def test_adaptive_corrector_emits_species_for_unambiguous_mapping() -> None:
    rows = [
        {
            "photo_id": 1,
            "taxon_id": 41970,
            "static_label": "felidae",
            "scientific_name": "Panthera onca",
            "common_name": "Jaguar",
        }
    ]
    payload = train_adaptive_corrector(rows)

    prediction = apply_adaptive_corrector(
        ModelPrediction("felidae", 0.6, "speciesnet"),
        payload,
        trained_at="now",
    )
    description = describe_adaptive_prediction(
        prediction, payload, source_label="felidae"
    )

    assert prediction.label == "Panthera onca"
    assert description["common_name"] == "Jaguar"
    assert description["ambiguous"] is False
    assert description["abstained"] is False


def test_unknown_static_label_falls_back_without_inventing_species() -> None:
    static = ModelPrediction("aves", 0.72, "speciesnet")

    prediction = apply_adaptive_corrector(static, {"counts": {}}, trained_at="now")

    assert prediction.label == "aves"
    assert prediction.confidence == 0.72


def test_legacy_taxonomy_keys_are_canonicalized_for_prediction_and_ambiguity() -> None:
    static = ModelPrediction(
        "uuid;aves;;;;;bird", 0.72, "speciesnet"
    )
    payload = {
        "counts": {
            "uuid;aves;;;;;bird": {
                "Buceros bicornis": 2,
                "Eumyias thalassinus": 1,
            }
        }
    }

    prediction = apply_adaptive_corrector(static, payload, trained_at="now")
    description = describe_adaptive_prediction(
        prediction, payload, source_label=static.label
    )

    assert prediction.label == "aves;bird"
    assert description["source_label"] == "aves;bird"
    assert description["candidate_count"] == 2
    assert description["ambiguous"] is True
    assert description["common_name"] == "Unidentified"


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


def test_visual_prediction_description_exposes_margin_without_probability_claim() -> None:
    payload = train_adaptive_corrector(
        [
            {
                "photo_id": 1,
                "taxon_id": 41970,
                "static_label": "animal",
                "scientific_name": "Panthera onca",
                "common_name": "Jaguar",
            },
            {
                "photo_id": 2,
                "taxon_id": 41977,
                "static_label": "animal",
                "scientific_name": "Leopardus pardalis",
                "common_name": "Ocelot",
            },
        ]
    )

    description = describe_visual_prediction(
        ModelPrediction(
            "Panthera onca", 0.08, "bioclip-vit-b16-selective-margin-0.075"
        ),
        payload,
    )

    assert description["scientific_name"] == "Panthera onca"
    assert description["common_name"] == "Jaguar"
    assert description["candidate_count"] == 2
    assert description["decision_margin"] == 0.08
    assert description["margin_threshold"] == BIOCLIP_MARGIN_THRESHOLD
    assert description["abstained"] is False
