import hashlib

import pytest

from roadid.training.datasets import (
    DatasetItem,
    assert_split_invariants,
    assign_splits,
    create_pseudo_tracks,
)


def item(index: int, *, identity: str | None = None, track: str | None = None) -> DatasetItem:
    payload = f"source-{index}".encode()
    return DatasetItem(
        item_id=f"item-{index}",
        image_path=f"images/{index}.jpg",
        source_id="fixture",
        source_version="1",
        source_terms_url="https://example.test/terms",
        source_license="fixture-only",
        source_sha256=hashlib.sha256(payload).hexdigest(),
        identity_id=identity,
        track_id=track,
        body_type="suv",
        make="toyota",
        model_family="rav4",
    )


def test_split_is_deterministic_and_identity_safe() -> None:
    source = [item(index, identity=f"vehicle-{index // 2}") for index in range(20)]
    first = assign_splits(source, seed=2608)
    second = assign_splits(reversed(source), seed=2608)

    assert {row.item_id: row.split for row in first} == {row.item_id: row.split for row in second}
    owners: dict[str, set[str]] = {}
    for row in first:
        owners.setdefault(row.ownership_key, set()).add(row.split or "")
    assert all(len(splits) == 1 for splits in owners.values())


def test_pseudo_tracks_require_split_and_retain_source_hash() -> None:
    source = [item(index) for index in range(8)]
    with pytest.raises(ValueError, match="split-owned"):
        create_pseudo_tracks(source, length=3, seed=1)

    split = assign_splits(source, seed=2608)
    pseudo = create_pseudo_tracks(split, length=3, seed=2608)
    assert len(pseudo) == len(split) * 3
    assert {row.source_sha256 for row in pseudo} == {row.source_sha256 for row in split}
    assert all(row.synthetic and row.pseudo_track_id for row in pseudo)
    assert_split_invariants(pseudo)


def test_explicit_cross_split_identity_is_rejected() -> None:
    source = item(1, identity="same-vehicle")
    with pytest.raises(ValueError, match="crosses splits"):
        assert_split_invariants(
            [
                DatasetItem(**{**source.to_dict(), "split": "train"}),
                DatasetItem(
                    **{
                        **item(2, identity="same-vehicle").to_dict(),
                        "split": "validation",
                    }
                ),
            ]
        )
