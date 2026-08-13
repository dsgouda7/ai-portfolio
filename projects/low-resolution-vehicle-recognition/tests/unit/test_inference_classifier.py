from types import SimpleNamespace

import numpy as np
from test_inference_bundle import _bundle

from roadid.inference.bundle import load_model_bundle
from roadid.inference.classifier import HuggingFaceHierarchicalResNetClassifier


class _ProcessorDouble:
    def __call__(self, *, images: np.ndarray, return_tensors: str):
        assert images[0, 0].tolist() == [30, 20, 10]
        return {"pixel_values": np.zeros((1, 3, 4, 4), dtype=np.float32)}


class _ModelDouble:
    def __call__(self, **inputs):
        return SimpleNamespace(logits=np.asarray([[0.0, 1.0, 2.0]]))


def test_hierarchical_resnet_is_lazy_and_supports_model_doubles(tmp_path) -> None:
    bundle = load_model_bundle(_bundle(tmp_path / "bundle"))
    classifier = HuggingFaceHierarchicalResNetClassifier(
        bundle,
        processor=_ProcessorDouble(),
        model=_ModelDouble(),
    )
    crop = np.zeros((4, 4, 3), dtype=np.uint8)
    crop[:] = [10, 20, 30]

    assert classifier.loaded
    scores = classifier.classify(crop)

    assert scores.body_type == scores.make == scores.model_family == (1.0,)
