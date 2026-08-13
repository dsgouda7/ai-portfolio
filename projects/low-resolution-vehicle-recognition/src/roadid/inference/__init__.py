"""RoadID inference components with no import-time model loading."""

from roadid.inference.bundle import BundleError, ModelBundle, ModelBundleLoader, load_model_bundle
from roadid.inference.calibration import HierarchicalDecisionEngine, TemperatureCalibrator
from roadid.inference.classifier import (
    DeterministicDemoClassifier,
    HierarchicalScores,
    HuggingFaceHierarchicalResNetClassifier,
    LabelSpace,
)
from roadid.inference.detector import DeterministicVehicleDetector, HuggingFaceDetrDetector
from roadid.inference.fusion import EvidenceLedger, TrackFuser, replay_evidence
from roadid.inference.pipeline import InferencePipeline, SynchronousInferencePipeline
from roadid.inference.privacy import DeterministicNoPIIRedactor, PrivacyGuard
from roadid.inference.quality import CropQualityScorer, QualityConfig
from roadid.inference.tracker import ByteTrackTracker, DeterministicTracker

__all__ = [
    "BundleError",
    "ByteTrackTracker",
    "CropQualityScorer",
    "DeterministicDemoClassifier",
    "DeterministicNoPIIRedactor",
    "DeterministicTracker",
    "DeterministicVehicleDetector",
    "EvidenceLedger",
    "HierarchicalDecisionEngine",
    "HierarchicalScores",
    "HuggingFaceDetrDetector",
    "HuggingFaceHierarchicalResNetClassifier",
    "InferencePipeline",
    "LabelSpace",
    "ModelBundle",
    "ModelBundleLoader",
    "PrivacyGuard",
    "QualityConfig",
    "SynchronousInferencePipeline",
    "TemperatureCalibrator",
    "TrackFuser",
    "load_model_bundle",
    "replay_evidence",
]
