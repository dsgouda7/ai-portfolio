# TrackLens Parallel Implementation Plan

## 1. Objective

Implement `projects/low-resolution-vehicle-recognition/` as a locally hosted Flask application with two clearly separated systems:

1. **Offline training pipeline:** adapt a generic Hugging Face ImageNet ResNet-50 to hierarchical vehicle body/make/model recognition under severe low-resolution degradation.
2. **Near-real-time inference pipeline:** consume frames from reviewed public-camera adapters, a local camera, or deterministic replay; detect and track vehicles; fuse multiple crop predictions; calibrate confidence; abstain when necessary; and stream graphical execution events to the Flask UI.

The project is complete only when both pipelines communicate through a versioned model-bundle contract rather than shared Python memory.

## 2. Product Decisions Frozen Before Parallel Work

These decisions prevent each agent from inventing incompatible architecture.

### 2.1 Project identity

- Directory: `projects/low-resolution-vehicle-recognition/`
- Python package: `roadid`
- Product name: `TrackLens`
- Flask port: `5000`, configurable by environment
- Default operation: local-only binding to `127.0.0.1`

### 2.2 Initial models

| Role | Hugging Face model | Initial policy |
|---|---|---|
| Broad vehicle detection | `facebook/detr-resnet-50`, preferably pinned to an explicit revision | Frozen; COCO vehicle classes only |
| Fine-grained classification | `microsoft/resnet-50`, pinned to an explicit revision | Replace head, freeze backbone, then selectively unfreeze late stages |

Do not silently change these defaults. Alternative models belong behind config with separate measured profiles.

### 2.3 Recognition hierarchy

```text
body_type -> make -> model_family
```

Model-year or exact trim is out of scope for version 1. Every child label names one valid parent. Predictions may stop at any hierarchy level.

### 2.4 Fusion baseline

Version 1 uses quality-weighted track fusion over frame logits or normalized frame embeddings. Do not add an LSTM or temporal Transformer until the baseline is implemented and measured.

### 2.5 Source policy

- The browser selects source IDs only.
- Remote source URLs live in server-owned configuration or provider discovery.
- No browser-supplied arbitrary URL.
- Replay is required and is the test/default demo.
- Local webcam is opt-in.
- TfL JamCam is the first candidate public provider, disabled until access and usage terms are validated.
- No Google Maps, Street View, or webcam-directory scraping.

### 2.6 Privacy boundary

No plate OCR, face recognition, person re-identification, owner inference, or driver/passenger identification. Browser overlays and exports blur faces/plates when detected by the privacy stage.

## 3. Target Structure

```text
projects/low-resolution-vehicle-recognition/
├── README.md
├── plan.md
├── pyproject.toml
├── requirements.txt
├── requirements-dev.txt
├── .env.example
├── .gitignore
├── configs/
│   ├── train_resnet50.yaml
│   ├── inference.yaml
│   └── camera_sources.example.yaml
├── scripts/
│   ├── prepare_dataset.py
│   ├── train.py
│   ├── evaluate.py
│   └── run_web.py
├── src/roadid/
│   ├── __init__.py
│   ├── contracts.py
│   ├── settings.py
│   ├── training/
│   │   ├── datasets.py
│   │   ├── degradations.py
│   │   ├── hierarchy.py
│   │   ├── model.py
│   │   ├── trainer.py
│   │   ├── calibration.py
│   │   ├── evaluation.py
│   │   └── packaging.py
│   ├── sources/
│   │   ├── base.py
│   │   ├── registry.py
│   │   ├── replay.py
│   │   ├── local_camera.py
│   │   ├── snapshot_http.py
│   │   └── tfl_jamcam.py
│   ├── inference/
│   │   ├── bundle.py
│   │   ├── detector.py
│   │   ├── tracker.py
│   │   ├── quality.py
│   │   ├── classifier.py
│   │   ├── fusion.py
│   │   ├── calibration.py
│   │   ├── privacy.py
│   │   └── pipeline.py
│   ├── telemetry/
│   │   ├── events.py
│   │   ├── recorder.py
│   │   └── metrics.py
│   └── web/
│       ├── __init__.py
│       ├── app.py
│       ├── api.py
│       ├── runs.py
│       └── serialization.py
├── templates/
│   └── index.html
├── static/
│   ├── css/app.css
│   ├── js/app.js
│   └── js/pipeline_graph.js
├── tests/
│   ├── fixtures/
│   ├── unit/
│   ├── integration/
│   ├── web/
│   └── browser/
└── artifacts/                  # gitignored
```

## 4. Shared Contracts

The coordinator owns `src/roadid/contracts.py`. Parallel agents may import these contracts but may not redefine them.

### 4.1 Camera source

```python
@dataclass(frozen=True)
class CameraSource:
    source_id: str
    name: str
    adapter_type: str
    enabled: bool
    attribution: str
    terms_url: str | None
    refresh_seconds: float
    location_label: str | None
    location_precision: str
```

### 4.2 Frame packet

```python
@dataclass(frozen=True)
class FramePacket:
    run_id: str
    source_id: str
    frame_id: int
    captured_at: datetime
    received_at: datetime
    image_bgr: np.ndarray
    source_metadata: Mapping[str, JSONValue]
```

`image_bgr` remains in process and is never serialized into telemetry.

### 4.3 Detection and observation

```python
@dataclass(frozen=True)
class Detection:
    frame_id: int
    bbox_xyxy: tuple[float, float, float, float]
    class_name: str
    confidence: float

@dataclass(frozen=True)
class TrackObservation:
    track_id: str
    frame_id: int
    bbox_xyxy: tuple[float, float, float, float]
    crop_bgr: np.ndarray
    quality: "CropQuality"
```

### 4.4 Crop quality

```python
@dataclass(frozen=True)
class CropQuality:
    apparent_height_px: int
    blur_score: float        # 0 sharp, 1 unusably blurred
    exposure_score: float    # 0 usable, 1 unusably exposed
    occlusion_score: float   # 0 visible, 1 fully occluded
    usable: bool
    rejection_reasons: tuple[str, ...]
    fusion_weight: float
```

All quality scores are constrained to `[0.0, 1.0]`, and `fusion_weight` is non-negative. Every `bbox_xyxy` contract uses original-frame pixel coordinates, never detector-resized coordinates.

### 4.5 Hierarchical prediction

```python
@dataclass(frozen=True)
class LabelPrediction:
    label: str | None
    confidence: float
    accepted: bool

@dataclass(frozen=True)
class VehiclePrediction:
    track_id: str
    body_type: LabelPrediction
    make: LabelPrediction
    model_family: LabelPrediction
    decision: "Decision"
    usable_frames: int
    disagreement: float
    model_version: str
```

`Decision` is a finite enum: `ACCEPT_BODY_MAKE_MODEL`, `ACCEPT_BODY_MAKE`, `ACCEPT_BODY_ONLY`, and `INSUFFICIENT_VISUAL_EVIDENCE`. Construction validates every accepted make/model against the immutable hierarchy loaded from `labels.json`.

### 4.6 Calibration contract

```python
@dataclass(frozen=True)
class CalibrationContract:
    method: str
    dataset_manifest_sha256: str
    validation_split_sha256: str
    test_split_sha256: str
    body_threshold: float
    make_threshold: float
    model_threshold: float
```

Calibration refuses overlap between validation and test identities. The inference loader verifies dataset and validation hashes before accepting thresholds.

### 4.7 Run state

```python
class RunState(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"
```

Valid transitions live in `contracts.py`. Pause stops source acquisition and retains at most one replaceable pending frame. Stop is idempotent and terminal.

### 4.8 Pipeline event

```python
@dataclass(frozen=True)
class PipelineEvent:
    event_id: str
    sequence_id: int
    run_id: str
    frame_id: int | None
    track_id: str | None
    stage: str
    status: Literal["pending", "running", "completed", "skipped", "warning", "failed"]
    started_at: datetime
    duration_ms: float | None
    input_summary: Mapping[str, JSONValue]
    output_summary: Mapping[str, JSONValue]
    warning: str | None
    error_code: str | None
```

Event summaries must remain bounded and privacy-safe.

### 4.9 Track evidence

```python
@dataclass(frozen=True)
class EvidenceItem:
    crop_id: str
    frame_id: int
    quality: CropQuality
    fusion_weight: float
    frame_prediction: tuple[float, ...]

@dataclass(frozen=True)
class TrackEvidence:
    track_id: str
    items: tuple[EvidenceItem, ...]
```

The bounded evidence ledger must reproduce the fused output within numerical tolerance.

### 4.10 Privacy redaction

```python
@dataclass(frozen=True)
class PrivacyRedactionResult:
    frame_id: int
    face_masks: tuple[tuple[int, int, int, int], ...]
    plate_masks: tuple[tuple[int, int, int, int], ...]
    redactor_version: str
    safe_for_display: bool
```

Public-camera display/export fails closed when the configured privacy redactor is unavailable or returns `safe_for_display=False`. Promotion requires retained face/plate redaction coverage results on a small labeled privacy hold-out. Local/replay sources may opt into raw display only through explicit local configuration.

### 4.11 Model bundle manifest

Required fields:

```json
{
  "schema_version": 1,
    "model_version": "roadid-resnet50-<dataset-hash-8>-<utc-build-id>",
  "base_model": "microsoft/resnet-50",
  "base_model_revision": "...",
  "dataset_manifest_sha256": "...",
  "label_hierarchy_sha256": "...",
  "classifier_sha256": "...",
  "image_processor_sha256": "...",
  "calibration_sha256": "...",
  "thresholds_sha256": "...",
  "metrics": {},
  "created_at": "...",
  "git_revision": "..."
}
```

The inference loader fails closed when any required file or hash differs.

Bundle writers refuse an existing version directory. Training writes intermediate state under `artifacts/training/<run-id>/`; validation writes under `artifacts/validation/<run-id>/` and never mutates a training run. Only an atomically finalized, validated bundle is copied to `artifacts/models/<model-version>/`. The UTC build ID uses a Windows-safe compact form such as `20260813T120001Z`.

## 5. Parallel Agent Ownership

Agents work in separate worktrees or obey the exclusive paths below. No agent may run a formatter across the whole project.

### Wave 0: Coordinator contracts and skeleton

**Owner: Coordinator only**

Creates:

- package skeleton;
- `contracts.py`;
- `settings.py`;
- placeholder package `__init__.py` files;
- dependency files;
- common test fixtures and model-bundle fixture;
- configuration schemas;
- CI-neutral setup/run scripts.

Validation gate:

- imports succeed;
- contracts serialize through explicit serializers;
- all settings load from a test config;
- no model download occurs during import.

### Wave 1: Independent implementation

#### Agent A: Data and training

Exclusive ownership:

```text
src/roadid/training/**
scripts/prepare_dataset.py
scripts/train.py
scripts/evaluate.py
configs/train_resnet50.yaml
tests/unit/test_training_*.py
tests/integration/test_training_pipeline.py
```

Responsibilities:

- dataset adapter protocol;
- license/provenance manifest;
- hierarchy normalization;
- vehicle/track-safe splitting;
- deterministic low-resolution degradation;
- synthetic pseudo-track construction;
- ResNet head replacement;
- frozen and partially unfrozen phases;
- frame and track fusion training/evaluation;
- calibration and abstention thresholds;
- model-bundle packaging;
- frame, track, resolution-bin, and selective-accuracy metrics.

Must prove:

- no source identity crosses splits;
- same seed reproduces degradation metadata;
- pseudo-tracks are created only after splitting and retain source hashes;
- synthetic and real-track metrics are reported separately;
- random-init, frozen-backbone, and partial-unfreeze baselines are comparable;
- frozen stages receive no gradients during head training, while selected late stages receive finite non-zero gradients after unfreezing;
- missing hierarchy labels are masked from their corresponding loss rather than treated as negative classes;
- accepted child labels always have accepted compatible parents;
- calibration artifacts name and match the dataset and validation split hashes;
- model bundle reload reproduces logits on a fixed probe batch.

#### Agent B: Camera sources and security

Exclusive ownership:

```text
src/roadid/sources/**
configs/camera_sources.example.yaml
tests/unit/test_source_*.py
tests/integration/test_source_registry.py
```

Responsibilities:

- typed source adapter interface;
- deterministic replay adapter;
- local-camera adapter with graceful missing-device behavior;
- secure HTTP snapshot adapter;
- optional TfL catalog adapter behind feature flag;
- source registry and health state;
- freshness, byte, content-type, resolution, timeout, redirect, DNS, and IP checks;
- provider attribution and terms metadata;
- bounded retry/backoff and typed errors.

Must prove:

- browser/client cannot submit arbitrary URLs;
- private, loopback, link-local, metadata-service, and disallowed redirects are rejected;
- oversized and stale frames fail before decode/inference;
- replay produces deterministic frame IDs and timestamps;
- a failing public provider does not prevent replay use.

#### Agent C: Inference and tracking

Exclusive ownership:

```text
src/roadid/inference/**
tests/unit/test_inference_*.py
tests/integration/test_inference_pipeline.py
```

Responsibilities:

- model-bundle loader and hash verification;
- HF DETR vehicle detector wrapper;
- ByteTrack association wrapper for production inference;
- deterministic IoU/centroid tracker double for tests only;
- crop quality scoring;
- HF ResNet classifier wrapper;
- quality-weighted track fusion;
- hierarchical consistency;
- calibrated confidence and abstention;
- privacy redaction hook;
- synchronous pipeline core with cancellation checkpoints.

Must prove:

- imports do not download models;
- missing or corrupt bundle fails closed;
- one replay track accumulates multiple observations;
- detector coordinates are restored to original-frame pixels before tracking;
- track expiration obeys configured `lost_track_buffer`;
- detector recall is reported by apparent-height bin on the named release profile;
- low-quality crops are excluded or down-weighted;
- fusion is invariant to observation order when using the baseline weighted average;
- replaying the bounded evidence ledger reproduces the fused result;
- exact model is withheld below its threshold;
- calibration hashes match the loaded dataset and validation identities;
- no plate/face text enters predictions or telemetry.
- public-source display fails closed without a validated `PrivacyRedactionResult`.

#### Agent D: Telemetry and graphical execution model

Exclusive ownership:

```text
src/roadid/telemetry/**
tests/unit/test_telemetry_*.py
tests/integration/test_event_ordering.py
```

Responsibilities:

- event lifecycle and monotonic sequence IDs;
- bounded in-memory event recorder;
- optional JSONL run export;
- per-stage latency aggregation;
- run/frame/track correlation;
- redaction and payload-size limits;
- event-to-graph snapshot serializer.

Frozen stage names:

```text
source_acquisition
frame_validation
vehicle_detection
track_association
crop_quality
frame_classification
track_fusion
calibration
hierarchy_decision
privacy_render
```

Must prove:

- `running` precedes a terminal state;
- every terminal event has a duration;
- failures stop or skip downstream stages predictably;
- bounded recorder evicts safely without reusing sequence IDs;
- JSONL sequence IDs are continuous except for explicitly recorded dropped-event gaps;
- exported events contain no image bytes, credentials, full source URLs, or plate text.

#### Agent E: Flask API and worker lifecycle

Exclusive ownership:

```text
src/roadid/web/**
tests/web/**
scripts/run_web.py
```

Responsibilities:

- Flask application factory;
- source, run, event, track, and report endpoints;
- background inference worker lifecycle;
- clean start/pause/resume/stop;
- SSE stream with heartbeat and disconnect handling;
- input validation and typed API errors;
- model/source readiness endpoint;
- no training execution from HTTP routes.

Must prove:

- multiple tabs can observe one run without starting duplicate workers;
- stop is idempotent;
- every run-state transition is valid and pause retains at most one replaceable pending frame;
- unknown source IDs and run IDs fail clearly;
- source URLs and credentials are never returned;
- SSE ordering matches recorder sequence IDs;
- SSE sends a heartbeat every configured 10 seconds and closes after three missed-heartbeat intervals;
- a worker failure produces a terminal run state.

#### Agent F: Browser UI

Exclusive ownership:

```text
templates/**
static/**
tests/browser/**
```

Responsibilities:

- operational first screen;
- source selector and run controls;
- camera frame/overlay surface;
- active track list;
- selected-track evidence strip;
- hierarchy confidence and abstention reason;
- live pipeline graph driven by SSE;
- stage status, duration, warning, and error detail;
- responsive desktop/mobile layout;
- accessible labels and keyboard controls;
- professional work-focused styling.

The pipeline graph should have stable dimensions so status changes do not shift the layout. Dynamic data must not resize controls or overlap the video surface.

Must prove with browser tests:

- source selection and run start;
- pipeline nodes transition visibly;
- selected track updates evidence and confidence;
- stop freezes updates and reaches terminal state;
- long labels and errors do not overflow;
- desktop and mobile screenshots contain no overlap;
- the primary video/trace surfaces are nonblank.

### Wave 2: Integration

#### Agent G: Integration and documentation

Exclusive ownership after Wave 1:

```text
pyproject.toml
requirements.txt
requirements-dev.txt
.env.example
.gitignore
README.md
projects/README.md
configs/inference.yaml
```

Responsibilities:

- reconcile dependency versions;
- connect Flask routes to source/inference/telemetry contracts;
- add project entry to `projects/README.md` with honest evidence status;
- finalize setup and run instructions;
- document model/dataset licenses and source adapters;
- preserve CPU-safe replay mode;
- keep public source adapters disabled until validated.

#### Agent H: End-to-end validation

Read/write scope limited to tests and narrowly required defect fixes.

Runs:

1. Static imports and lint.
2. Unit suite.
3. Training smoke run on a tiny synthetic fixture.
4. Model bundle reload parity.
5. Replay inference integration run.
6. Flask API tests.
7. SSE event ordering tests.
8. Browser workflow and screenshots.
9. Security tests for source allowlisting and SSRF defenses.
10. `git diff --check` and whole-worktree status review.

## 6. Training Pipeline Acceptance Plan

### 6.1 Dataset preparation

Command:

```powershell
python scripts/prepare_dataset.py --config configs/train_resnet50.yaml
```

Outputs:

```text
artifacts/datasets/<dataset-version>/
├── manifest.json
├── labels.json
├── train.jsonl
├── validation.jsonl
└── test.jsonl
```

Checks:

- source terms URL present;
- every item has source ID and hash;
- no source vehicle/track overlap;
- synthetic variants retain their source identity and are generated only after splitting;
- class hierarchy valid;
- class-count report generated;
- excluded items and reasons retained.

### 6.2 Training

Command:

```powershell
python scripts/train.py --config configs/train_resnet50.yaml
```

Required baselines:

1. Randomly initialized ResNet head/backbone profile.
2. Frozen ImageNet backbone with trained heads.
3. Partial unfreeze of late ResNet stages.
4. Single-frame classifier.
5. Best-frame track classifier.
6. Uniform fusion.
7. Quality-weighted fusion.

A baseline may be skipped only with an explicit resource reason in the report.

One-step gradient tests assert frozen stages receive no gradients during head training and selected late stages receive finite non-zero gradients after unfreezing.

### 6.3 Evaluation and calibration

Command:

```powershell
python scripts/evaluate.py --bundle artifacts/models/<version>
```

Required report slices:

- apparent height bins;
- blur bins;
- viewpoint when available;
- body type, make, and model family;
- source dataset/camera;
- a named camera-disjoint hold-out;
- clean versus degraded;
- synthetic pseudo-tracks versus real tracks;
- single frame versus track;
- accepted coverage versus selective accuracy.

The sealed test set is opened only after calibration and thresholds are frozen.

## 7. Inference Pipeline Acceptance Plan

### 7.1 Deterministic replay

Replay is the first complete integration target:

```text
fixture video
-> deterministic frames
-> stub or real detector profile
-> tracks
-> classifier fixture or model bundle
-> fusion
-> hierarchy decision
-> SSE events
-> Flask browser
```

Tests use lightweight deterministic doubles. A separate opt-in test exercises real HF models when weights are available locally.

The required CI fixture is generated locally from deterministic synthetic frames: 30 frames, one vehicle-like track visible from frames 3-24, fixed 100 ms capture intervals, and no network access. It validates source, tracking, fusion, SSE, and UI contracts with model doubles. A separately downloaded, hash-pinned, rights-reviewed short traffic clip is required for the opt-in real-model smoke test.

### 7.2 Real model smoke test

With model cache available:

- load pinned detector revision;
- load TrackLens classifier bundle;
- process a short licensed clip;
- verify at least one vehicle track;
- report detector precision/recall by apparent-height bin and require at least 90% vehicle recall on the named release profile;
- verify finite confidence and hierarchy;
- record stage latency and device;
- retain a privacy-safe report.

### 7.3 Public-camera adapter

A provider adapter is promoted from disabled to enabled only after:

- official API/source documentation is recorded;
- image-use and caching terms are reviewed;
- attribution is rendered in the UI;
- source refresh behavior is measured;
- rate limits and retry rules are configured;
- provider outage and stale-frame tests pass;
- no page scraping is used.

## 8. Graphical Execution Requirements

The graph is part of the product contract, not decorative UI.

### 8.1 Stable graph

Nodes are created once from frozen stage names. Events update state, duration, and summaries. The graph does not rebuild or reorder itself per frame.

`sequence_id`, not wall-clock timestamp, orders events. Any dropped sequence range is reported explicitly.

### 8.2 Frame and track scope

The UI can switch between:

- latest frame execution;
- selected track execution history;
- run-level aggregate latency and failures.

### 8.3 Failure representation

If detection fails:

```text
source_acquisition: completed
frame_validation: completed
vehicle_detection: failed
track_association onward: skipped
```

If no vehicle is present, detection completes successfully with zero results and downstream stages skip without showing a failure.

### 8.4 Bounded telemetry

Graph payloads contain counts, dimensions, accepted labels, confidence summaries, durations, and typed errors. They exclude raw tensors, images, credentials, full URLs, and personal identifiers.

## 9. Test Matrix

| Layer | Required tests |
|---|---|
| Contracts | serialization, hierarchy, bounded summaries |
| Dataset | license fields, hashes, leakage, labels, deterministic degradation |
| Training | transfer freeze/unfreeze, calibration isolation, bundle parity |
| Sources | replay, local-camera absence, timeout, stale/oversized frame, SSRF |
| Detector | vehicle class filtering, original-coordinate scaling, recall by apparent-height bin, no-result case |
| Tracker | ByteTrack association, expiration, identity switches, IDF1/HOTA where labeled, cancellation |
| Quality | blur/resolution/exposure thresholds and weights |
| Fusion | order invariance, disagreement, minimum evidence |
| Decision | hierarchy consistency, calibration, abstention |
| Privacy | mask bounds, fail-closed public display, labeled hold-out coverage, redacted export |
| Telemetry | ordering, redaction, bounded retention, export |
| Flask | endpoint schemas, worker lifecycle, SSE, typed errors |
| Browser | source/run workflow, graph states, track evidence, responsive layout |
| End to end | replay -> prediction -> graph -> report |

## 10. Performance Profiles

### CPU replay profile

- Reduced detector input size.
- Poll or sample a low FPS.
- One inference worker.
- Small bounded track window.
- Quality-weighted fusion.
- Guaranteed to run without CUDA, with measured latency disclosed.

The release report compares detector recall at the CPU resize with the default detector resize. Input reduction is accepted only with the accuracy delta disclosed and the 90% release-profile recall gate still met.

### CUDA profile

- Full detector profile.
- Higher processing FPS.
- Batched crop classification.
- Larger active-track budget.
- Mixed precision only after parity checks.

Do not claim real-time FPS before measurement on named hardware.

## 11. Security Checklist

- No arbitrary URL endpoint.
- Server-owned source IDs only.
- URL scheme allowlist.
- DNS resolution checked before request.
- Private, loopback, link-local, multicast, and metadata-service addresses rejected.
- Redirect destination revalidated.
- Response byte and dimension limits.
- Content-type validation.
- Fetch and decode timeouts.
- Credentials from environment, never sent to browser.
- Terms and attribution retained.
- No raw frame retention by default.
- Plates/faces blurred in display and export.
- SSE summaries bounded and redacted.
- Model and dataset bundles hash-verified.

## 12. Definition of Done

TrackLens version 1 is done only when:

- the training and inference packages import independently;
- training produces a hash-verified reloadable bundle;
- model selection and calibration use validation data only;
- track-level fusion is compared with single/best-frame baselines;
- the inference worker runs from replay without network access;
- at least one opt-in local or approved public source uses the same source contract;
- the Flask UI shows camera frames, tracks, evidence, hierarchy confidence, and abstention;
- the pipeline graph updates from real structured events and displays failures/skips correctly;
- stop/restart and source failure paths are tested;
- privacy and SSRF controls pass;
- browser screenshots pass desktop and mobile layout checks;
- model, data, camera, and code provenance appear in the final report;
- synthetic and real-track metrics are reported separately;
- calibration hashes bind thresholds to the validation and dataset identities;
- detector recall meets the named release-profile gate, or detector adaptation remains a documented blocker;
- all claims in README and `projects/README.md` match retained validation evidence.

## 13. Explicit Non-Goals

- Scraping Google Maps, Street View, or webcam directories.
- Arbitrary user-provided remote URLs.
- Plate OCR or person identification.
- Forced exact model predictions from insufficient evidence.
- Treating super-resolved pixels as factual observations.
- Training inside a Flask request.
- Multi-node training or production camera fleet orchestration.
- Commercial surveillance deployment claims.
- Replacing provider terms review with a technical access check.
