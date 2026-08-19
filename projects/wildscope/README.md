# WildScope: Test-Then-Train Wildlife Identification

WildScope is a local learning project that demonstrates how labeled observations can move through
an online inference and incremental supervised-training loop without confusing confidence,
training agreement, and future-data accuracy.

It uses recent iNaturalist observations from ten tropical protected areas, runs SpeciesNet on each
image, compares generated predictions with iNaturalist community identifications, and builds a
feed-specific species candidate catalog from data received after the deployed model's watermark.
Pretrained BioCLIP then performs selective visual classification over that catalog.

## Learning question

> How should a model be evaluated on newly arriving labeled data before that same data becomes the
> training batch for the next model version?

WildScope answers with a **test-then-train** lifecycle rather than a random train/test split.

```text
deployed version N
	-> predicts newly arrived batch N+1
	-> is scored against obtained iNaturalist identifications
	-> all eligible images in batch N+1 train version N+1
	-> version N+1 waits for batch N+2 before earning live accuracy
```

## Two connected portals

### Observation Review (`/`)

This is the inference surface. It:

- fetches observations created in the last 24 hours;
- shows the iNaturalist-obtained identification beside the selected model prediction;
- compares the static SpeciesNet baseline with the currently deployed selective BioCLIP model;
- shows SpeciesNet confidence and BioCLIP cosine margin separately, never treating either as accuracy;
- scores baseline and deployed predictions on eligible labels newer than the deployed watermark;
- shows how many unseen labels are waiting for the next training run;
- maps only public iNaturalist coordinates and preserves source obscuration flags;
- links to the Training Portal rather than starting training itself.

### Training Portal (`/training`)

This is the data and model-lifecycle surface. It:

- uses checkboxes to select one or more protected-area datasets;
- aggregates selected manifests and metrics without merging feed-specific models;
- names the deployed model being evaluated;
- identifies the SpeciesNet engine and stored prediction artifact versions;
- shows the exact exclusive/inclusive batch window;
- lists every image, timestamp, obtained target, baseline prediction, and deployed prediction;
- reports total observations, eligible labels, exclusion counts, and target distribution;
- reports baseline and deployed exact-identification accuracy before training starts;
- streams fetch, download, preprocessing, baseline inference, evaluation, training, and persistence stages;
- trains the next version on every eligible image in the displayed batch;
- records evaluated-version -> trained-version lineage;
- labels same-batch post-training performance as **training agreement**, not accuracy;
- leaves the new version without live accuracy until unseen labels arrive.

The pages share the same model ID, watermark, pending-batch count, and batch API. They are separate
workspaces, not separate products.

## Data roles

| Value | Source | Role |
|---|---|---|
| Original image | iNaturalist photo asset | Model input and visual evidence |
| Obtained identification | iNaturalist community taxon | Supervised target and evaluation reference |
| Quality grade | iNaturalist | Eligibility gate (`research` only) |
| Baseline prediction | SpeciesNet | Static comparison model |
| Deployed prediction | Latest selective BioCLIP version | Version evaluated on unseen post-watermark data |
| Training batch | Eligible observations newer than the deployed watermark | Entire input to the next candidate-catalog version |

The obtained identification is not called unquestionable ground truth. Community identifications
can be wrong. WildScope measures **agreement with the obtained research-grade identification**.

## Test-then-train contract

For a deployed model with watermark $w_N$:

1. Query iNaturalist for observations newer than $w_N$.
2. Download and preprocess uncached images.
3. Run SpeciesNet for images without a baseline prediction.
4. Run deployed version $N$ before changing it.
5. Keep only licensed, `research`-grade targets for scoring/training.
6. Calculate exact identification agreement for SpeciesNet and version $N$.
7. Persist the evaluated model ID and exact sample manifest.
8. Train version $N+1$ on **all** eligible rows in that batch.
9. Set $w_{N+1}$ to the newest `created_at` timestamp consumed.
10. Apply version $N+1$ for inference, but mark its live accuracy as awaiting new data.

The training interval is:

$$
w_N < \text{observation.created\_at} \leq w_{N+1}
$$

If there are no eligible rows after $w_N$, training fails without replacing the model.

### Legacy bootstrap exception

A model created before `test-then-train-v1` has no auditable evaluated-version -> trained-version
lineage. For that one migration only, the Training Portal exposes all cached licensed,
`research`-grade labels as a **bootstrap batch**, scores SpeciesNet as the baseline, and trains the
first lifecycle-compatible model. After that model is written, normal post-watermark rules apply
and older labels cannot be consumed again.

## Models

### SpeciesNet baseline

WildScope runs SpeciesNet 5.0.5, an ensemble that combines MegaDetector with a species classifier.
SpeciesNet may return a species, family, class, or generic `animal` label depending on its evidence.

### Selective BioCLIP species model

The deployed WildScope component combines a pretrained biodiversity encoder with a deliberately
small feed-specific artifact:

```text
image pixels -> pretrained BioCLIP -> feed candidate species -> margin gate
```

BioCLIP is a ViT-B/16 image-text model trained on TreeOfLife-10M. WildScope does **not** fine-tune
BioCLIP or SpeciesNet on the sparse per-feed batches. The incremental training step accumulates the
licensed research-grade scientific names that the next feed version may consider, while retaining
the exact image manifest, watermark, and target metadata. At inference, BioCLIP compares a full
frame with prompts for those known species.

The model emits a species only when the top-1 minus top-2 cosine-similarity margin is at least
`0.075`; otherwise it emits `Unidentified`. The margin is a ranking separation, not a calibrated
probability. A newly arrived species outside the deployed candidate catalog cannot be emitted until
its obtained label is consumed by a later version.

On the 34-image Monteverde development sample, unrestricted BioCLIP selected the obtained species
for 26 images. The initial `0.05` full-frame gate emitted 18 labels and all 18 matched, but a later
live Monteverde batch exposed one wrong label at margin `0.0616`. The gate was therefore raised to
`0.075`; this would have retained 13/34 correct development-sample labels and rejected that live
error. These are small-sample operating observations, **not** future-data accuracy. WildScope's
test-then-train ledger remains the source of live performance on unseen observations.

Generated species are stored canonically by scientific name. WildScope also stores the iNaturalist
taxon ID and preferred common name when those are present in the observation response. Legacy model
payloads without that catalog are enriched through a cached iNaturalist `/v1/taxa` lookup by exact
scientific name. This lookup changes presentation metadata only; it does not change the model's
classification or make an incorrect species prediction correct.

The visual candidate catalog is versioned as `test-then-train-v3-bioclip-selective`. Models created
before this format are not deleted; the Training Portal marks them for a one-time bootstrap
using existing eligible labels. Images, SpeciesNet predictions, watermarks, and run history remain
intact. Once rebuilt, common/scientific target metadata is part of the deployed model payload.

Legacy categorical payloads remain safe: when one coarse SpeciesNet source label maps to several
observed species (for example `aves;bird`), that fallback **abstains** instead of selecting the most
frequent target. Review cards display only `Unidentified` with no confidence or implied taxon. A v3
bootstrap adds the visual candidate catalog without deleting images, SpeciesNet predictions,
watermarks, or run history.

## Metric definitions

### Exact identification accuracy

For a pending batch evaluated before training:

$$
	ext{accuracy} =
\frac{\text{predictions matching obtained scientific or common name}}
{\text{eligible batch images}}
$$

Matching normalizes case and punctuation and compares the leaf SpeciesNet taxonomy label. This is a
strict species/name-level metric. A correct higher taxon such as `bird` does not count as an exact
match for `Eumyias thalassinus`.

### Confidence and visual margin

SpeciesNet confidence is displayed as its model score. BioCLIP displays the top-1/top-2 cosine
margin instead of relabeling that value as confidence. Neither is an accuracy claim.

### Training agreement

After training, the new corrector may be run against the batch it just consumed. That value is
reported only as **training agreement**. It is not evidence of generalization and does not become the
new version's live accuracy.

### Additional meaningful metrics

WildScope also reports metrics that can be derived honestly from the available labels and model
outputs:

- **correct / error counts**: the numerator and residual behind exact accuracy;
- **prediction coverage**: fraction of eligible rows with a generated label;
- **mean confidence**: average confidence for covered predictions, kept separate from accuracy;
- **target diversity**: number and distribution of eligible scientific-name targets;
- **dataset composition**: total, cached, licensed, research-grade, and excluded rows;
- **training agreement**: same-batch agreement after fitting, explicitly not generalization;
- **duration and stage throughput**: observable execution rather than model quality;
- **version lineage and watermark window**: which model met which data and what version followed.

Macro-F1, per-class recall, and confusion matrices are not shown by default. Post-watermark batches
are often tiny and contain many singleton species, so those values would be undefined, unstable, or
visually persuasive without being meaningful. They become appropriate once repeated targets exist
in a sufficiently large evaluated batch.

## Pointed decisions and constraints

| Decision | Reason and consequence |
|---|---|
| Use iNaturalist observations, not “live cameras” | No stable public API was found for ten tropical camera archives. These are near-real-time community observations. |
| Keep inference and training on separate pages | It makes model use and model change distinct while preserving one linked lifecycle. |
| Use dataset checkboxes on the Training Portal | Users can inspect several pending feeds together; each selected feed still builds its own candidate catalog sequentially because observed taxa are feed-specific. |
| Do not use a random train/test split | The teaching goal is online learning. The next naturally arriving labeled batch evaluates the deployed version before training. |
| Train on the whole post-watermark batch | Every image shown in the Training Portal is consumed by the next version; there is no hidden partition. |
| Allow one explicit legacy bootstrap | Existing labels remain usable after adopting the lifecycle protocol, but the UI does not misrepresent them as unseen deployed-model evaluation data. |
| Defer new-version accuracy | A model cannot honestly earn future-data accuracy from the batch that trained it. |
| Require `research` quality and a photo license | Lower-confidence community IDs and unlicensed photos remain reviewable but are not supervised targets. |
| Preserve filtered taxa in storage | Insects, arachnids, and reptiles are hidden from review by preference, but source records are not silently deleted. |
| Use strict exact-name accuracy | It is simple and auditable, but under-credits correct higher-taxonomy predictions. Hierarchical accuracy is a future extension. |
| Use pretrained BioCLIP without image-model fine-tuning | A biodiversity-specific encoder adds real visual evidence, while the available per-feed batches remain too small and imbalanced for a defensible fine-tune. |
| Resolve common names as metadata, not predictions | Scientific names remain canonical targets. Cached iNaturalist taxon lookups add readable common names but never alter correctness. |
| Abstain below a visual margin | If BioCLIP's top-two candidate separation is below `0.075`, the public result is `Unidentified`; margin and candidate provenance remain auditable. |
| Request original images first | If unavailable or over 15 MB, use the large derivative and retain that processing provenance. |
| Enhance only low-resolution inputs | Images below 1280x720 use CLAHE, bicubic x2 enlargement, and bilateral denoising. High-resolution images are not upscaled. |
| Deduplicate by SHA-256 | Reposted identical images should not inflate evaluation or training counts. |
| Display only public coordinates | WildScope never attempts to recover obscured or private locations. |

## Image pipeline

Each frame can expose four inspectable stages:

1. iNaturalist original or large fallback;
2. EXIF-corrected RGB normalization;
3. low-resolution enhancement or high-resolution passthrough;
4. obtained identification, SpeciesNet output, and selective BioCLIP output.

The UI uses `object-fit: contain` so the visual evidence is not cropped.

## Ten protected-area feeds

- Monteverde Cloud Forest, Costa Rica
- Tambopata, Peru
- Yasuni National Park, Ecuador
- Tortuguero National Park, Costa Rica
- Daintree, Australia
- Tijuca National Park, Brazil
- Kinabalu Park, Malaysia
- Taman Negara, Malaysia
- Khao Yai National Park, Thailand
- Bwindi Impenetrable Forest, Uganda

Daily volume varies. A feed may have no eligible observations in a 24-hour window.

## API

| Endpoint | Purpose |
|---|---|
| `GET /api/status` | Service and static-model readiness |
| `GET /api/feeds` | Protected-area feeds and deployed model metadata |
| `POST /api/feeds/<id>/sync` | Fetch and classify the last 24 hours |
| `GET /api/jobs/<id>` | Sync or training job status |
| `GET /api/feeds/<id>/frames?page=N` | Ten review frames per page |
| `GET /api/feeds/<id>/locations` | Public coordinate groups |
| `GET /api/feeds/<id>/locations/<photo_id>/frames` | Images sharing one public coordinate |
| `GET /api/frames/<photo_id>` | Frame provenance and processing stages |
| `GET /api/images/<photo_id>?stage=<stage>` | Source, normalized, enhanced, or model-input image |
| `GET /api/feeds/<id>/training` | Deployed model, pending batch manifest, and version history |
| `POST /api/feeds/<id>/train` | Test deployed version, then train the next version on the pending batch |

## Project structure

```text
configs/feeds.yaml                 protected-area source definitions
src/wildscope/feeds.py             iNaturalist pagination and contracts
src/wildscope/preprocessing.py     inspectable image preparation
src/wildscope/inference.py         SpeciesNet runner, BioCLIP classifier, and matching
src/wildscope/storage.py           SQLite observations, predictions, versions, and runs
src/wildscope/service.py           sync and test-then-train orchestration
src/wildscope/web/app.py           Flask API and page routes
templates/observations.html        inference workspace
templates/training.html            data and model-lifecycle workspace
static/js/observations.js          inference interactions
static/js/training-portal.js       multi-feed selection, pipeline, and lineage interactions
tests/backend/                     Python behavior and API tests
tests/browser/app.spec.js          desktop/mobile browser contract
```

## Setup

From `projects/wildscope`:

```powershell
python -m pip install -e ".[dev]"
python -m wildscope.web
```

Open `http://127.0.0.1:5000`. The first SpeciesNet and BioCLIP runs download their official model
weights. Subsequent runs reuse cached models and images under ignored
`artifacts/runtime/`.

## Validation

```powershell
python -m pytest tests/backend -q
python -m ruff check src/wildscope tests/backend
cd tests/browser
npx playwright test
```

## Honest limitations

- Exact-name agreement is not biological truth and does not measure hierarchical correctness.
- Small and imbalanced post-watermark batches can make live accuracy volatile.
- BioCLIP candidates are limited to species in the deployed feed catalog; open-world species remain unidentified until a later version includes them.
- The `0.075` margin threshold is based on small development and live samples and must be re-evaluated as future unseen batches accumulate.
- A newly trained version has no live accuracy until another eligible batch arrives.
- iNaturalist `created_at` is the ingestion boundary; observation dates may be older.
- Provider rate limits, missing licenses, duplicate images, and sparse feeds reduce usable data.

