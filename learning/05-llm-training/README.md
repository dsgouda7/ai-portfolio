# Stage 05: LLM Training

**Contents:** [one-line map of this directory](CONTENTS.md)

**Previous:** [Stage 04 · Transformers](../04-transformers/README.md)

**Current:** Stage 05 · LLM Training

**Next:** [Stage 06 · LLM Fine-Tuning](../06-llm-finetuning/README.md)

Stage 04 built the model contract. This stage asks what must be preserved when that model learns
from a real token stream: data lineage, held-out selection, complete pause state, deterministic
resume, and a reloadable final package.

## Required Route

| Part | Artifact | Start with the job | End with the evidence |
|---|---|---|---|
| [1. Pretraining Data Pipeline](01-pretraining-data-pipeline.ipynb) · [Theory](01-pretraining-data-pipeline-theory.md) | Decide what token stream the model will practice | Splits, duplicates, tokenizer fitting, document boundaries, packing, and manifests | Trustworthy, reproducible train and validation shards |
| [2. Pretraining Theory](02-pretraining-theory.ipynb) · [Theory notes](02-pretraining-theory.md) | Decide what evidence makes a run trustworthy | Update lifecycle, held-out selection, complete pause state, and artifact lineage | A testable training and checkpoint contract |
| [3. Pretrain a Base Model Lab](03-pretrain-a-base-model-lab.ipynb) | Turn random weights into a resumable package | Manual training and validation, generation, complete checkpoints, resume parity, and lineage | A selected, reloadable base-model package |

Run Parts 1–3 in order. They exchange only validated files under `artifacts/base-lm/`. The
architecture contract is recreated by Part 1 from Stage 04's final design, then the data and
training notebooks extend the same package.

## Optional Extensions

- [Lightning translation](optional/lightning/training-a-base-model-with-lightning.ipynb) maps the
  understood manual lifecycle to Lightning hooks and callbacks.
- [Full-model audit](optional/full-model-audit/tiny-transformer-for-ai-engineers.ipynb) checks the
  completed data, model, training, evaluation, package, and resume contracts without modifying
  required artifacts.

Run `setup.ps1` on Windows or `setup.sh` on Linux/macOS. Either script creates or reuses this
stage's `.venv`, installs the adjacent `requirements.txt`, registers the stage-unique
`learning-05-llm-training` Jupyter kernel, and assigns it recursively to required and optional
notebooks.

The default Part 3 profile is CPU-safe and intentionally small. It proves mechanics and artifact
completeness; it does not claim that the checkpoint acquired broad knowledge, reasoning, or
assistant behavior.

## Evidence Ladder

Use these labels without promoting one into another: **Implemented source**, **Local fixture**,
**Local measured**, **Modeled or simulated**, **Customer-validated**, **Live-cloud validated**, and
**Independently reviewed**. Parts 1–3 can produce mechanics evidence through the first four rungs.
They do not supply the last three capability/live-evidence rungs.

## One Required Bounded Intervention per Part

Predict first; change only the named variable; measure; diagnose; then state whether the bounded
claim is supported.

| Part | Runnable section and one-variable test | Evidence-safe claim to judge |
|---|---|---|
| 1 | [`01-pretraining-data-pipeline.ipynb`](01-pretraining-data-pipeline.ipynb), **“Insert Explicit Document Boundaries”**: change only `preview_chars`; measure the EOD token count and boundary decode | **Local measured mechanics:** preview width does not remove the atomic EOD boundary |
| 2 | [`02-pretraining-theory.ipynb`](02-pretraining-theory.ipynb), **“Validation chooses; training loss diagnoses”**: change only AdamW `lr` from `0.08` to `0.02`; measure final train and validation loss | **Local measured mechanics:** the update policy changes the measured trajectory; one miniature run does not establish general capability |
| 3 | [`03-pretrain-a-base-model-lab.ipynb`](03-pretrain-a-base-model-lab.ipynb), **“Same Mechanism, Larger Budget”**: double only the `100M` profile's `tokens_per_step`; measure estimated state memory and rough FLOPs | **Modeled or simulated:** token budget changes estimated compute while parameter-state memory stays fixed; this is not a live hardware result |
