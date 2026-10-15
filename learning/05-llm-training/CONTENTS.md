# Contents: LLM Training

Preserve the data and training state needed to trust, resume, and inspect a model-building run.

## Required route

- [Stage 05 LLM Training, Part 1: Build the Pretraining Data Stream](01-pretraining-data-pipeline.ipynb) — Build the text stream a model practices on while preserving document boundaries, data splits, and source history.
- [Stage 05, Part 1 — Pretraining Data Pipeline: Handwritten Theory Notes](01-pretraining-data-pipeline-theory.md) — Review how duplicate control, source history, and text boundaries affect what a training run can claim.
- [Stage 05 LLM Training, Part 2: What Pretraining Must Preserve](02-pretraining-theory.ipynb) — Determine what a weight update, a check on unused examples, and a complete saved pause state must preserve in a trustworthy training run.
- [Stage 05, Part 2 — Pretraining Theory: Handwritten Theory Notes](02-pretraining-theory.md) — Review why learning progress, fair model selection, and exact resume behavior require different evidence.
- [Stage 05 LLM Training, Part 3: Pretrain a Base Model — PyTorch Lab](03-pretrain-a-base-model-lab.ipynb) — Train from random weights, pause and resume the run, and reload a final model package with inspectable data history.

## Optional

- [Optional Training Branches](optional/CONTENTS.md) — (optional) Revisit the manual loop through a framework translation or a full tiny-model package audit.
