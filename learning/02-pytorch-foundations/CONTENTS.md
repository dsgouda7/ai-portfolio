# Contents: PyTorch Foundations

Follow the loop from sensor readings to batches, validation, and saved state.

## Required route

- [The Deep-Learning Map](01-deep-learning-map/README.md) — Separate prediction, error measurement, error tracing, and weight updates in one tiny learning run.
- [Tensors and Autograd](02-tensors-and-autograd/README.md) — Track array shapes and memory placement, then let PyTorch calculate how each value affects an error.
- [Models and Training Loops](03-models-and-training-loops/README.md) — Move from loose arrays to a model that owns its weights and a loop with explicit update steps.
- [Data, Checkpoints, and Hyperparameters](04-data-checkpoints-and-hyperparameters/README.md) — Check batch coverage, reload predictions, and write a training-state snapshot.
- [The Modern Training Lifecycle](05-modern-training-lifecycle/README.md) — Separate validation, select and reload the best model, and check snapshot fields—not interruption recovery.

## Optional

- [Framework translation and vision](optional/CONTENTS.md) — Translate familiar Keras code or inspect filters and image reconstruction.

## Shortcuts

- [Training Skeleton Quick Refresher](quick-refresher/README.md) — Refresh the core loop; use its links for lifecycle topics it does not run.
