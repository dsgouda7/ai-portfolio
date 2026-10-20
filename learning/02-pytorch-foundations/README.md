# Stage 02: PyTorch Foundations

Learn the PyTorch loop, named tensor axes, batches, validation, and saving state.
[Contents](CONTENTS.md)

## Choose your route

1. [Deep-Learning Map](01-deep-learning-map/README.md) — train a curved sensor response.
2. [Tensors and Autograd](02-tensors-and-autograd/README.md) — add text-shaped axes to see shape mistakes.
3. [Models and Training Loops](03-models-and-training-loops/README.md) — return to two sensor columns and one fuel target.
4. [Data, Checkpoints, and Hyperparameters](04-data-checkpoints-and-hyperparameters/README.md) — batch rows, reload weights, and save training state.
5. [Modern Training Lifecycle](05-modern-training-lifecycle/README.md) — keep validation separate and reload the best model.

Each chapter links its notebook, theory notes, and setup. Run setup from that chapter's directory.

Already know PyTorch? The [quick refresher](quick-refresher/README.md) checks the core loop,
not validation or checkpoint recovery. Its table points back to gaps.
[Framework translation](optional/framework-translation/README.md) and
[vision](optional/vision/README.md) are optional.

The snapshots in Parts 04–05 save selected training state; they do not prove the same next update
after an interruption. Text-shaped tensors check mechanics, not language learning.

**Previous:** [ML Foundations](../01-ml-foundations/README.md)
**Next:** [Sequence Memory](../03-language-model-foundations/01-sequence-memory/README.md) — reuse the loop, but add a way to remember earlier inputs.
