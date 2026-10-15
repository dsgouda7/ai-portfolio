# Stage 02: PyTorch Foundations

**Contents:** [one-line map of this directory](CONTENTS.md)



This stage turns Stage 01's conceptual contracts into one visible PyTorch training path. Take the
full route when any responsibility in the loop is still fuzzy; use the quick refresher only when
you can already explain the whole skeleton.

**Ownership:** you bring Stage 01's reasons for learning and checking a model. This stage owns tensors (number containers with named dimensions), autograd (automatic tracing of how error depends on those numbers), modules (homes for learned state), and data, training, and checkpoint mechanics. Language-shaped examples preview these mechanics; [Stage 03](../03-language-model-foundations/README.md) owns what the text pieces mean and how sequence tasks are built.

## Choose your route

**Full route**

1. [Deep-Learning Map](01-deep-learning-map/01-deep-learning-map.ipynb)
2. [Tensors and Autograd](02-tensors-and-autograd/02-tensors-and-autograd.ipynb)
3. [Models and Training Loops](03-models-and-training-loops/03-models-and-training-loops.ipynb)
4. [Data, Checkpoints, and Hyperparameters](04-data-checkpoints-and-hyperparameters/04-data-checkpoints-and-hyperparameters.ipynb)
5. [Modern Training Lifecycle](05-modern-training-lifecycle/05-modern-training-lifecycle.ipynb)

**Returning-reader shortcut**

Run [PyTorch Training Skeleton: Quick Refresher](quick-refresher/pytorch-training-skeleton.ipynb).
Its closing readiness table links directly to the full chapter that repairs each gap.

## Optional revision branches

- [Framework Translation](optional/framework-translation/README.md)
- [Vision](optional/vision/README.md)

Run setup from the chapter directory you are about to use. Each script creates or reuses a
chapter-local `.venv`, installs the adjacent `requirements.txt`, and assigns a stage-specific
kernel to that chapter's notebooks.

**Previous:** [Stage 01 — ML Foundations](../01-ml-foundations/README.md)  
**Current:** Stage 02 — PyTorch Foundations  
**Next:** [Stage 03 — Language-Model Foundations](../03-language-model-foundations/README.md)


The trainer can now update and reload a model, but a single sensor row contains no memory of earlier rows. [Sequence Memory](../03-language-model-foundations/01-sequence-memory/README.md) asks how the prediction can carry useful information from earlier steps. Bring the training loop and named-axis habit, not an assumption that a language-shaped tensor already understands language.
