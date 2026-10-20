# Part 04: Data, Checkpoints, and Hyperparameters

Run [Part 03](../03-models-and-training-loops/README.md) first. Reload its fuel model, check
batch coverage, save a training-state snapshot, and compare learning rates.

Weight reload checks predictions. Saving optimizer state too does not prove the same next update;
this lab does not run an interrupted-training comparison.

- [Notebook](04-data-checkpoints-and-hyperparameters.ipynb) · [Theory](04-data-checkpoints-and-hyperparameters-theory.md) · [Contents](CONTENTS.md)
- Setup from this directory: [Windows](setup.ps1) (`.\setup.ps1`) or [Linux/macOS](setup.sh) (`bash ./setup.sh`).
- Next: [Modern Training Lifecycle](../05-modern-training-lifecycle/README.md) adds validation and model selection.
