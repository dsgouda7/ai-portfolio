# Part 04: The Modern Training Lifecycle

- [Notebook](04-modern-training-lifecycle.ipynb)
- [Handwritten theory](04-modern-training-lifecycle-theory.md)

This chapter decomposes modern training into small readable units: split-before-normalization,
loaders, train/eval modes, separate train and validation loops, scheduling, mixed-precision
boundaries, early stopping, best-model checkpoints, resumable checkpoints, reload, and inference
in original units.

Lightning and `torch.compile` are optional mappings after the manual lifecycle is understood.

Run `./setup.ps1` on Windows or `bash ./setup.sh` on Linux/macOS.
