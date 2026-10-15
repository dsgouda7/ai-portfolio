# Part 05: The Modern Training Lifecycle

- [Notebook](05-modern-training-lifecycle.ipynb)
- [Handwritten theory](05-modern-training-lifecycle-theory.md)

This chapter decomposes modern training into small readable units: split-before-normalization,
loaders, train/eval modes, separate train and validation loops, scheduling, mixed-precision
boundaries, early stopping, best-model checkpoints, resumable checkpoints, reload, and inference
in original units.

Lightning and `torch.compile` are optional mappings after the manual lifecycle is understood.

Run `./setup.ps1` on Windows or `bash ./setup.sh` on Linux/macOS.

## Required Bounded Intervention

Run **“Your turn: change only patience”** in
[`05-modern-training-lifecycle.ipynb`](05-modern-training-lifecycle.ipynb). Predict the stopping
step, change only `YOUR_PATIENCE`, measure the reported stop point, diagnose how many non-improving
epochs were tolerated, and explain whether the evidence supports the stopping-rule claim without
claiming a better model.
