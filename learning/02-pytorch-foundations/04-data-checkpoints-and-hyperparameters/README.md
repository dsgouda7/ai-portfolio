# Part 04: Data, Checkpoints, and Hyperparameters

- [Notebook](04-data-checkpoints-and-hyperparameters.ipynb)
- [Handwritten theory](04-data-checkpoints-and-hyperparameters-theory.md)

Run Part 03 first. This chapter reloads its saved state from a fresh kernel, then separates manual
batching, `Dataset`, `DataLoader`, model state, optimizer state, experiment configuration, and
controlled hyperparameter comparisons.

Run `./setup.ps1` on Windows or `bash ./setup.sh` on Linux/macOS.

## Required Bounded Intervention

Run **“Your turn: change only the batch size”** in
[`04-data-checkpoints-and-hyperparameters.ipynb`](04-data-checkpoints-and-hyperparameters.ipynb).
Predict the batch-count and final-batch shape, change only `YOUR_BATCH_SIZE`, measure the emitted
batch sizes, diagnose any remainder batch, and explain whether the evidence supports the loader's
sample-preservation contract.
