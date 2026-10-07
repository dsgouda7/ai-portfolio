# Sequence Memory

This required PyTorch-first arc replaces the executable Keras sequence route while preserving its strongest evidence.

1. [05A - Sequence Memory Theory](05A-sequence-memory-theory.ipynb) makes recurrence necessary, measures vanishing gradients, and opens the LSTM gates.
2. [05B - Sequence Memory PyTorch Lab](05B-sequence-memory-pytorch-lab.ipynb) moves from forecasting windows to token IDs, packed sequences, padding-aware loss, causal invariance, and clipping.
3. [05C - Cinematic Piano Memory](05C-cinematic-piano-memory.ipynb) turns the same memory question into an audible capstone.
4. [Sequence Memory Theory Notes](05A-sequence-memory-theory.md) is the concise handwritten companion.

The notebooks share only the small, readable helpers in [`sequence_helpers.py`](sequence_helpers.py). Kernels do not share state; each notebook recreates or reloads everything it needs.

Run `setup.ps1` on Windows or `setup.sh` on Linux/macOS. The script creates or reuses this chapter's `.venv`, installs `requirements.txt`, registers the unique `pytorch-llms-05-sequence-memory` kernel, and assigns it to all three notebooks. Use `-SkipKernel` or `--skip-kernel` when only the local environment and dependencies are needed.
