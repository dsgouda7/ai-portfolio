# Chapter 01: Sequence Memory

**Contents:** [one-line map of this directory](CONTENTS.md)


This required PyTorch-first arc preserves one memory question across three fresh-kernel notebooks.

> **Preview contract:** 01B uses token IDs, padding, shifted targets, and causal invariance to make
> recurrent memory measurable. It does not yet explain how a tokenizer creates those IDs or why
> padding, loss, and causal masks are different mechanisms. Chapter 02 returns to each preview and
> gives it formal ownership.

1. [01A — Sequence Memory Theory](01A-sequence-memory-theory.ipynb) makes recurrence necessary, measures vanishing gradients, and opens the LSTM gates.
2. [01B — Sequence Memory PyTorch Lab](01B-sequence-memory-pytorch-lab.ipynb) moves from forecasting windows to token IDs, packed sequences, padding-aware loss, causal invariance, and clipping.
3. [01C — Cinematic Piano Memory](01C-cinematic-piano-memory.ipynb) turns the same memory question into an audible capstone.
4. [Sequence Memory Theory Notes](01A-sequence-memory-theory.md) is the concise handwritten companion.

| Preview used here | Formal treatment in Chapter 02 |
|---|---|
| Integer token IDs | BPE pieces and vocabulary addresses |
| Padded sequence batches | Padding positions and ignored loss targets |
| Shifted next-token targets | Sequence logits and target alignment |
| Causal invariance | Causal access versus padding and loss masking |

The notebooks share only the readable helpers in [`sequence_helpers.py`](sequence_helpers.py).
Kernels do not share state; each notebook recreates or reloads everything it needs.

Run `setup.ps1` on Windows or `setup.sh` on Linux/macOS. The script creates this chapter's
`.venv`, installs `requirements.txt`, and assigns the
`stage03-language-model-01-sequence-memory` kernel.

Continue to [Chapter 02: Tokenization and Embeddings](../02-tokenization-and-embeddings/README.md).

## Required Bounded Intervention

Run **“Your turn: change the retention factor”** in
[`01A-sequence-memory-theory.ipynb`](01A-sequence-memory-theory.ipynb). Predict the first distance
below 1%, change only `retention_to_test`, measure the crossing, diagnose why it moved, and explain
whether the result supports the bounded claim that a stronger repeated memory route carries a
teaching signal farther.
