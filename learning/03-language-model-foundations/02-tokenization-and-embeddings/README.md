# Chapter 02: Tokenization and Embeddings

**Contents:** [directory map](CONTENTS.md)

Turn a pet-sitter's note into pieces, integer addresses, and learned rows.
Then check why blank batch slots must not count as evidence.

## Before you start

Use [Stage 02 — PyTorch Foundations](../../02-pytorch-foundations/README.md), or the
[training-loop refresher](../../02-pytorch-foundations/quick-refresher/pytorch-training-skeleton.ipynb)
if you can pass its closing check.
[Chapter 01B](../01-sequence-memory/01B-sequence-memory-pytorch-lab.ipynb) already used
IDs, shifted targets, and masks to study memory; here you explain and test those choices.

## Read in order

1. [02A — Tokenization and Embeddings Theory](02A-tokenization-and-embeddings-theory.ipynb) ·
   [handwritten notes](02A-tokenization-and-embeddings-theory.md) — Discover BPE, separate
   addresses from rows, and expose a padding failure.
2. [02B — Tokenization and Embeddings PyTorch Lab](02B-tokenization-and-embeddings-pytorch-lab.ipynb) —
   Build the small BPE loop, inspect GPT-2 with `tiktoken`, train an embedding table,
   and check which rows and targets receive learning signals.

## Run it

From this directory, run [`setup.ps1`](setup.ps1) on Windows or [`setup.sh`](setup.sh) on
Linux/macOS. It creates a chapter `.venv`, installs [requirements](requirements.txt),
and assigns kernel `stage03-language-model-02-tokenization`.
Restart and run each notebook independently. The lab uses GPT-2's tokenizer, not its
pretrained model weights; `tiktoken` may fetch its vocabulary on first use.

**Next:** [Chapter 03 — Language Models](../03-language-models/README.md).
[03A](../03-language-models/03A-training-skeleton-and-word-objectives.ipynb) ·
[theory notes](../03-language-models/03A-training-skeleton-and-word-objectives-theory.md)
asks how ordinary text can supply the practice questions that train those rows.
