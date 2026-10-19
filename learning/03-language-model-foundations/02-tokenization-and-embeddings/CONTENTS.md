# Contents: Tokenization and Embeddings

Turn text into model inputs without confusing the pieces, their addresses, and their learned representations.

## Required route

- [Chapter 02: Tokenization and Embeddings](README.md) — Check the shared legal-text example, prerequisites, and route through the two representation lessons.
- [02A - Tokenization and Embeddings Theory](02A-tokenization-and-embeddings-theory.ipynb) — Split unfamiliar words into reusable pieces and distinguish piece identifiers from the learned rows they select.
- [02B - Tokenization and Embeddings PyTorch Lab](02B-tokenization-and-embeddings-pytorch-lab.ipynb) — Build a small text splitter, inspect a real tokenizer, and check which learned rows and blank positions receive training signals.
- [Tokenization and Embeddings: Handwritten Theory Notes](02A-tokenization-and-embeddings-theory.md) — Review why text pieces, identifiers, learned rows, and rules for ignoring blanks have different jobs.

## Optional

No optional lessons are listed in this directory.

## Continue to model architectures

- [Chapter 03: Language Models](../03-language-models/README.md) — The former 02C is preserved as 03A, followed by ordered windows, RNN/LSTM text models, and encoder-decoder attention.
- [03A - Training Skeleton and Word Objectives](../03-language-models/03A-training-skeleton-and-word-objectives.ipynb) — Reuse one training loop while changing from predicting neighboring words to reading one sequence and writing another.
- [03A - Theory Notes](../03-language-models/03A-training-skeleton-and-word-objectives-theory.md) — Review how the prediction target changes a learned word representation and a reader-to-writer handoff.
