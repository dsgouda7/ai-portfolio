# Contents: Tokenization and Embeddings

Turn text into model inputs without confusing the pieces, their addresses, and their learned representations.

## Required route

- [Chapter 02: Tokenization and Embeddings](README.md) — Check the shared legal-text example, prerequisites, and route through the three prediction lessons.
- [02A - Tokenization and Embeddings Theory](02A-tokenization-and-embeddings-theory.ipynb) — Split unfamiliar words into reusable pieces and distinguish piece identifiers from the learned rows they select.
- [02B - Tokenization and Embeddings PyTorch Lab](02B-tokenization-and-embeddings-pytorch-lab.ipynb) — Build a small text splitter, inspect a real tokenizer, and check which learned rows and blank positions receive training signals.
- [02C - One Training Skeleton, Three Language Objectives](02C-context-windows-to-static-embeddings.ipynb) — Reuse one training loop while changing from predicting neighboring words to reading one sequence and writing another.
- [One Training Skeleton, Three Language Objectives](02C-context-windows-to-static-embeddings-theory.md) — Review how the prediction target changes a learned word representation and a reader-to-writer handoff.
- [Tokenization and Embeddings: Handwritten Theory Notes](02A-tokenization-and-embeddings-theory.md) — Review why text pieces, identifiers, learned rows, and rules for ignoring blanks have different jobs.

## Optional

No optional lessons are listed in this directory.
