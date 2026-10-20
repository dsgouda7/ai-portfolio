# Stage 03: Language-Model Foundations

**Contents:** [directory map](CONTENTS.md)

You know the PyTorch training loop. Now ask what a model needs to remember, how text becomes
numbers, and why a writer might need to look back at its input.

## Start here

Finish [Stage 02 — PyTorch Foundations](../02-pytorch-foundations/README.md), then follow this order.
Each chapter has its own setup script and notebook kernel; its README tells you what to run.

| Chapter | Start with | What changes |
|---|---|---|
| [01. Sequence memory](01-sequence-memory/README.md) | [01A notebook](01-sequence-memory/01A-sequence-memory-theory.ipynb) · [theory notes](01-sequence-memory/01A-sequence-memory-theory.md) | Carry a clue through time; test RNN and LSTM memory |
| [02. Tokenization and embeddings](02-tokenization-and-embeddings/README.md) | [02A notebook](02-tokenization-and-embeddings/02A-tokenization-and-embeddings-theory.ipynb) · [theory notes](02-tokenization-and-embeddings/02A-tokenization-and-embeddings-theory.md) | Turn pet-sitter's notes into pieces, IDs, learned rows, and padded batches |
| [03. Language models](03-language-models/README.md) | [03A notebook](03-language-models/03A-training-skeleton-and-word-objectives.ipynb) · [theory notes](03-language-models/03A-training-skeleton-and-word-objectives-theory.md) | Make practice questions from text, then follow fixed windows, recurrent memory, and source attention |

Chapter 01B uses token IDs and masks to test memory before Chapter 02 explains how they work.
That preview is intentional: you don't need to master tokenization before studying recurrence.

![Piano notes become token pieces, integer addresses, and learned rows, then shifted next-note targets, padding, and separate input and loss masks. All are first used in the Chapter 01B lab and formally explained in Chapter 02.](images/03-first-use-concept-map.png)

[Embedding history](optional/embedding-history/README.md) is optional background, not a prerequisite.
The old model-evolution lessons now live in Chapter 03 as 03B–03D; follow the links there.

**Next:** [Stage 04 — Transformers](../04-transformers/README.md) replaces recurrent reading
with attention and position information.
