# Stage 03: Language-Model Foundations

**Contents:** [one-line map of this directory](CONTENTS.md)


This stage changes the shape of the problem. The PyTorch loop is now familiar; the new complaints
are memory across a sequence and useful representations for discrete tokens.

## Required route

1. [Sequence Memory](01-sequence-memory/README.md)
2. [Tokenization and Embeddings](02-tokenization-and-embeddings/README.md)
3. [Language Models: Learned Rows to Attention](03-language-models/README.md)

## Recommended reader route

1. [Stage 02 — PyTorch Foundations](../02-pytorch-foundations/README.md) gives you the tensor, batch, and training-loop contracts used throughout this stage.
2. [Chapter 01 — Sequence Memory](01-sequence-memory/README.md) first studies recurrence on numeric sequences, so memory has a clear job before text machinery appears.
3. [Chapter 02 — Tokenization and Embeddings](02-tokenization-and-embeddings/README.md) formally explains the token pieces, integer addresses, learned rows, and padding that Chapter 01B previews.
4. [03A — Training Skeleton and Word Objectives](03-language-models/03A-training-skeleton-and-word-objectives.ipynb) · [theory notes](03-language-models/03A-training-skeleton-and-word-objectives-theory.md) preserves the former 02C in full: jointly learned embeddings, CBOW and Skip-gram objectives, a shared inherited training loop, and a destination preview of source-conditioned writing.
5. [03B — Counts and Fixed-Window Language Models](03-language-models/03B-counts-and-fixed-window-language-models.ipynb) · [theory notes](03-language-models/03B-counts-and-fixed-window-language-models-theory.md) preserves word order, then shows what a bounded window cannot see.
6. [03C — Fixed Windows to Recurrent Language Models](03-language-models/03C-fixed-windows-to-recurrent-language-models.ipynb) · [theory notes](03-language-models/03C-fixed-windows-to-recurrent-language-models-theory.md) reuses recurrent memory for text, adds an inherited-loop LSTM implementation, and measures joint embedding training and causality.
7. [Recurrent Encoder-Decoder to Attention](03-language-models/03D-recurrent-encoder-decoder-to-attention.ipynb) · [theory notes](03-language-models/03D-recurrent-encoder-decoder-to-attention-theory.md) asks what changes when the writer can look back at the reader's notes for each source position.
8. [Stage 04 — Transformers](../04-transformers/README.md) remains the canonical route for attention, position, and complete Transformer mechanisms.

The [Chapter 03 README](03-language-models/README.md) collects all four lessons, their same-stem
Python companions, and the shared training base. The [optional embedding-history branch](optional/embedding-history/README.md) is adjacent background, not a prerequisite for attention.

![Piano notes become token pieces, integer addresses, and learned rows, then shifted next-note targets, padding, and separate input and loss masks. All are first used in the Chapter 01B lab and formally explained in Chapter 02.](images/03-first-use-concept-map.png)
*where token pieces, IDs, learned rows, padding, shifted targets, and masks first become necessary along the Stage 03 route.*


## Preview, Then Formalize

Chapter 01 uses integer token IDs, padding, shifted targets, and causal checks so recurrence can be
measured in a language-shaped task. Treat those as preview contracts, not as the tokenization
lesson. Chapter 02 then explains where IDs come from and separates the jobs of padding, loss, and
causal masks before reusing the same shapes.

```text
Chapter 01: use the sequence contract to study memory
-> Chapter 02: explain and test the token representation contract
-> Chapter 03: train word objectives, RNN/LSTM predictors, and source-conditioned models
-> Stage 04: replace the recurrent bottleneck with attention and position
```

## Optional branch

- [Embedding History](optional/embedding-history/README.md)

**Previous:** [Stage 02 — PyTorch Foundations](../02-pytorch-foundations/README.md)  
**Current:** Stage 03 — Language-Model Foundations  
**Next:** [Stage 04 — Transformers](../04-transformers/README.md)
