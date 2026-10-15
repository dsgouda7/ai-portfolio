# Stage 03: Language-Model Foundations

This stage changes the shape of the problem. The PyTorch loop is now familiar; the new complaints
are memory across a sequence and useful representations for discrete tokens.

## Required route

1. [Sequence Memory](01-sequence-memory/README.md)
2. [Tokenization and Embeddings](02-tokenization-and-embeddings/README.md)

## Preview, Then Formalize

Chapter 01 uses integer token IDs, padding, shifted targets, and causal checks so recurrence can be
measured in a language-shaped task. Treat those as preview contracts, not as the tokenization
lesson. Chapter 02 then explains where IDs come from and separates the jobs of padding, loss, and
causal masks before reusing the same shapes.

```text
Chapter 01: use the sequence contract to study memory
-> Chapter 02: explain and test the token representation contract
-> Stage 04: replace the recurrent bottleneck with attention and position
```

## Optional branch

- [Embedding History](optional/embedding-history/README.md)

**Previous:** [Stage 02 — PyTorch Foundations](../02-pytorch-foundations/README.md)  
**Current:** Stage 03 — Language-Model Foundations  
**Next:** [Stage 04 — Transformers](../04-transformers/README.md)
