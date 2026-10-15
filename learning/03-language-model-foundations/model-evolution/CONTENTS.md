# Contents: Language-Model Evolution

Follow the failures that lead from counting continuations to a writer that can look back at the input.

## Required route

- [Model Evolution Route Guide](README.md) — Check prerequisites and the three comparison lessons before the handoff to Transformers.
- [Counts and Fixed-Window Language Models](01-counts-and-fixed-window-language-models.ipynb) — Compare counted continuations with a learned fixed-size text window and expose a clue that neither window can see.
- [Counts and Fixed-Window Theory Notes](01-counts-and-fixed-window-language-models-theory.md) — Review why learning reusable features does not remove a fixed context boundary.
- [Fixed Windows to Recurrent Language Models](02-fixed-windows-to-recurrent-language-models.ipynb) — Reuse the same text task to compare a fixed window with a running memory and reconnect to the earlier memory lesson.
- [Fixed Windows to Recurrent Models: Theory Notes](02-fixed-windows-to-recurrent-language-models-theory.md) — Review what a running text memory changes and why it still need not retain every early clue.
- [Recurrent Reader-and-Writer Models to Attention](03-recurrent-encoder-decoder-to-attention.ipynb) — Compare a writer given one learned summary with a writer that can inspect every source position during practice and generation.
- [Recurrent Reader-and-Writer Models to Attention: Theory Notes](03-recurrent-encoder-decoder-to-attention-theory.md) — Review why looking back at source notes differs from remembering one summary and from receiving correct earlier words during practice.

## Optional

No optional lessons are listed in this directory.
