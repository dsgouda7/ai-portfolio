# Contents: Transformers

Build context routing, reusable blocks, and complete model families while preserving honest prediction boundaries.

## Required route

- [Stage 04 Transformers, Part 1: Token Embeddings to Transformers](01-token-embeddings-to-transformers.ipynb) — Carry text pieces and learned lookup rows into the problem of giving a word meaning from its surrounding text.
- [Token Embeddings to Transformers: Handwritten Theory Notes](01-token-embeddings-to-transformers-theory.md) — Review why identifying a word is different from interpreting it in a sentence.
- [Stage 04 Transformers, Part 2: Attention, Position, and RoPE](02-attention-and-position.ipynb) — Let each text position retrieve useful information from other positions, then show why word order must also be represented.
- [Attention, Position, and RoPE: Handwritten Theory Notes](02-attention-and-position-theory.md) — Review how context selection, word order, and learned lookup questions work together.
- [Stage 04 Transformers, Part 3: Build the Complete Transformer Block](03-complete-transformer-block.ipynb) — Combine several views of context with per-position processing and protected learning paths to build a reusable block.
- [The Complete Transformer Block: Handwritten Theory Notes](03-complete-transformer-block-theory.md) — Review why selecting context alone is not a complete trainable processing block.
- [Stage 04 Transformers, Part 4A: Why a Decoder-Only Model Works](04a-decoder-only-language-model-theory.ipynb) — Hide future answers during next-word practice and separate prediction rules from the rules for choosing a generated word.
- [Decoder-Only Language Models: Handwritten Theory Notes](04-decoder-only-language-model-theory.md) — Review how one growing text sequence becomes honest training examples and step-by-step generation.
- [Stage 04 Transformers, Part 4B: Decoder-Only PyTorch Lab](04b-decoder-only-language-model-lab.ipynb) — Implement unequal text batches, next-word training, and generation that reuses earlier calculations without changing its answers.
- [Stage 04 Transformers, Part 5A: Why Cross-Attention Exists](05a-encoder-decoder-and-cross-attention-theory.ipynb) — Give a writer access to separate source positions instead of forcing the input through one pooled summary.
- [Encoder-Decoder Transformers and Cross-Attention: Handwritten Theory Notes](05-encoder-decoder-and-cross-attention-theory.md) — Review the difference between reading the source, hiding future target words, and looking back while writing.
- [Stage 04 Transformers, Part 5B: Encoder-Decoder PyTorch Lab](05b-encoder-decoder-and-cross-attention-lab.ipynb) — Train a reader-and-writer model to reverse sequences and compare practice using correct earlier words with generation using its own guesses.
- [Stage 04 Transformers, Part 6: Inside a Modern Decoder-Only LLM](06-modern-decoder-only-llm.ipynb) — Inspect modern block changes that improve scaling, context handling, and memory use while preserving next-word prediction.
- [Modern Decoder-Only LLMs: Handwritten Theory Notes](06-modern-decoder-only-llm-theory.md) — Review which block refinements change computation and which prediction contract stays the same.

## Optional

- [Focused Transformer Route](focused/README.md) — (optional) Study Transformer-specific routing and block families after checking that the language-model prerequisites are secure.
