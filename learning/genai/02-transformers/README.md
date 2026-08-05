# Transformer Foundations

This three-notebook series preserves the original `the cat sat on the mat` journey while separating the architectural decisions into manageable chapters.

1. [Attention and Transformer Blocks](01-attention-and-transformer-blocks.ipynb) builds embeddings, position, Q/K/V attention, multi-head attention, feed-forward layers, residuals, normalization, and the reusable Pre-LN block.
2. [Decoder-Only Language Models](02-decoder-only-language-model.ipynb) adds causal next-token training, per-position loss, autoregressive generation, W_V interpretation, the causal accumulation tower, and DistilGPT-2 internals.
3. [Encoder-Decoder and Cross-Attention](03-encoder-decoder-and-cross-attention.ipynb) builds bidirectional source encoding, cross-attention, teacher forcing, free-running decoding, beam-search intuition, and a T5/BART bridge.

Run `setup.ps1` on Windows or `setup.sh` on Linux/macOS. Either script creates or reuses this chapter's `.venv`, installs the adjacent `requirements.txt`, registers the chapter-unique `genai-02-transformers` Jupyter kernel, and assigns it to all three notebooks. Parts 2 and 3 contain compact executable recaps so each notebook can start in a fresh kernel without repeating the exploratory proofs from Part 1.
