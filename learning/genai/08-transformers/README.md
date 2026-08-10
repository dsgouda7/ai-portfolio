# Transformer Foundations

This three-notebook series follows Riverside's manuscript line `aria heard the signal aboard meridian` while separating the architectural decisions into manageable chapters. Synthetic sequence reversal appears only as an instrumented routing diagnostic for cross-attention.

**Prerequisite:** complete [`../../genai-prerequisites/06-tokenization/tokenization-and-embeddings.ipynb`](../../genai-prerequisites/06-tokenization/tokenization-and-embeddings.ipynb), especially Part 4, followed by the [`../../genai-prerequisites/07-pytorch-rnn-bridge/01-pytorch-rnn-bridge.ipynb`](../../genai-prerequisites/07-pytorch-rnn-bridge/01-pytorch-rnn-bridge.ipynb). Together they establish learned lookup rows plus the PyTorch sequence, loss, and generation contracts this series reuses while attention is under inspection.

1. [Attention and Transformer Blocks](01-attention-and-transformer-blocks.ipynb) · [Theory notes](01-attention-and-transformer-blocks-theory.md)
2. [Decoder-Only Language Models](02-decoder-only-language-model.ipynb) · [Theory notes](02-decoder-only-language-model-theory.md)
3. [Encoder-Decoder and Cross-Attention](03-encoder-decoder-and-cross-attention.ipynb) · [Theory notes](03-encoder-decoder-and-cross-attention-theory.md)

Run `setup.ps1` on Windows or `setup.sh` on Linux/macOS. Either script creates or reuses this chapter's `.venv`, installs the adjacent `requirements.txt`, registers the chapter-unique `genai-08-transformers` Jupyter kernel, and assigns it to all three notebooks. Parts 2 and 3 contain compact executable recaps so each notebook can start in a fresh kernel without repeating the exploratory proofs from Part 1.
