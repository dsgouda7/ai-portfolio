# Transformer Foundations and Base LLM Construction

This six-notebook series follows Riverside's manuscript line `aria heard the signal aboard meridian` from attention mechanics through a small pretrained base model. Synthetic sequence reversal appears only as an instrumented routing diagnostic for cross-attention; the final three notebooks use the committed Riverside corpus to connect a modern decoder, pretraining data, and random-weight training end to end.

**Prerequisite:** complete [`../../genai-prerequisites/06-tokenization/tokenization-and-embeddings.ipynb`](../../genai-prerequisites/06-tokenization/tokenization-and-embeddings.ipynb), especially Part 4, followed by the [`../../genai-prerequisites/07-pytorch-rnn-bridge/01-pytorch-rnn-bridge.ipynb`](../../genai-prerequisites/07-pytorch-rnn-bridge/01-pytorch-rnn-bridge.ipynb). Together they establish learned lookup rows plus the PyTorch sequence, loss, and generation contracts this series reuses while attention is under inspection.

1. [Attention and Transformer Blocks](01-attention-and-transformer-blocks.ipynb) · [Theory notes](01-attention-and-transformer-blocks-theory.md)
2. [Decoder-Only Language Models](02-decoder-only-language-model.ipynb) · [Theory notes](02-decoder-only-language-model-theory.md)
3. [Encoder-Decoder and Cross-Attention](03-encoder-decoder-and-cross-attention.ipynb) · [Theory notes](03-encoder-decoder-and-cross-attention-theory.md)
4. [Modern Decoder-Only LLM](04-modern-decoder-only-llm.ipynb) · [Theory notes](04-modern-decoder-only-llm-theory.md)
5. [Pretraining Data Pipeline](05-pretraining-data-pipeline.ipynb) · [Theory notes](05-pretraining-data-pipeline-theory.md)
6. [Pretrain a Base Model](06-pretrain-a-base-model.ipynb) · [Theory notes](06-pretrain-a-base-model-theory.md)

Run `setup.ps1` on Windows or `setup.sh` on Linux/macOS. Either script creates or reuses this chapter's `.venv`, installs the adjacent `requirements.txt`, registers the chapter-unique `genai-01-transformers` Jupyter kernel, and assigns it to all six notebooks. Parts 2 and 3 contain compact executable recaps. Parts 4–6 also start in fresh kernels and exchange only validated artifacts under `artifacts/base-lm/`; run them in order because Part 5 creates the tokenizer and shards that Part 6 consumes.

The default Part 6 profile is a CPU-safe construction lab. It proves the complete mechanism and artifact boundaries; it does not claim that the small checkpoint acquired broad knowledge, reasoning, or assistant behavior.
