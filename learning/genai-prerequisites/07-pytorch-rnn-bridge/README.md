# PyTorch RNN Bridge

This chapter closes the recurrent sequence with two notebooks:

1. [The PyTorch RNN Bridge](01-pytorch-rnn-bridge.ipynb) translates the established Keras RNN, tokenization, embedding, padding, and masked-loss contracts into PyTorch.
2. [Cinematic Piano Memory](02-cinematic-piano-memory.ipynb) trains transparent vanilla RNN and LSTM cells on an original Dm-Bb-F-C motif, then makes checkpoint learning, long-horizon accuracy, and gradient retention audible.

Complete them in order before continuing to [Transformer Foundations](../../genai/01-transformers/README.md).

Run `setup.ps1` on Windows or `setup.sh` on Linux/macOS. Either script creates or reuses this chapter's `.venv`, installs the adjacent `requirements.txt`, registers the chapter-unique `genai-prereq-07-pytorch-rnn` Jupyter kernel, and assigns it to the notebook.
