# Tokenization and Embeddings

This required PyTorch-first chapter keeps one legal-corpus example across two fresh-kernel notebooks:

1. [06A - Tokenization and Embeddings Theory](06A-tokenization-and-embeddings-theory.ipynb) makes BPE, token addresses, padding, and masking necessary before introducing APIs.
2. [06B - Tokenization and Embeddings PyTorch Lab](06B-tokenization-and-embeddings-pytorch-lab.ipynb) implements the compact BPE loop, inspects GPT-2 `tiktoken`, trains `nn.Embedding`, and proves the masking loss contract.
3. [Tokenization Theory Notes](06A-tokenization-and-embeddings-theory.md) is the concise handwritten companion.

Run each notebook independently. Kernels do not share state.

Run `setup.ps1` on Windows or `setup.sh` on Linux/macOS. The script creates or reuses this chapter's `.venv`, installs `requirements.txt`, registers the unique `pytorch-llms-06-tokenization` kernel, and assigns it to both notebooks. Use `-SkipKernel` or `--skip-kernel` when only the local environment and dependencies are needed.
