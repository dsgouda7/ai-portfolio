# Tokenization Migration Note

The Keras-first notebook in this directory is preserved for its original legal-corpus narrative and visual evidence. It is no longer the required executable implementation.

Use the PyTorch-first split:

1. [06A - Tokenization and Embeddings Theory](../../pytorch-for-llms/06-tokenization-and-embeddings/06A-tokenization-and-embeddings-theory.ipynb) owns the BPE, token-address, static/contextual, padding, and masking mental models.
2. [06B - Tokenization and Embeddings PyTorch Lab](../../pytorch-for-llms/06-tokenization-and-embeddings/06B-tokenization-and-embeddings-pytorch-lab.ipynb) owns `tiktoken`, `torch.nn.Embedding`, padded batches, PyTorch masks, and `ignore_index` loss.
3. [Optional Embedding History](../../pytorch-for-llms/optional/embedding-history/01-from-counts-to-static-geometry.ipynb) owns Word2Vec, GloVe, analogies, and the static/contextual boundary.

Sentiment classification and semantic search are not part of the optional history branch. Semantic search belongs in RAG, where retrieval has a concrete job.
