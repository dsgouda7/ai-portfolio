# Tokenization Migration Note

The Keras-first notebook in this directory is preserved for its original legal-corpus narrative and visual evidence. It is no longer the required executable implementation.

Use the PyTorch-first split:

1. [Tokenization and Embeddings](../../../learning/03-language-model-foundations/02-tokenization-and-embeddings/) owns the BPE, token-address, static/contextual, padding, masking, `tiktoken`, `torch.nn.Embedding`, padded batches, PyTorch masks, and `ignore_index` loss.
2. [Optional Embedding History](../../../learning/03-language-model-foundations/optional/embedding-history/) owns Word2Vec, GloVe, analogies, and the static/contextual boundary.

Sentiment classification and semantic search are not part of the optional history branch. Semantic search belongs in RAG, where retrieval has a concrete job.
