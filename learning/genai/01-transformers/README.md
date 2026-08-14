# Transformer Foundations and Base LLM Construction

This eight-notebook series follows one idea all the way through a Transformer: turn text into token IDs, turn IDs into vectors, expose the need for position, let attention exchange information, let feed-forward layers refine each token, turn final vectors into vocabulary scores, and use prediction error to improve every learned step.

The first chapters use small sentences such as `the cat sat on the mat` because the mechanism should be understandable without a fictional business wrapper. Later chapters use the repository's local text corpus only when real data artifacts are necessary to build and train a small base model.

## The Learning Spine

Keep this path in view throughout the track:

```text
text
-> tokens and token IDs
-> token embeddings + position
-> Q/K/V attention: gather useful context
-> feed-forward network: refine each token privately
-> vocabulary logits and probabilities
-> next-token prediction and loss
-> backpropagation updates embeddings, attention, FFNs, and the output head
```

Every chapter zooms in on part of this path, then reconnects it to the whole. Equations appear only after the corresponding information movement is visible in words, shapes, or a measured example.

**Prerequisite:** complete the [PyTorch fundamentals](../../genai-prerequisites/03-pytorch-fundamentals/01-keras-to-pytorch-antarctic-field-guide.ipynb). The broader [tokenization prerequisite](../../genai-prerequisites/06-tokenization/tokenization-and-embeddings.ipynb) remains useful, but Part 1 now rebuilds the Transformer-specific tokenizer, embedding, and gradient contracts directly.

| Part | Start with the job | Open the mechanism | End with the reason it fits |
|---|---|---|---|
| [1. Tokenization and Embeddings](01-tokenization-and-embeddings.ipynb) · [Theory](01-tokenization-and-embeddings-theory.md) | Turn text into stable model inputs | Pieces, IDs, trainable lookup rows, gradient updates, and the ordering failure | Establishes the substrate every Transformer consumes |
| [2. Attention, Position, and RoPE](02-attention-and-position.ipynb) · [Theory](02-attention-and-position-theory.md) | Let token positions retrieve useful context | Minimal attention, permutation failure, additive position, RoPE, Q/K/V, and scaling | Makes contextual routing and relative position visible before block complexity |
| [3. The Complete Transformer Block](03-transformer-block.ipynb) · [Theory](03-transformer-block-theory.md) | Build one reusable shape-preserving unit | Multi-head attention, FFN, normalization, residuals, logits, loss, and backpropagation | Shared foundation for every Transformer family |
| [4. Decoder-Only Language Models](04-decoder-only-language-model.ipynb) · [Theory](04-decoder-only-language-model-theory.md) | Continue one growing sequence | Causal masking, shifted targets, prediction, generation, and cache intuition | Natural fit for completion, chat, and open-ended generation |
| [5. Encoder-Decoder and Cross-Attention](05-encoder-decoder-and-cross-attention.ipynb) · [Theory](05-encoder-decoder-and-cross-attention-theory.md) | Read one sequence and write another | Bidirectional encoder memory, causal decoder, and cross-attention | Natural fit for translation, summarization, and structured transformation |
| [6. Modern Decoder-Only LLM](06-modern-decoder-only-llm.ipynb) · [Theory](06-modern-decoder-only-llm-theory.md) | Keep next-token prediction but improve the block | RMSNorm, RoPE, grouped-query attention, and SwiGLU | Better training and inference trade-offs at current LLM scale |
| [7. Pretraining Data Pipeline](07-pretraining-data-pipeline.ipynb) · [Theory](07-pretraining-data-pipeline-theory.md) | Decide what token stream the model will practice | Splits, duplicates, tokenizer fitting, document boundaries, packing, and manifests | Makes training evidence trustworthy and reproducible |
| [8. Pretrain a Base Model](08-pretrain-a-base-model.ipynb) · [Theory](08-pretrain-a-base-model-theory.md) | Turn random weights into a next-token predictor | Logits, loss, gradients, optimizer updates, validation, and checkpoints | Connects the complete forward path to learning and artifact lineage |

## Architecture Choice in One View

| Family | What it does | How information moves | Ideal when |
|---|---|---|---|
| Encoder-only | Produces contextual representations of a complete input | Every input token can read every other input token | Classification, tagging, extraction, and retrieval embeddings |
| Decoder-only | Predicts and appends the next token | Each token can read only the prefix to its left | Continuation, chat, code generation, and general autoregressive tasks |
| Encoder-decoder | Reads a source, then generates a separate target | A bidirectional reader builds source memory; a causal writer queries it with cross-attention | Translation, summarization, and source-to-target transformation |

Run `setup.ps1` on Windows or `setup.sh` on Linux/macOS. Either script creates or reuses this chapter's `.venv`, installs the adjacent `requirements.txt`, registers the chapter-unique `genai-01-transformers` Jupyter kernel, and assigns it to all eight notebooks. Every part starts in a fresh kernel. Parts 6–8 exchange only validated artifacts under `artifacts/base-lm/`; run those three in order because Part 6 writes the model contract, Part 7 creates the tokenizer and shards, and Part 8 consumes both.

The default Part 8 profile is a CPU-safe construction lab. It proves the complete mechanism and artifact boundaries; it does not claim that the small checkpoint acquired broad knowledge, reasoning, or assistant behavior.
