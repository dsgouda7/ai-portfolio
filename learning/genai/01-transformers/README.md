# Transformer Foundations and Base LLM Construction

This track follows one idea all the way through a Transformer: turn text into token IDs, turn IDs into vectors, expose the need for position, let attention exchange information, let feed-forward layers refine each token, turn final vectors into vocabulary scores, and use prediction error to improve every learned step.

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

**Prerequisite:** complete the repository's PyTorch and tokenization foundations. Part 1 keeps a compact Transformer-specific recap so the canonical example and gradient contracts are available in a fresh kernel.

| Part | Artifact | Start with the job | End with the evidence |
|---|---|---|---|
| [1. Tokenization and Embeddings](01-tokenization-and-embeddings.ipynb) · [Theory](01-tokenization-and-embeddings-theory.md) | Turn text into stable model inputs | Pieces, IDs, trainable lookup rows, gradient updates, and the ordering failure | Establishes the substrate every Transformer consumes |
| [2. Attention, Position, and RoPE](02-attention-and-position.ipynb) · [Theory](02-attention-and-position-theory.md) | Let token positions retrieve useful context | Minimal attention, permutation failure, additive position, RoPE, Q/K/V, and scaling | Makes contextual routing and relative position visible before block complexity |
| [3. The Complete Transformer Block](03-transformer-block.ipynb) · [Theory](03-transformer-block-theory.md) | Build one reusable shape-preserving unit | Multi-head attention, FFN, normalization, residuals, logits, loss, and backpropagation | Shared foundation for every Transformer family |
| 4A | [Decoder-Only Theory Notebook](04a-decoder-only-language-model-theory.ipynb) · [Handwritten Theory](04-decoder-only-language-model-theory.md) | Make one growing tape future-blind | Causal objective, shifted labels, three mask jobs, serial generation, temperature/top-k, and cache economics |
| 4B | [Decoder-Only PyTorch Lab](04b-decoder-only-language-model-lab.ipynb) | Implement the Part 4A contracts | `Dataset`/`DataLoader`/collation, loss flattening, generation boundaries, and cache parity |
| 5A | [Encoder-Decoder Theory Notebook](05a-encoder-decoder-and-cross-attention-theory.ipynb) · [Handwritten Theory](05-encoder-decoder-and-cross-attention-theory.md) | Read one sequence and write another | Mean-pooling bottleneck, addressable source memory, cross-attention, exposure bias, and cache roles |
| 5B | [Encoder-Decoder PyTorch Lab](05b-encoder-decoder-and-cross-attention-lab.ipynb) | Make source-to-target routing inspectable | Reversal training, rectangular cross-attention, learned routing, and teacher-forced/free-running evaluation |
| [6. Modern Decoder-Only LLM](06-modern-decoder-only-llm.ipynb) · [Theory](06-modern-decoder-only-llm-theory.md) | Keep next-token prediction but improve the block | RMSNorm, RoPE, grouped-query attention, and SwiGLU | Better training and inference trade-offs at current LLM scale |
| [7. Pretraining Data Pipeline](07-pretraining-data-pipeline.ipynb) · [Theory](07-pretraining-data-pipeline-theory.md) | Decide what token stream the model will practice | Splits, duplicates, tokenizer fitting, document boundaries, packing, and manifests | Makes training evidence trustworthy and reproducible |
| 8A | [Pretraining Theory Notebook](08a-pretrain-a-base-model-theory.ipynb) · [Handwritten Theory](08-pretrain-a-base-model-theory.md) | Decide what evidence makes a run trustworthy | Update lifecycle, held-out selection, full pause state, and artifact lineage |
| 8B | [Pretraining PyTorch Lab](08b-pretrain-a-base-model-lab.ipynb) | Turn random weights into a resumable base-model package | Manual training/validation, generation, full checkpoints, resume parity, and reloadable lineage |

## Optional Extensions

These notebooks are revision branches, not prerequisites for the core route:

| Notebook | Purpose | Acceptance evidence |
|---|---|---|
| [9. Training a Base Model with Lightning](09-training-a-base-model-with-lightning.ipynb) | Translate the understood manual lifecycle into Lightning hooks and callbacks | Same-batch shape/loss parity and checkpoint reload-logit parity |
| [10. Tiny Transformer for AI Engineers](10-tiny-transformer-for-ai-engineers.ipynb) | Audit the complete data, model, training, evaluation, and artifact contracts | Hash, shape, selection, package, and resume-manifest checks |

## Architecture Choice in One View

| Family | What it does | How information moves | Ideal when |
|---|---|---|---|
| Encoder-only | Produces contextual representations of a complete input | Every input token can read every other input token | Classification, tagging, extraction, and retrieval embeddings |
| Decoder-only | Predicts and appends the next token | Each token can read only the prefix to its left | Continuation, chat, code generation, and general autoregressive tasks |
| Encoder-decoder | Reads a source, then generates a separate target | A bidirectional reader builds source memory; a causal writer queries it with cross-attention | Translation, summarization, and source-to-target transformation |

Run `setup.ps1` on Windows or `setup.sh` on Linux/macOS. Either script creates or reuses this chapter's `.venv`, installs the adjacent `requirements.txt`, registers the chapter-unique `genai-01-transformers` Jupyter kernel, and assigns it to every notebook in this directory. Every notebook starts in a fresh kernel.

Parts 6–8 exchange only validated artifacts under `artifacts/base-lm/`; run Part 6, Part 7, then Part 8A/8B because Part 6 writes the model contract, Part 7 creates the tokenizer and shards, and Part 8 consumes both. Optional Part 9 is self-contained. Optional Part 10 reads the completed Parts 6–8 package without modifying it.

The default Part 8B profile is a CPU-safe construction lab. It proves the complete mechanism and artifact boundaries; it does not claim that the small checkpoint acquired broad knowledge, reasoning, or assistant behavior.
