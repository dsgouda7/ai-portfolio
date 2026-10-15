# Stage 04: Transformers

**Previous:** [Stage 03 · Language-Model Foundations](../03-language-model-foundations/README.md)

**Current:** Stage 04 · Transformers

**Next:** [Stage 05 · LLM Training](../05-llm-training/README.md)

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

## Entry Check: Full Route or Shortcut

Before Part 1, choose the route that matches your current memory:

- **Full route:** complete [Stage 02 · PyTorch Foundations](../02-pytorch-foundations/README.md), including the
  [tokenization and embedding bridge](../03-language-model-foundations/README.md).
- **Shortcut route:** run the
  [PyTorch Training Skeleton refresher](../02-pytorch-foundations/quick-refresher/pytorch-training-skeleton.ipynb).
  If its readiness check passes, continue through
  [Stage 03](../03-language-model-foundations/README.md) and return here.

Part 1 keeps a compact Transformer-specific recap so the canonical example and gradient contracts
are available in a fresh kernel. It does not reteach `nn.Module`, `DataLoader`, loss, backward, or
optimizer responsibilities; the refresher is the quick recovery point for those mechanics.

| Part | Artifact | Start with the job | End with the evidence |
|---|---|---|---|
| [1. Token Embeddings to Transformers](01-token-embeddings-to-transformers.ipynb) · [Theory](01-token-embeddings-to-transformers-theory.md) | Carry token IDs and learned vectors into contextual modeling | Pieces, IDs, trainable lookup rows, gradient updates, and the ordering failure | Establishes the substrate every Transformer consumes |
| [2. Attention, Position, and RoPE](02-attention-and-position.ipynb) · [Theory](02-attention-and-position-theory.md) | Let token positions retrieve useful context | Minimal attention, permutation failure, additive position, RoPE, Q/K/V, and scaling | Makes contextual routing and relative position visible before block complexity |
| [3. The Complete Transformer Block](03-complete-transformer-block.ipynb) · [Theory](03-complete-transformer-block-theory.md) | Build one reusable shape-preserving unit | Multi-head attention, FFN, normalization, residuals, logits, loss, and backpropagation | Shared foundation for every Transformer family |
| 4A | [Decoder-Only Theory Notebook](04a-decoder-only-language-model-theory.ipynb) · [Handwritten Theory](04-decoder-only-language-model-theory.md) | Make one growing tape future-blind | Causal objective, shifted labels, three mask jobs, serial generation, temperature/top-k, and cache economics |
| 4B | [Decoder-Only PyTorch Lab](04b-decoder-only-language-model-lab.ipynb) | Implement the Part 4A contracts | `Dataset`/`DataLoader`/collation, loss flattening, generation boundaries, and cache parity |
| 5A | [Encoder-Decoder Theory Notebook](05a-encoder-decoder-and-cross-attention-theory.ipynb) · [Handwritten Theory](05-encoder-decoder-and-cross-attention-theory.md) | Read one sequence and write another | Mean-pooling bottleneck, addressable source memory, cross-attention, exposure bias, and cache roles |
| 5B | [Encoder-Decoder PyTorch Lab](05b-encoder-decoder-and-cross-attention-lab.ipynb) | Make source-to-target routing inspectable | Reversal training, rectangular cross-attention, learned routing, and teacher-forced/free-running evaluation |
| [6. Modern Decoder-Only LLM](06-modern-decoder-only-llm.ipynb) · [Theory](06-modern-decoder-only-llm-theory.md) | Keep next-token prediction but improve the block | RMSNorm, RoPE, grouped-query attention, and SwiGLU | Better training and inference trade-offs at current LLM scale |

## Architecture Choice in One View

| Family | What it does | How information moves | Ideal when |
|---|---|---|---|
| Encoder-only | Produces contextual representations of a complete input | Every input token can read every other input token | Classification, tagging, extraction, and retrieval embeddings |
| Decoder-only | Predicts and appends the next token | Each token can read only the prefix to its left | Continuation, chat, code generation, and general autoregressive tasks |
| Encoder-decoder | Reads a source, then generates a separate target | A bidirectional reader builds source memory; a causal writer queries it with cross-attention | Translation, summarization, and source-to-target transformation |

Run `setup.ps1` on Windows or `setup.sh` on Linux/macOS. Either script creates or reuses this stage's `.venv`, installs the adjacent `requirements.txt`, registers the stage-unique `learning-04-transformers` Jupyter kernel, and assigns it to every notebook in this directory. Every notebook starts in a fresh kernel.

Stage 04 ends with a complete modern decoder-only architecture. Continue to
[Stage 05 · LLM Training](../05-llm-training/README.md) to build its pretraining data contract,
train from random weights, and package resumable evidence.

## One Required Bounded Intervention per Chapter

Each row is one intervention, not a worksheet. Predict first; change only the named variable; run
the named notebook section; measure the stated output; diagnose the mechanism; then write one
sentence saying whether that evidence supports the bounded claim.

| Chapter | Runnable section and one-variable test | Bounded claim to judge |
|---|---|---|
| 1 | [`01-token-embeddings-to-transformers.ipynb`](01-token-embeddings-to-transformers.ipynb), **“Chapter 1 Checkpoint”**: replace only `cat`/`mat` with `dog`/`rug`; measure gradient-bearing rows and the swapped-order mean | lookup rows receive local gradients, while mean pooling still loses order |
| 2 | [`02-attention-and-position.ipynb`](02-attention-and-position.ipynb), **“Chapter 2 Checkpoint”**: change only the minimal-attention query from `cat` to `mat`; measure the strongest non-self route | changing the query changes contextual routing |
| 3 | [`03-complete-transformer-block.ipynb`](03-complete-transformer-block.ipynb), **“Chapter 3 Checkpoint”**: change only `NUM_HEADS` from `4` to `2`; measure head width and score-table shape | head count repartitions a fixed model width |
| 4 | [`04b-decoder-only-language-model-lab.ipynb`](04b-decoder-only-language-model-lab.ipynb), **“Part 4B scorecard”**: change only `top_k` from `3` to `2`; measure the removed candidate | top-k changes the allowed sampling set, not learned weights |
| 5 | [`05b-encoder-decoder-and-cross-attention-lab.ipynb`](05b-encoder-decoder-and-cross-attention-lab.ipynb), **“Part 5B scorecard”**: change only `SOURCE_LENGTH` from `4` to `5`; measure cross-attention shape | source length changes the addressable memory axis |
| 6 | [`06-modern-decoder-only-llm.ipynb`](06-modern-decoder-only-llm.ipynb), **GQA “Your turn”**: change only `trial_kv_heads` among `1, 2, 4, 8`; measure cache elements | fewer KV heads reduce cache state while query-head count stays fixed |
