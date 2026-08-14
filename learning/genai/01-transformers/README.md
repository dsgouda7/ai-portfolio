# Transformer Foundations and Base LLM Construction

This six-notebook series follows one idea all the way through a Transformer: turn text into token IDs, turn IDs into positioned vectors, let attention exchange information, let feed-forward layers refine each token, turn the final vectors into vocabulary scores, and use prediction error to improve every earlier step.

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

**Prerequisite:** complete [`../../genai-prerequisites/06-tokenization/tokenization-and-embeddings.ipynb`](../../genai-prerequisites/06-tokenization/tokenization-and-embeddings.ipynb), especially Part 4, followed by the [`../../genai-prerequisites/07-pytorch-rnn-bridge/01-pytorch-rnn-bridge.ipynb`](../../genai-prerequisites/07-pytorch-rnn-bridge/01-pytorch-rnn-bridge.ipynb). Together they establish learned lookup rows plus the PyTorch sequence, loss, and generation contracts this series reuses while attention is under inspection.

| Part | Start with the job | Open the mechanism | End with the reason it fits |
|---|---|---|---|
| [1. Attention and Transformer Blocks](01-attention-and-transformer-blocks.ipynb) · [Theory](01-attention-and-transformer-blocks-theory.md) | Give every token useful context | Tokens, embeddings, position, Q/K/V, attention, FFN, logits, loss, and one backward pass | Shared foundation for every Transformer family |
| [2. Decoder-Only Language Models](02-decoder-only-language-model.ipynb) · [Theory](02-decoder-only-language-model-theory.md) | Continue one growing sequence | Causal masking, shifted targets, prediction, generation, and cache intuition | Natural fit for completion, chat, and open-ended generation |
| [3. Encoder-Decoder and Cross-Attention](03-encoder-decoder-and-cross-attention.ipynb) · [Theory](03-encoder-decoder-and-cross-attention-theory.md) | Read one sequence and write another | Bidirectional encoder memory, causal decoder, and cross-attention | Natural fit for translation, summarization, and structured transformation |
| [4. Modern Decoder-Only LLM](04-modern-decoder-only-llm.ipynb) · [Theory](04-modern-decoder-only-llm-theory.md) | Keep next-token prediction but improve the block | RMSNorm, RoPE, grouped-query attention, and SwiGLU | Better training and inference trade-offs at current LLM scale |
| [5. Pretraining Data Pipeline](05-pretraining-data-pipeline.ipynb) · [Theory](05-pretraining-data-pipeline-theory.md) | Decide what token stream the model will practice | Splits, duplicates, tokenizer fitting, document boundaries, packing, and manifests | Makes training evidence trustworthy and reproducible |
| [6. Pretrain a Base Model](06-pretrain-a-base-model.ipynb) · [Theory](06-pretrain-a-base-model-theory.md) | Turn random weights into a next-token predictor | Logits, loss, gradients, optimizer updates, validation, and checkpoints | Connects the complete forward path to learning and artifact lineage |

## Architecture Choice in One View

| Family | What it does | How information moves | Ideal when |
|---|---|---|---|
| Encoder-only | Produces contextual representations of a complete input | Every input token can read every other input token | Classification, tagging, extraction, and retrieval embeddings |
| Decoder-only | Predicts and appends the next token | Each token can read only the prefix to its left | Continuation, chat, code generation, and general autoregressive tasks |
| Encoder-decoder | Reads a source, then generates a separate target | A bidirectional reader builds source memory; a causal writer queries it with cross-attention | Translation, summarization, and source-to-target transformation |

Run `setup.ps1` on Windows or `setup.sh` on Linux/macOS. Either script creates or reuses this chapter's `.venv`, installs the adjacent `requirements.txt`, registers the chapter-unique `genai-01-transformers` Jupyter kernel, and assigns it to all six notebooks. Parts 2 and 3 contain compact executable recaps. Parts 4–6 also start in fresh kernels and exchange only validated artifacts under `artifacts/base-lm/`; run them in order because Part 5 creates the tokenizer and shards that Part 6 consumes.

The default Part 6 profile is a CPU-safe construction lab. It proves the complete mechanism and artifact boundaries; it does not claim that the small checkpoint acquired broad knowledge, reasoning, or assistant behavior.
