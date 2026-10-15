# Chapter 02: Tokenization and Embeddings

This required chapter keeps one legal-corpus example across three fresh-kernel notebooks.

**Prerequisite route:** complete Stage 02, or use the
[PyTorch Training Skeleton quick refresher](../../02-pytorch-foundations/quick-refresher/pytorch-training-skeleton.ipynb)
when its closing readiness check passes.

## Revisit the Chapter 01 Preview

Chapter 01 deliberately used several language-shaped tensors before explaining their origin. This
chapter now owns those contracts:

| Preview from sequence memory | What this chapter adds |
|---|---|
| Token IDs already existed | BPE creates pieces; a vocabulary assigns addresses |
| `<pad>` appeared in a batch | Padding is bookkeeping and ignored targets need an explicit loss contract |
| Inputs and targets were shifted | Each logit position must align with the token it predicts |
| Earlier outputs ignored a future edit | Causality, padding visibility, and loss participation are separate questions |

1. [02A — Tokenization and Embeddings Theory](02A-tokenization-and-embeddings-theory.ipynb) makes BPE, token addresses, padding, and masking necessary before introducing APIs.
2. [02B — Tokenization and Embeddings PyTorch Lab](02B-tokenization-and-embeddings-pytorch-lab.ipynb) implements the compact BPE loop, inspects GPT-2 `tiktoken`, trains `nn.Embedding`, and proves the masking loss contract.
3. [02C — One Training Skeleton, Three Language Objectives](02C-context-windows-to-static-embeddings.ipynb) · [Theory notes](02C-context-windows-to-static-embeddings-theory.md) keeps one `DataLoader` and fit contract visible while CBOW, Skip-gram, and a tiny encoder-decoder change only the prediction job and target alignment.
4. [Tokenization Theory Notes](02A-tokenization-and-embeddings-theory.md) is the concise handwritten companion for 02A.

Run `setup.ps1` on Windows or `setup.sh` on Linux/macOS. The script creates this chapter's
`.venv`, installs `requirements.txt`, and assigns the
`stage03-language-model-02-tokenization` kernel.

Continue to [Stage 04: Transformers](../../04-transformers/README.md).

## Required Bounded Intervention

Run **“Your turn — change the merge budget”** in
[`02B-tokenization-and-embeddings-pytorch-lab.ipynb`](02B-tokenization-and-embeddings-pytorch-lab.ipynb).
Predict the target piece count, change only `merge_budget`, measure the resulting pieces, diagnose
which learned merge became available or unavailable, and explain whether the evidence supports the
claim that adding already-learned merges cannot increase this word's piece count.
