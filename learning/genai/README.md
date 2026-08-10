# GenAI Learning Arc

This track builds sequence-modeling and generative-AI fundamentals from first principles.
Each chapter is a concept-building notebook or notebook series. Every numbered chapter owns
its `requirements.txt`, local `.venv`, setup scripts, and Jupyter kernel so dependencies stay
isolated and notebook kernel selection is reproducible.

Applied mini-projects that build on these foundations (conversation analysis,
conversational AI, image captioning, translation, voice assistant) now live under
[`/projects`](../../projects/README.md) as standalone apps, each with its own
`requirements.txt` and install script.

See [authoring-guide.md](authoring-guide.md) for notebook conventions, cell-tagging rules,
and how to add new content to this track.

The shared [Riverside House fiction corpus](content/README.md) lives at `content/` so every
chapter can reuse one canonical manuscript world. Chapter directories should reference this
root rather than own duplicate manuscript trees.

> **Notebooks are PyTorch-only.** Every notebook in this track is built with PyTorch +
> HuggingFace (LoRA/PEFT, DPO/TRL for fine-tuning) — the actual tools used in production.
> Earlier revisions of this track also shipped parallel TF/Keras notebooks; those were
> removed since testing and development here is PyTorch-based. Each notebook's code cells
> that use a PyTorch API are annotated with a note on what the call does and its Keras
> equivalent, so a Keras-background reader can still follow along without a separate
> Keras notebook.

---

## Contents

| # | Directory | Topic | What you build | What you can do when done | Prerequisites |
|---|-----------|-------|----------------|--------------------------|---------------|
| 2 | `02-transformers/` | Transformer foundations in three parts | Attention and reusable blocks; decoder-only LM training; encoder-decoder cross-attention | Explain and modify each architecture without carrying one 150-cell notebook in working memory | [Prerequisite 06](../genai-prerequisites/06-tokenization/tokenization-and-embeddings.ipynb), [Prerequisite 07](../genai-prerequisites/07-pytorch-rnn-bridge/01-pytorch-rnn-bridge.ipynb) |
| 3 | `03-llm-finetuning/` | LLM adaptation | CPT, SFT, DPO, full tuning, freezing, LoRA, QLoRA, and evidence-based model selection | Choose what behavior to teach, where to store the update, and what evidence supports release | `02-transformers/` |
| 4 | `04-rag/` | Retrieval-augmented generation | Hybrid retrieval, reranking, boundary checks, and a RAG evaluation harness | Retrieve current authorized evidence and diagnose retriever versus generator failure | `03-llm-finetuning/` |
| 5 | `05-llm-evaluation/` | LLM evaluation in depth | Automated metrics, LLM-as-judge, human evaluation, safety, hallucination detection, and calibration | Build a regression-aware evaluation pipeline and reason about evaluator uncertainty | `03-llm-finetuning/`, `04-rag/` |
| 6 | `06-llm-gateway/` | LLM request control plane | Provider normalization, routing, rate limiting, fallback, caching, and cost controls | Operate multiple model providers behind one observable application contract | `03-llm-finetuning/`, `04-rag/` |

---

## Transformer Foundations Route

The prerequisite track is separately numbered 00–07; after prerequisite 07, the GenAI track begins at chapter 02. Before entering this series, complete the ordered [prerequisite sequence](../genai-prerequisites/README.md), including tokenization and the PyTorch RNN bridge.

1. [Attention and Transformer Blocks](02-transformers/01-attention-and-transformer-blocks.ipynb) · [Theory notes](02-transformers/01-attention-and-transformer-blocks-theory.md)
2. [Decoder-Only Language Model](02-transformers/02-decoder-only-language-model.ipynb) · [Theory notes](02-transformers/02-decoder-only-language-model-theory.md)
3. [Encoder-Decoder and Cross-Attention](02-transformers/03-encoder-decoder-and-cross-attention.ipynb) · [Theory notes](02-transformers/03-encoder-decoder-and-cross-attention-theory.md)

The first notebook remains the mechanistic gold standard. The three notebooks use Riverside's
*The Weight of Distant Light* as their shared narrative world while making each architectural
decision easier to learn and revisit.

---

## Learning path summary

```
../genai-prerequisites (00–07) -> 02-transformers -> 03-llm-finetuning -> 04-rag -> 05-llm-evaluation -> 06-llm-gateway
```
