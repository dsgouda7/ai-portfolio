# GenAI Learning Arc

This track builds sequence-modeling and generative-AI fundamentals from first principles.
Each chapter is a concept-building notebook or notebook series. Foundation chapters keep
their own `requirements.txt` and setup scripts. The applied LLM chapters share the
`_llm-shared/` environment so fine-tuning, RAG, and gateway notebooks can live in clear
topic directories without duplicating one broad dependency stack.

Applied mini-projects that build on these foundations (conversation analysis,
conversational AI, image captioning, translation, voice assistant) now live under
[`/projects`](../../projects/README.md) as standalone apps, each with its own
`requirements.txt` and install script.

See [authoring-guide.md](authoring-guide.md) for notebook conventions, cell-tagging rules,
and how to add new content to this track.

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
| 0 | `00-pytorch-fundamentals/` | Keras to PyTorch foundations | Antarctic Field Guide, a CC0 Palmer Penguins species classifier | Translate Keras training habits into explicit PyTorch tensor, model, autograd, and inference contracts | `genai-prerequisites/` |
| 1 | `01-rnns/` | Recurrent Neural Networks | PyTorch next-token music model with LSTM state and autoregressive generation | Translate known RNN/LSTM concepts into PyTorch; explain the recurrent path that motivates attention | `00-pytorch-fundamentals/`, `genai-prerequisites/04-rnn-sequence-modeling/` |
| 2 | `02-transformers/` | Transformer foundations in three parts | Attention and reusable blocks; decoder-only LM training; encoder-decoder cross-attention | Explain and modify each architecture without carrying one 150-cell notebook in working memory | `01-rnns/01-pytorch-rnn-bridge.ipynb` |
| 4 | `04-llm-finetuning/` | LLM adaptation | CPT, SFT, DPO, full tuning, freezing, LoRA, QLoRA, and evidence-based model selection | Choose what behavior to teach, where to store the update, and what evidence supports release | `02-transformers/` |
| 5 | `05-rag/` | Retrieval-augmented generation | Hybrid retrieval, reranking, boundary checks, and a RAG evaluation harness | Retrieve current authorized evidence and diagnose retriever versus generator failure | `04-llm-finetuning/` |
| 5E | `05-llm-evaluation/` | LLM evaluation in depth | Automated metrics, LLM-as-judge, human evaluation, safety, hallucination detection, and calibration | Build a regression-aware evaluation pipeline and reason about evaluator uncertainty | `04-llm-finetuning/`, `05-rag/` |
| 6 | `06-llm-gateway/` | LLM request control plane | Provider normalization, routing, rate limiting, fallback, caching, and cost controls | Operate multiple model providers behind one observable application contract | `04-llm-finetuning/`, `05-rag/` |

---

## Transformer Foundations Route

1. [`01-attention-and-transformer-blocks.ipynb`](02-transformers/01-attention-and-transformer-blocks.ipynb)
2. [`02-decoder-only-language-model.ipynb`](02-transformers/02-decoder-only-language-model.ipynb)
3. [`03-encoder-decoder-and-cross-attention.ipynb`](02-transformers/03-encoder-decoder-and-cross-attention.ipynb)

The first notebook remains the mechanistic gold standard. The three notebooks preserve the
original running example and complete content while making each architectural decision easier to
learn and revisit.

---

## Learning path summary

```
genai-prerequisites -> 00-pytorch-fundamentals -> 01-rnns -> 02-transformers -> 04-llm-finetuning -> 05-rag -> 05-llm-evaluation -> 06-llm-gateway
```
