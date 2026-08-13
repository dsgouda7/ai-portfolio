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
| 08 | `01-transformers/` | Transformer foundations and base LLM construction in six parts | Attention and reusable blocks; decoder-only and encoder-decoder architectures; a modern Llama-style block; pretraining data; a randomly initialized base model | Explain each architecture and trace raw documents through next-token training into a validated base checkpoint | [Prerequisite 06](../genai-prerequisites/06-tokenization/tokenization-and-embeddings.ipynb), [Prerequisite 07](../genai-prerequisites/07-pytorch-rnn-bridge/01-pytorch-rnn-bridge.ipynb) |
| 09 | `02-llm-finetuning/` | LLM adaptation | CPT, SFT, DPO, full tuning, freezing, LoRA, QLoRA, and evidence-based model selection | Choose what behavior to teach, where to store the update, and what evidence supports release | `01-transformers/` |
| 10 | `03-rag/` | Retrieval-augmented generation | Hybrid retrieval, reranking, boundary checks, and a RAG evaluation harness | Retrieve current authorized evidence and diagnose retriever versus generator failure | `02-llm-finetuning/` |
| 11 | `04-llm-evaluation/` | LLM evaluation in depth | Automated metrics, LLM-as-judge, human evaluation, safety, hallucination detection, and calibration | Build a regression-aware evaluation pipeline and reason about evaluator uncertainty | `02-llm-finetuning/`, `03-rag/` |
| 12 | `05-llm-gateway/` | LLM request control plane | Provider normalization, routing, rate limiting, fallback, caching, and cost controls | Operate multiple model providers behind one observable application contract | `02-llm-finetuning/`, `03-rag/` |

---

## Transformer Foundations Route

The curriculum is continuous: prerequisites are chapters 00–07, and the GenAI track continues with chapters 08–12. Before entering chapter 08, complete the ordered [prerequisite sequence](../genai-prerequisites/README.md), including tokenization and the PyTorch RNN bridge.

1. [Attention and Transformer Blocks](01-transformers/01-attention-and-transformer-blocks.ipynb) · [Theory notes](01-transformers/01-attention-and-transformer-blocks-theory.md)
2. [Decoder-Only Language Model](01-transformers/02-decoder-only-language-model.ipynb) · [Theory notes](01-transformers/02-decoder-only-language-model-theory.md)
3. [Encoder-Decoder and Cross-Attention](01-transformers/03-encoder-decoder-and-cross-attention.ipynb) · [Theory notes](01-transformers/03-encoder-decoder-and-cross-attention-theory.md)
4. [Modern Decoder-Only LLM](01-transformers/04-modern-decoder-only-llm.ipynb) · [Theory notes](01-transformers/04-modern-decoder-only-llm-theory.md)
5. [Pretraining Data Pipeline](01-transformers/05-pretraining-data-pipeline.ipynb) · [Theory notes](01-transformers/05-pretraining-data-pipeline-theory.md)
6. [Pretrain a Base Model](01-transformers/06-pretrain-a-base-model.ipynb) · [Theory notes](01-transformers/06-pretrain-a-base-model-theory.md)

The first notebook remains the mechanistic gold standard. The first three notebooks use Riverside's
*The Weight of Distant Light* as their shared narrative world while making each architectural
decision easier to learn and revisit. The final three notebooks continue that world into a modern
decoder, a reproducible pretraining stream, and a saved base-model checkpoint.

---

## Learning path summary

```
../genai-prerequisites (00–07) -> 01-transformers -> 02-llm-finetuning -> 03-rag -> 04-llm-evaluation -> 05-llm-gateway
```
