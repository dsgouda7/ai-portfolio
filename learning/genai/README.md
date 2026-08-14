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

## Learning Contract

Every mechanism in this track follows the recovered Transformer gold-standard rhythm:

```text
crude attempt -> visible failure -> named complaint -> minimal fix
-> measured payoff -> next complaint -> real-system bridge
```

Notebooks keep one running example long enough for the learner to stop spending attention on new nouns. Essential animations have static storyboards, consequential claims have executable measurements, and `Your turn` drills change one known variable beside the mechanism they exercise. Theory companions preserve the same discovery chain in concise prose that can be copied directly into handwritten notes.

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
| 08 | `01-transformers/` | Transformer foundations and base LLM construction in eight parts | Tokenization and embeddings; attention and position; reusable blocks; decoder-only and encoder-decoder architectures; a modern Llama-style block; pretraining data; a randomly initialized base model | Trace text through every forward stage and backpropagation into a validated base checkpoint | [Prerequisite 03](../genai-prerequisites/03-pytorch-fundamentals/01-keras-to-pytorch-antarctic-field-guide.ipynb) |
| 09 | `02-llm-finetuning/` | LLM adaptation | CPT, SFT, DPO, full tuning, freezing, LoRA, QLoRA, and evidence-based model selection | Choose what behavior to teach, where to store the update, and what evidence supports release | `01-transformers/` |
| 10 | `03-rag/` | Retrieval-augmented generation | Hybrid retrieval, reranking, boundary checks, and a RAG evaluation harness | Retrieve current authorized evidence and diagnose retriever versus generator failure | `02-llm-finetuning/` |
| 11 | `04-llm-evaluation/` | LLM evaluation in depth | Automated metrics, LLM-as-judge, human evaluation, safety, hallucination detection, and calibration | Build a regression-aware evaluation pipeline and reason about evaluator uncertainty | `02-llm-finetuning/`, `03-rag/` |
| 12 | `05-llm-gateway/` | LLM request control plane | Provider normalization, routing, rate limiting, fallback, caching, and cost controls | Operate multiple model providers behind one observable application contract | `02-llm-finetuning/`, `03-rag/` |

---

## Transformer Foundations Route

The curriculum is continuous: prerequisites are chapters 00–07, and the GenAI track continues with chapters 08–12. Before entering chapter 08, complete the ordered [prerequisite sequence](../genai-prerequisites/README.md), including tokenization and the PyTorch RNN bridge.

1. [Tokenization and Embeddings](01-transformers/01-tokenization-and-embeddings.ipynb) · [Theory notes](01-transformers/01-tokenization-and-embeddings-theory.md)
2. [Attention, Position, and RoPE](01-transformers/02-attention-and-position.ipynb) · [Theory notes](01-transformers/02-attention-and-position-theory.md)
3. [The Complete Transformer Block](01-transformers/03-transformer-block.ipynb) · [Theory notes](01-transformers/03-transformer-block-theory.md)
4. [Decoder-Only Language Model](01-transformers/04-decoder-only-language-model.ipynb) · [Theory notes](01-transformers/04-decoder-only-language-model-theory.md)
5. [Encoder-Decoder and Cross-Attention](01-transformers/05-encoder-decoder-and-cross-attention.ipynb) · [Theory notes](01-transformers/05-encoder-decoder-and-cross-attention-theory.md)
6. [Modern Decoder-Only LLM](01-transformers/06-modern-decoder-only-llm.ipynb) · [Theory notes](01-transformers/06-modern-decoder-only-llm-theory.md)
7. [Pretraining Data Pipeline](01-transformers/07-pretraining-data-pipeline.ipynb) · [Theory notes](01-transformers/07-pretraining-data-pipeline-theory.md)
8. [Pretrain a Base Model](01-transformers/08-pretrain-a-base-model.ipynb) · [Theory notes](01-transformers/08-pretrain-a-base-model-theory.md)

The first three notebooks form the mechanistic foundation and use `the cat sat on the mat` so every information movement stays inspectable. Parts 4–6 compare architecture families and modernize the decoder block. Parts 7–8 use real local corpus artifacts to build a reproducible pretraining stream and saved base-model checkpoint.

---

## Learning path summary

```
../genai-prerequisites (00–07) -> 01-transformers -> 02-llm-finetuning -> 03-rag -> 04-llm-evaluation -> 05-llm-gateway
```
