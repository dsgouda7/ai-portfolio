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

> **Notebooks are PyTorch-first.** Complete [PyTorch for LLMs](../pytorch-for-llms/README.md)
> before this track. Keras translation remains optional reference material rather than a parallel
> implementation path.

---

## Contents

| Directory | Topic | What you build | Prerequisites |
|-----------|-------|----------------|---------------|
| [01-transformers/](01-transformers/README.md) | Transformer foundations and base-model construction | Mechanisms, decoder and encoder-decoder labs, modern blocks, verified data, and a restorable base checkpoint | [PyTorch for LLMs](../pytorch-for-llms/README.md) |
| [02-llm-finetuning/](02-llm-finetuning/README.md) | LLM adaptation | CPT, SFT, DPO, full tuning, freezing, LoRA, QLoRA, comparison, and staged GPU practice | `01-transformers/` |
| [03-rag/](03-rag/README.md) | Retrieval-augmented generation | Hybrid retrieval, reranking, authorization, failure localization, and oracle-context diagnosis | `02-llm-finetuning/` |
| [04-llm-evaluation/](04-llm-evaluation/README.md) | LLM evaluation | Metrics, judge/safety controls, hallucination localization, and calibration/abstention | `02-llm-finetuning/`, `03-rag/` |
| [05-llm-gateway/](05-llm-gateway/README.md) | LLM request control plane | Normalization, routing, limits, budgets, fallback, caching, and telemetry | `02-llm-finetuning/`, `03-rag/` |

---

## Transformer Foundations Route

Before entering Transformers, complete the [foundation prerequisites](../genai-prerequisites/README.md)
and [PyTorch for LLMs](../pytorch-for-llms/README.md).

1. [Tokenization and Embeddings](01-transformers/01-tokenization-and-embeddings.ipynb) · [Theory notes](01-transformers/01-tokenization-and-embeddings-theory.md)
2. [Attention, Position, and RoPE](01-transformers/02-attention-and-position.ipynb) · [Theory notes](01-transformers/02-attention-and-position-theory.md)
3. [The Complete Transformer Block](01-transformers/03-transformer-block.ipynb) · [Theory notes](01-transformers/03-transformer-block-theory.md)
4. [Decoder-Only Theory](01-transformers/04a-decoder-only-language-model-theory.ipynb) · [PyTorch lab](01-transformers/04b-decoder-only-language-model-lab.ipynb) · [Handwritten notes](01-transformers/04-decoder-only-language-model-theory.md)
5. [Encoder-Decoder Theory](01-transformers/05a-encoder-decoder-and-cross-attention-theory.ipynb) · [PyTorch lab](01-transformers/05b-encoder-decoder-and-cross-attention-lab.ipynb) · [Handwritten notes](01-transformers/05-encoder-decoder-and-cross-attention-theory.md)
6. [Modern Decoder-Only LLM](01-transformers/06-modern-decoder-only-llm.ipynb) · [Theory notes](01-transformers/06-modern-decoder-only-llm-theory.md)
7. [Pretraining Data Pipeline](01-transformers/07-pretraining-data-pipeline.ipynb) · [Theory notes](01-transformers/07-pretraining-data-pipeline-theory.md)
8. [Pretraining Theory](01-transformers/08a-pretrain-a-base-model-theory.ipynb) · [Manual PyTorch lab](01-transformers/08b-pretrain-a-base-model-lab.ipynb) · [Handwritten notes](01-transformers/08-pretrain-a-base-model-theory.md)

Optional continuations:

- [Train the same contract with Lightning](01-transformers/09-training-a-base-model-with-lightning.ipynb)
- [Audit a tiny Transformer as an AI engineer](01-transformers/10-tiny-transformer-for-ai-engineers.ipynb)

The first three notebooks form the mechanistic foundation and use `the cat sat on the mat` so every information movement stays inspectable. Parts 4–6 compare architecture families and modernize the decoder block. Parts 7–8 use real local corpus artifacts to build a reproducible pretraining stream and saved base-model checkpoint.

---

## Learning path summary

```
../genai-prerequisites -> ../pytorch-for-llms -> 01-transformers -> 02-llm-finetuning -> 03-rag -> 04-llm-evaluation -> 05-llm-gateway
```
