# GenAI Learning Arc

This track builds sequence-modeling and generative-AI fundamentals from first principles.
Each chapter is a concept-building notebook (or set of notebooks) with its own
`requirements.txt` and `setup.ps1` / `setup.sh` that creates a local `.venv` and
registers a Jupyter kernel for that chapter's notebooks. Run the setup script once per
chapter before opening its notebook(s); the kernelspec is already wired to the
matching kernel name, so VS Code should pick it automatically.

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
| 2 | `02-transformers/` | Transformer architecture | Scaled dot-product attention and full encoder stack from first principles | Read and modify transformer code; explain every component mathematically | `01-rnns/01-pytorch-rnn-bridge.ipynb` |
| 3 | `03-encoder-decoder/` | Encoder-Decoder architecture | Seq2seq model with cross-attention for translation | Build and train encoder-decoder models; tune beam search | `02-transformers/` |
| 4 | `04-llm/` | Applied LLM patterns + fine-tuning | Hybrid search pipeline, LLM gateway, RAG evaluation harness, LoRA/PEFT/DPO fine-tuning | Wire together retrieval + generation; evaluate answer quality quantitatively; fine-tune a causal LM on domain data | `03-encoder-decoder/` |
| 5 | `05-llm-evaluation/` | LLM evaluation in depth | Automated metrics (BLEU/ROUGE/BERTScore), LLM-as-judge (G-Eval, pairwise), human evaluation, safety eval, hallucination detection (SelfCheckGPT, NLI, entity-gap), and model calibration (ECE, temperature scaling, selective prediction) | Measure any LLM's quality rigorously; detect hallucination at inference time; publish calibrated confidence scores; build a regression-aware eval pipeline | `04-llm/` |

---

## Gold-standard notebook

`02-transformers/transformers.ipynb` is the reference implementation for the entire track. It is the most heavily annotated notebook and
demonstrates the authoring conventions all other notebooks should follow.

---

## Learning path summary

```
genai-prerequisites -> 00-pytorch-fundamentals -> 01-rnns -> 02-transformers -> 03-encoder-decoder -> 04-llm -> 05-llm-evaluation
```
