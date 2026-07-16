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

---

## Contents

| # | Directory | Topic | What you build | What you can do when done | Prerequisites |
|---|-----------|-------|----------------|--------------------------|---------------|
| 0 | `00-pytorch-primer/` | Keras -> PyTorch primer | The same CNN MNIST classifier built cell-by-cell in both Keras and PyTorch | Translate Keras models/training loops to PyTorch; avoid the NHWC/NCHW, `zero_grad()`, and train/eval-mode gotchas | Basic Python, one prior Keras model (any) |
| 1 | `01-rnns/` | Recurrent Neural Networks | Character-level RNN in PyTorch and Keras/TF | Implement sequence models from scratch; explain vanishing gradients | Basic Python, NumPy |
| 2 | `02-transformers/` | Transformer architecture | Scaled dot-product attention and full encoder stack from first principles | Read and modify transformer code; explain every component mathematically | `01-rnns/` |
| 3 | `03-encoder-decoder/` | Encoder-Decoder architecture | Seq2seq model with cross-attention for translation | Build and train encoder-decoder models; tune beam search | `02-transformers/` |
| 4 | `04-llm/` | Applied LLM patterns + fine-tuning | Hybrid search pipeline, LLM gateway, RAG evaluation harness, LoRA/PEFT/DPO fine-tuning | Wire together retrieval + generation; evaluate answer quality quantitatively; fine-tune a causal LM on domain data | `03-encoder-decoder/` |

---

## Gold-standard notebook

`02-transformers/transformers.ipynb` (or `transformers-keras.ipynb`) is the reference
implementation for the entire track. It is the most heavily annotated notebook and
demonstrates the authoring conventions all other notebooks should follow.

---

## Learning path summary

```
00-pytorch-primer -> 01-rnns -> 02-transformers -> 03-encoder-decoder -> 04-llm
```

`00-pytorch-primer/` is optional -- skip it if you're already comfortable writing raw PyTorch
training loops (as opposed to only having used Keras's `model.fit()`).
