# Chapter 03: Language Models - From Learned Rows to Attention

**Contents:** [directory map](CONTENTS.md)

The core question in 03A is **how embedding values learn when their input IDs stay
fixed**. Train one CBOW model on text, then compare its actual embedding rows
before and after learning.

## Before you start

Use [Stage 02](../../02-pytorch-foundations/README.md) for tensors and optimization,
[Chapter 01](../01-sequence-memory/README.md) for recurrent memory, and
[Chapter 02](../02-tokenization-and-embeddings/README.md) for pieces, IDs, and rows.
03A builds its vocabulary and training questions inline; earlier notebook state
is not required.

Run [setup.ps1](setup.ps1) on Windows or [setup.sh](setup.sh) on Linux/macOS.
They create a local `.venv`, install [requirements](requirements.txt), and register
kernel `stage03-language-model-03-models` (Python 3.10+).
Select that kernel, restart, and run notebooks independently.

The TinyStories model needs internet for its first pinned download, then verifies
a local cache. The conceptual explanation and static illustrations remain readable offline.
03B-03D use generated local examples and one CPU thread.

## 03A: one core path, then optional extensions

Start with the space-probe catalogue: object IDs identify records; aligned sensor
profiles let you compare their properties with Earth's. The notebook explains
that scenario and maps it to token IDs and learned rows before introducing CBOW
as a separate training task. Words have no physical sensors; prediction feedback
must teach their profiles. The rich celestial illustration and shared-word-features
picture sit beside the explanation, with the schematic channel table retained.
The analogy reaches semantic relatedness, reusable features, and the separate
roles of context-based prediction and stochastic decoding.

1. Follow the catalogue-to-embedding analogy and one numbered training flow.
2. Load text and build sentence-local NumPy context/answer windows. Reserve
   the next 200 stories with a fixed training vocabulary.
3. Create `nn.Embedding(vocabulary_size, 32)` inside the real CBOW model.
   Allocation and lookup shapes are annotated directly in that model's code.
   A static storyboard and four-frame animation then record one real window:
   IDs select vectors, CBOW averages the vectors, backward fills selected-row
   gradients, and a separate fresh-SGD step updates the copied model.
4. Train the model; print actual before/after embedding features and verify that
   the input IDs stayed fixed. No standalone three-word model is required.
5. Check reserved-story loss and neighbours. Hold one gap's probabilities
   fixed to compare greedy selection with seeded sampling, without confusing
   neighbours with likely answers or independent gap samples with generated text.
6. Expose the trained model's remaining order blindness.

The **Core checkpoint** marks a complete stopping point before 03B.
Notebook subsections use `2.1`, `3.1`, and so on within core sections 1-5;
optional extensions use A-D with subsections such as `A.1` and `D.9`.
The Python companion's cell labels follow those same section numbers.
There is one embedding-learning path, not a separate toy or parameter-registry tour.
The routing animation uses a copy of that actual CBOW and leaves its training state
and random seed untouched. Its exact unused-row result is scoped to fresh SGD,
no weight decay, and untied weights; the text distinguishes this from Adam history.
Running the animation cell rebuilds its PNG/GIF and displays a JavaScript replay;
the static storyboard remains readable without JavaScript.
The original rich illustrations remain beside the conceptual explanation.

Extensions retain fused lookup, visualization, Skip-gram, the recurrent reader/writer,
and the larger three-objective comparison. **Run All and the full Python script
still execute these extensions.** The 8,000-story comparison uses fresh models
and its own whole-story split; its losses are not the core model's evaluation.
Attention and Transformer internals are deferred to their actual chapters.

## Read in order

| Notebook | The next complaint | Handwritten notes | Code-only view |
|---|---|---|---|
| [03A. Training skeleton and word objectives](03A-training-skeleton-and-word-objectives.ipynb) | How do fixed IDs select rows that learn? | [Theory](03A-training-skeleton-and-word-objectives-theory.md) | [Python](03A-training-skeleton-and-word-objectives.py) |
| [03B. Counts and fixed windows](03B-counts-and-fixed-window-language-models.ipynb) | Ordered seats preserve order, but a clue outside the window is still invisible | [Theory](03B-counts-and-fixed-window-language-models-theory.md) | [Python](03B-counts-and-fixed-window-language-models.py) |
| [03C. Fixed windows to recurrent models](03C-fixed-windows-to-recurrent-language-models.ipynb) | A running note can carry the clue, but memory has limits | [Theory](03C-fixed-windows-to-recurrent-language-models-theory.md) | [Python](03C-fixed-windows-to-recurrent-language-models.py) |
| [03D. Recurrent reader/writer to attention](03D-recurrent-encoder-decoder-to-attention.ipynb) | The writer needs access beyond one final summary | [Theory](03D-recurrent-encoder-decoder-to-attention-theory.md) | [Python](03D-recurrent-encoder-decoder-to-attention.py) |

03B-03C deliberately switch to short color-clue sentences; 03D uses digit
reversal. Those controlled tasks expose information access, not competing corpora.

## Run and inspect

The same-stem Python companions contain VS Code `# %%` cells. Run individual
cells to stop at the core checkpoint, or execute the whole script:

```powershell
.\.venv\Scripts\python.exe .\03A-training-skeleton-and-word-objectives.py
```

Set `MPLBACKEND=Agg` for noninteractive plotting. The full cached lesson takes a
few CPU minutes; printed timing and losses are measurements, not quality guarantees.

The [trainer](language_model.py) centralizes repeated optimization.
The [corpus/comparison helper](real_text_embeddings.py) verifies the pinned data
and records the extended comparison. Offline [tests](test_real_text_embeddings.py)
cover windows, objective alignment, the real CBOW's allocation/lookup/training
contract, repeated-ID gradient accumulation, optimizer-history boundaries, measured
storyboard values, reading order, sampling, and reserved-story evaluation without downloading a corpus:

```powershell
.\.venv\Scripts\python.exe -m unittest test_real_text_embeddings -v
```

Rebuild static teaching diagrams locally with
[`generate_sensor_analogy_diagrams.py`](generate_sensor_analogy_diagrams.py).
