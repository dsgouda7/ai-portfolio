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

## 03A: visible CBOW learning, then the section 6 reader/writer

Start with the space-probe catalogue: object IDs identify records; aligned sensor
profiles let you compare their properties with Earth's. The notebook explains
that scenario and maps it to token IDs and learned rows before introducing CBOW
as a separate training task. Words have no physical sensors; prediction feedback
must teach their profiles. The rich celestial illustration and shared-word-features
picture sit beside the explanation, with the schematic channel table retained.
The analogy reaches semantic relatedness, reusable features, and the distinction
between learned word profiles and context-based prediction scores.

1. Follow the catalogue-to-embedding analogy and one numbered training flow.
2. Load text and build sentence-local NumPy context/answer windows. Reserve
   the next 200 stories with a fixed training vocabulary.
3. Create `nn.Embedding(vocabulary_size, 32)` inside the real CBOW model.
   Allocation and lookup shapes are annotated directly in that model's code.
   Section 3.1 separates model setup/shape checks, diagnostic helper definitions,
   and the three familiar `cat_contexts` questions into distinct code blocks.
   These save a before-training baseline; **3.3 trains that same CBOW now**.
   Its epoch-0-to-10 table reports practice/held-out loss, cat probability/rank,
   and actual cat-row movement, followed by before/after feature values.
   One cat-probe animation carries actual values through lookup, mean, Linear,
   ReLU, vocabulary logits, and cross-entropy, then carries gradients back through
   each operation. The active layer lights up and enlarges, with large input/output
   readouts; the saved row IDs finally route feedback to the embedding table.
4. Train the model; print actual before/after embedding features and verify that
   the input IDs stayed fixed. No standalone three-word model is required.
5. Inspect actual word neighbours and compare the familiar questions' probabilities
   and ranks with reserved-story loss. Neighbour inspection and prediction
   evaluation answer different questions; decoding strategies belong in later lessons.
6. Use a short cat/dog example to explain order-blindness: the same clue words
   average to the same context, so their highest-scoring answer is unchanged.
   This is a plain-English conclusion, not another code experiment.

The **Core checkpoint** marks a complete stopping point before 03B.
Notebook subsections use `2.1`, `3.1`, and so on within core sections 1-5;
optional A/B asides are retained. The encoder-decoder and its original larger
comparison are now **section 6.1-6.13**, not optional C/D subsections.
The Python companion's cell labels follow those same section numbers.
There is one embedding-learning path, not a separate toy or parameter-registry tour.
The routing animation uses a copy of that actual CBOW and leaves its training state
and random seed untouched. Its exact unused-row result is scoped to fresh SGD,
no weight decay, and untied weights; the text distinguishes this from Adam history.
The recorder cell reloads the local helper module before importing its function,
so rerunning it after helper edits does not require restarting the kernel.
Running the animation cell rebuilds one static checkpoint and a GIF, then displays
one JavaScript replay. Cyan marks forward values, orange gradients, and green
the separate fresh-SGD update. The recorder
checks the full mean-gradient split and reconstructs the complete table gradient
from saved addresses; retaining intermediate gradients is for inspection only.
The original rich illustrations remain beside the conceptual explanation.

Section 6 explains source lookup, the encoder's final note, teacher forcing,
per-token loss, feedback through both GRUs, and accumulation into one shared
embedding table. Its own six-epoch training run prints progress and source,
decoder-input, BOS and EOS row changes. EOS is checked as an answer but is never
an input lookup in this small untied model; its unchanged input row is expected.

All fused-lookup, visualization, Skip-gram, reader/writer, and comparison experiments
are retained. **Run All and the full Python script still execute all of them.**
The three training runs are separate: main CBOW in **3.3**, fresh reader/writer
in **6.3**, and the larger three-objective comparison in **6.5-6.13**. That
8,000-story comparison uses fresh models and its own whole-story split; it does
not continue either earlier model, and its losses are not their evaluations.
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
storyboard values, reading order, per-epoch training equivalence, shared
encoder/decoder gradient routes, teacher-forcing alignment, and reserved-story
evaluation without downloading a corpus:

```powershell
.\.venv\Scripts\python.exe -m unittest test_real_text_embeddings -v
```

Rebuild static teaching diagrams locally with
[`generate_sensor_analogy_diagrams.py`](generate_sensor_analogy_diagrams.py).
