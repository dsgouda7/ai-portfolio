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

## 03A: visible CBOW learning, then the section 5 reader/writer

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
   The shared word-model base owns the lookup and prediction layers; CBOW
   overrides only how the fetched rows are combined.
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
   Section 3.4 explains why predicting the same missing word from
   otherwise identical cat/dog clues can encourage reusable features in both
   profiles. Similar usage can produce similar profiles without a separate
   similarity target; related words are not guaranteed to become nearest neighbours.
6. Use a short cat/dog example to explain order-blindness: the same clue words
   average to the same context, so their highest-scoring answer is unchanged.
   This closes section 3 in plain English, not another code experiment.

Section 3's closing statement marks a complete stopping point before 03B.
Notebook subsections use `2.1`, `3.1`, and so on within core sections 1-3.
The actual windows have a saved storyboard beside section **2.2**; section **3.4**
has a saved before/after view of the core CBOW's word profiles. Neither needs
visualization code to render in the notebook. Both profile panels use one fixed
projection, with full-profile similarity reported separately from the 2D sketch.
Section 4 now contains only optional Skip-gram in **4.1-4.2**. The redundant
EmbeddingBag shortcut and old visualization cells are removed.
The encoder-decoder preview is now **5.1-5.5**. Its original larger comparison
is a [separate optional lab](03A-word-objective-comparison-lab.ipynb).
To skip the optional follow-ups, jump to the
[encoder-decoder preview](03A-training-skeleton-and-word-objectives.ipynb#encoder-decoder-preview).
To run it without CBOW training, run Setup and section 2, then the
[shared model definitions](03A-training-skeleton-and-word-objectives.ipynb#shared-model-definitions)
and [measurement helpers](03A-training-skeleton-and-word-objectives.ipynb#prediction-diagnostics).
The shared definitions now include the training base, word-model base, and concrete
CBOW class in consecutive executable cells. Section 2 also prepares the shared
reserved-story windows, so this preview route needs no CBOW training.
The preview needs neither CBOW nor Skip-gram training. The comparison lab imports
the same tested concrete model classes and runs independently from a fresh kernel.
03B is the next-chapter route, not the jump destination for this preview.
The Python companion's cell labels follow those same section numbers.
There is one embedding-learning path, not a separate toy or parameter-registry tour.
The routing animation uses a copy of that actual CBOW and leaves its training state
and random seed untouched. Its exact unused-row result is scoped to fresh SGD,
no weight decay, and untied weights; the text distinguishes this from Adam history.
The saved GIF renders directly in the notebook without running code, with a static
checkpoint available underneath. Run All and the Python companion do not record
or regenerate this replay. Cyan marks forward values, orange gradients, and green
the separate fresh-SGD update. The helper's offline tests still check the full
mean-gradient split and reconstruct the complete table gradient from saved
addresses; retaining intermediate gradients is for inspection only.
The original rich illustrations remain beside the conceptual explanation.

## Focused reader/writer preview

The new question is **why `cat`'s input row changes when the answer being graded
is `sat`**, not how to build a complete text generator.

1. **5.1:** bridge from supplied piano profiles to a trainable word lookup table.
   The [GRU foundation](../01-sequence-memory/01A-sequence-memory-theory.ipynb#gru-hidden-state-mechanics)
   owns gate mechanics; the detailed cat/BOS layer replay remains optional.
   The animation itself is an embedded GIF attachment, visible without a kernel
   or JavaScript and portable with the notebook. A visible, independent display
   cell reads the saved HTML for optional controls without loading data or training.
   JavaScript controls need a trusted renderer; separate-player and static links remain.
   The external replay exporter also refreshes this GIF attachment when rebuilding it.
2. **5.2:** keep teacher forcing to its input rule and a three-row alignment table.
   The full plain-English reference remains in the lab appendix.
3. **5.3:** follow prediction feedback into the shared table. A saved animation
   and a checked model-copy intervention show why the reader's `cat` row learns.
   That diagnostic update does not alter the real model.
4. **5.4:** run the original six Adam epochs. The inline callback delegates reporting
   to `EncoderDecoderProbeRecorder`; there is still only one optimizer loop.
   Compare actual row movement, full-row cat/dog versus cat/mat similarities,
   and reserved-story prediction. All belong to this same run.
5. **5.5:** reverse the same input rows without updates. Their mean agrees up to
   rounding, while recurrent states and next-word scores change. This proves
   order sensitivity, not understanding. The concise attention prelude explains
   the next limitation: the writer receives only a final summary. Attention would
   allow selective access to all saved source states; recurrence supplies order
   here, while Transformer attention needs positional information.

Stored rows, temporary context-dependent states, and vocabulary scores stay
distinct. Movement is not proof of rich semantics; actual wrong guesses remain
visible. EOS's untied input row remains unchanged because it is never looked up.
No attention is implemented here, and all notebook visuals are saved references.

**Run All in 03A no longer runs the larger comparison.** The
[optional lab](03A-word-objective-comparison-lab.ipynb), with its
[Python companion](03A-word-objective-comparison-lab.py), preserves all three
original 8,000-story, shared-initialization, 16-epoch experiments, including plots,
shortlists, best/final checks and optional extra seeds. It uses its own split and
fresh models; its results are not the small preview's before/after evidence.

## Engineering contracts inside the lesson

Keep the responsibilities separate rather than rebuilding them in each cell:

| Component | Shared behavior | Objective-specific behavior |
|---|---|---|
| [WindowDataLoader](language_data.py) | Validate/split windows, align tensors, seeded batching | CBOW returns clues/center; Skip-gram repeats each center against its neighbours |
| [WordEmbeddingLanguageModel](language_model.py) | Allocate layers; lookup, combine, and score in one order | CBOW averages input rows; Skip-gram keeps its one input row |
| [TrainableLanguageModel](language_model.py) | Configure loss/optimizer once; predict, align targets, calculate loss, backward, step | Models provide predictions; the reader/writer also shifts checking targets |
| [ModelValidator](language_validation.py) | Read-only evaluation and before/after reporting on fixed questions | Each objective's data preparation and target alignment determine what is checked |

The loader helper/base and concrete CBOW/Skip-gram pairing classes are defined
directly in the notebook, as are the training and word-model bases. Their module
counterparts remain for reuse; tests compare their ASTs so the learning code
cannot silently drift. Corpus and plotting utilities remain external.
Prediction readout helpers and `ModelProbeRecorder` live in
[language_validation.py](language_validation.py); their vocabulary is explicit,
not a hidden notebook global. The notebook imports the recorder and `word_probe`.
Data, model, and training definitions remain inline.

The recorder captures fixed probe questions and detached embedding values before
and after a run, then reports the prediction and row changes through short calls.
Per-epoch callbacks still live in the notebook; recording does not own training.
Concrete CBOW, Skip-gram and encoder-decoder definitions also have AST-matched
module counterparts for the standalone lab. `compare_source_order` and
`greedy_decode` are read-only inspections, not hidden training loops.

`prepare_data()` returns exactly `(input_ids, target_ids)`.
`loader.make_dataloader()` uses the object's data, batch size, and seed.
`WindowDataLoader.validate_window_ids` is a static window check, and its shared
`batch_tensors` static helper handles already-prepared tensors, including the
reader/writer and larger comparison. These operations belong to the base class,
not standalone functions, and still use standard PyTorch batching.
The same CBOW adapter prepares the
practice text, familiar probes, and reserved-story evaluation.

`train_model()` returns the loss history and elapsed time. Epoch callbacks read
measurements and print the existing progress tables; they do not implement another
training loop. `fit_language_model()` remains the lower-level path for callers
that supply their own loss and optimizer, including the larger comparison.
Required prediction/pairing hooks are enforced by abstract base classes.
Notebook cells print evidence without inline assertions; the offline tests own
those checks, while shared classes still reject invalid inputs explicitly.
PyTorch's `model.train()` still only changes training mode; it is not this
optimization entry point.

## Validation after each training run

Validation internals stay in [language_validation.py](language_validation.py).
Section 2 prepares the next 200 stories with the fixed training vocabulary.
Each model records a validation baseline immediately before training; a separate
cell after its training block calls `compare_after` with the actual epoch count:

```python
validator.record_before(model)
run = model.train_model(loader, epochs=10)
validator.compare_after(model, epochs=len(run.losses))
```

The report shows loss, answer accuracy, number of checked answers, and one actual
input/answer/prediction example before and after. The same 2,048 retained windows
supply 2,048 CBOW center answers, 8,192 Skip-gram neighbour pairs, or 6,144
reader/writer next-token answers including stop. Compare improvement within each
objective, not raw losses across different jobs.

Reader/writer validation is teacher-forced: it supplies correct previous words.
Free-running generation is a small boundary check in 5.5. Evaluation preserves model
modes, weights, gradients, and RNG state. The larger comparison reuses this same
evaluation implementation on its independent split.

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

The loader and training-template contracts have a focused offline suite:

```powershell
.\.venv\Scripts\python.exe -m unittest test_language_contracts -v
```

Validation has its own offline tests:

```powershell
.\.venv\Scripts\python.exe -m unittest test_language_validation -v
```

Rebuild static teaching diagrams locally with
[`generate_sensor_analogy_diagrams.py`](generate_sensor_analogy_diagrams.py).

Rebuild the two saved CBOW captures from the current core notebook code and its
verified corpus cache with
[`generate_cbow_static_views.py`](generate_cbow_static_views.py):

```powershell
.\.venv\Scripts\python.exe .\generate_cbow_static_views.py
```

The generator runs the real core training in a separate process, not the notebook's
active kernel. PNG metadata records the source fingerprint so stale captures are detected.

Rebuild the measured state-flow animation, HTML replay, and static fallback from
the current preview's actual six-epoch training with
[`encoder_decoder_feedback.py`](encoder_decoder_feedback.py):

```powershell
.\.venv\Scripts\python.exe .\encoder_decoder_feedback.py
.\.venv\Scripts\python.exe -m unittest test_encoder_decoder_feedback -v
```

The exporter runs the preview in its own process using the cached corpus, then
records copies of that checkpoint. The diagnostic matches explicit recurrent
steps to the actual fused GRU call and its parameter gradients. It runs one
fresh-SGD update on independent copies, never on the main model. Generation
feeds back its own actual guesses, unlike teacher-forced inputs.
Matplotlib `to_jshtml` supplies replay controls without FFmpeg; the GIF renders
without running a notebook cell. This proves mechanics, not language quality.
