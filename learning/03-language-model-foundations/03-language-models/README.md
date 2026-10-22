# Chapter 03: Language Models — From Learned Rows to Attention

**Contents:** [directory map](CONTENTS.md)

You can fetch an embedding row and train a network. **What should it guess, and what does it
need to see to get the answer right?** Keep that question while the training loop stays familiar.

## Before you start

Use [Stage 02](../../02-pytorch-foundations/README.md) for tensors and optimization,
[Chapter 01](../01-sequence-memory/README.md) for recurrent memory, and
[Chapter 02](../02-tokenization-and-embeddings/README.md) for pieces, IDs, and learned rows.

From this directory, run [`setup.ps1`](setup.ps1) on Windows or [`setup.sh`](setup.sh) on
Linux/macOS. The scripts create a local `.venv`, install [requirements](requirements.txt),
and assign kernel `stage03-language-model-03-models` (Python 3.10+).
Select that kernel, restart, and run each notebook independently; kernels don't share state.
03A needs internet for its first TinyStories download, then reuses its checked local cache.
03B–03D use local, generated examples and one CPU thread.

## Resume the two embedding examples here

03A's opening checkpoint repeats everything needed from the **lookup microscope**
(`the ___ sat`, with hand-picked two-number rows) and the **learning game**
(`cat/dog -> sat/slept; bird -> flew/sang`, six supplied cards). You do not need
to reopen earlier notebooks. A side-by-side table then turns the same
`the sleepy cat sat on` window into CBOW, Skip-gram, and reader/writer questions.

Experiment-state labels distinguish arithmetic, random lookup previews, fresh
32-number training tables, and the separate 48-number held-out comparison.
The mechanism stays fixed; the questions and context processing change.

Model-specific headings name CBOW, Skip-gram, or the encoder-decoder; shared setup
and the three-model comparison are marked separately. Section 3 uses a short CBOW
sliding-window animation with a static storyboard. We share `make_document_windows`
because these lessons repeat the same cutting step, not because every language
model needs five-word windows or a hidden middle word.

The embedding intuition uses registry IDs and abstract learned sensor profiles.
03A's forward look follows one static profile into a fleet with one craft per
token occurrence: learned communication builds contextual working states without
rewriting the registry during inference. The `bank`/`river` and `bank`/`money`
examples illustrate the limitation, not a measured disambiguation result.
The encoder-decoder becomes an observatory using known profiles, partial readings
for a newly discovered object, and a question to write a likely report. This
chapter's GRU passes one final briefing; later Transformer cross-attention can
consult per-position source states. These are analogies for token prediction,
not an astronomy model. Rebuild the existing diagrams locally without network calls:

```powershell
python .\generate_sensor_analogy_diagrams.py
```

## Read in order

| Notebook | The next complaint | Handwritten notes | Code-only view |
|---|---|---|---|
| [03A. Training skeleton and word objectives](03A-training-skeleton-and-word-objectives.ipynb) | Lookup gives you rows, but what teaches them? Make practice questions from text and train rows with the predictor | [Theory](03A-training-skeleton-and-word-objectives-theory.md) | [Python](03A-training-skeleton-and-word-objectives.py) |
| [03B. Counts and fixed windows](03B-counts-and-fixed-window-language-models.ipynb) | Ordered seats preserve word order, but a clue outside the window is still invisible | [Theory](03B-counts-and-fixed-window-language-models-theory.md) | [Python](03B-counts-and-fixed-window-language-models.py) |
| [03C. Fixed windows to recurrent models](03C-fixed-windows-to-recurrent-language-models.ipynb) | A running note can carry the clue; RNN and LSTM memory still have limits | [Theory](03C-fixed-windows-to-recurrent-language-models-theory.md) | [Python](03C-fixed-windows-to-recurrent-language-models.py) |
| [03D. Recurrent reader/writer to attention](03D-recurrent-encoder-decoder-to-attention.ipynb) | One final summary may not be enough; let the writer consult earlier source notes | [Theory](03D-recurrent-encoder-decoder-to-attention-theory.md) | [Python](03D-recurrent-encoder-decoder-to-attention.py) |

03A starts learning from a small sample of children's stories from
[TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories), with a separate familiar
cat document for the close-up examples. Its [helper](real_text_embeddings.py) downloads and
checks the pinned corpus, builds practice pairs, and records the embedding comparison.
Its document-safe window builder is shared by the introductory run and held-out comparison:
CBOW, Skip-gram, and the writer reuse raw windows, then choose their own inputs and answers.
03A also takes one isolated optimizer step to show that embedding weights change while
the word IDs stay fixed.
Read the notebook's printed budget and measured losses rather than treating a tiny CPU run
as a model-quality benchmark. The later held-out comparison trains fresh models separately
and keeps the limits visible. The full notebook took about 3–5 minutes on the authoring CPU;
your machine may differ.

Then we switch to eight color-clue sentences in 03B–03C: you can see exactly which word
decides the answer and whether the model can read it. In 03D, reversing digits gives the
reader and writer separate jobs and known source positions to check.
These are deliberate diagnostic tasks, not competing text corpora.

## Keep the code handy

The same-stem `.py` files are ordinary scripts with `# %%` sections. For example:

```powershell
.\.venv\Scripts\python.exe .\03C-fixed-windows-to-recurrent-language-models.py
```

Set `MPLBACKEND=Agg` for noninteractive plots. The notebooks remain the best view for inline
figures and HTML animations. Saved notebooks have no outputs; run them to see the measurements.

The shared [training base](language_model.py) keeps `fit_language_model → training_loss →
forward → loss → backward → step` in one place. 03A shows it inline, 03C reuses it for
RNN/LSTM prediction, and 03D connects a reader/writer through a small wrapper.
The original comparison loops stay visible too; the wrapper doesn't replace their experiments.

The offline [window and objective checks](test_real_text_embeddings.py) need no corpus download:

```powershell
.\.venv\Scripts\python.exe -m unittest test_real_text_embeddings -v
```

The [03A illustration manifest](images/03a-illustration-prompts.json) and
[generator](generate_illustrations.py) are only for rebuilding the intuition sketches.
Running the lessons doesn't call the image generator; running it separately uses Azure
CLI sign-in and incurs the deployment's normal charges.

## What comes next

This route isn't a leaderboard or a mandatory family tree. The look-back model also has more
weights, so its result isn't attention alone. Practice gets correct earlier answers;
generation must live with its own guesses. Check both.

The writer can now look back, but the reader still works one token at a time.
Continue to [Stage 04 — Transformers](../../04-transformers/README.md):
**what if attention does the reading too?**
