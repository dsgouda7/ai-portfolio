# Model evolution: follow the complaint

**Contents:** [one-line map of this directory](CONTENTS.md)

You can fit a network and carry a sequence state. Now test what each predictor can actually see.
This bounded comparison branch is not a history survey or a replacement for existing lessons.
Eight color-clue sentences stay fixed across the first two notebooks. The third changes to integer
reversal because input and answer now have separate roles.

## Reader route

| Read | What you can explain afterward | Handwritten notes |
|---|---|---|
| [1. Counts and fixed windows](01-counts-and-fixed-window-language-models.ipynb) | Why identical windows force identical predictions, even with learned ordered features | [Theory](01-counts-and-fixed-window-language-models-theory.md) |
| [2. Fixed windows to recurrent language models](02-fixed-windows-to-recurrent-language-models.ipynb) | How a running note can carry a missing clue without promising unlimited recall | [Theory](02-fixed-windows-to-recurrent-language-models-theory.md) |
| [3. Recurrent reader/writer to attention](03-recurrent-encoder-decoder-to-attention.ipynb) | Why one learned card differs from looking back, and generation needs its own test | [Theory](03-recurrent-encoder-decoder-to-attention-theory.md) |

## The route is not a leaderboard

**Image pending:** `images/03-evo-01-model-evolution-map.png`
*What this will show:* A route of complaints, not a ranking of newer models.

> **Copilot image prompt**
> - **Target file:** `images/03-evo-01-model-evolution-map.png`
> - **Teaches:** A route of complaints, not a ranking of newer models.
> - **Aspect ratio:** 16:9
> - **Prompt:** Arrange cards: counts, ordered window, running note (RNN), reader and writer with one card, writer looks back, Transformer. Above counts show red wait quietly now warm. Label arrows: unseen combinations; clue outside window; different answer sequence; cannot look back; reading remains sequential. Add protected note (LSTM) as a side refinement from running note labelled memory can fade, not a required predecessor of attention. Use blue input, orange notes, green output, clean hand-drawn whiteboard style.
> - **Must show:** Seven legible cards; LSTM side refinement; arrows name complaints; no accuracy claims.
> - **Must not show:** invented numbers, unreadable text, decorative clutter, logos, copyrighted characters
> - **Alt text:** Counts lead to a bounded window, a running note, and a reader and writer. Looking back removes a source-access restriction.

LSTM refines a memory route; it is not a required rung toward attention. Attention predates
Transformers. Encoder and decoder mean reading and writing roles, implemented by different layers.
CBOW and skip-gram learn word rows under different prediction jobs; they are an adjacent branch,
not required evolutionary ancestors. Convolutional sequence models offer another route outside
this bounded comparison.

## Prerequisites and setup

Use [Stage 02](../../02-pytorch-foundations/README.md) for tensors and optimization.
Use [sequence memory](../01-sequence-memory/README.md) for running notes and LSTM gates.
[Tokenization and embeddings](../02-tokenization-and-embeddings/README.md) formally owns token
pieces, addresses, and learned rows. Each notebook still gives a short first-use explanation.
Before Notebook 3, visit the reader/writer sections in
[02C](../02-tokenization-and-embeddings/02C-context-windows-to-static-embeddings.ipynb).

Run `./setup.ps1` in PowerShell or `bash setup.sh` on Linux/macOS from this directory.
The scripts are adapted from sequence-memory. They create a local `.venv`, install
[requirements](requirements.txt), and assign kernel `stage03-language-model-evolution`, displayed
as `Python (Stage 03 Language Models Model Evolution .venv)`. Python 3.10+ is required.
Select that kernel, restart, and Run All for each notebook independently.
No downloaded models, data, checkpoints, internet calls, or shared notebook state are needed.
One CPU thread and small tensors keep each run under about two minutes on the validation machine.

Source outputs are intentionally clear. Read printed scorecards after execution; claims branch
on measured results. Splits and unequal parameter counts are documented. Reversal tests unseen
sequences at trained lengths, not length extrapolation. Artwork remains explicitly pending;
text diagrams and measured plots keep lessons usable. See [image prompts](images/PROMPTS.md).

## Required bounded intervention

Predict whether `visible_words = 4` reveals color in Notebook 1, then change only that variable.
Explain why revealing a clue is not fitting a four-word model. In Notebook 2, change only
`clue_word`. In Notebook 3, change only the first source digit and compare free-running output
with its expected reversal.

## Handoff

The next complaint is not simply more memory. The writer needs source addresses, and recurrent
reading remains sequential. Continue to the unchanged, self-contained
[Stage 04 route](../../04-transformers/README.md):
**what if we drop recurrence and let attention do all the reading?**
