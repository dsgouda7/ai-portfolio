# Chapter 01: Sequence Memory

**Contents:** [directory map](CONTENTS.md)

How can an opening clue still matter at the end of a sequence?
Start after [Stage 02 — PyTorch Foundations](../../02-pytorch-foundations/README.md).

## Read in order

1. [01A — Sequence Memory Theory](01A-sequence-memory-theory.ipynb) ·
   [handwritten notes](01A-sequence-memory-theory.md) — Build a running note, watch its
   teaching signal fade, and discover LSTM keep/write/show and
   [GRU reset/keep/replace mechanics](01A-sequence-memory-theory.ipynb#gru-hidden-state-mechanics).
   Then hear a tiny GRU echo an opening after twenty distracting notes, before
   and after fitting, and erase only its opening state.
2. [01B — Sequence Memory PyTorch Lab](01B-sequence-memory-pytorch-lab.ipynb) — Train
   forecasting and token models with packed batches, padding-aware loss, and causal checks.
3. [01C — Cinematic Piano Memory](01C-cinematic-piano-memory.ipynb) — Hear and measure
   the same memory question in one original piano phrase played in two keys.
   Change only its opening cue and inspect the ending.

The notebooks now use [static illustrated stories](images) rather than named-box flowcharts:
follow `D3 A3 D4` through ordered seats, a changing notecard, backward feedback, and
LSTM and GRU controls. The lab shows the exact shifted note pairs, what blank seats do to
a final state, and why stopping the reader and skipping blank answers are separate jobs.
Each illustration has a plain-English reading guide next to its measured check.
Essential explanations remain visible without Mermaid or JavaScript support.

01A references saved feedback/state charts and LSTM/GRU gate-flow animations,
with embedded static fallbacks and linked controllable HTML replays.
Both follow the same repeated `D3 A3 D4` phrase. Chosen scalar controls expose
retained, written, carried and exposed values; GRU interventions differentiate
the complete scalar update. A two-value `GRUCell` check then computes actual
input/state-derived gates and compares the resulting state at every note.
These are mechanism demonstrations, not claims that initialized weights learned music.
The [one-token layer replay](images/01a-gru-token-layer-flow.html) opens the actual
A3 step after D3: two input/state Linear projections, gates, content blending,
and the fork into output and next-token hidden state. Those two routes carry
identical values in one unidirectional layer; a separate model head produces note
scores. Repeated time steps are not additional stacked layers.

The [musical memory walkthrough](images/01a-musical-gru-memory.html) connects those
mechanics to actual predictions. A twelve-value GRU learns two cue-echo examples
in 240 optimizer steps. The audio-clocked replay plays both reference openings,
actual initialized/fitted guesses, a no-carry intervention, all twenty distracting
notes, and an opening-state-erasure comparison. It uses synthesized tones, not a
recorded piano. The [static scorecard](images/01a-musical-gru-memory.png),
[silent GIF](images/01a-musical-gru-memory.gif), and
[same WAV](images/01a-musical-gru-memory.wav) remain available without notebook JavaScript.
This is fitted recall, not composition, held-out accuracy, or a GRU/RNN benchmark.

01B uses token IDs, shifted answers, and padding before explaining where they come from.
That's a preview for testing memory; [Chapter 02](../02-tokenization-and-embeddings/README.md)
then explains pieces, addresses, and the different jobs of input, loss, and causal masks.

## Run it

From this directory, run [`setup.ps1`](setup.ps1) on Windows or [`setup.sh`](setup.sh) on
Linux/macOS. It creates a chapter `.venv`, installs [requirements](requirements.txt),
and assigns kernel `stage03-language-model-01-sequence-memory`.
Run each notebook from a fresh kernel. They share readable
[helpers](sequence_helpers.py), not live notebook state.
Small CPU examples use one PyTorch thread to avoid spending more time scheduling than learning.

01A's visuals require no kernel execution: static diagrams are notebook attachments,
GIFs render inline, and linked HTML files provide playback controls and synchronized
sound in a browser. The notebook contains no plotting, animation generation, or
replay-display code. **Run All** from a fresh kernel only for its learning experiments
and training; the setup supports both repository-root and chapter working directories.
Changing an experiment changes its printed evidence, not the saved default captures.
Sound starts only after clicking the linked replay's native audio Play button. All replays
are offline, use visible text buttons rather than an external icon font, fit narrow
output panes, and show an initial image even when scripts are disabled.

## Rebuild the illustrations

The [local diagram generator](generate_memory_diagrams.py) uses Matplotlib only:
no image service, credentials, or network calls. From this chapter directory:

```powershell
.\.venv\Scripts\python.exe .\generate_memory_diagrams.py
```

The pictures are intuition sketches, not fabricated training results. 01A's saved
curves and animations are rebuilt externally from its measured computations;
its musical exporter saves actual prediction audio. The later piano notebook
computes its own audio and key-cue intervention.
The piano experiment fits both key versions; it does not test composing or unseen-key quality.

Rebuild the measured gate-flow captures from the notebook's recorded updates:

```powershell
.\.venv\Scripts\python.exe .\memory_gate_flows.py
.\.venv\Scripts\python.exe .\musical_memory.py
.\.venv\Scripts\python.exe -m unittest test_memory_gate_flows test_musical_memory -v
.\.venv\Scripts\python.exe -m unittest test_gru_layer_flow -v
```

The gate-flow generator saves feedback-decay, LSTM-state and retention-comparison
charts alongside the paired gate replays. It uses the same recorded arithmetic as
the lesson and adds source fingerprints; it never trains a model. The musical
generator trains the tiny cue-echo GRU and exports measured guesses and state
arithmetic, audio, and playback timing with provenance metadata. Neither downloads data.
Both explicit exporters also refresh the notebook's embedded PNG attachments
without changing cell sources, outputs, or metadata.
The gate exporter also rebuilds the one-token layer replay using the shared
[GRU layer recorder/renderer](gru_layer_flow.py), which the encoder/decoder preview
reuses for the same cell rule with learned word profiles.

**Next:** [Chapter 02 — Tokenization and Embeddings](../02-tokenization-and-embeddings/README.md).
