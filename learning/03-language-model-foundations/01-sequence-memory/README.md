# Chapter 01: Sequence Memory

**Contents:** [directory map](CONTENTS.md)

How can an opening clue still matter at the end of a sequence?
Start after [Stage 02 — PyTorch Foundations](../../02-pytorch-foundations/README.md).

## Read in order

1. [01A — Sequence Memory Theory](01A-sequence-memory-theory.ipynb) ·
   [handwritten notes](01A-sequence-memory-theory.md) — Build a running note, watch its
   teaching signal fade, and learn LSTM first: keep/write/show, real two-state
   cell arithmetic and audible delayed echoes. Then discover
   [GRU reset/keep/replace mechanics](01A-sequence-memory-theory.ipynb#gru-hidden-state-mechanics).
   Reuse the same echo task and trainer for GRU; erase both LSTM states or the
   GRU's single state. Finish with the boundary between cells, encoder-decoder
   wiring and optional attention.
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
01A adds short local keys for symbols, gate letters, subscripts and multiplication.
Essential explanations remain visible without Mermaid or JavaScript support.

Section 2 now opens the training trail explicitly: create h1, h2, h3, make
a wrong note-four guess, inspect those saved cards in reverse, then update
the shared weights once. Every step keeps its note and incoming state visible:
D3 + h0 makes h1, A3 + h1 makes h2, and D4 + h2 makes h3; h0 starts blank.
The same input pairs stay beside the saved cards during backward inspection.
Every produced h has two visible uses: carry it into the next read, and pass
it to a separate next-note prediction layer (the head). The RNN cell makes h
first; that extra layer then reads h with its own `W_out` weights to score
the next note. The same prediction layer is reused at every step, not only
the last. The replay shows real initialized guesses
D3, A3, A3 for notes two, three, four; only note four's loss is traced.
The [cinematic studio renderer](sequence_studio.py) uses piano-key inputs,
folded memory cards, luminous carry ribbons and separate prediction beams.
The [numeric notecard close-up](images/01a-rnn-notecard-closeup.png) opens the
actual D4 update: the current input and old h2 pass through their displayed
shared matrices, combine, and become h3. The card contains computed features,
not written notes.
Its [restrained studio storyboard](images/01a-backprop-saved-cards.png)
is paired with a [measured local replay](images/01a-backprop-saved-cards.html).
The replay reuses the notebook's two-value recurrence and a hand-picked
prediction head. Detaching after h1 preserves all forward cards and the guess
while stopping feedback into h1. This is a training horizon, not a claim that
an RNN's carried information has a fixed-age context window.
The backward window and detach boundary are training-loop choices, not
`nn.RNN`, `nn.LSTM` or `nn.GRU` constructor settings.

All lesson visuals are embedded in 01A, including the full-resolution studio
storyboard, PNG/GIF fallbacks, SVG companions, and all ten HTML players.
The musical players embed their WAV recordings; the notebook's audio links
jump to those players rather than requiring separate files. The notebook has
a 65 MB limit to retain these original assets and the LSTM role illustrations without reducing their quality.

Sections 2-4 now build gated-memory intuition from **vanishing and exploding
gradients**, rather than treating them as a warning after the gates.
The [measured gradient probes](gradient_memory.py) show tanh saturation,
using the same tiny +0.0001 nudge near zero and near a flat edge. These are
changes to an internal feature, not prediction-versus-target errors.
The [interactive tanh tracker](images/01a-tanh-value-tracker.html) replaces
the Python nudge exercise: slide or enter a value, or play a sweep, to inspect
the output, local backward factor and the same +0.0001 nudge response.
They then show
the actual two-feature reader's backward sensitivity across repeated A3 inputs,
and a zero-centred scalar RNN whose bounded states coexist with amplified
feedback. The [error-feedback tracker](images/01a-error-feedback-tracker.html)
uses a draggable memory-weight scale with compact Vanishing/Exploding presets.
Panel 1 shows weights generating the output and error; panel 2 shows the
backward gradient. All 71 slider positions use actual autograd recordings,
not interpolated gradients. Input, other weights and target stay fixed. Its
[recorder](error_feedback_flow.py) uses PyTorch autograd and a half-squared-error
loss, not note classification. The Python verification is optional.
The older [backward replay](images/01a-gradient-sensitivity-replay.html)
and repeated numerical checks are now in an optional technical appendix,
outside the main reading flow. Two short explanations beside the primary
animation distinguish vanishing from exploding gradients. The slider changes
the recurrent weight; the cards show the stored hidden state h, not weights.
The appendix's replay remains a per-unit reference with 21 positions:
the values are sensitivity magnitudes relative to each final card, not hidden
values, memory amounts or probabilities. Color-matched decimal labels and a
printed raw-versus-normalized table explain its logarithmic scale.
Actual writing-weight SGD updates
show that clipping reduces a surge but does not enlarge fading credit.

The LSTM refinement first tries an [unsquashed total](images/01a-unsquashed-total.png):
different note orders still collide despite perfect first-write feedback.
It then introduces **c as a fixed-width numeric workspace**, not raw input
history, weights or a transcript. Its [memory timeline](images/01a-lstm-cell-vs-hidden.png)
shows one deliberately chosen coordinate being written, kept through later
notes, and overwritten. That toy coordinate's meaning is hand-picked; real
training learns distributed numerical patterns. Only then does the lesson
introduce h as the gated readout and next-step control context.
The [controlled carry comparison](images/01a-protected-cell-feedback.png)
separates direct c feedback from its bounded h readout. GRU follows the same
complaint: [old h bypasses the proposal's squash](images/01a-gru-protected-update.png),
without requiring a separate c. Native LSTMCell and GRUCell stress tests keep
KEEP at 0.99 and change only a candidate recurrent weight, exposing another
route that can amplify the full gradient. These are controlled sensitivity
experiments, not trained-architecture rankings or recommended weight settings.
The [complete A3 step](images/01a-lstm-two-state-step.png) shows the actual
keep/add arithmetic and **both states** travelling onward, with only h
feeding the note head directly. The
[measured role probe](lstm_state_roles.py) holds one incoming vector fixed
while changing the other, and closes only SHOW to demonstrate unchanged c
with zero h. The [numeric comparison](images/01a-lstm-state-roles-trace.png)
contains actual LSTMCell values, not generated measurements.
All three LSTM figures share the RNN studio's navy backdrop, piano-key inputs,
folded cards, soft carry ribbons and circular controls. Warm paper makes the
c numbers prominent; cyan marks h, and violet separates the note-head branch.
The music-studio cues stay around the mechanism, without scenery competing
with labels. Regenerate them locally with [lstm_state_roles.py](lstm_state_roles.py);
the measured labels stay exact and no model-service call is needed.

01A references saved feedback/state charts and LSTM/GRU gate-flow animations,
with embedded static fallbacks and embedded controllable HTML replays.
The recurrent replays use spacious, staged motion graphics: rounded memory
cards, per-feature gate meters, and moving markers rather than a complete
wiring diagram in every frame. Labels are checked at a 640-pixel pane width.
The four primary GIFs are embedded notebook attachments, so relative-path
resolution and JavaScript are not needed for playback. Their SVG companions
remain crisp when zoomed. Motion interpolates marker positions, never invented
intermediate gate or state values.
Eight recorded HTML players are stored in notebook display outputs; the two
interactive trackers are embedded directly in Markdown cells, with no code-input box.
None uses a relative iframe URL. Each runs in its own sandboxed frame, so its
controls cannot collide with another replay. PNG frames are packed as lossless
WebP: every pixel and all playback/audio timing remain unchanged.
The tanh and error-feedback trackers use self-contained SVG and JavaScript.
The latter scales recorded unit-sensitivity trajectories by the current signed error.
Both follow the same repeated `D3 A3 D4` phrase. Chosen scalar controls expose
retained, written, carried and exposed values; GRU interventions differentiate
the complete scalar update. Two-value `LSTMCell` and `GRUCell` checks compute
actual input/state-derived gates and compare both LSTM states or GRU's single
state with PyTorch at every note.
These are mechanism demonstrations, not claims that initialized weights learned music.
The [LSTM layer replay](images/01a-lstm-token-layer-flow.html) opens A3 after D3:
input and old h choose controls, old c follows the keep lane, and output h is
exposed while **both (h, c)** travel to the next token.
The [GRU layer replay](images/01a-gru-token-layer-flow.html) contrasts its single
carried/exposed h. A separate head produces note scores in either model.
Repeated time steps are not additional stacked layers.

The [LSTM musical walkthrough](images/01a-musical-lstm-memory.html), then the
[GRU walkthrough](images/01a-musical-gru-memory.html), connect those mechanics to
actual predictions. Both learn the same two openings after twenty distractions:
480 updates for LSTM, 240 for the original unchanged GRU run.
The LSTM's shorter 240-update trial still confuses the cues; the notebook's
one-variable drill reproduces it. Equal hidden widths do not mean equal parameter
counts, and different fit budgets are not an architecture ranking.
Each audio-clocked replay plays both reference openings,
actual initialized/fitted guesses, a no-carry intervention, all twenty distracting
notes, and an opening-state-erasure comparison. It uses synthesized tones, not a
recorded piano. LSTM's [scorecard](images/01a-musical-lstm-memory.png),
[silent GIF](images/01a-musical-lstm-memory.gif) and
[WAV](images/01a-musical-lstm-memory.wav), plus GRU's
[static scorecard](images/01a-musical-gru-memory.png),
[silent GIF](images/01a-musical-gru-memory.gif), and
[same WAV](images/01a-musical-gru-memory.wav) remain available without notebook JavaScript.
This is fitted recall, not composition, held-out accuracy, or an architecture benchmark.

A concise [access diagram](images/01a-memory-vs-attention.png) separates three
questions: a cell updates memory, encoder-decoder wiring assigns reader/writer
jobs, and optional source attention revisits saved source positions before an
output. Equal access does not mean equal weight. No attention model is trained
in 01A; that implementation belongs in
[03D](../03-language-models/03D-recurrent-encoder-decoder-to-attention.ipynb).

01A now has short native audio players in the learning flow: hear D3, A3 and D4
separately and as the opening phrase, compare the actual original/reversed inputs,
then hear both reference openings and the model's real before/after, erased-memory
and no-carry echoes. Those latter clips use the current run's recorded guesses.
Players require a click and use local synthesized tones; no sound autoplays.
Early gate/gradient blocks do not predict notes, so their internal numbers are
not converted into fake melodies. Dense per-note vector dumps are replaced by
shorter memory/parity takeaways; the saved visual traces retain the detailed values.

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

01A's saved visuals require no training run: static diagrams and GIF fallbacks
are notebook attachments, and ten saved HTML players appear beside their
explanations. **Trust the notebook outputs** to use interactive HTML; viewers
that block scripts or frames can use the PNG/GIF fallbacks. The short hidden
display cells only read saved players, not generate images or animations.
The tanh and error-feedback trackers are Markdown HTML, not empty Python cells.
VS Code renders these widgets in a trusted workspace; untrusted renderers retain
the static figures and comparisons. The shared exporter refreshes their embedded HTML.
Rerun setup and the recorded-player display cells to reload those players without training.
**Run All** from a fresh kernel for the learning experiments and training;
the setup supports both repository-root and chapter working directories.
Rerunning 01A's setup reloads its local musical helper, so an existing kernel
does not keep an older copy missing newly added audio functions.
The first recurrent-reader cell also reloads its backprop helper before
importing the shared note-head weights. After helper edits, rerun that cell
to refresh a cached module without restarting the kernel.
Changing an experiment changes its printed evidence, not the saved default captures.
Sound starts only after clicking an embedded player's native audio Play button. All replays
are offline, use visible text buttons rather than an external icon font, fit narrow
output panes, and show an initial image or static tracker readout when scripts are disabled.

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
.\.venv\Scripts\python.exe .\backprop_memory_flow.py
.\.venv\Scripts\python.exe .\gradient_memory.py
.\.venv\Scripts\python.exe .\error_feedback_flow.py
.\.venv\Scripts\python.exe .\lstm_state_roles.py
.\.venv\Scripts\python.exe .\lstm_layer_flow.py
.\.venv\Scripts\python.exe .\musical_memory.py
.\.venv\Scripts\python.exe -m unittest test_memory_gate_flows test_musical_memory test_gru_layer_flow test_lstm_layer_flow -v
.\.venv\Scripts\python.exe -m unittest test_memory_visuals -v
.\.venv\Scripts\python.exe -m unittest test_notebook_replays -v
.\.venv\Scripts\python.exe -m unittest test_backprop_memory_flow -v
.\.venv\Scripts\python.exe -m unittest test_lstm_state_roles -v
.\.venv\Scripts\python.exe -m unittest test_gradient_memory -v
.\.venv\Scripts\python.exe -m unittest test_error_feedback_flow -v
```

The gate-flow generator saves feedback-decay, LSTM-state and retention-comparison
charts alongside the paired gate replays. It uses the same recorded arithmetic as
the lesson and adds source fingerprints; it never trains a model. The musical
generator trains both tiny cue-echo models and exports measured guesses and state
arithmetic, audio, and playback timing with provenance metadata. Neither downloads data.
The backprop exporter refreshes only its own captures and marked player.
The other explicit exporters refresh notebook PNG/GIF attachments and the ten marked
HTML-player embeds. They preserve learning prose/code, experiment outputs,
and unrelated metadata. Player snapshots include source/output hashes so a
rebuild refreshes changed saved players rather than leaving stale HTML.
The gradient exporter rebuilds its six measured figures, static/GIF/HTML
backward replay and the quiet section-2 storyboard without changing model
weights or running the echo trainer. It refreshes its notebook attachments and
marked player while preserving experiment sources and outputs. Viewing the
saved notebook requires no separate image files.
The error-feedback recorder regenerates its HTML player from the template and
actual forward/backward values in `error_feedback_flow.py`. Edit that generator,
not the generated HTML. Run it before `memory_gate_flows.py` to embed a revised
player; it records gradients without fitting a model.
The gate exporter also rebuilds the one-token layer replay using the shared
[GRU layer recorder/renderer](gru_layer_flow.py), which the encoder/decoder preview
reuses for the same cell rule with learned word profiles.
The [LSTM recorder/exporter](lstm_layer_flow.py) separately verifies the native
(h, c) contract and builds the LSTM close-up plus conceptual attention-access
diagram. Musical provenance fingerprints cover both the experiment and this
LSTM recorder.
The shared [motion-graphics renderer](memory_visuals.py) uses the recorded
values from those helpers. [Spacing checks](test_memory_visuals.py) enforce
contained text, non-overlapping labels, readable type, and connectors that do
not cross text. No cloud image model is required to rebuild these graphics.

**Next:** [Chapter 02 — Tokenization and Embeddings](../02-tokenization-and-embeddings/README.md).
