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

01A opens with a small rule: predict note four by repeating note one.
Two phrases share the same last two notes but need different answers, so the
first note must survive later reads. The longer delayed echo comes after the
LSTM mechanics, with its repeated middle notes shown explicitly.
Each change of scale is labeled: the two-feature reader, one-feature gradient microscope, hand-controlled storage,
initialized native cells, and twelve-feature fitted echo. The
[complete LSTM read](01A-sequence-memory-theory.ipynb#lstm-complete-read)
is the main forward walkthrough; the
[technical checks](01A-sequence-memory-theory.ipynb#gradient-technical-checks)
retain the deeper derivative and retention experiments.

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
storyboard, PNG/GIF fallbacks, SVG companions, and all nine HTML players.
The musical players embed their WAV recordings; the notebook's audio links
jump to those players rather than requiring separate files. The notebook has
a 65 MB limit to retain these original assets and the LSTM role illustrations without reducing their quality.

Sections 2-4 now build gated-memory intuition from **vanishing and exploding
gradients**, rather than treating them as a warning after the gates.
The [measured gradient probes](gradient_memory.py) introduce tanh with one
[static curve and comparison](images/01a-tanh-saturation.png): the same
+0.0001 input change passes almost unchanged near zero, but only about 1%
passes near a flat edge. These are internal feature changes, not
prediction-versus-target errors. This replaces the separate tanh tracker.
The RNN still needs a nonlinear update; small states do not guarantee stable
feedback. The later LSTM/GRU diagrams retain their essential distinction:
old memory can bypass repeated squashing even though the cells still use tanh.
The remaining experiments show
the actual two-feature reader's backward sensitivity across repeated A3 inputs,
and a zero-centred scalar RNN whose bounded states coexist with amplified
feedback. The [error-feedback tracker](images/01a-error-feedback-tracker.html)
now follows **D3 → A3 → D4**, using the same two-number note codes but one
hidden feature so each multiplication fits on screen. Every note has a
visible cell labelled with its incoming/outgoing h: h0 → h1 → h2 → h3.
The **Previous-memory multiplier** slider changes the shared recurrent
weight, not an input, a hidden state or a fresh random initialization.
The label now explicitly says **recurrent weight**. Each block separately
names its **Input multipliers**, rather than showing an unexplained w.
Starting input/head weights are hand-picked and fixed during the sweep.
Each RNN shows `previous h * m` and its actual contribution. A branch after
every note displays the head's guess, the actual next note, its probability,
and cross-entropy loss. Targets follow A3, D4, D3 as the phrase advances.
Earlier losses are dashed **previews only**; only the gold ending loss
drives backpropagation. The final head guesses A3; its correct next note is D3.
The default diagram spells loss out as **Error**, and replaces bare g labels
with **Ending-error response at h**. Selecting a block shows two actual
forward reruns: saved h slightly lower and slightly higher, with weights fixed.
The visible word equation divides **ending-error change by saved-memory
change** to measure the nearby response independently of the gradient.
Positive means increasing h raises error; near-zero means little guidance
reaches that state. Backprop computes the tiny-change limit by chaining
local rates, rather than performing these trial reruns as its training algorithm.
The response is a rate, not another error or a weight update; mathematical
L/g notation remains only in the expandable arithmetic.
Each backward factor is `m * (1 - h_t**2)`; earlier states cross more factors.
Ten decimal places expose the middle head's very small probability changes.
The first head stays fixed during the sweep because its incoming h0 is zero.
Click any note to inspect live forward/backward substitutions and that
cell's parameter-gradient contributions. The player sums contributions
before one actual SGD update, then reruns the same phrase with the new
weights and reports the new guess, D3 probability and loss.
Three notes show the arithmetic, not a dramatic explosion. The optional
seven-fold repetition supplies 21 individually inspectable note cells:
twenty backward links compound the factors. Both lengths have 71 genuine
autograd recordings, not interpolated gradients. The automatic label
compares first/last state feedback; within 10% of 1 is this toy's
nearly-unchanged band, not a universal cutoff.
The main BPTT/gradient explanation now uses one
[3200x5600 dark handwritten walkthrough](images/01a-bptt-memory-weight-explainer.png),
a one-line takeaway, the existing slider, and a short measured
opening/ending feedback comparison. Direct and inherited effects expand into repeated
powers of the same memory weight. It keeps the full **differentiate ->
expand -> substitute -> distribute** sequence rather than replacing
derivatives with a new shorthand. Each equation has its verbal reason:
error depends on the prediction, the prediction depends on memory, and
old memory also depends on the shared weight. The sheet labels its **linear sketch**
separately from the **RNN handoff**, which includes the local squeeze response;
a multiplier above one alone
does not guarantee exploding feedback, and a large gradient is not
necessarily a large prediction error.

[MAI-Image-2.6 paper artwork](images/01a-bptt-notebook-paper-mai.png) supplies
the original notebook texture. The [local compositor](bptt_summary.py)
renders every equation, circle and label in Segoe Print at the final pixel
resolution, rather than trusting image-generated mathematics.
The retained native artwork is 1216x1792; the final high-definition composite
is 3200x5600, not native 4K MAI output. Its paper texture is locally color-graded
to the existing dark navy theme, with white/cyan/gold/coral handwriting, and smoothed
and its PNG palette optimized to fit the notebook budget. This requested
handwritten style now matches the dark measured charts. No additional image-model
request is needed for this color grade.
[Prompt/provenance](images/01a-bptt-summary-prompts.json) ·
[checks](test_bptt_summary.py).

The optional, collapsed arithmetic uses the original two-number inputs and chosen input
weights: note contributions are 0.001, -0.0005 and 0.0005.
At memory weights 0.8/1.2, the linear three-read responses are 0.0011/0.0019;
the exact tanh responses, ending feedback and recurrent-weight gradients
are checked against native autograd.
The numerical example also expands the head's two factors, `-1/p` and
`dp/dh3`, so its starting feedback is derived rather than asserted.
The primary path distinguishes a weight's combined effects from feedback
travelling through saved states. The remaining animations and recordings are preserved.

The [static comparison export](images/01a-note-gradient-chain.png) is available
separately, but its redundant snapshot section is no longer embedded in the
notebook; the dark handwritten sheet supplies the static explanation.
The older [forward math](images/01a-bptt-forward-math.png),
[derivative math](images/01a-bptt-backward-math.png), and
[worked card](images/01a-bptt-recurrent-gradient.png), plus the manual BPTT
check, are collapsed or placed in the technical appendix rather than
repeating the main explanation. [Detailed exporter](bptt_math.py) ·
[checks](test_bptt_math.py). The actual writer-gradient clipping experiment
still shows that clipping caps a surge but does not restore fading guidance.
It is explicitly a separate scalar target-1, half-squared-error test with
SGD step size 0.01, not the preceding note cross-entropy experiment.

The LSTM refinement first tries an [unsquashed total](images/01a-unsquashed-total.png):
it reuses the earlier two-feature reader's fixed input weights to make
decimal writes: D3 `[0.70, 0.30]`, A3 `[-0.20, 0.80]`, D4 `[0.50, 1.10]`.
These are write contributions, not replacement note codes. Both orders reach
`[0.80, 3.00]`; first-write feedback passes unchanged in each written feature.
The +0.01 first-write nudge makes that 100% carry-through concrete. Order is
lost because this trial uses plain commutative addition, not because every
linear recurrence must lose order.
The diagram and printed running totals consistently show two decimal places.
The gradient prose distinguishes the ending mistake from correction guidance:
early guidance can be muffled or amplified before contributions reach the shared update.
It then introduces **c as a fixed-width numeric workspace**, not raw input
history, weights or a transcript. Its [memory timeline](images/01a-lstm-cell-vs-hidden.png)
shows one deliberately chosen coordinate being written, kept through later
notes, and overwritten. That toy coordinate's meaning is hand-picked; real
training learns distributed numerical patterns. Its +0.8/-0.8 values are
internal writes, not replacement encodings for the two-number note inputs.
Only then does the lesson
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
The [LSTM forward walkthrough](images/01a-lstm-forward-walkthrough.html) follows
the same native cell through D3, A3 and D4. Note input and old h choose the
controls; old c takes the KEEP lane. The new c feeds a SHOW-controlled bounded
readout h, and only that h enters the next-note head. Both states continue to
the next read. Shared weights compute the displayed controls; the fractions
are not the trainable parameters themselves.
[MAI-generated gold/cyan objects](images/01a-lstm-state-objects.png)
mark equal-width workspace/readout roles, with all labels outside the artwork.
Its [static companion](images/01a-lstm-forward-walkthrough.png) preserves the
routing without JavaScript. Play/pause, manual stages and note selection use
recorded values, not interpolated states. An optional isolated SHOW-closed
check keeps c fixed while h becomes zero; our bias-free head then has equal
note scores. This is initialized arithmetic, not trained musical performance.
The three existing LSTM role figures share the RNN studio's navy backdrop, piano-key inputs,
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
Seven recorded HTML players are stored in notebook display outputs; the two
interactive trackers and the LSTM forward walkthrough are embedded directly
in Markdown cells, with no code-input box.
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
The tanh/error-feedback trackers and LSTM forward walkthrough are Markdown
HTML, not empty Python cells.
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
.\.venv\Scripts\python.exe .\bptt_math.py
.\.venv\Scripts\python.exe .\bptt_summary.py
.\.venv\Scripts\python.exe .\memory_gate_flows.py
.\.venv\Scripts\python.exe .\backprop_memory_flow.py
.\.venv\Scripts\python.exe .\gradient_memory.py
.\.venv\Scripts\python.exe .\error_feedback_flow.py
.\.venv\Scripts\python.exe .\lstm_state_roles.py
.\.venv\Scripts\python.exe .\lstm_layer_flow.py
.\.venv\Scripts\python.exe .\lstm_forward_flow.py
.\.venv\Scripts\python.exe .\musical_memory.py
.\.venv\Scripts\python.exe -m unittest test_memory_gate_flows test_musical_memory test_gru_layer_flow test_lstm_layer_flow -v
.\.venv\Scripts\python.exe -m unittest test_memory_visuals -v
.\.venv\Scripts\python.exe -m unittest test_notebook_replays -v
.\.venv\Scripts\python.exe -m unittest test_backprop_memory_flow -v
.\.venv\Scripts\python.exe -m unittest test_lstm_state_roles -v
.\.venv\Scripts\python.exe -m unittest test_gradient_memory -v
.\.venv\Scripts\python.exe -m unittest test_error_feedback_flow -v
.\.venv\Scripts\python.exe -m unittest test_lstm_forward_flow -v
.\.venv\Scripts\python.exe -m unittest test_bptt_math -v
.\.venv\Scripts\python.exe -m unittest test_bptt_summary -v
```

The math-card exporter uses Windows **Segoe Print** for handwriting; viewing
the embedded PNGs needs no installed font. Run it before the gate-flow
exporter, which refreshes the notebook's image attachments. The equations
are rendered by Matplotlib, not an image model; copyable notation and
rounded worked tables remain in the lesson.

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
The gradient exporter rebuilds its five measured figures and the quiet
section-2 storyboard without changing model weights or running the echo
trainer. It refreshes only its notebook image attachments while preserving
experiment sources, outputs and players. Viewing the
saved notebook requires no separate image files.
The error-feedback recorder regenerates its HTML player and paired static figure
from actual per-note forward/backward values in `error_feedback_flow.py`. Edit that generator,
not the generated HTML. Run it before `memory_gate_flows.py` to embed a revised
player; it records each trial and one SGD preview, not a training loop.
The card image is embedded as a resized lossless WebP, so the player remains
self-contained and offline. Rebuilding the player needs no cloud call.
To regenerate only the artwork with MAI-Image-2.6 and your Azure CLI sign-in:

```powershell
.\.venv\Scripts\python.exe ..\03-language-models\generate_illustrations.py --manifest .\images\01a-gradient-card-prompts.json --only 01a-gradient-memory-card.png --force
```

The [prompt manifest](images/01a-gradient-card-prompts.json) records the intended
card surface and prohibits generated numbers. Regenerate the player and
notebook embed afterward.
The [LSTM forward recorder](lstm_forward_flow.py) reuses the lesson's native
initialization, note codes and existing gate recorder; it does not change any
older captures. Its [MAI manifest](images/01a-lstm-forward-prompts.json) keeps
the source image and reviewed crop boxes. To regenerate that source art:

```powershell
.\.venv\Scripts\python.exe ..\03-language-models\generate_illustrations.py --manifest .\images\01a-lstm-forward-prompts.json --only 01a-lstm-state-objects.png --force
```

Inspect the two object crops after regeneration, then rerun the forward
recorder and refresh its marked notebook embed. Viewing/rebuilding its saved
player uses no model service.
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
