# Chapter 01: Sequence Memory

**Contents:** [directory map](CONTENTS.md)

How can an opening clue still matter at the end of a sequence?
Start after [Stage 02 — PyTorch Foundations](../../02-pytorch-foundations/README.md).

## Read in order

1. [01A — Sequence Memory Theory](01A-sequence-memory-theory.ipynb) ·
   [handwritten notes](01A-sequence-memory-theory.md) — Follow ordered notes,
   backward feedback, the LSTM's animated data/control flow and GRU by contrast, then fit all three
   models on the same original musical phrases. The main notebook has exactly
   six teaching code blocks: three reusable models and three musical applications.
   Its actual before/after predictions are playable without rerunning training.
2. [01B — Sequence Memory PyTorch Lab](01B-sequence-memory-pytorch-lab.ipynb) — Train
   forecasting and token models with packed batches, padding-aware loss, and causal checks.
3. [01C — Cinematic Piano Memory](01C-cinematic-piano-memory.ipynb) — Hear and measure
   the same memory question in one original piano phrase played in two keys.
   Change only its opening cue and inspect the ending.

**Optional closer look:** [01A1 — Worked details](01A1-sequence-memory-worked-details.ipynb)
retains the original complete calculations, one-variable code checks, detailed
pictures, old replays and two-opening echo experiment. It has its own plain-English
reading guide, the same running example, and a fresh-kernel setup. No images or
unique explanations were deleted to shorten the main route.

## What changed in the main route

The main explanations establish **why** a mechanism is needed before its
metaphor or implementation: counts cannot choose different required answers,
hidden state carries a concise current-phrase summary, backward feedback
teaches earlier reads, and gates separate keeping old content from using or
rewriting it. The plain-English reasoning from the preceding versions stays in the main
route, including how past context changes a read and why weak responses
compound. Repeated explanations are consolidated; the companion holds the
longer worked calculations, not the only explanation of the concept.

The main route also opens the actual reads and teaching decisions: what h1
passes to h2, why backward multiplies handoffs but adds a shared weight's uses,
how an A3 read updates each gated state, and what the held-out music scores
do and do not establish. These explanations are not deferred to an optional
link or left for the reader to infer from a figure.

The mathematical spine starts with ordinary gradient descent: which change
in a shared weight will reduce the D3 prediction loss? The symbolic proof
lives in the [loss-to-weight sheet](images/01a-rnn-gradient-discovery.png)
and [longer-chain continuation](images/01a-rnn-gradient-chain.png).
They keep `softmax(scores)` as a named block rather than expanding its
probability algebra. The notebook text plugs in recorded forward/backward
values and checks the direction and effect of one weight update.
Their [recorder/exporter](gradient_discovery.py) verifies the numbers with
autograd and finite differences; running it rebuilds both static sheets.

Section 9 is a [quick memory-design cheat sheet](01A-sequence-memory-theory.ipynb#sequence-memory-cheat-code):
RNN rewrites; LSTM stores, then shows; GRU keeps, then blends. Each recipe
names the earlier drawback it addresses and the control it trades away.
[System boundaries and the scope ledger](01A1-sequence-memory-worked-details.ipynb#sequence-memory-reference)
stay in the optional worked-details route.

### Handwritten calculation panels

The [three-case gradient player](images/01a-gradient-three-cases.html) remains
an optional [worked-details reference](01A1-sequence-memory-worked-details.ipynb#rnn-gradient-microscope),
with the [static snapshot](images/01a-gradient-three-cases.png) available separately.
The main route omits the repeated recorded-values recap and table/image;
it keeps the chain-rule proof and the numerical examples that explain it.
The [inline LSTM GIF](images/01a-lstm-forward-walkthrough.gif) plays directly
in section 6 without JavaScript or a Play button. It follows A3 after D3:
input and old h choose the three gate controls and a proposal; KEEP and WRITE
make c; SHOW makes h; only h enters the note head. Moving contents and dashed
control signals reveal results only after their inputs arrive. Visited blocks
and links stay bright; moving markers identify the current operation.
Spaced lanes and heavier arrows keep connections clear of unrelated blocks.
A SHOW=0 comparison preserves c, then the normal c/h pair continues into D4.
Both recurrent GIFs explicitly separate the learned **Linear/Dense note head**
from **softmax and next-note selection**. The head maps two h features to three
actual note scores; only after those arrive do the probabilities and chosen
note appear. The LSTM example has no head bias; the GRU example includes one.
The [optional interactive player](01A1-sequence-memory-worked-details.ipynb#lstm-complete-read)
lives only in the worked-details companion, keeping the main lesson to one GIF.
It retains pause, exact arithmetic and all three reads. It starts paused; explicit
Play enables motion even when reduced-motion preferences are active.

The [matching GRU GIF](images/01a-gru-forward-walkthrough.gif) uses the same
spaced, cumulative style. RESET filters only the proposal's old-h context;
KEEP preserves old amounts directly; WRITE=1-KEEP admits new content.
Their blend is the one h read by both the head and the next step. A RESET=0
comparison preserves the kept branch. The
[optional interactive GRU board](01A1-sequence-memory-worked-details.ipynb#gru-forward-details)
retains full forward/backward arithmetic in the companion, not a duplicate
player in the main route. Both GIFs use recorded values, not fabricated
animation intermediates.

[MAI-Image-2.6 generated the blank paper](images/01a-handwritten-notes-prompts.json),
not the mathematical content. The [shared renderer](handwritten_notes.py)
embeds the paper and the [Kalam handwriting font](images/kalam/OFL.txt) directly
in each offline player. Equations and substituted numbers still come from
the recorded calculations, with expandable exact text and full traces.
There are no model-invented numbers, external font requests or handwritten
screenshots that stop matching the current playback step.

Rebuild these three players with `gradient_cases.py` and
`gated_sequence_storyboards.py`, then refresh their notebook embeds using
the existing replay helper. The MAI generation manifest is only needed when
intentionally replacing the paper surface.

The optional [full LSTM board](01A1-sequence-memory-worked-details.ipynb#lstm-fixed-board-details)
and [GRU](images/01a-gru-sequence-storyboard.html) keep all three note columns
and saved states on one board during forward and backward inspection. The active
arithmetic is also shown in full-sized text outside the drawing. Static boards
remain available without JavaScript. Their [recorder](gated_sequence_storyboards.py)
uses actual PyTorch/autograd values, not invented animation intermediates.

The [shared music experiment](sequence_music_comparison.py) uses 384 original
32-event phrases, a fixed 307/77 whole-phrase split and the same settings for
RNN, LSTM and GRU. Each uses 48 memory numbers and 320 updates; parameter counts
still differ. Saved panels show the reference, initial guesses and trained guesses
from the same held-out opening. All three improve on this corpus, but GRU does
not beat RNN in this particular run. The lesson does not manufacture an
architecture ranking or claim professional music quality.

The main GRU explanation separates an ordinary RNN's remixed h from a GRU's
per-number kept/written blend. It uses a taped-off-whiteboard picture in words,
not the misleading claim that LSTM's c and h were simply merged.

[Main/companion checks](test_sequence_notebook_flow.py) verify the main LSTM
and GRU GIFs with companion-only interactive replays, six-block
contract, that training uses the classes actually shown, shared comparison data,
saved playable results and the retained worked route.

## Worked-details reference and media maintenance

The remaining detailed descriptions refer to the preserved **01A1 companion**.
Its original part numbers and measured examples remain available; they are not
extra required steps in the concise main route.

The companion opens with a small rule: predict note four by repeating note one.
Two phrases share the same last two notes but need different answers, so the
first note must survive later reads. The longer delayed echo comes after the
LSTM mechanics, with its repeated middle notes shown explicitly.
The dark opening illustration handles the counting/order comparison without
a code drill. Counts keep frequency, not sequence order. The RNN then reads
the original ordered notes; it cannot reconstruct order from their counts.
Its first runnable example demonstrates memory updates and actual head guesses.
Each change of scale is labeled: the two-feature reader, one-feature gradient microscope, hand-controlled storage,
initialized native cells, and twelve-feature fitted echo. The
[complete LSTM read](01A1-sequence-memory-worked-details.ipynb#lstm-complete-read)
is the main forward walkthrough. The
[copy-before-showing comparison](01A1-sequence-memory-worked-details.ipynb#lstm-copy-before-showing)
explains why c and h have separate jobs. It compares repeatedly squashing
the carried number with keeping c and making a fresh h, without claiming
that c stores the raw past. The
[three-case gradient lesson](01A1-sequence-memory-worked-details.ipynb#rnn-gradient-microscope)
uses the same six calculations for stable, exploding and vanishing feedback.
The fixed-fraction drills and optional technical appendix are removed.

The notebooks now use [static illustrated stories](images) rather than named-box flowcharts:
follow `D3 A3 D4` through ordered seats, a changing notecard, backward feedback, and
LSTM and GRU controls. The lab shows the exact shifted note pairs, what blank seats do to
a final state, and why stopping the reader and skipping blank answers are separate jobs.
Each illustration has a plain-English reading guide next to its measured check.
01A adds short local keys for symbols, gate letters, subscripts and multiplication.
Essential explanations remain visible without Mermaid or JavaScript support.

The [illustrated LSTM forward pass](images/01a-lstm-forward-illustrated.png)
replaces the introductory hand-controlled code block. One D4 step shows
KEEP/WRITE making c, tanh/SHOW making h, both states continuing, and only h
feeding the note head. The same measured six-note arithmetic stays in the
[replay helper](memory_gate_flows.py); the later full PyTorch example remains runnable.

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

All original lesson visuals are embedded in 01A1, including the full-resolution studio
storyboard, PNG/GIF fallbacks, SVG companions, and all nine HTML players.
The musical players embed their WAV recordings; the notebook's audio links
jump to those players rather than requiring separate files. The notebook has
a 65 MB limit to retain these original assets and the LSTM role illustrations without reducing their quality.

The nine question-led parts follow forward memory, backward teaching,
feedback failure, three memory-design attempts, LSTM, GRU, a shared echo
experiment, and the final boundaries/scorecard. Native stress tests come
after both real gate calculations, rather than interrupting the first copy-lane example.
The reading prose calls a one-number memory exactly that, rather than a
"scalar reader". Each longer check explains its count before running:
20 updates do not mean 20 features. The echo task separately introduces
its twelve memory numbers and four-number input code. Formulas, measured
tables and runnable code stay unchanged.
The [frozen-weight what-if](01A1-sequence-memory-worked-details.ipynb#gradient-what-if)
uses the original two-feature reader before introducing scalar gradient cases.
It measures error change per memory change, explains positive/negative
responses, and checks the result against autograd without changing the saved card.
The [measured gradient probes](gradient_memory.py) introduce tanh with one
[static curve and comparison](images/01a-tanh-saturation.png): the same
+0.0001 input change passes almost unchanged near zero, but only about 1%
passes near a flat edge. These are internal feature changes, not
prediction-versus-target errors. This replaces the separate tanh tracker.
The RNN still needs a nonlinear update; small states do not guarantee stable
feedback. The later LSTM/GRU diagrams retain their essential distinction:
old memory can bypass repeated squashing even though the cells still use tanh.
The [three-case player](images/01a-gradient-three-cases.html) replaces the
slider and repeated numerical substitutions. Its three columns are stable
(`W_h=1.0`), exploding (`1.2`) and vanishing (`0.8`). They use identical
21-note inputs, input/head weights and ending target. Only `W_h` changes.
Six shared formula rows reveal the forward read, prediction/error, head
feedback, backward handoffs, summed weight contributions and one SGD update.
The underlying RNN includes tanh; the panel arithmetic is recorded by
autograd, not interpolated. The old saved-state feedback and shared-weight
gradient are explicitly different questions: multiply along a state route;
add contributions from repeated uses of a weight.
Each formula appears once above three numerical substitutions. Shared
table rows keep the cases horizontally aligned even when numbers wrap;
narrow panes scroll the table rather than stacking the cases.
A static Markdown version in the notebook preserves the comparison without
JavaScript or images. All values are initially shown; Play/pause, Next and
Reset optionally reveal the steps. Per-case inspectors retain all 21 reads.
There is no parameter slider or random restart.

Both former gradient images are removed from the notebook. Their
**differentiate -> expand -> substitute -> distribute** reasoning remains
as text in the local vector-detail block, using standard `x_t`, `h_t`,
`W_x`, `W_h`, `W_out`, prediction and loss notation. The linear sketch is
clearly separate from the actual RNN handoff. A weight above one alone
does not guarantee amplification, and gradient growth is not error growth.

The [current proof exporter](gradient_discovery.py) renders verified equations
on the shared [white MAI paper](images/01a-handwritten-paper.png), using the
bundled [Kalam font](images/kalam/OFL.txt). The
[proof and numerical checks](test_gradient_discovery.py) cover this version;
the superseded math cards, dark reference sheet and their exporters are retired.

The optional technical appendix and its fixed-fraction drills are no longer
in the notebook. Their unique teaching is restored locally: a
[complete three-read contribution check](01A1-sequence-memory-worked-details.ipynb#shared-gradient-check)
shows why the final use is not the whole shared gradient, and a short manual
backward loop checks the total against PyTorch. The
[adjacent vector detail](01A1-sequence-memory-worked-details.ipynb#vector-backprop)
contains matrix/input/head/bias derivatives, selected-head loss masking,
and the direct/inherited counting distinction. Batch/time/features are
explained beside the real-dimensions bridge, not in the counting introduction.
The [mathematical checks](test_rnn_gradients.py) retain scalar and matrix
autograd comparisons, finite differences and shared-weight verification;
they are not prerequisites for following the lesson.
The actual writer-gradient clipping experiment still shows that clipping
caps a surge but does not restore fading guidance.
It is explicitly a separate scalar target-1, half-squared-error test with
SGD step size 0.01, not the preceding note cross-entropy experiment.

The LSTM refinement first tries an [unsquashed total](images/01a-unsquashed-total.png):
it reuses the earlier two-feature reader's fixed input weights to make
decimal writes: D3 `[0.70, 0.30]`, A3 `[-0.20, 0.80]`, D4 `[0.50, 1.10]`.
These are write contributions, not replacement note codes. One running
trace shows that later writes are admitted without selection, even though
first-write feedback passes unchanged. The +0.01 nudge makes that 100%
carry-through concrete. The next refinement closes unwanted writes; it does
not repeat the earlier note-count argument. A compact two-opening comparison
still proves that the additive final card cannot distinguish the needed cues.
Plain addition loses order,
but a linear recurrence with recurrent mixing need not.
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
The [optional LSTM forward walkthrough](01A1-sequence-memory-worked-details.ipynb#lstm-complete-read) follows
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
Seven recorded HTML players are stored in notebook display outputs; the
three-case gradient player and LSTM forward walkthrough are embedded directly
in Markdown cells, with no code-input box.
None uses a relative iframe URL. Each runs in its own sandboxed frame, so its
controls cannot collide with another replay. PNG frames are packed as lossless
WebP: every pixel and all playback/audio timing remain unchanged.
The three-case gradient player uses self-contained HTML and JavaScript.
Its numeric recordings follow the same repeated `D3 A3 D4` phrase. Chosen scalar controls expose
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
are notebook attachments, and nine saved HTML players appear beside their
explanations. **Trust the notebook outputs** to use interactive HTML; viewers
that block scripts or frames can use the PNG/GIF fallbacks. The short hidden
display cells only read saved players, not generate images or animations.
The three-case gradient player and LSTM forward walkthrough are Markdown
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
.\.venv\Scripts\python.exe .\gradient_discovery.py
.\.venv\Scripts\python.exe .\memory_gate_flows.py
.\.venv\Scripts\python.exe .\backprop_memory_flow.py
.\.venv\Scripts\python.exe .\gradient_memory.py
.\.venv\Scripts\python.exe .\gradient_cases.py
.\.venv\Scripts\python.exe .\lstm_state_roles.py
.\.venv\Scripts\python.exe .\lstm_layer_flow.py
.\.venv\Scripts\python.exe .\lstm_forward_flow.py
.\.venv\Scripts\python.exe .\gru_forward_flow.py
.\.venv\Scripts\python.exe .\musical_memory.py
.\.venv\Scripts\python.exe -m unittest test_memory_gate_flows test_musical_memory test_gru_layer_flow test_lstm_layer_flow -v
.\.venv\Scripts\python.exe -m unittest test_memory_visuals -v
.\.venv\Scripts\python.exe -m unittest test_notebook_replays -v
.\.venv\Scripts\python.exe -m unittest test_backprop_memory_flow -v
.\.venv\Scripts\python.exe -m unittest test_lstm_state_roles -v
.\.venv\Scripts\python.exe -m unittest test_gradient_memory -v
.\.venv\Scripts\python.exe -m unittest test_error_feedback_flow -v
.\.venv\Scripts\python.exe -m unittest test_lstm_forward_flow -v
.\.venv\Scripts\python.exe -m unittest test_rnn_gradients test_gradient_discovery -v
.\.venv\Scripts\python.exe -m unittest test_gradient_cases -v
```

The proof exporter uses the bundled **Kalam** font; viewing
the embedded PNGs needs no installed font. Run it before the gate-flow
exporter, which refreshes the notebook's image attachments. The equations
are rendered by Matplotlib, not an image model;
the aligned calculation rows reuse that notation for actual measured values.

The gate-flow generator saves feedback-decay, LSTM-state and retention-comparison
charts alongside the paired gate replays. It uses the same recorded arithmetic as
the lesson and adds source fingerprints; it never trains a model. The musical
generator trains both tiny cue-echo models and exports measured guesses and state
arithmetic, audio, and playback timing with provenance metadata. Neither downloads data.
The backprop exporter refreshes only its own captures and marked player.
The other explicit exporters refresh notebook PNG/GIF attachments and the nine marked
HTML-player embeds. They preserve learning prose/code, experiment outputs,
and unrelated metadata. Player snapshots include source/output hashes so a
rebuild refreshes changed saved players rather than leaving stale HTML.
The gradient exporter rebuilds its five measured figures and the quiet
section-2 storyboard without changing model weights or running the echo
trainer. It refreshes only its notebook image attachments while preserving
experiment sources, outputs and players. Viewing the
saved notebook requires no separate image files.
The [three-case generator](gradient_cases.py) records its values through
[the shared RNN probe](error_feedback_flow.py). Edit the generator, not the
generated HTML. Run it before refreshing embeds. It records real forward,
backward and optimizer calculations, not a training curve. The previous
slider exporter remains standalone for regression checks; it is not embedded.
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
