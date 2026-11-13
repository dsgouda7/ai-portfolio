# Contents: Sequence Memory

Follow an early clue through increasingly demanding memory tasks.

## Required route

- [Chapter 01: Sequence Memory](README.md) — Check the three-notebook route and which text-input ideas are previews for the next chapter.
- [01A - Sequence Memory Theory](01A-sequence-memory-theory.ipynb) — Follow order and fading feedback into a complete LSTM story, then [GRU by contrast](01A-sequence-memory-theory.ipynb#gru-hidden-state-mechanics). Close with the [memory-design cheat code](01A-sequence-memory-theory.ipynb#sequence-memory-cheat-code); deeper boundaries remain optional.
- [01A1 - Optional worked details](01A1-sequence-memory-worked-details.ipynb) — The complete original arithmetic, experiments, detailed images and replays remain in their teaching order, with a contextual reading guide. Not another required chapter.
- [01B - Sequence Memory PyTorch Lab](01B-sequence-memory-pytorch-lab.ipynb) — Shift exact note pairs, measure the padded-state failure, separate packing from loss masking, and edit a future note to test causal access.
- [01C - Cinematic Piano Memory](01C-cinematic-piano-memory.ipynb) — Fit a phrase in two keys, swap only the opening cue, and hear and measure what happens at the ending.
- [Sequence Memory: Handwritten Theory Notes](01A-sequence-memory-theory.md) — Review why running memory forgets and how controlled memory gates help preserve useful signals.

In 01A, start with the [fourth-note prediction puzzle](01A-sequence-memory-theory.ipynb#memory-challenge),
use the [complete LSTM read](01A-sequence-memory-theory.ipynb#lstm-complete-read)
as the main forward view, and revisit the
[three-case gradient lesson](01A-sequence-memory-theory.ipynb#rnn-gradient-microscope)
before the three memory-design attempts. Both native cells precede the shared
music experiment; a concise RNN/LSTM/GRU revision closes the notebook.

## Visuals

- [Inline LSTM GIF](images/01a-lstm-forward-walkthrough.gif) · [companion-only interactive flow](01A1-sequence-memory-worked-details.ipynb#lstm-complete-read) · [static A3 read](images/01a-lstm-forward-walkthrough.png) · [recorder/exporter](lstm_forward_flow.py) · [checks](test_lstm_forward_flow.py) — Automatic, JavaScript-free playback shows each gate's job, c/h generation, h-only predictions, SHOW=0 without erasing c, and both states entering the next read. Exact arithmetic and manual stepping remain optional in the companion rather than repeating the main GIF.
- [Inline GRU GIF](images/01a-gru-forward-walkthrough.gif) · [static route map](images/01a-gru-forward-walkthrough.png) · [recorder/exporter](gru_forward_flow.py) · [checks](test_gru_layer_flow.py) — Matching spaced, cumulative playback shows the separate KEEP and RESET/proposal routes, complementary WRITE, the h-only prediction head and the same h carried onward. RESET=0 changes the proposal without erasing the kept amount.
- [Optional full LSTM forward/backward board](01A1-sequence-memory-worked-details.ipynb#lstm-fixed-board-details) · [optional full GRU board](01A1-sequence-memory-worked-details.ipynb#gru-forward-details) · [recorder/checks](gated_sequence_storyboards.py) — The deeper route retains stationary saved cards and actual backward feedback without crowding the main forward animations.
- [Shared original-music experiment](sequence_music_comparison.py) · [checks](test_sequence_music_comparison.py) — Three reusable models, the same whole-phrase split and budget, honest before/after held-out measurements and playable actual continuations.
- [Illustrated stories](images) — Static sketches with the same note names, cards, colors, and plain-English distinctions used in the notebooks.
- [Saved-card backprop replay](images/01a-backprop-saved-cards.html) · [studio storyboard](images/01a-backprop-saved-cards.png) · [recorder/exporter](backprop_memory_flow.py) · [checks](test_backprop_memory_flow.py) — Inspect h3, h2, h1 before one shared-weight update; contrast full feedback with a detach boundary that preserves forward cards.
- [Gradient-to-memory evidence](gradient_memory.py) · [checks](test_gradient_memory.py) — Build saturation, vanishing/exploding feedback and clipping intuition; fail with an unsquashed total, then measure protected LSTM c and GRU h routes alongside full-cell amplification stress tests.
- [One soft-squeeze comparison](images/01a-tanh-saturation.png) — A static curve and the same tiny nudge near its middle and flat edge explain saturation without a separate tanh tracker.
- [Optional worked-details gradient calculations](images/01a-gradient-three-cases.html) · [standalone snapshot](images/01a-gradient-three-cases.png) · [generator](gradient_cases.py) · [checks](test_gradient_cases.py) — Full numerical references remain available without repeating the recap, comparison table or snapshot in the main route.
- [Handwritten numerical working](handwritten_notes.py) · [MAI paper prompt](images/01a-handwritten-notes-prompts.json) · [font license](images/kalam/OFL.txt) — Shared white-paper styling for gradient and LSTM/GRU calculations. The paper is generated; all mathematical content remains tied to the real recorded state.
- [What a gradient means](01A-sequence-memory-theory.ipynb#gradient-what-if) gives the short bridge from memory sensitivity to changing weights. The [optional worked check in the companion](01A1-sequence-memory-worked-details.ipynb#memory-nudge-visual) holds the two-copy diagram and numerical comparison; its [recorder/exporter](gradient_what_if.py) remains available. The [shared-weight discovery](01A-sequence-memory-theory.ipynb#shared-gradient-check) carries the main derivation; [vector/head/bias calculations](01A1-sequence-memory-worked-details.ipynb#vector-backprop) remain optional detail.
- [Loss-to-weight proof](images/01a-rnn-gradient-discovery.png) · [longer-chain proof](images/01a-rnn-gradient-chain.png) · [recorder/exporter](gradient_discovery.py) · [checks](test_gradient_discovery.py) — The images carry the symbolic chain-rule proof with `softmax(scores)` kept compact. The text supplies the actual memory updates, probabilities, feedback contributions and weight changes, then checks fading/growing guidance numerically.
- [Full vector, head and bias calculations](01A1-sequence-memory-worked-details.ipynb#vector-backprop) · [numerical checks](test_rnn_gradients.py) — The companion retains the expanded recurrence and shared-weight reasoning; superseded standalone math cards and the dark reference sheet are retired.
- [Cinematic music studio](sequence_studio.py) — Show every read's carried h and exposed h-to-head prediction branch with real initialized next-note guesses, piano-key inputs and flowing memory ribbons.
- [Open a numerical notecard](images/01a-rnn-notecard-closeup.png) — Inspect the actual D4 input, shared weight matrices, incoming h2, their contributions, new h3 and separate next-note scores.
- [Local diagram generator](generate_memory_diagrams.py) — Rebuild the sketches without a model or network; measured outputs still come from executing the lessons.
- [LSTM/GRU gate-flow generator](memory_gate_flows.py) — Rebuild matching animations, controllable replays and static captures from the lesson's actual recorded state arithmetic.
- [Gate-flow checks](test_memory_gate_flows.py) — Verify carry/replacement/exposure differences, reset-route consequences, GRUCell parity and source-aware capture contracts.
- [One-token GRU layer](images/01a-gru-token-layer-flow.html) · [shared recorder](gru_layer_flow.py) · [checks](test_gru_layer_flow.py) — Open the input/state Linear projections and show why output and next-token hidden have the same values in one layer.
- [One-token LSTM layer](images/01a-lstm-token-layer-flow.html) · [recorder/exporter](lstm_layer_flow.py) · [checks](test_lstm_layer_flow.py) — Follow input, h and c through real gates; expose h and carry both states.
- [Cell-state memory timeline](images/01a-lstm-cell-vs-hidden.png) · [one full two-state step](images/01a-lstm-two-state-step.png) · [memory/role probes](lstm_state_roles.py) · [checks](test_lstm_state_roles.py) — First see c as one evolving numeric workspace, then distinguish it from h; piano inputs, folded cards and soft ribbons match the RNN studio without obscuring the numbers.
- [Illustrated LSTM forward pass](images/01a-lstm-forward-illustrated.png) — Replaces the introductory code block with one measured, hand-controlled D4 step: keep and write to make c, show h, carry both, and send only h to the note head.
- [Optional LSTM forward walkthrough](01A1-sequence-memory-worked-details.ipynb#lstm-complete-read) · [static route map](images/01a-lstm-forward-walkthrough.png) · [recorder](lstm_forward_flow.py) · [checks](test_lstm_forward_flow.py) — Follow the actual D3/A3/D4 reads: input and old h select controls; old c supplies contents; new c produces a gated h; only h enters the prediction head. Gold/cyan MAI role objects stay separate from the measured labels. Both two-feature states continue onward.
- [Audible LSTM walkthrough](images/01a-musical-lstm-memory.html) — Fit the same delayed echo, hear actual guesses, and erase both h and c.
- [Audible GRU walkthrough](images/01a-musical-gru-memory.html) — Hear actual guesses before/after fitting and after erasing opening memory; synchronized piano keys and real gate/state values.
- [Musical experiment and exporter](musical_memory.py) · [checks](test_musical_memory.py) — Rebuild offline audio/animation, verify both echoes, state interventions, named pitches, timing, and notebook/helper parity.
- [Memory versus attention](images/01a-memory-vs-attention.png) — Conceptual final-summary versus source-position look-back; access is not equal weighting.
- [Motion-graphics renderer](memory_visuals.py) · [visual checks](test_memory_visuals.py) — Staged memory cards, per-feature gate meters, moving value markers, and measured text/connector spacing.

**Next:** [Chapter 02 — Tokenization and Embeddings](../02-tokenization-and-embeddings/README.md).
