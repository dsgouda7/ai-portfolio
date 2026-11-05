# Contents: Sequence Memory

Follow an early clue through increasingly demanding memory tasks.

## Required route

- [Chapter 01: Sequence Memory](README.md) — Check the three-notebook route and which text-input ideas are previews for the next chapter.
- [01A - Sequence Memory Theory](01A-sequence-memory-theory.ipynb) — Follow order and fading feedback into a complete LSTM story, then [GRU by contrast](01A-sequence-memory-theory.ipynb#gru-hidden-state-mechanics): real cells, audible echoes, and a concise encoder-decoder/attention boundary.
- [01B - Sequence Memory PyTorch Lab](01B-sequence-memory-pytorch-lab.ipynb) — Shift exact note pairs, measure the padded-state failure, separate packing from loss masking, and edit a future note to test causal access.
- [01C - Cinematic Piano Memory](01C-cinematic-piano-memory.ipynb) — Fit a phrase in two keys, swap only the opening cue, and hear and measure what happens at the ending.
- [Sequence Memory: Handwritten Theory Notes](01A-sequence-memory-theory.md) — Review why running memory forgets and how controlled memory gates help preserve useful signals.

## Visuals

- [Illustrated stories](images) — Static sketches with the same note names, cards, colors, and plain-English distinctions used in the notebooks.
- [Saved-card backprop replay](images/01a-backprop-saved-cards.html) · [MAI storyboard](images/01a-backprop-saved-cards.png) · [recorder/exporter](backprop_memory_flow.py) · [checks](test_backprop_memory_flow.py) — Inspect h3, h2, h1 before one shared-weight update; contrast full feedback with a detach boundary that preserves forward cards.
- [Cinematic music studio](sequence_studio.py) — Show every read's carried h and exposed h-to-head prediction branch with real initialized next-note guesses, piano-key inputs and flowing memory ribbons.
- [Local diagram generator](generate_memory_diagrams.py) — Rebuild the sketches without a model or network; measured outputs still come from executing the lessons.
- [LSTM/GRU gate-flow generator](memory_gate_flows.py) — Rebuild matching animations, controllable replays and static captures from the lesson's actual recorded state arithmetic.
- [Gate-flow checks](test_memory_gate_flows.py) — Verify carry/replacement/exposure differences, reset-route consequences, GRUCell parity and source-aware capture contracts.
- [One-token GRU layer](images/01a-gru-token-layer-flow.html) · [shared recorder](gru_layer_flow.py) · [checks](test_gru_layer_flow.py) — Open the input/state Linear projections and show why output and next-token hidden have the same values in one layer.
- [One-token LSTM layer](images/01a-lstm-token-layer-flow.html) · [recorder/exporter](lstm_layer_flow.py) · [checks](test_lstm_layer_flow.py) — Follow input, h and c through real gates; expose h and carry both states.
- [Audible LSTM walkthrough](images/01a-musical-lstm-memory.html) — Fit the same delayed echo, hear actual guesses, and erase both h and c.
- [Audible GRU walkthrough](images/01a-musical-gru-memory.html) — Hear actual guesses before/after fitting and after erasing opening memory; synchronized piano keys and real gate/state values.
- [Musical experiment and exporter](musical_memory.py) · [checks](test_musical_memory.py) — Rebuild offline audio/animation, verify both echoes, state interventions, named pitches, timing, and notebook/helper parity.
- [Memory versus attention](images/01a-memory-vs-attention.png) — Conceptual final-summary versus source-position look-back; access is not equal weighting.
- [Motion-graphics renderer](memory_visuals.py) · [visual checks](test_memory_visuals.py) — Staged memory cards, per-feature gate meters, moving value markers, and measured text/connector spacing.

**Next:** [Chapter 02 — Tokenization and Embeddings](../02-tokenization-and-embeddings/README.md).
