# Contents: Chapter 03 - Language Models

Follow the failures that lead from counting continuations to a writer that can look back at the input.

## Required route

- [Chapter 03 Route Guide](README.md) — Follow the three lessons from fetching word rows to a writer that can look back.
- [03A - Training Skeleton and Word Objectives](03A-training-skeleton-and-word-objectives.ipynb) — Follow saved window/profile evidence, optional Skip-gram, and the [compact encoder-decoder preview](03A-training-skeleton-and-word-objectives.ipynb#encoder-decoder-preview): feedback teaches a shared input table; recurrent order leads to an attention prelude.
- [03A - Theory Notes](03A-training-skeleton-and-word-objectives-theory.md) — Keep lookup, word guessing, running notes, and reading/writing jobs straight.
- [03B - Counts to Recurrent Language Models](03B-counts-to-recurrent-language-models.ipynb) — Follow controlled cat/dog continuations from counts to ordered windows and recurrent memory, then check a word LSTM's training and causality.
- [03B - Theory Notes](03B-counts-to-recurrent-language-models-theory.md) — Review what order and carried memory fix, what the state intervention demonstrates, and which limits remain.
- [03C - Recurrent Reader-and-Writer Models to Attention](03C-recurrent-encoder-decoder-to-attention.ipynb) — Follow the reader/writer handoff, how the answer teaches the encoding and teacher forcing; three compact code cells train the fixed-summary model. End with the need for attention and links to the dedicated Transformer lessons, not an attention implementation.
- [03C - Theory Notes](03C-recurrent-encoder-decoder-to-attention-theory.md) — Review why looking back at source notes differs from remembering one summary and from receiving correct earlier words during practice.

## Optional comparison lab

- [03C1 - Encoder-Decoder Worked Details](03C1-encoder-decoder-worked-details.ipynb) · [Python](03C1-encoder-decoder-worked-details.py) — Preserve the original full `ReaderWriter`, initialized-role and teacher-input probes, gradient interventions, encoder-only proof, two-model attention benchmark, held-out evaluation and inherited-trainer adapter.
- [Word-objective comparison lab](03A-word-objective-comparison-lab.ipynb) · [Python](03A-word-objective-comparison-lab.py) — Run the original three-model experiment independently; inspect geometry, held-out curves, shortlists and best/final checkpoints. The appendix preserves the full teacher-forcing reference.
- [Lab checks](test_word_objective_comparison_lab.py) — Verify fresh-kernel dependencies, companion parity and exact training equivalence with the inline model definitions.

## Code-only companions

- [Common training base](language_model.py) — Keep one practice loop while each model chooses how to guess and which answers to grade.
- [Word-data loader contracts](language_data.py) — Keep shared window validation and seeded batching while CBOW and Skip-gram supply their own aligned input/target pairs.
- [Engineering contract checks](test_language_contracts.py) — Verify loader shapes/order, abstract hooks, training-step order, and explicit failure behavior offline.
- [Validation comparisons](language_validation.py) — Evaluate CBOW, Skip-gram, and reader/writer answers through one read-only pipeline; compare each model before and after its training run.
- [Validation contract checks](test_language_validation.py) — Verify checking-answer counts, target alignment, before/after reports, and preserved model state offline.
- [Encoder-decoder embedding feedback](encoder_decoder_feedback.py) — Rebuild saved shared-table feedback, same-run input-row evidence and measured order/attention storyboard; retain optional token-layer and teacher-forcing replays.
- [Recurrent feedback checks](test_encoder_decoder_feedback.py) — Verify state/gradient parity, untouched main-model state, and why source-only embeddings need the encoder-to-decoder backward route.
- [03A - Python](03A-training-skeleton-and-word-objectives.py) — Visible CBOW, optional Skip-gram and compact reader/writer training; the larger comparison no longer runs here.
- [CBOW static-capture generator](generate_cbow_static_views.py) — Rebuild the saved word-profile comparison and window storyboard from the actual core notebook states.
- [Shared cat-story illustrations](generate_model_evolution_views.py) — Rebuild all six counts/window/memory/source-access sketches from the canonical word sequence; conceptual sketches do not invent measured values.
- [Real-text experiment helper](real_text_embeddings.py) — Cache the checked TinyStories file, build practice questions, and record losses and embedding diagnostics.
- [Window and objective checks](test_real_text_embeddings.py) — Check document boundaries, short documents, unknown slots, and each objective's question/answer alignment offline.
- [03B - Python](03B-counts-to-recurrent-language-models.py) — One shared color corpus and window baseline, followed by RNN evidence and inherited LSTM training and generation.
- [03C - Python](03C-recurrent-encoder-decoder-to-attention.py) — Only the main notebook's three cells: frozen data, small encoder-decoder and training.
- [Encoder-decoder example data](encoder_decoder_example.py) — Load the checked [frozen source split](data/03c-cat-reversal-corpus.json), convert word IDs, shift previous-answer inputs, replay matched batches and retain the former main section's bounded generation helper; no model or hidden training loop.
- [Encoder-decoder walkthrough recorder](encoder_decoder_walkthrough.py) — Offline recorder/exporter: run the main three-cell model once and capture its actual states, predictions, probabilities, gradients and updates for the paused, step-through player. Notebook Run All does not export the recording.
- [Encoder-decoder background prompts](images/03c-encoder-decoder-animation-prompts.json) — Describe the decorative background; measured overlays come from the offline recorder, not image generation.
- [Shared cat-example checks](test_shared_cat_examples.py) — Preserve original source-access and gradient behavior; verify the compact model's explicit training shift, separate source reading, carried generation state and EOS/budget stopping.

**Next:** [Stage 04 — Transformers](../../04-transformers/README.md).
