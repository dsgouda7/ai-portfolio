# Contents: Chapter 03 - Language Models

Follow the failures that lead from counting continuations to a writer that can look back at the input.

## Required route

- [Chapter 03 Route Guide](README.md) — Follow the four lessons from fetching word rows to a writer that can look back.
- [03A - Training Skeleton and Word Objectives](03A-training-skeleton-and-word-objectives.ipynb) — Follow the saved window storyboard in section 2 and before/after word profiles in section 3; section 4 adds optional Skip-gram, or jump to the [section 5 encoder-decoder preview](03A-training-skeleton-and-word-objectives.ipynb#encoder-decoder-preview), followed by the independent three-model comparison.
- [03A - Theory Notes](03A-training-skeleton-and-word-objectives-theory.md) — Keep lookup, word guessing, running notes, and reading/writing jobs straight.
- [Counts and Fixed-Window Language Models](03B-counts-and-fixed-window-language-models.ipynb) — Compare counted continuations with a learned fixed-size text window and expose a clue that neither window can see.
- [Counts and Fixed-Window Theory Notes](03B-counts-and-fixed-window-language-models-theory.md) — Review why learning reusable features does not remove a fixed context boundary.
- [Fixed Windows to Recurrent Language Models](03C-fixed-windows-to-recurrent-language-models.ipynb) — Reuse the same text task to compare a fixed window with a running memory and reconnect to the earlier memory lesson.
- [Fixed Windows to Recurrent Models: Theory Notes](03C-fixed-windows-to-recurrent-language-models-theory.md) — Review what a running text memory changes and why it still need not retain every early clue.
- [Recurrent Reader-and-Writer Models to Attention](03D-recurrent-encoder-decoder-to-attention.ipynb) — Compare a writer given one learned summary with a writer that can inspect every source position during practice and generation.
- [Recurrent Reader-and-Writer Models to Attention: Theory Notes](03D-recurrent-encoder-decoder-to-attention-theory.md) — Review why looking back at source notes differs from remembering one summary and from receiving correct earlier words during practice.

## Code-only companions

- [Common training base](language_model.py) — Keep one practice loop while each model chooses how to guess and which answers to grade.
- [Word-data loader contracts](language_data.py) — Keep shared window validation and seeded batching while CBOW and Skip-gram supply their own aligned input/target pairs.
- [Engineering contract checks](test_language_contracts.py) — Verify loader shapes/order, abstract hooks, training-step order, and explicit failure behavior offline.
- [Validation comparisons](language_validation.py) — Evaluate CBOW, Skip-gram, and reader/writer answers through one read-only pipeline; compare each model before and after its training run.
- [Validation contract checks](test_language_validation.py) — Verify checking-answer counts, target alignment, before/after reports, and preserved model state offline.
- [Encoder-decoder embedding feedback](encoder_decoder_feedback.py) — Animate actual cat-probe reset/update gates, kept/written hidden-state arithmetic, state/word chains and feedback; rebuild the GIF, replay, gate close-up and static route.
- [Recurrent feedback checks](test_encoder_decoder_feedback.py) — Verify state/gradient parity, untouched main-model state, and why source-only embeddings need the encoder-to-decoder backward route.
- [03A - Python](03A-training-skeleton-and-word-objectives.py) — Visible CBOW training, section 4 Skip-gram, section 5 reader/writer training, and the retained story-trained embedding comparison.
- [CBOW static-capture generator](generate_cbow_static_views.py) — Rebuild the saved word-profile comparison and window storyboard from the actual core notebook states.
- [Real-text experiment helper](real_text_embeddings.py) — Cache the checked TinyStories file, build practice questions, and record losses and embedding diagnostics.
- [Window and objective checks](test_real_text_embeddings.py) — Check document boundaries, short documents, unknown slots, and each objective's question/answer alignment offline.
- [03B - Python](03B-counts-and-fixed-window-language-models.py) — Counts and an ordered fixed-window predictor.
- [03C - Python](03C-fixed-windows-to-recurrent-language-models.py) — Original RNN experiments plus inherited LSTM training and generation.
- [03D - Python](03D-recurrent-encoder-decoder-to-attention.py) — Fixed-summary and attention experiments plus an inherited-training adapter.

**Next:** [Stage 04 — Transformers](../../04-transformers/README.md).
