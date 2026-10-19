# Contents: Chapter 03 - Language Models

Follow the failures that lead from counting continuations to a writer that can look back at the input.

## Required route

- [Chapter 03 Route Guide](README.md) — Follow the four lessons from fetching word rows to a writer that can look back.
- [03A - Training Skeleton and Word Objectives](03A-training-skeleton-and-word-objectives.ipynb) — Make practice questions from text, train word rows and a predictor together, and reuse one training loop.
- [03A - Theory Notes](03A-training-skeleton-and-word-objectives-theory.md) — Keep lookup, word guessing, running notes, and reading/writing jobs straight.
- [Counts and Fixed-Window Language Models](03B-counts-and-fixed-window-language-models.ipynb) — Compare counted continuations with a learned fixed-size text window and expose a clue that neither window can see.
- [Counts and Fixed-Window Theory Notes](03B-counts-and-fixed-window-language-models-theory.md) — Review why learning reusable features does not remove a fixed context boundary.
- [Fixed Windows to Recurrent Language Models](03C-fixed-windows-to-recurrent-language-models.ipynb) — Reuse the same text task to compare a fixed window with a running memory and reconnect to the earlier memory lesson.
- [Fixed Windows to Recurrent Models: Theory Notes](03C-fixed-windows-to-recurrent-language-models-theory.md) — Review what a running text memory changes and why it still need not retain every early clue.
- [Recurrent Reader-and-Writer Models to Attention](03D-recurrent-encoder-decoder-to-attention.ipynb) — Compare a writer given one learned summary with a writer that can inspect every source position during practice and generation.
- [Recurrent Reader-and-Writer Models to Attention: Theory Notes](03D-recurrent-encoder-decoder-to-attention-theory.md) — Review why looking back at source notes differs from remembering one summary and from receiving correct earlier words during practice.

## Optional

No optional lessons are listed in this directory.

## Code-only companions

- [Common training base](language_model.py) — Keep one practice loop while each model chooses how to guess and which answers to grade.
- [03A - Python](03A-training-skeleton-and-word-objectives.py) — CBOW, Skip-gram, and the small encoder-decoder preview.
- [03B - Python](03B-counts-and-fixed-window-language-models.py) — Counts and an ordered fixed-window predictor.
- [03C - Python](03C-fixed-windows-to-recurrent-language-models.py) — Original RNN experiments plus inherited LSTM training and generation.
- [03D - Python](03D-recurrent-encoder-decoder-to-attention.py) — Fixed-summary and attention experiments plus an inherited-training adapter.
