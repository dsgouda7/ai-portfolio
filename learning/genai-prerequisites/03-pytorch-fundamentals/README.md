# Legacy PyTorch Bridges

The required PyTorch route now lives in
[PyTorch for LLMs 00-04](../../pytorch-for-llms/00-deep-learning-map/). This directory preserves
evidence that has not yet reached its final optional home.

## Preserved Shape and Sequence Evidence

- [Arrays to Model Tensors](00-arrays-to-model-tensors.ipynb)
- [Handwritten theory](00-arrays-to-model-tensors-theory.md)

Its table-shape, indexing, reshape, transpose, broadcasting, and matrix-operation evidence has
been merged into required Part 01. Its sequence-window evidence remains here until the approved
sequence-memory and tokenization chapters are implemented.

## Optional Keras Translation

- [Keras to PyTorch: Antarctic Field Guide](optional/01-keras-to-pytorch-antarctic-field-guide.ipynb)

This notebook preserves Keras/PyTorch vocabulary, `fit()` versus explicit-loop comparison, and
evaluation-mode differences. It is no longer on the required route.

## Setup

Run `setup.ps1` on Windows or `setup.sh` on Linux/macOS for these preserved notebooks. New
learners should instead use the setup script in each canonical PyTorch chapter.
