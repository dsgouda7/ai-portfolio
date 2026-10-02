# PyTorch Fundamentals

This chapter closes the gap between labeled data and framework code in two steps:

1. [Arrays to Model Tensors](00-arrays-to-model-tensors.ipynb) · [Theory notes](00-arrays-to-model-tensors-theory.md) starts with NumPy indexing, slicing, reshape, transpose, broadcasting, and matrix multiplication, then proves the same classifier and sequence-input contracts in TensorFlow and PyTorch.
2. [Keras to PyTorch: Antarctic Field Guide](01-keras-to-pytorch-antarctic-field-guide.ipynb) carries those shape and dtype contracts into explicit PyTorch modules, autograd, optimization, and inference.

Complete both notebooks in order. The first makes every axis visible; the second makes the training state visible.

Run `setup.ps1` on Windows or `setup.sh` on Linux/macOS. Either script creates or reuses this chapter's `.venv`, installs the adjacent `requirements.txt`, registers the chapter-unique `genai-prereq-03-pytorch` Jupyter kernel, and assigns it to both notebooks.
