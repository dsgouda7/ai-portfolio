# Optional Framework Translation

**Contents:** [one-line map of this directory](CONTENTS.md)



[Keras to PyTorch: Antarctic Field Guide](01-keras-to-pytorch-antarctic-field-guide.ipynb)
preserves the framework-translation evidence from the former required route:

- Keras/PyTorch vocabulary;
- dense-layer weight-layout parity;
- `GradientTape` versus autograd;
- `fit()` versus the explicit PyTorch loop;
- evaluation-mode differences.

Use it only when translating existing Keras knowledge. The canonical route is
[Stage 02: PyTorch Foundations](../../01-deep-learning-map/).

This optional branch was extracted from the former
`learning/genai-prerequisites/03-pytorch-fundamentals/optional` directory. Its notebook,
five visual assets, dependency manifest, and setup scripts moved together.
