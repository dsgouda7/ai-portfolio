# Optional Framework Translation

[Keras to PyTorch: Antarctic Field Guide](01-keras-to-pytorch-antarctic-field-guide.ipynb)
preserves the framework-translation evidence from the former required route:

- Keras/PyTorch vocabulary;
- dense-layer weight-layout parity;
- `GradientTape` versus autograd;
- `fit()` versus the explicit PyTorch loop;
- evaluation-mode differences.

Use it only when translating existing Keras knowledge. The canonical route is
[PyTorch for LLMs 00-04](../../../pytorch-for-llms/00-deep-learning-map/).
