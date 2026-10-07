# GenAI Prerequisites

This sequence builds the mathematical, machine-learning, and backpropagation intuition required
before the code-first [PyTorch for LLMs](../pytorch-for-llms/README.md) spine.

## Core Route

| # | Chapter | Outcome |
|---|---|---|
| 00 | [Math Foundations](00-math-foundations/math-foundations-for-ml.ipynb) | Build motion from local change, read vectors and probability, then use local derivatives for constrained gradient descent |
| 01 | [ML Basics](01-ml-basics/ml-basics.ipynb) | Build train/validation/test, optimization, classification, and generalization contracts without duplicating a framework course |
| 02 | [Neural Networks and Backpropagation](02-neural-networks/README.md) | Expose XOR's linear failure, hidden representations, responsibility, and verified gradients |

Then complete [PyTorch for LLMs](../pytorch-for-llms/README.md) Parts 00-06.

## Optional Branches

- [Melodyne audible backpropagation](02-neural-networks/optional/README.md)
- [PyTorch vision: CNNs and autoencoders](optional-vision/README.md)
- [Keras-to-PyTorch translation reference](03-pytorch-fundamentals/optional/README.md)

The former CNN, Keras-first RNN/tokenization, arrays bridge, and RNN bridge directories remain as
migration references. Their required learning outcomes now have one canonical PyTorch owner.

## Chapter Setup

Run setup from each chapter directory you plan to use. On Windows run `.\setup.ps1`; on Linux or
macOS run `bash ./setup.sh`.

After the foundations, continue to [PyTorch for LLMs](../pytorch-for-llms/README.md).
