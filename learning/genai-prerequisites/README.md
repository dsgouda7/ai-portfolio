# GenAI Prerequisites

This sequence builds the mathematical, machine-learning, framework, and sequence contracts required by the GenAI track. Complete the chapters in order unless the route below marks a branch optional.

## Core Route

| # | Chapter | Primary framework | Outcome |
|---|---|---|---|
| 00 | [Math Foundations](00-math-foundations/math-foundations-for-ml.ipynb) | NumPy + SciPy | Build motion from local change, read vectors and probability, then use local derivatives for constrained gradient descent |
| 01 | [ML Basics](01-ml-basics/ml-basics.ipynb) | NumPy + scikit-learn | Build train/validation/test, optimization, classification, and generalization contracts |
| 02 | [Neural Networks and Backpropagation](02-neural-networks/neural-networks-and-backprop.ipynb) | TensorFlow/Keras | Build dense networks, derive backpropagation, and distinguish train from inference behavior |
| 03 | [Keras to PyTorch: Antarctic Field Guide](03-pytorch-fundamentals/01-keras-to-pytorch-antarctic-field-guide.ipynb) | TensorFlow/Keras ↔ PyTorch | Translate familiar model, loss, autograd, optimizer, dtype, and device contracts into PyTorch |
| 04 | [Convolutional Neural Networks](04-cnns/convolutional-neural-networks.ipynb) | TensorFlow/Keras ↔ PyTorch | Optional vision branch: convolution, receptive fields, residual paths, and transfer learning |
| 05 | [RNN/LSTM Sequence Modeling](05-rnn-sequence-modeling/rnn-sequence-modeling.ipynb) | TensorFlow/Keras ↔ PyTorch | Derive recurrent state, BPTT, vanishing gradients, and LSTM gating |
| 06 | [Tokenization and Embeddings](06-tokenization/tokenization-and-embeddings.ipynb) | TensorFlow/Keras ↔ PyTorch | Build BPE intuition, train embedding rows, and handle padding and masked loss |
| 07 | [PyTorch RNN Bridge](07-pytorch-rnn-bridge/01-pytorch-rnn-bridge.ipynb) | TensorFlow/Keras ↔ PyTorch | Carry the sequence contract into PyTorch before attention replaces recurrence |

For the language-model route, chapter 04 is optional: follow `00 → 01 → 02 → 03 → 05 → 06 → 07`. Complete chapter 04 when vision, multimodal work, or convolution-heavy systems are relevant.

## Chapter Setup

Run setup from each chapter directory you plan to use. On Windows run `.\setup.ps1`; on Linux or macOS run `bash ./setup.sh`. Each script creates or reuses the chapter-local `.venv`, installs dependencies, registers the chapter kernel, and assigns that kernelspec to the chapter notebook.

## Comparison Policy

Every notebook contains at least one compact TensorFlow/Keras ↔ PyTorch comparison block after the underlying concept is taught. These blocks emphasize invariant computation and the API or tensor-layout difference most likely to cause translation bugs. TensorFlow-first notebooks keep PyTorch snippets in Markdown so their executable dependency set remains unchanged; chapters 03 and 07 provide the runnable PyTorch practice.

After chapter 07, continue to [Transformer Foundations](../genai/01-transformers/README.md).
