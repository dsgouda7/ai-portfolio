# PyTorch for LLMs

This is the required code-first spine between the conceptual GenAI prerequisites and the
Transformer track. It teaches one canonical PyTorch implementation path instead of repeating the
same mechanics in NumPy, Keras, and PyTorch.

Complete the chapters in order:

| Part | Chapter | What becomes routine |
|---|---|---|
| 00 | [Deep-Learning Map](00-deep-learning-map/00-deep-learning-map.ipynb) | Trace data through `nn.Module`, forward, loss, backward, optimizer, epochs, and inference |
| 01 | [Tensors and Autograd](01-tensors-and-autograd/01-tensors-and-autograd.ipynb) | Read shape, dtype, device, broadcasting, matrix multiplication, computation graphs, and gradient state |
| 02 | [Models and Training Loops](02-models-and-training-loops/02-models-and-training-loops.ipynb) | Move from visible tensor math to `nn.Module` without hiding the update lifecycle |
| 03 | [Data, Checkpoints, and Hyperparameters](03-data-checkpoints-and-hyperparameters/03-data-checkpoints-and-hyperparameters.ipynb) | Use `Dataset`, `DataLoader`, state dictionaries, resumable state, and controlled comparisons |
| 04 | [Modern Training Lifecycle](04-modern-training-lifecycle/04-modern-training-lifecycle.ipynb) | Separate training, validation, selection, resume, mixed precision, and inference contracts |
| 05A | [Sequence-Memory Theory](05-sequence-memory/05A-sequence-memory-theory.ipynb) | Explain recurrence, BPTT, vanishing gradients, and LSTM gates |
| 05B | [Sequence-Memory PyTorch Lab](05-sequence-memory/05B-sequence-memory-pytorch-lab.ipynb) | Build forecasting and token-sequence models with packing, masked loss, invariance checks, and clipping |
| 05C | [Cinematic Piano Memory](05-sequence-memory/05C-cinematic-piano-memory.ipynb) | Hear and measure the difference between short and durable recurrent memory |
| 06A | [Tokenization and Embeddings Theory](06-tokenization-and-embeddings/06A-tokenization-and-embeddings-theory.ipynb) | Explain BPE, token addresses, embedding rows, padding, and masking failures |
| 06B | [Tokenization and Embeddings PyTorch Lab](06-tokenization-and-embeddings/06B-tokenization-and-embeddings-pytorch-lab.ipynb) | Implement BPE evidence, `nn.Embedding`, padding masks, `padding_idx`, and ignored loss targets |

After Part 06, continue to [Transformer Foundations](../genai/01-transformers/README.md).

## Optional Revision Branches

- [Embedding history](optional/embedding-history/README.md): one-hot and bag-of-words through
  Word2Vec, GloVe, and the static/contextual boundary.
- [Vision](../genai-prerequisites/optional-vision/README.md): PyTorch CNNs and autoencoders for
  learners continuing into multimodal work.
- Transformer Foundations provides optional Lightning and AI-engineering revision notebooks after
  the manual training path is understood.

## Setup

Run setup from the chapter directory you are about to use. Each setup script creates or reuses a
chapter-local `.venv`, installs the adjacent `requirements.txt`, registers a chapter-specific
kernel, and assigns that kernel to the chapter notebooks.

On Windows:

```powershell
.\setup.ps1
```

On Linux or macOS:

```bash
bash ./setup.sh
```

The notebooks are committed without outputs or execution counts. Setup and execution artifacts
remain local.
