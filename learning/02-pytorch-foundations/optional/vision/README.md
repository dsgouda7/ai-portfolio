# Optional Vision

**Contents:** [one-line map of this directory](CONTENTS.md)



Vision is useful revision, but it is not a prerequisite for the LLM route.

1. [PyTorch CNNs](01-pytorch-cnns.ipynb) preserves convolution, Sobel filters, stride/pooling, feature maps, receptive fields, trained-filter specialization, and the ResNet gradient proof.
2. [PyTorch Autoencoders](02-pytorch-autoencoders.ipynb) follows with encoder/decoder reconstruction, latent-space inspection, and compression evidence.

Both notebooks generate small synthetic image datasets locally, so execution does not depend on a network download. VAE behavior is named as optional reading only; this branch does not pretend an ordinary autoencoder is variational.

Run `setup.ps1` on Windows or `setup.sh` on Linux/macOS. The script creates or reuses this branch's `.venv`, installs `requirements.txt`, registers the unique `pytorch-foundations-optional-vision` kernel, and assigns it to both notebooks. Use `-SkipKernel` or `--skip-kernel` when only the local environment and dependencies are needed.

This branch moved intact from the former `learning/genai-prerequisites/optional-vision`
location. The notebooks remain optional; they are not the next required step after Stage 02.


After reconstruction, return to [Sequence Memory](../../../03-language-model-foundations/01-sequence-memory/README.md) if you are following the language route. The unresolved complaint is ordered history, not another image architecture: a frame's compact code does not by itself remember earlier frames.
