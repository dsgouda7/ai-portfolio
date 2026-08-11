# Chapter 02: Neural Networks and Backpropagation

This chapter uses two self-contained examples. SmartVal establishes the neural-network mechanics; Melodyne revises backpropagation through an audible parameter-recovery problem.

## Recommended Order

### 1. [SmartVal Neural Networks and Backpropagation](01-smartval-neural-networks-and-backprop.ipynb)

**Owns:** nonlinear hidden representations, manual backpropagation, `tf.GradientTape`, depth versus width, and training versus inference behavior.

SmartVal remains one connected case study because its later experiments reuse the dense-network vocabulary and training loop established by XOR. The added forward/backward animation makes the responsibility route visible without replacing the manual derivative check.

### 2. [Melodyne Backprop Synthesizer](02-melodyne-backprop-synthesizer.ipynb)

**Owns:** differentiable audio rendering, finite-difference gradient verification, and learning interpretable synth controls from target audio.

The notes are fixed while bass, mid, treble, and drive are learned. This isolates backpropagation from sequence generation: the later Melodyne RNN notebook predicts what comes next, while this notebook learns how known notes should sound.

## Setup

Run the setup script in this directory:

- Windows: `./setup.ps1`
- Linux or macOS: `bash ./setup.sh`

The script installs the shared dependencies and assigns the `neural-networks` kernel to both notebooks.

## Chapter Handoff

After both notebooks, continue to [Keras to PyTorch: Antarctic Field Guide](../03-pytorch-fundamentals/01-keras-to-pytorch-antarctic-field-guide.ipynb). The framework vocabulary changes, but forward pass, loss, backpropagation, and optimizer responsibilities remain the same.
