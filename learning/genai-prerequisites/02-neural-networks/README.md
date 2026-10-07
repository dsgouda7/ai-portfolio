# Chapter 02: Neural Networks and Backpropagation

The required chapter is theory-first. It uses XOR to force a hidden representation, then follows
one loss backward into a four-weight responsibility report and verifies that report with finite
differences.

## Required Notebook

### [SmartVal: Why Hidden Layers and Backpropagation Exist](01-smartval-neural-networks-and-backprop.ipynb)

**Owns:** XOR failure, hidden representations, ReLU gates, chain-rule responsibility, gradient
verification, and concise boundaries for depth, width, dropout, and normalization.

It deliberately does not teach another framework training loop. Those mechanics begin in the
canonical PyTorch route.

## Optional Audible Capstone

### [Melodyne Backprop Lab](optional/02-melodyne-backprop-synthesizer.ipynb)

**Preserves:** differentiable rendering, the four-gradient audit, recovery gates, and audible
evidence. It is useful after SmartVal, but it is not required to continue.

## Setup

Run the setup script in this directory:

- Windows: `./setup.ps1`
- Linux or macOS: `bash ./setup.sh`

The script installs the shared dependencies and assigns the `neural-networks` kernel to the
notebooks in this chapter.

## Chapter Handoff

Continue to
[PyTorch for LLMs 00: The Deep-Learning Map](../../pytorch-for-llms/00-deep-learning-map/00-deep-learning-map.ipynb).
The manual responsibility report becomes a real `loss.backward()` call there.
