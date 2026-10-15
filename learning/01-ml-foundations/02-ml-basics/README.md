# ML Contracts

This condensed foundation chapter uses one SmartVal dataset to separate the contracts that are
often blurred together:

- train, validation, and test evidence;
- regression loss versus human-readable metrics;
- probability versus thresholded decisions;
- training fit versus held-out generalization.

## Required Notebook

- [ML Contracts: Evidence Before Optimization](ml-basics.ipynb)

The notebook uses one compact scikit-learn implementation. PyTorch mechanics are deliberately
deferred to [Stage 02: PyTorch Foundations](../../02-pytorch-foundations/README.md).

## Setup

- Windows: `./setup.ps1`
- Linux or macOS: `bash ./setup.sh`

## Handoff

Continue to [SmartVal: Why Hidden Layers and Backpropagation Exist](../03-neural-networks-and-backprop/01-smartval-neural-networks-and-backprop.ipynb).

## Required Bounded Intervention

Run **“Your turn: change one decision, not the whole experiment”** in
[`ml-basics.ipynb`](ml-basics.ipynb). Predict the precision/recall direction, change only
`YOUR_THRESHOLD`, measure both metrics before and after, diagnose the trade-off, and explain whether
the result supports the claim that a threshold changes decisions without retraining the model.
