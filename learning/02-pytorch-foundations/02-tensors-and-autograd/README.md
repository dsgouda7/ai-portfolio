# Part 02: Tensors and Autograd

- [Notebook](02-tensors-and-autograd.ipynb)
- [Handwritten theory](02-tensors-and-autograd-theory.md)

This chapter merges the external tensor/autograd route with the repository's stronger shape-first
evidence: named axes, visible failures, reshape/transpose distinctions, broadcasting repairs,
matrix contraction, gradient accumulation, and an LM-shaped graph.

Run `./setup.ps1` on Windows or `bash ./setup.sh` on Linux/macOS.

## Required Bounded Intervention

Run **“Your turn: change one known axis”** in
[`02-tensors-and-autograd.ipynb`](02-tensors-and-autograd.ipynb). Predict the head width, change only
`YOUR_HEADS`, measure the printed before/after shapes, diagnose divisibility failures if any, and
explain whether the evidence supports the claim that reshaping changes the axis interpretation
without changing the feature count.
