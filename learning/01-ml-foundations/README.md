# Stage 01: ML Foundations

Start with a free kick, then use SmartVal to ask which predictions to trust and how a small
weight change would affect an error. This stage builds the intuition; Stage 02 gives the
training state and repeated updates explicit homes in PyTorch.

**Contents:** [lesson map](CONTENTS.md). Each chapter has its own setup scripts.

## Required route

1. [Math Foundations](01-math-foundations/README.md)
2. [ML Basics](02-ml-basics/README.md)
3. [Neural Networks and Backpropagation](03-neural-networks-and-backprop/README.md)

**Optional:** [audible backpropagation lab](03-neural-networks-and-backprop/optional/README.md),
after Chapter 03. It does not block the required route.

The examples stay small on purpose: the kick follows known physics; the rental listings
are synthetic; XOR's useful features are hand-chosen. The optional synth actually optimizes
four controls. Each notebook separates what its code checks from what it only explains.

**Next:** [Stage 02 — PyTorch Foundations](../02-pytorch-foundations/README.md).
You can trace an error back to its weights. Next, give those weights, gradients, and updates
explicit homes in code.
