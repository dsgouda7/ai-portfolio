# Stage 01: ML Foundations

**Contents:** [one-line map of this directory](CONTENTS.md)



You can compute a prediction, but why should you trust it or know which adjustable number to change? Start with a free kick: small changes build a path. Then carry that habit into SmartVal, where a good training score can hide poor predictions on new examples.

**Ownership:** this stage owns optimization (changing model settings to reduce mistakes), generalization (working on unseen examples), neural networks (learned layers of features), and backpropagation (tracing each setting's effect on error). Stage 02 owns the PyTorch objects that implement those ideas.

This stage builds the conceptual contracts that the code-first stages rely on: the math of change,
the evidence rules of machine learning, and the responsibility flow behind backpropagation.

## Required route

1. [Math Foundations](01-math-foundations/README.md)
2. [ML Basics](02-ml-basics/README.md)
3. [Neural Networks and Backpropagation](03-neural-networks-and-backprop/README.md)

The audible backpropagation lab remains an optional branch inside Chapter 03.

**Previous:** start of the curriculum  
**Current:** Stage 01 — ML Foundations  
**Next:** [Stage 02 — PyTorch Foundations](../02-pytorch-foundations/README.md)


You can now trace a prediction's error back to its weights. The remaining complaint is practical: who stores those weights, clears yesterday's gradients, and applies today's update? [Stage 02, Chapter 01](../02-pytorch-foundations/01-deep-learning-map/01-deep-learning-map.ipynb) turns that responsibility report into a working training loop.
