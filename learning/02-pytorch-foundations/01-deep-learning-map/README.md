# Chapter 01: The Deep-Learning Map

Start here for the canonical PyTorch code-first route.

- [Notebook](01-deep-learning-map.ipynb)
- [Handwritten theory](01-deep-learning-map-theory.md)
- [PyTorch Training Skeleton: Quick Refresher](../quick-refresher/pytorch-training-skeleton.ipynb) for returning readers who need the complete training skeleton in one place

The chapter keeps one tiny nonlinear regression problem visible while connecting:

`data -> module -> forward -> loss -> backward -> optimizer -> epoch -> inference`

DataLoader, checkpoints, and production abstractions are deliberately deferred.

The refresher is an alternate entry point, not the next required chapter. Use its closing
readiness check to decide whether to resume at one targeted chapter.

Run `./setup.ps1` on Windows or `bash ./setup.sh` on Linux/macOS.

## Required Bounded Intervention

Run **“Your turn: change only the epoch budget”** in
[`01-deep-learning-map.ipynb`](01-deep-learning-map.ipynb). Predict the final-loss direction, change
only `YOUR_EPOCHS`, measure the original and trial loss, diagnose under-training versus convergence,
and explain whether this run supports the bounded claim that more updates helped this fixed tiny
problem.
