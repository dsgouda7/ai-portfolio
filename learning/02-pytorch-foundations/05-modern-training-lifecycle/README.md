# Part 05: The Modern Training Lifecycle

Keep the fuel task, but separate training from validation, choose the best epoch, and return
predictions to fuel units. Mixed precision runs only on CUDA; Lightning and compile are explained, not run.

The best-model file reloads for inference. The separate training-state snapshot gets a key check,
not a same-next-update recovery test.

- [Notebook](05-modern-training-lifecycle.ipynb) · [Theory](05-modern-training-lifecycle-theory.md) · [Contents](CONTENTS.md)
- Setup from this directory: [Windows](setup.ps1) (`.\setup.ps1`) or [Linux/macOS](setup.sh) (`bash ./setup.sh`).
- Next: [Sequence Memory](../../03-language-model-foundations/01-sequence-memory/README.md) adds ordered history; no fuel artifact is needed.
