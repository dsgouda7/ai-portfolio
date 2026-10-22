# Chapter 01: Sequence Memory

**Contents:** [directory map](CONTENTS.md)

How can an opening clue still matter at the end of a sequence?
Start after [Stage 02 — PyTorch Foundations](../../02-pytorch-foundations/README.md).

## Read in order

1. [01A — Sequence Memory Theory](01A-sequence-memory-theory.ipynb) ·
   [handwritten notes](01A-sequence-memory-theory.md) — Build a running note, watch its
   teaching signal fade, and discover the LSTM's keep, write, and show controls.
2. [01B — Sequence Memory PyTorch Lab](01B-sequence-memory-pytorch-lab.ipynb) — Train
   forecasting and token models with packed batches, padding-aware loss, and causal checks.
3. [01C — Cinematic Piano Memory](01C-cinematic-piano-memory.ipynb) — Hear and measure
   the same memory question in one original piano phrase played in two keys.
   Change only its opening cue and inspect the ending.

The notebooks now use [static illustrated stories](images) rather than named-box flowcharts:
follow `D3 A3 D4` through ordered seats, a changing notecard, backward feedback, and
LSTM controls. The lab shows the exact shifted note pairs, what blank seats do to
a final state, and why stopping the reader and skipping blank answers are separate jobs.
Each illustration has a plain-English reading guide next to its measured check.
Essential explanations remain visible without Mermaid or JavaScript support.

01B uses token IDs, shifted answers, and padding before explaining where they come from.
That's a preview for testing memory; [Chapter 02](../02-tokenization-and-embeddings/README.md)
then explains pieces, addresses, and the different jobs of input, loss, and causal masks.

## Run it

From this directory, run [`setup.ps1`](setup.ps1) on Windows or [`setup.sh`](setup.sh) on
Linux/macOS. It creates a chapter `.venv`, installs [requirements](requirements.txt),
and assigns kernel `stage03-language-model-01-sequence-memory`.
Run each notebook from a fresh kernel. They share readable
[helpers](sequence_helpers.py), not live notebook state.
Small CPU examples use one PyTorch thread to avoid spending more time scheduling than learning.

## Rebuild the illustrations

The [local diagram generator](generate_memory_diagrams.py) uses Matplotlib only:
no image service, credentials, or network calls. From this chapter directory:

```powershell
.\.venv\Scripts\python.exe .\generate_memory_diagrams.py
```

The pictures are intuition sketches, not fabricated training results. Curves, animations,
predicted audio, and the key-cue intervention are computed by the notebooks themselves.
The piano experiment fits both key versions; it does not test composing or unseen-key quality.

**Next:** [Chapter 02 — Tokenization and Embeddings](../02-tokenization-and-embeddings/README.md).
