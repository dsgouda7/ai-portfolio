# RNN Sequence Modeling Migration Note

The Keras-first notebook in this directory is preserved because it contains unique vanishing-gradient and LSTM-gate evidence. It is no longer the required executable implementation.

The PyTorch-first sequence-memory arc now owns this material:

1. [05A - Sequence Memory Theory](../../pytorch-for-llms/05-sequence-memory/05A-sequence-memory-theory.ipynb) preserves the manual recurrence, BPTT complaint, gradient-decay static evidence and animation, LSTM gate/cell-state static evidence and animation, and controlled RNN-versus-LSTM comparison.
2. [05B - Sequence Memory PyTorch Lab](../../pytorch-for-llms/05-sequence-memory/05B-sequence-memory-pytorch-lab.ipynb) owns framework execution.
3. [05C - Cinematic Piano Memory](../../pytorch-for-llms/05-sequence-memory/05C-cinematic-piano-memory.ipynb) owns the audible capstone.

Keep this directory when studying the historical Keras translation, but do not follow it as a parallel required route.
