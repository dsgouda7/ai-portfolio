# RNN Sequence Modeling Migration Note

The Keras-first notebook in this directory is preserved because it contains unique vanishing-gradient and LSTM-gate evidence. It is no longer the required executable implementation.

The PyTorch-first sequence-memory arc now owns this material:

1. [Sequence Memory Theory](../../../learning/03-language-model-foundations/01-sequence-memory/) preserves the manual recurrence, BPTT complaint, gradient-decay static evidence and animation, LSTM gate/cell-state static evidence and animation, and controlled RNN-versus-LSTM comparison.
2. [Sequence Memory PyTorch Lab](../../../learning/03-language-model-foundations/01-sequence-memory/) owns framework execution.
3. [Cinematic Piano Memory](../../../learning/03-language-model-foundations/01-sequence-memory/) owns the audible capstone.

Keep this directory when studying the historical Keras translation, but do not follow it as a parallel required route.
