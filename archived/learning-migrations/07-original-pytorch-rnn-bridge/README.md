# PyTorch RNN Bridge Migration Note

The two notebooks in this directory are preserved as the original PyTorch translation and cinematic evidence. This standalone bridge is no longer a separate required stop.

Its contracts now live in the consolidated sequence-memory arc:

1. [Sequence Memory Theory](../../../learning/03-language-model-foundations/01-sequence-memory/)
2. [Sequence Memory PyTorch Lab](../../../learning/03-language-model-foundations/01-sequence-memory/)
3. [Cinematic Piano Memory](../../../learning/03-language-model-foundations/01-sequence-memory/)

The merged lab adds the external forecasting problem, sliding windows, temporal split, `Dataset`/`DataLoader`, prediction evaluation, shifted token targets, packed sequences, padding-aware loss, causal invariance, and clipping. The capstone reuses canonical training helpers instead of reteaching their APIs.

Keep [the original bridge](01-pytorch-rnn-bridge.ipynb) and [the original piano notebook](02-cinematic-piano-memory.ipynb) when comparing the migration or recovering unique historical evidence.
