# PyTorch RNN Bridge Migration Note

The two notebooks in this directory are preserved as the original PyTorch translation and cinematic evidence. This standalone bridge is no longer a separate required stop.

Its contracts now live in the consolidated sequence-memory arc:

1. [05A - Sequence Memory Theory](../../pytorch-for-llms/05-sequence-memory/05A-sequence-memory-theory.ipynb)
2. [05B - Sequence Memory PyTorch Lab](../../pytorch-for-llms/05-sequence-memory/05B-sequence-memory-pytorch-lab.ipynb)
3. [05C - Cinematic Piano Memory](../../pytorch-for-llms/05-sequence-memory/05C-cinematic-piano-memory.ipynb)

The merged lab adds the external forecasting problem, sliding windows, temporal split, `Dataset`/`DataLoader`, prediction evaluation, shifted token targets, packed sequences, padding-aware loss, causal invariance, and clipping. The capstone reuses canonical training helpers instead of reteaching their APIs.

Keep [the original bridge](01-pytorch-rnn-bridge.ipynb) and [the original piano notebook](02-cinematic-piano-memory.ipynb) when comparing the migration or recovering unique historical evidence.
