# Optional Neural-Network Capstones

**Start:** [Melodyne Backprop Lab](02-melodyne-backprop-synthesizer.ipynb).
Hear a differentiable renderer recover four synth controls, with gradient checks and recovery gates.
Theory, audio explanations, and experiments are together in the notebook.

Follow 8,960 fixed samples through FFT tone shaping and smooth drive. Check the four raw
control slopes before Adam changes them, then compare actual checkpoints every 25 updates
and at update 220. The recovery assertions set targets for a run; the prose does not claim
that they have passed without execution. Matching one phrase is not general audio restoration.

**Before this:** [required SmartVal chapter](../README.md).
**Contents:** [branch map](CONTENTS.md).

## Setup

Use the [parent chapter's setup scripts](../README.md#setup) and
`stage01-neural-networks` kernel. TensorFlow is only this optional lab's implementation;
you do not need it to continue the main route.

**Next:** [the Deep-Learning Map](../../../02-pytorch-foundations/01-deep-learning-map/01-deep-learning-map.ipynb).
Give parameters, gradients, and updates explicit homes in PyTorch.
