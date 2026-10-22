# Chapter 03 Visual Briefs

**Authoring only:** the current briefs live beside the explanations in the notebooks.
They are placeholders, not generated images or links to future files. Existing images
and executable plots remain in place.

## SmartVal

See [the notebook](01-smartval-neural-networks-and-backprop.ipynb).

- **Fold the XOR square without relabelling it** — Track the four original cases into
  hidden coordinates `[0,0]`, `[1,0]`, `[1,0]`, and `[2,1]`. A straight separator becomes
  possible without changing labels. The weights are hand-chosen, not learned.
- **A gradient stops at the closed gate** — Use the actual one-case calculation:
  `[1,0]`, pre-activations `[0.9,-0.6]`, output `0.68`, loss `0.0512`, and first weight
  gradient `-0.224`. Keep gradient reporting separate from an optimizer update.
- **Keep the average, not every light** — Contrast dropped activations at training
  with untouched evaluation activations. Kept values scale by `1/0.6`; preservation
  holds in expectation, not for every sampled mask.

The current SmartVal lab uses a linear output and half-squared error in its manual
backward example. It does not use sigmoid plus binary cross-entropy, train XOR weights,
compare spiral models, or animate a framework training loop.

## Optional Melodyne

See [the notebook](optional/02-melodyne-backprop-synthesizer.ipynb).

- **Overlapping tone stencils** — Show smoothly blended bass, mid, and treble masks,
  including the explicitly illustrative `0.8/0.2/0` bin calculation.
- **Two knob scales, one backward path** — Show raw zero mapping to bass gain `1.00`,
  gradients returning through that linkage, and a separate optimizer action.
- **A matching sound need not identify the preset** — Keep waveform, spectrum, and
  known-control agreement separate. High correlation alone does not establish unique
  settings or perceptual equivalence.

The existing [fixed-notes illustration record](optional/images/PROMPTS.md) remains
available. The actual audio, waveforms, frequency spectra, loss history, and knob
trajectories must continue to come from the executable cells, not invented pictures.
