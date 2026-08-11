# Chapter 02 Image Plan

This plan covers:

- `01-smartval-neural-networks-and-backprop.ipynb`
- `02-melodyne-backprop-synthesizer.ipynb`

## Medium Policy

Keep measured artifacts executable: XOR and spiral boundaries, hidden activations, gradient values, finite-difference comparisons, waveforms, spectra, loss curves, and knob trajectories must come from code. Use Copilot-generated images only for stable mechanism diagrams that establish intuition before the measurements.

## Existing Visuals

| Asset or output | Disposition | Reason |
|---|---|---|
| `images/xor-not-linearly-separable.png` | KEEP | Establishes the geometric failure of one linear boundary. |
| `images/neural-network-forward-pass.png` | KEEP | Introduces the 2-to-2-to-1 architecture. |
| `images/depth-vs-width-decision-boundary.png` | KEEP pending visual audit | Companion code reproduces the measured comparison. |
| SmartVal forward/backward `FuncAnimation` | KEEP EXECUTABLE | Animation state follows the notebook explanation and can be revised in code. |
| Melodyne waveform, spectrum, loss, and knob plots | KEEP EXECUTABLE | These are measured evidence, not illustration. |
| Melodyne target, damaged, and restored audio players | KEEP EXECUTABLE | Listening evidence must use the generated arrays. |

## Copilot Image Queue

### IMG-01 - Forward Calculation and Backward Responsibility

- **Notebook:** SmartVal
- **Anchor:** `Watch One Prediction Become Four Responsibility Reports`
- **Action:** ADD
- **Target:** `images/backprop-forward-backward-responsibility.png`
- **Medium:** Copilot image
- **Aspect ratio:** 16:9
- **Prompt:** Create a clean computational graph for one 2-input, 2-hidden-unit, 1-output XOR prediction. Use blue arrows from inputs through affine transforms, ReLU, sigmoid, and scalar binary cross-entropy loss. Overlay coral arrows returning from loss through the same recorded operations, ending in a separate signed gradient report for each weight. Place a small green optimizer update after all gradient reports, not inside the backward pass. Show one inactive ReLU as a closed responsibility gate. Use short labels, generous spacing, and a light neutral background.
- **Acceptance:** Forward and backward arrows are directionally distinct; the optimizer is separate from backpropagation; no target label flows backward as input data; all layers remain readable at notebook width.
- **Prohibited:** Brain imagery, an optimizer changing weights during the backward arrows, fabricated numerical gradients, or claims that backpropagation itself performs gradient descent.
- **Alt text:** A neural-network graph first carries a prediction forward to loss, then carries responsibility backward to each weight before a separate optimizer update.
- **Status:** TODO

### IMG-02 - XOR Input Space Becomes Hidden Space

- **Notebook:** SmartVal
- **Anchor:** `Hidden Space: XOR Becomes Linearly Separable`
- **Action:** ADD
- **Target:** `images/xor-input-hidden-space-transformation.png`
- **Medium:** Copilot image
- **Aspect ratio:** 16:9
- **Prompt:** Show the same four XOR points in two coordinate spaces. On the left, opposite-corner classes cannot be separated by one straight line. In the center, two learned ReLU features reshape the coordinate system. On the right, the transformed points are separated by one straight output boundary. Preserve point identity with consistent shapes and colors across all three stages. Make the transformation look like a learned remapping, not physical motion through a tunnel.
- **Acceptance:** All four points can be tracked; input space is visibly inseparable; hidden space is visibly separable; no invented activations or accuracy values appear.
- **Prohibited:** Curved final classifier, extra data points, three-dimensional perspective that hides geometry, or a claim that ReLU always makes arbitrary data separable.
- **Alt text:** Four XOR points move through two learned ReLU features from an inseparable input arrangement to a linearly separable hidden arrangement.
- **Status:** TODO

### IMG-03 - Differentiable Synth Signal Chain

- **Notebook:** Melodyne
- **Anchor:** `Build the Differentiable Forward Pass`
- **Action:** ADD
- **Target:** `images/melodyne-differentiable-signal-chain.png`
- **Medium:** Copilot image
- **Aspect ratio:** 16:9
- **Prompt:** Create a left-to-right software synthesizer signal chain: fixed note events create a dry waveform; four raw trainable values become bounded bass, mid, treble, and drive knobs; an FFT separates frequency content; three overlapping frequency masks apply the tone controls; inverse FFT returns a waveform; smooth tanh drive produces candidate audio; target comparison produces one scalar loss. Add a thin coral backward path from loss to all four raw values. Use familiar studio controls but keep the computation graph technically explicit.
- **Acceptance:** Note events are fixed; exactly four trainable controls are shown; FFT and inverse FFT appear in the correct order; loss compares candidate with reference audio; gradients reach raw controls.
- **Prohibited:** Microphone training data, nondifferentiable hard clipping, MIDI generation, a deep neural network not present in the lab, or gradients flowing into the target recording.
- **Alt text:** Fixed notes pass through differentiable tone and drive controls to produce audio, while loss sends gradients back to four trainable synth parameters.
- **Status:** TODO

### IMG-04 - Reference, Damaged, Restored

- **Notebook:** Melodyne
- **Anchor:** `Hide a Preset, Then Damage It`
- **Action:** ADD
- **Target:** `images/melodyne-restoration-listening-workflow.png`
- **Medium:** Copilot image
- **Aspect ratio:** 3:2
- **Prompt:** Design a restrained three-stage listening workflow for the same short phrase. Reference audio establishes the target tone; a damaged preset shows visibly misplaced bass, mid, treble, and drive controls; a learned preset moves those controls close to the hidden target and produces restored audio. Connect the stages with loss, backpropagation, and optimizer-update labels. Emphasize that the notes and timing remain identical in all three clips.
- **Acceptance:** The same phrase appears in all stages; only timbre controls change; learned and target knobs are close but conceptually distinct; the workflow reads left to right.
- **Prohibited:** Different songs, instrument-transfer claims, fake loss numbers, or visual normalization that implies all clips have identical loudness.
- **Alt text:** The same phrase is heard as a reference, through damaged controls, and after learned controls restore its timbre.
- **Status:** TODO

### IMG-05 - Generate Then Render

- **Notebook:** Melodyne
- **Anchor:** `Bridge: From Rendering a Phrase to Generating One`
- **Action:** ADD
- **Target:** `images/melodyne-generate-then-render-bridge.png`
- **Medium:** Copilot image
- **Aspect ratio:** 16:9
- **Prompt:** Show two connected but separately trained systems. A sequence model receives previous note tokens and predicts the next note and duration tokens. Those events feed a differentiable synthesizer renderer with learned tone controls, producing audio. Label the sequence model question as what comes next and the renderer question as how known notes should sound. Keep separate losses and backward paths so the diagram does not imply that argmax note selection is differentiable end to end.
- **Acceptance:** Content generation and timbre rendering are distinct; event tokens connect them; separate objectives are visible; no future notes leak into the sequence input.
- **Prohibited:** A single unexplained end-to-end gradient through argmax, guitar-to-sitar generalization claims, copyrighted song artwork, or attention architecture not taught here.
- **Alt text:** A sequence model predicts note events, then a separate differentiable synthesizer renders those events with learned timbre controls.
- **Status:** TODO

## Integration Checks

- Verify every generated file is a real PNG and not JPEG data with a `.png` suffix.
- Insert images only at the named semantic anchors.
- Preserve executable evidence immediately after each intuition image.
- Use descriptive alt text and avoid text too small for notebook width.
- Re-run notebook JSON validation and Markdown link checks after integration.
