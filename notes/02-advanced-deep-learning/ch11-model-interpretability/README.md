# Ch.11 — Model Interpretability

> **The story.** You have trained a 6.8 MB ProductionCV model that runs at 35ms on a $99 Jetson Nano and meets all five production constraints. The operations team wants to deploy it across 50 retail stores. The store manager at the pilot site asks one question: *"Why did it flag shelf position 14C as a stockout?"* Without an answer, the system doesn't get past the pilot. This chapter gives you three auditing tools that turn a "black box" into an auditable system.
>
> **Where you are in the curriculum.** All five constraints are met (Ch.10). This chapter addresses a sixth, implicit constraint: **auditability** — the ability to explain a model's specific decisions to a non-technical stakeholder. It is not a stretch goal; for enterprise retail deployment it is a gate.

---

## 0 · The Challenge — Where We Are

> 🎯 **The mission**: ProductionCV is production-ready on metrics. Make it **auditable**.

The 5 constraints from Ch.1–10 are all met:
- ✅ mAP@0.5 = 82.1%
- ✅ IoU = 71.2%
- ✅ Latency = 35ms
- ✅ Model size = 6.8 MB
- ✅ Trained on 850 labeled images

**New constraint:**
- ❌ **Auditability:** Can you explain *which shelf regions* drove a specific detection decision? Without this, enterprise procurement won't sign off.

**What this chapter unlocks:** Two families of auditing tools — one that reveals what the model *has learned*, and one that explains *why it made a specific call*:

| Family | Technique | Question it answers | Direction |
|---|---|---|---|
| **Feature Visualization** | Activation maps | What intermediate representations did shelf 14C produce at each layer? | Forward — no gradients needed |
| **Feature Visualization** | Filter visualization (gradient ascent) | What input pattern maximally excites each learned filter? | Backward — optimise input |
| **Attribution Maps** | Grad-CAM | Which pixels of shelf 14C caused the stockout flag? | Backward — one backward pass |

The distinction matters: Feature Visualization answers *"what has the model learned?"* (model-wide); Attribution Maps answer *"why this specific prediction?"* (instance-specific). The store manager needs Attribution Maps. Feature Visualization helps you trust — or distrust — the model before deployment.

---

## 1 · Family 1 — Feature Visualization

### 1a · Activation Maps

> _What intermediate representations does each layer produce for shelf image 14C?_

Build a multi-output Keras `Model` that returns every `Conv2D` layer's output simultaneously, then run shelf 14C through it. Plot each channel as a 2D grid.

**What you'll see in ProductionCV:**
- **Block 1** — edge detectors fire on shelf edges and product outlines; most channels active
- **Block 3+** — channels become sparse; the few that fire encode product-shape templates specific to the training classes
- **Block 5** — almost entirely dark except the channels encoding "empty slot" texture

Uniform or noisy activations in early blocks signal a training problem — fix data augmentation or learning rate before tuning the head.

### 1b · Filter Visualization via Gradient Ascent

> _What input pattern maximally activates each learned filter?_

Start with a random-noise image, then repeatedly nudge pixel values in the direction that increases one filter's mean activation (gradient ascent). Normalise the gradients at each step to prevent runaway magnitudes; stop after ~30 iterations.

**What you'll see in ProductionCV:**
- **Block 1 filters** — oriented edges and colour gradients (universal low-level features)
- **Block 3 filters** — repeating texture patches (bar-code stripes, label colours)
- **Block 5 filters** — abstract blobs that vaguely resemble product silhouettes

> Gradient ascent on very deep layers often converges to noise — the feature space is too abstract to be human-interpretable. Stick to blocks 1–4 for useful visualisations.

---

## 2 · Family 2 — Attribution Maps (Grad-CAM)

> _Which pixels of shelf 14C caused the stockout prediction?_

Grad-CAM weights each spatial channel of the last convolutional layer by the gradient of the target class score, then sums across channels to produce a single `[H × W]` heatmap. Resize the heatmap to the original image size and overlay it at ~40% opacity.

**What you'll see in ProductionCV:**
- The hot region concentrates over the **empty slot at position 14C** — the model fires on missing product, not surrounding products
- Adjacent full-product slots are cold, confirming the model distinguishes "empty" from "stocked"

A heatmap concentrated on the store floor or ceiling signals a **spurious correlation** — something no accuracy metric would reveal, but that Grad-CAM surfaces immediately before a bad model reaches production.

---

## Progress Check

After this chapter:
- You can build a multi-output feature-extractor from any `Functional` API model
- You can visualize per-layer activation maps for any input image (Feature Visualization)
- You can implement gradient ascent to reveal what each filter has learned to detect
- You can implement Grad-CAM and overlay the attribution heatmap on the original shelf image
- You know when to reach for Feature Visualization ("is the model learning sensible features?") vs. Attribution Maps ("why did it flag this specific image?")
- You can explain a specific detection decision to a non-technical stakeholder

---

## Bridge

➡️ This chapter completes the ProductionCV system. The full arc: ResNets (Ch.1) → Compression (Ch.10) → Auditability (Ch.11). A model that is fast, small, accurate, data-efficient, *and* explainable is production-ready in enterprise environments.
