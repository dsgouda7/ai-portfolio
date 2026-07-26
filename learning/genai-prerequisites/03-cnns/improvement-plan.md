# Improvement Plan — Convolutional Neural Networks

**Audited:** 2026-07-26 | **Audience fit:** 6/10

## Overall Assessment

Strong bones: the digit-"3" running example is threaded throughout, "Predict first" exercises in Parts 1 and 4 are right, the Sobel-before-API sequencing is correct, and the empirical gradient-ratio proof beats any theorem. The notebook weakens on three fronts that matter most to this audience: trained feature maps are never shown (only random-weight noise), the vanishing-gradient explanation leads with Jacobian algebra that appears *before* the "Predict first" exercise (giving away the answer), and the transfer-learning section explains freezing as an API call without building the mental model. Six of eight teaching code cells exceed 30 lines without mid-cell narrative breaks.

---

## Strengths (preserve these)

- **"Predict first" exercises in Parts 1 and 4** — the notebook's best pedagogical feature; preserve both
- **Sobel-before-API sequencing** — Part 1 correctly applies a hand-coded filter before `torch.nn.functional.conv2d`
- **Empirical gradient ratio proof (Part 4)** — printing actual `pg`, `rg`, ratio values is more convincing than citing He et al.
- **Transfer learning phase table** — decision-ready reference for engineers; do not compress or remove
- **Tier ledger** — explicit, honest coverage scoping
- **Closing decision cell** — five concrete design rules; effective payoff
- **TinyCNN → ResNet-18 → ResNet-50 → ViT-B/16 scale bridge** — connects toy model to production architectures

---

## Gaps & Recommended Changes

### Gap 1 — Trained feature maps are never shown; only random-weight noise — Priority: High

**Problem:** The eight feature-map panels in Part 2 are from a randomly initialised network. The code correctly labels them "random init (before training)" — but never shows what trained filters look like. The audience is told "after training they specialise" but never shown it.

**Justification:** Seeing a trained filter light up on horizontal arcs vs. vertical strokes is the single most powerful "aha" in CNNs. Without it, the reader takes the author's word that convolution is useful.

**Recommendation:** Add a short training step (≤3 epochs of TinyCNN on MNIST) followed by a second feature-map grid using trained `conv1` weights. Label each panel with a one-word description ("horizontal edge", "diagonal /", etc.). Alternative: commit a `tiny_cnn_trained.pth` checkpoint and `torch.load` it rather than re-training.

---

### Gap 2 — Jacobian explanation precedes the "Predict first" exercise, giving away the answer — Priority: High

**Problem:** Part 4's markdown opens with "each layer's Jacobian has values < 1 on average. After 20 multiplications the gradient can be 10⁻⁸" — then presents the `+1` term — **before** the "Predict first" exercise. A reader who reads sequentially already knows the answer to (b) before being asked to predict.

**Recommendation:** Reorder Part 4: (1) degradation-problem motivation → (2) "Predict first" exercise → (3) empirical measurement → (4) "Why did ResNet win? Here's the math." Lead the mathematical section with an analogy first: "Imagine the gradient as a whispered message passed backward through 20 relay stations. Each plain layer hears only 70% of the previous station's signal — after 20 relays, 0.70²⁰ ≈ 0.08% of the original survives. The skip connection gives every station a direct line to the source."

---

### Gap 3 — Transfer learning: no diagram and no mental model for why ImageNet filters apply to digit strokes — Priority: High

**Problem:** Part 5 asserts "early layers learn universal low-level features: edges, textures" and immediately goes to the API call. The connection "ImageNet edges → MNIST strokes" is made in one sentence without demonstration.

**Recommendation:** Before the `binary_subset` code cell, add a ~15-line matplotlib cell: load the pretrained ResNet-18, extract two or three `layer1.0.conv1` filters, apply them to `example` (the digit "3"), show original + filter weights + activation in a three-panel row. Caption: "These weights were trained on cars and dogs — they fire on digit strokes because strokes are edges. That's the entire justification for freezing."

---

### Gap 4 — Part 4 proof cell reaches 70 lines with 5 conceptual units — Priority: Medium

**Problem:** The Part 4 cell defines two module classes, a factory function, and a measurement function, then runs two full forward+backward passes and prints a multi-line conclusion. An engineer who sees 70 lines will scroll to the output and miss the `+ x` line in `ResBlock.forward` that is the entire point.

**Recommendation:** Split at the natural boundary between class/function definitions and experiment execution — two cells of ~35 lines each. Apply the same split to Part 6 between `TinyCNN` definition and the parameter-count table.

---

### Gap 5 — Max-pooling has no diagram; "why the maximum?" goes unanswered — Priority: Medium

**Problem:** No visual shows what a 2×2 sliding window does to a feature map, and no comparison of max vs. average pooling outcomes is provided. The design choice feels arbitrary.

**Recommendation:** Add a ~12-line matplotlib cell before the size-progression table: a synthetic 4×4 feature map, four overlaid 2×2 windows drawn as colored rectangles, and the resulting 2×2 max-pool output next to a 2×2 avg-pool output. Caption: "Max-pool keeps the strongest signal in each neighbourhood — stroke shifted one pixel gives the same maximum. Average-pool blurs — useful before a classifier head."

---

### Gap 6 — No single-pixel walkthrough before the Sobel cell — Priority: Low

**Problem:** "Sliding dot product" is stated but never demonstrated at single-step granularity. An engineer who hasn't traced one computation manually will treat `conv2d` as a black box.

**Recommendation:** Add a ~10-line code cell immediately before the Sobel cell that extracts the 3×3 patch from the digit "3" at one specific position, multiplies it element-wise with `sobel_h.squeeze()`, and prints the products and their sum: "One output pixel = sum of element-wise products = X.XX."

---

## Do NOT Change

- "Predict first" exercise format in both Parts 1 and 4 (🔮 prefix, multiple-choice lettering)
- The Sobel filter selection as the first convolution example
- `grad_norm_at_layer1` measurement function and its role as primary evidence for skip connections
- Transfer learning phase table (four rows, three columns, dataset-size guidance)
- Tier ledger structure (all existing entries)
- Closing decision cell's five design rules
- UnifiedAI running example and P-4 transition sentence
