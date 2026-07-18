# Plan: `TF-Part1-Intro-keras.ipynb` → Authoring Guide Parity

## 1. Current State

**Topic**: TensorFlow/Keras fundamentals (tensors, computation graphs, Keras layers,
`GradientTape` autodiff, gradient descent), *not* actually RNNs despite living in the
`01-rnns` folder and inheriting MIT's original lab title ("Music Generation with RNNs").
The notebook itself explains this pivot: the field moved from from-scratch LSTMs to
pretrained Transformers (MusicGen, TunesFormer) for music generation, so Part 2 uses
those instead of hand-rolling an RNN. This mirrors the sibling `PT-Part1-Intro.ipynb`
(PyTorch version, house-price running example) — both are "Part 1: framework
fundamentals" notebooks; RNNs proper are deferred to "Lab 2."

**Structure** (68 cells before this pass):
- Title/copyright → "What You'll Learn" roadmap cell (9-step table) → running example
  established (2D circle-vs-square binary classification, 200 points).
- Part 1 (sections 1.0–1.5): tensors & shapes → computation graphs (with a hand-traced
  DAG proof) → Keras layers (`OurDenseLayer`, `Sequential`, subclassed `ResidualClassifier`
  with a skip connection) → `GradientTape` autodiff (verified against hand-coded
  derivatives for x², sin(x), eˣ²) → gradient descent (1D bowl, then the real classifier)
  → toy→ResNet-50 parameter-count bridge.
- Part 2: pretrained Transformer inference (MusicGen text→audio, TunesFormer ABC-seed→
  ABC-notation), switchable via a `MUSIC_MODEL` flag.
- Summary: completed roadmap table + "Key Insights to Keep" bullets.

**Existing pedagogy already at parity**: roadmap table, single running example threaded
throughout Part 1, 🔮/🧪 predict-and-verify cadence in several places, reflection cells
("What just happened — and what's missing"), Code Walkthrough cells after dense blocks,
a toy→real parameter-mapping table, deterministic seeding, `# ──` section-banner
comments, math-mirroring variable names (`W`, `b`, `d_k`-style), a completion-only
`generate_abc()` helper with a proper docstring (already satisfies §9.2).

## 2. Checklist Audit (§6, §8.8, §9.5, §10.8)

| Item | Status | Notes |
|---|---|---|
| Title + roadmap table | ✅ | present, accurate |
| Single running example threaded through | ✅ | circle-vs-square, Part 1 only (Part 2 necessarily switches domains to audio/ABC — acceptable, bridged explicitly) |
| Claims measured, not asserted | ⚠️ | width/depth claim proven via exercise; **nonlinearity claim in §1.3 ("sigmoid/relu needed or boundary can't curve") is stated but never measured** — gap |
| 🔮 Predict-first before non-obvious reveals | ⚠️ | present twice but **missing the 🔮 emoji marker** (inconsistent with the rest of the notebook's own convention) |
| 🧪 Exercises near the concept, not new content | ❌ | the width exercise trains via `tf.GradientTape()` **before** §1.4 ever explains what `GradientTape` is — violates "exercises drill an already-explained mechanism" |
| Reflection cells close each part | ✅ | present throughout |
| Toy→real bridge table | ✅ | classifier vs. ResNet-50 |
| Math always glossed | ✅ | |
| Visualizations: legends per subplot | ⚠️ | main loss+boundary figure: loss-curve panel has no legend; boundary panel's learned-boundary contour line has no legend entry (only a `clabel`) |
| Section-banner code comments, pedagogical prints | ✅ | |
| Completed roadmap + Key Insights at the end | ✅ | |
| Seeds deterministic wherever numbers are quoted | ✅ | |
| §8.8 narrative framing | N/A | this is a from-scratch fundamentals notebook, not a business-scenario notebook — §8 doesn't apply here (no named client/brief to bind to); judged out of scope |
| §9.5 code clarity: >30-line multi-concept cells get a walkthrough or are split | ❌ | the TunesFormer cell (load model → define helper → generate → export MIDI, ~80 lines, 4 separable steps) needs splitting per §10.6, not just a walkthrough |
| §9.5 `generate()`-style helpers return only completion tokens + docstring | ✅ | `generate_abc()` already does this |
| §10.8 TOC for notebooks >~40 cells | ❌ | 68 cells, no TOC |
| §10.8 cross-reference hygiene (no stale "next cell") | ❌ | "Change `MUSIC_MODEL` in the next cell to switch" — the model-selector cell is actually **two** cells down (a pip-install cell intervenes) |
| §10.8 qualitative combo matrix for 2 orthogonal axes | N/A | notebook has no 2-axis technique comparison (MusicGen vs. TunesFormer is an either/or choice, not a combinable grid) |
| §10.8 "state shorthand, then correct" | N/A | no existing oversimplified-shorthand correction needed beyond what's already handled by the eager/graph predict-first cell |
| §10.8 grounded "expected outcome" claims | ✅ | claims about ResNet-50 param count, circle geometry etc. are all real, computed numbers |

## 3. Ordered Changes Implemented

1. **Add a Table of Contents** cell immediately after the title/roadmap cell (before the
   first code cell) — 68 cells clears the ~40-cell TOC threshold (§10.1).
2. **Add 🔮 emoji** to the two existing "Predict First" markdown headers (Eager vs. Graph
   Execution; Sequential vs. skip connection) for consistency with the notebook's own
   load-bearing-signpost convention (§3).
3. **Fix the cross-reference bug**: "Change `MUSIC_MODEL` in the next cell" → rephrased to
   name the target directly instead of a stale cell-distance claim (§10.7/10.8).
4. **Prove, don't assert, the nonlinearity claim**: insert a new 🔮 predict-first cell +
   two code cells right after the main classifier's loss/boundary plot. Trains an
   identical-architecture model with a **linear** hidden-layer activation side-by-side
   with the existing **ReLU** model (same seed, same data, same training loop), plots
   both learned decision boundaries, and prints a measured verdict (final loss + boundary
   shape) instead of just asserting "you need a nonlinearity." This directly demonstrates
   the claim already made in §1.3's markdown.
5. **Fix the exercise-ordering violation**: move the "🧪 Your turn — how wide does the
   hidden layer need to be?" exercise (+ its Code Walkthrough) from *before* `GradientTape`
   is ever explained to *after* the main classifier training loop and the new nonlinearity
   proof — so the exercise now drills a mechanism the reader has actually seen explained,
   per §5.6/§6.
6. **Add missing subplot legends** to the main loss+decision-boundary figure: a one-entry
   legend on the loss-curve panel, and a `Line2D` legend entry for the learned-boundary
   contour on the decision-boundary panel (§10.2/§10.8).
7. **Split the TunesFormer cell** (~80 lines, 4 separable steps: load model+tokenizer →
   define `generate_abc()` → generate → export MIDI) into 4 small cells, each preceded by
   a short 2–4 sentence intro, per §10.6's progressive-disclosure pattern.
8. Re-grep the notebook for cell-distance phrases after all moves/splits to confirm no new
   staleness was introduced (§10.7).

## 4. Deliberately Deferred (out of scope for this pass)

- Splitting the MusicGen cell (~42 lines) the same way — it's a single cohesive
  load→generate→save→display sequence behind one `try/except`, borderline over the
  30-line threshold but less clearly multi-concept than the TunesFormer cell; lower
  priority given time budget.
- Converting the unconditional `!pip install transformers accelerate ... --quiet` cell to
  a `try: import / except ImportError: pip install` guard (§2's ideal pattern) — cosmetic,
  low-risk-to-defer, and changing installer behavior without a kernel to verify felt
  riskier than leaving Colab's already-idiomatic pattern alone.
- No changes to the Part 2 pretrained-model narrative/business framing (§8) — this
  notebook is a mechanics-first fundamentals lab, not a scenario-driven notebook, so §8
  is treated as not applicable rather than force-fitted.
- Not adding an actual from-scratch RNN/LSTM implementation despite the folder name —
  confirmed via the sibling `PT-Part1-Intro.ipynb` that this is intentional scope for
  both "Part 1" notebooks in this lab pair; RNNs are Lab 2's job.
