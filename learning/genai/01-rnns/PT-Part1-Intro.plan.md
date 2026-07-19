# `PT-Part1-Intro.ipynb` — Authoring Plan

Scoped to this notebook only. Written against `learning/genai/authoring-guide.md` (all 10
sections), per Section 7's process.

## 1 · Current state

**Topic**: despite living in the `01-rnns/` folder, this notebook (MIT Intro to Deep Learning,
Lab 1 Part 1) is **not** about RNNs — it is a from-scratch tour of PyTorch fundamentals (tensors,
computation graphs, `nn.Module`, autograd, gradient descent), plus a "bonus" Part 6 that runs
pre-trained HuggingFace music-generation models. Its sibling `TF-Part1-Intro-keras.ipynb` (being
upgraded separately) covers the same ground in TensorFlow/Keras with a different running example.
The RNN/LSTM content proper lives in Lab 2, outside this notebook's scope — the closing cells of
Part 5/6 already say this explicitly ("the next lab applies the same machinery to sequence
modelling"), so no renaming/rescoping is needed here.

**Structure (79 cells before this pass)**: title + roadmap table → Setup → **Part 1** tensors as
data containers → **Part 2** computation graphs → **Part 3** `nn.Module` (three ways: hand-written
`OurDenseLayer`, `nn.Sequential`, subclassing with `nn.Linear`) → **Part 4** autograd + gradient
descent → **Part 5** toy-to-production scaling (ResNet-18 param count) → **Part 6** pre-trained
music generation (MusicGen / TunesFormer) → Summary.

**Existing pedagogy — already strong, already at parity with most of Sections 1-9**:
- Title + roadmap table, restated as a completed table in the Summary, plus a "Key Insights to
  Keep" bullet list. ✅
- A single running example (5-house price-prediction toy dataset) threaded through Parts 1-5. ✅
- `🔮`/"Predict first" and "Predict before you run" cells before every non-obvious reveal
  (tensor speed/safety/GPU, house-price hand computation, `nn.Module` autograd registration,
  `y=x²` gradient, gradient-descent convergence, ResNet-18 param count). ✅
- "What just happened — and what's missing" reflection cells closing every Part. ✅
- "Don't assert, demonstrate": tensor speed/safety/GPU claims are measured, not stated; the
  toy→production bridge is a real `torchvision.models.resnet18` param count, not an invented
  number. ✅
- Two "Code Walkthrough" cells already exist (tensor-vs-list reasons; manual gradient descent). ✅
- Toy→real parameter-mapping table (Part 5) mapping the 3-parameter toy model to ResNet-18/GPT-2/
  Stable Diffusion parameter counts. ✅
- Completion-only generation helper (`generate_abc`) already strips the prompt correctly and has a
  proper docstring, matching Section 9.2. ✅

## 2 · Checklist scoring

**Section 6 (core checklist)** — effectively fully met already; no gaps found worth flagging.

**Section 8.8 (narrative framing)** — not applicable. This notebook is a mechanics tour, not a
named-client scenario like `01-llm-finetuning.ipynb`; forcing a business framing onto "learn what a
tensor is" would be artificial. No action.

**Section 9.5 (code-clarity addendum)** — gaps found:
- [ ] Several code cells exceed ~30 lines and combine multiple separately-nameable steps without a
      walkthrough or split (see 10.6 below — same root cause, addressed there).
- [x] `generate()`/`generate_abc()` helpers already return completion-only tokens with docstrings.
- [ ] No orthogonal 2-axis technique grid exists yet (see 10.3 below — partially applicable).
- [x] No token-position/step-axis heatmap exists that needs `FuncAnimation` — not applicable here.

**Section 10.8 (navigation/consistency addendum)** — gaps found:
- [ ] **No Table of Contents.** 79 cells is well past the ~40-cell threshold in 10.1.
- [x] Per-subplot legends: the one multi-panel plot (house-price scatter, 2 subplots) already has
      a `plt.colorbar(..., label=...)` on each subplot (continuous color-coding) — compliant. The
      gradient-descent convergence plot has a single legend for its two series — compliant. No
      action needed here.
- [ ] **No qualitative pros/cons comparison table** exists for either of the two natural
      "≥2 things being compared" spots in this notebook: (a) the three ways to define the same
      layer in Part 3 (`OurDenseLayer` subclass vs. `nn.Sequential` vs. subclassing with
      `nn.Linear`), and (b) the two Part 6 pre-trained model options (MusicGen vs. TunesFormer).
      Neither is a true M×N orthogonal-axis grid like the fine-tuning notebook's data-objective ×
      parameter-strategy matrix (Section 9.3/10.3) — both are single-axis "pick one of N" choices —
      so a straightforward pros/cons **table** (Section 3's "tables over prose") is the right-sized
      pattern here, not a forced 2D grid.
- [ ] **"State the common mental model, then correct it" (10.4) not yet applied anywhere.** The
      strongest candidate: the gradient-descent section lets `L = (x - x_f)²` converge to the
      global minimum every time, which quietly reinforces the common (and wrong-in-general)
      shorthand "gradient descent finds *the* minimum." This toy loss is convex (one bowl, one
      minimum), so it always works here — but the identical update rule on a real neural network's
      non-convex loss surface is only guaranteed to reach *a* stationary point, not the global one.
      This gap should be named explicitly, right where the reader is most likely to have absorbed
      the oversimplified version.
- [x] Section 10.5 (grounding "expected outcome" claims in real facts): all "predict" cells already
      predict values the notebook itself computes/verifies (matmul speedup, `y=x²` derivative,
      ResNet-18's real parameter count, measured mean error) — nothing invented. No action.
- [ ] **Progressive disclosure (10.6)** — cells combining multiple separately-nameable steps that
      should be split into small intro'd cells rather than left as one dense block:
  - `# ── WHY tensors? Three reasons, measured` (~34 lines: speed test / shape-safety test / GPU
    check — three independent, separately-nameable proofs bundled in one cell). This is the guide's
    *own* worked example for a walkthrough cell — but per 10.6, a walkthrough-after is now the
    fallback for cells that can't be split, and this one clearly can.
  - `# ── Our Running Example — House Price Prediction` (~41 lines: data definition/print table,
    then a 2-subplot plot — two natural checkpoints).
  - `if MUSIC_MODEL == "musicgen":` cell (~33 lines: load pipeline → generate → normalise/save →
    display — four steps).
  - `if MUSIC_MODEL == "tunesformer":` cell (~83 lines, the densest cell in the notebook: load
    tokenizer/model → define `generate_abc()` → run generation → convert to MIDI — four steps).
- [ ] **Cross-reference hygiene (10.7)** — grep for `cell below|cells below|cell above|cells
      above|next cell|following cell|cells? (right )?after` found 9 hits. 7 are the sanctioned
      "Predict before you run" / "Predict first" adjacent-pair pattern from Section 3 (predict cell
      immediately followed by its own reveal cell) — safe, left alone. 2 are genuinely fragile:
      "Run the cell below when `MUSIC_MODEL = "musicgen"`" and "...`"tunesformer"`" — both will
      become stale the moment those two cells are split per 10.6 above, so both need rewording to
      distance-free language as part of the same edit.

## 3 · Ordered changes implemented

1. **Table of Contents** — new Markdown cell inserted immediately after the title/roadmap cell,
   before the first code cell. Numbered top-level entries for every `##`-level Part/Setup/Summary
   section, nested entries for the handful of `###` subsections worth a direct jump (both Code
   Walkthrough cells, the Part 6 "why pretrained" cell, and the Option A/B cells). Includes the
   `Ctrl+F`/outline-panel fallback caveat verbatim per 10.1.
2. **Common-mental-model correction** — new Markdown cell inserted right after the
   `L = (x - x_f)²` setup cell and before the "Predict before you run" (GD convergence) cell.
   States the shorthand ("gradient descent finds *the* minimum"), shows why it's true here (convex
   parabola, one bowl), then names the gap for real networks (non-convex loss surfaces → only a
   stationary point is guaranteed, not the global minimum) — the three-beat 10.4 structure.
3. **Pros/cons table — three ways to define a model** — new Markdown cell inserted after the
   `LinearWithSigmoidActivation` test cell and before its "What just happened" reflection, comparing
   `OurDenseLayer` (manual `nn.Parameter`) vs. `nn.Sequential` vs. subclassing with `nn.Linear`.
4. **Pros/cons table — MusicGen vs. TunesFormer** — new Markdown cell inserted after the existing
   Part 6 facts table, before the "Why pre-trained models" cell.
5. **Split the "three reasons tensors beat lists" cell** into three small cells (speed / shape
   safety / GPU), each preceded by a one-line intro. The existing Code Walkthrough cell is kept but
   retitled "..., Recapped" with its opening line reworded so it no longer claims the three checks
   were "combined into one cell" (10.6's explicit guidance for this exact situation).
6. **Split the house-price dataset + plot cell** into a data/print cell and a plotting cell, with a
   short intro before the plot.
7. **Split the MusicGen (`if MUSIC_MODEL == "musicgen"`) cell** into load/generate+save/display
   steps, each preceded by a short intro, preserving the `if` guard on each piece.
8. **Split the TunesFormer (`if MUSIC_MODEL == "tunesformer"`) cell** — the densest cell in the
   notebook — into load/define-helper/generate/convert-to-MIDI steps, each preceded by a short
   intro, preserving the `if` guard on each piece.
9. **Cross-reference fixes** — reworded both "Run the cell below when `MUSIC_MODEL = ...`" phrases
   (Option A/B intro cells) to name the upcoming steps instead of counting cells, since step 7/8
   above turn each single target cell into several.
10. **Final grep re-check** — after all insertions/splits, re-run the cross-reference grep across
    the whole notebook to confirm no new "next cell"/"cell below" phrasing was introduced that
    doesn't satisfy the adjacent-pair exception, and that the two fixed phrases are gone.

Deliberately **not** doing: forcing an M×N quantitative combination grid (no real orthogonal
2-axis technique choice exists in this notebook — see 2 above); rewriting the RNN mislabel in the
folder name (out of scope, a repo-organization concern, not this notebook's content); adding a
named-client narrative framing (Section 8 doesn't fit a from-scratch mechanics tour and would read
as forced).
