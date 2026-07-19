# Plan: `keras-to-pytorch-primer.ipynb` → authoring-guide parity

## 1. Current state

**Topic:** A side-by-side primer — the exact same MNIST CNN (`Conv2D→Pool→Conv2D→Pool→Flatten→Dense→Dense`)
built once in Keras and once in raw PyTorch, cell pair by cell pair, ending in a Keras→PyTorch
cheat sheet and a completed roadmap. 40 cells total (21 markdown / 19 code), none executed.

**Structure already in place (good news — this notebook is closer to parity than most):**
- Title cell with pitch + "what you'll build" + "what you'll be able to do."
- A roadmap table (Section 2 skeleton) listing all 7 steps + the cheat sheet.
- 7 numbered `## Section N` parts, each opening with a Keras-vs-PyTorch contrast paragraph,
  followed immediately by a Keras code cell then the equivalent PyTorch code cell.
- Three 🔮 "Predict first" cells (channel-order shapes, flatten size, zero_grad ablation outcome).
- One 🧪 "Your turn" exercise (prove the `zero_grad()` gotcha, with a `# 👉 CHANGE` knob).
- Reflection ("What just happened") cells after the flatten-size proof, the training loop, and
  the inference comparison.
- One Code Walkthrough cell (the four-step training loop).
- A real "don't assert, demonstrate" habit already used well: the `64*5*5` flatten size is
  verified with a dummy forward pass instead of only trusted from a comment, and the
  `zero_grad()` gotcha is proven with a live ablation rather than just stated as a warning.
- Ends with a completed roadmap table + a "Key insights to keep" bullet list (not a bare recap).
- Deterministic seeding (`tf.random.set_seed`, `torch.manual_seed`) at the top and again before
  the ablation experiment.

**What's missing / not yet at parity**, mapped to the sections below.

## 2. Checklist scoring

### Section 6 (core parity checklist)
- [x] Title + roadmap table.
- [x] Single running example (MNIST CNN) threaded throughout — no topic drift.
- [~] Claims measured, not asserted — mostly yes (flatten size, zero_grad, final accuracy
  comparison), but the `nn.CrossEntropyLoss() # combines log-softmax + NLL` comment (Section 4)
  is asserted, never proven numerically. **Gap.**
- [x] 🔮 Predict-first used before non-obvious reveals.
- [~] 🧪 Your turn exercises — only one exists (zero_grad). The flatten-size section (Section 3)
  ends with a proof but no exercise letting the reader turn a knob themselves. **Gap.**
- [x] Reflection cells close key sections.
- [ ] Toy→real bridge table — **N/A for this topic** (there is no toy-space/production-space
  split; the whole point is Keras vs. PyTorch on one fixed task). Not scored as a gap.
- [ ] Math never left unglossed — **N/A**, no LaTeX/formulas in this notebook.
- [~] Visualizations — only two plots exist (sample-digit grid, final accuracy bar chart).
  Both already self-label their categories directly on the axis (digit labels as subplot titles;
  "Keras"/"PyTorch" as x-tick labels under each bar), which satisfies the spirit of Section
  10.2's legend requirement without a redundant `Patch` legend. No gap worth forcing.
- [~] Code-cell banner comments — uses a consistent `# --- Keras ---` / `# --- PyTorch ---`
  convention instead of the gold standard's `# ── Description ──` banner style. This is a
  deliberate, topic-appropriate convention (it's the load-bearing signal for the whole
  notebook's structure) and is used with full consistency — not treated as a gap.
- [x] Ends with completed roadmap + "Key insights to keep."
- [x] Deterministic seeding wherever quoted numbers appear.

### Section 8.8 (narrative framing) — **not applicable**
This is a mechanical framework-comparison primer, not a business-scenario notebook. Per the
task's own guidance, forcing a "Riverside House"-style named client/constraint framing onto a
Keras-vs-PyTorch cheat sheet would be artificial. Skipped deliberately, not a gap.

### Section 9.5 (code clarity)
- [x] The one cell over ~30 lines that combines multiple steps *and already has a walkthrough*
  (the manual training loop) has its Code Walkthrough cell.
- [ ] Two more cells combine multiple separately-nameable steps but have **no** walkthrough or
  split: `SimpleCNN` class-definition-plus-instantiation-plus-param-count (~30 lines), and the
  `zero_grad` ablation's function-definition-plus-two-runs-plus-comparison-print (~36 lines).
  Per Section 10.6 these are progressive-disclosure split candidates rather than walkthrough
  candidates (see below). **Gap.**
- [x] `generate()`-style helper convention — N/A, no text generation in this notebook.
- [x] Combination grid (9.3) — N/A, no M×N technique axes here.
- [x] `FuncAnimation` — N/A, no token-position/step data to animate.

### Section 10.8 (navigation & consistency)
- [ ] **No Table of Contents.** The notebook is at 40 cells, right at the guide's "~40 cells"
  threshold, and will exceed it once the changes below are applied. **Gap.**
- [x] Legends — see Section 6 note above; both plots already self-label adequately. No gap.
- [x] Qualitative pros/cons matrix — N/A, no orthogonal technique axes to compare.
- [ ] **"State the shorthand first" pattern not used.** Section 7's raw-logits-vs-softmax
  explanation currently jumps straight to "PyTorch returns raw logits" without first naming the
  Keras-trained instinct it's correcting (a reader coming from
  `Dense(10, activation="softmax")` reasonably expects to need an explicit softmax layer in
  PyTorch too). **Gap.**
- [~] "Expected outcome" grounding — N/A in the corpus sense (no text corpus), but the notebook
  already avoids inventing fake expected-accuracy numbers up front; it only reports measured
  results. No gap.
- [ ] **Two dense multi-step cells not yet split** (see 9.5 above). **Gap.**
- [ ] **9 stale cell-distance phrases** ("cells below," "next two cells," "cell below," "cell
  right after it," "next cell") found via grep — every one needs to become distance-free or name
  its target directly. **Gap** (mechanical, exhaustive fix required per Section 10.7).

## 3. Concrete, ordered changes for this notebook

1. **Insert a Table of Contents cell** right after the title cell (before the Roadmap cell),
   linking every `## Section N` header plus a handful of named subsections readers would want to
   jump to directly (the NHWC/NCHW gotcha, "Inspecting the model," the training-loop Code
   Walkthrough, "Gotchas worth remembering," "Next steps"). Include the Ctrl+F/outline-panel
   fallback caveat line.
2. **Add a "prove it" cell for the `CrossEntropyLoss` claim** (Section 4): a short markdown
   bridge + a code cell that computes `F.log_softmax` + `F.nll_loss` by hand on random dummy
   logits and asserts it matches `nn.CrossEntropyLoss()` numerically. This turns an asserted
   comment into a measured claim and sets up change #3.
3. **Restructure the Section 7 raw-logits explanation** using the Section 10.4 three-beat
   pattern: (a) name the common Keras-trained shorthand ("I need an explicit softmax layer"),
   (b) show what the PyTorch code actually does (raw logits + `CrossEntropyLoss`'s internal
   log-softmax — pointing back at the proof cell from change #2), (c) resolve where the softmax
   intuition actually lives (inside the loss, not the model) and why `argmax` alone is enough at
   inference time.
4. **Split the `SimpleCNN` cell** (class definition + instantiation + parameter-count print) into
   two smaller cells — a short intro before the class, a short intro before instantiation/param
   count — rather than one ~30-line cell.
5. **Split the `zero_grad` ablation cell** (`train_n_steps` helper + two calls + comparison print,
   ~36 lines) into: a short intro + helper-function cell, then a short intro + run-both-and-compare
   cell. The existing Code Walkthrough cell right after it stays as-is (it explains generic
   training-loop mechanics that apply to both halves, and doesn't claim "N steps combined into one
   cell," so it needs no retitling per 10.6).
6. **Add one more 🧪 exercise** after the flatten-size "What just happened" cell (Section 3): let
   the reader flip a `PADDING` knob on a probe conv stack and re-measure the flattened size with
   the same dummy-forward-pass technique already established, reinforcing the "measure, don't
   trust arithmetic" habit with a second, reader-driven example.
7. **Fix all 9 stale cell-distance phrases** found by grep (`cells below`, `next two cells`,
   `cell below` ×3, `cell right after it`, `next cell`) — replace with directional/named language
   ("further down," "the exercise further down," naming the artifact). Re-grep after edits to
   confirm zero remaining hits.

Not planned: toy/real bridge tables, narrative business framing, combination grids, animations —
all genuinely not applicable to this notebook's topic.
