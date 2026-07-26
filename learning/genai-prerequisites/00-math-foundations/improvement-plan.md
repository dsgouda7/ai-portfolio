# Improvement Plan — Math Foundations for ML

**Audited:** 2026-07-26 | **Audience fit:** 6/10

## Overall Assessment

Strong bones — the free-kick scenario runs through all six Parts and the closing decision is genuinely satisfying. But for an audience with weak math background, the notebook repeatedly fails its own goal: in Parts 1, 4, 5, and 6 the LaTeX formula is the *first* thing the reader encounters, before any intuition is established. That is the opposite of the intended order. The most important line in the notebook — `theta = theta - lr * grad` — has no "why" comment. Diagram density is thin (3 images across 6 Parts, none in Parts 1 or 4). A non-math reader will follow Parts 1–3 reasonably, hit Part 4 at full formula speed, and finish Part 6 with a pretty plot they don't quite own.

---

## Strengths (preserve these)

- **"Predict first" prompts in Parts 2 and 3** — the notebook's best pedagogical feature; force commitment before the code reveals the answer
- **Closing Decision cell** — pulls gradient descent output, scoreable window, and P(scoring) together; deeply satisfying
- **Summary table** — Part | Tool | Free kick result | ML connection mapping
- **"What just happened — and what's missing"** cell at end of Part 3 — plants the right question for Part 5
- **Chain-rule computation graph image** — correct placement in Part 5
- **Free-kick narrative thread** — holds through all 6 Parts without feeling forced

---

## Gaps & Recommended Changes

### Gap 1 — Formula-before-intuition in Parts 1, 4, 5, and 6 — Priority: High

**Problem:** Four of six Parts open with LaTeX as their first substantive line before one sentence of physical intuition.

- Part 1: dot product formula before "alignment" means anything
- Part 4: `y = Wx + b` in the first sentence
- Part 5: chain rule fraction before any "composed functions" analogy
- Part 6: Gaussian CDF formula before any noise analogy

**Justification:** Engineers with strong graphical memory and weak symbol fluency disengage when formulas arrive before the hook. The formula should be *revealed* after the intuition makes it inevitable.

**Recommendation:** Add 2–3 sentences of physical intuition before the formula in each of these four Parts. Examples:
- Part 1: "Imagine rotating your kick direction toward the goal. At exactly the right angle they're pointing the same way — that's maximum overlap. The dot product measures that overlap as a single number."
- Part 4: "Each number in the weight matrix is a dial. A matrix multiply is just all those dials applied at once." — then `y = Wx + b`.
- Part 5: "Functions can be chained: goal_height depends on wall_clearance which depends on angle. To know how much goal_height changes when we tweak angle, we multiply the individual sensitivities together." — then the fraction.
- Part 6: "A real kick won't land at exactly 22°. Wind and muscle jitter push it left or right. The bell curve describes how spread out those misses are." — then the formula.

---

### Gap 2 — `theta = theta - lr * grad` has no "why" comment — Priority: High

**Problem:** The single most important line in the notebook appears in the gradient descent loop with no geometric comment. A reader sees arithmetic, not concept.

**Justification:** This is the entire concept of gradient descent. Skipping its explanation while annotating mundane lines is a priority inversion.

**Recommendation:**
```python
# gradient > 0 means loss rises if we increase theta → step LEFT (subtract)
# gradient < 0 means loss rises if we decrease theta → step RIGHT  
# lr controls how big each step is — too large → overshoot, too small → slow
theta = theta - lr * grad   # walk one step downhill on the loss surface
```

---

### Gap 3 — No diagram for vector alignment in Part 1 — Priority: Medium

**Problem:** Part 1 computes dot product = 0.9962 with no visual showing two arrows and the angle between them. The cos θ = 1 / 0 / −1 interpretation is stated in text only.

**Justification:** The audience has strong graphical memory. "Alignment" stays abstract without a picture. This is also the concept that directly becomes attention scores.

**Recommendation:** Add `![...](images/vector-alignment-dot-product.png)` after the formula, or a matplotlib cell showing kick vector and goal vector as arrows with the angle labeled. Even a three-panel "cos θ = 1 | 0 | -1" diagram resolves this.

---

### Gap 4 — Part 4 free-kick connection is the weakest Part — Priority: Medium

**Problem:** The feature vector `[0.4, 0.8]` (angle=40%, speed=80%) feels contrived. The grid-transformation uses a generic `W_vis` matrix with no free-kick context.

**Recommendation:** Frame Part 4 as "Gradient descent found the optimal angle. Now we use that angle as one of many inputs to a fuller model." Then use `features = np.array([optimal_angle / 90, v0 / 30])` — the actual gradient-descent result — as the input vector, making the connection to Part 3 explicit.

---

### Gap 5 — Tier ledger placement and tone — Priority: Low

**Problem:** The "What This Covered and Didn't" section lists Hessians, Taylor series, information theory — reading as a list of things the learner *didn't* learn, deflating after a satisfying closing cell.

**Recommendation:** Move to the very end (after "What's Next"), or collapse to a one-liner: "Topics intentionally deferred: Hessians, Taylor series, KL divergence — not needed for first-principles gradient descent."

---

## Do NOT Change

- The "Predict first" prompts in Parts 2 and 3
- The "Closing Decision" cell — the notebook's emotional endpoint
- The Summary table structure (Part | Tool | Free kick result | ML connection)
- The "What just happened — and what's missing" transition cell at end of Part 3
- The chain-rule autograd vs. numerical gradient verification in Part 5
- The `images/chain-rule-computation-graph.png` reference placement in Part 5
- The "When to Use What" lookup table
- The `→ Next:` navigation cell
