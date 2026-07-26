# Improvement Plan — ML Basics

**Audited:** 2026-07-26 | **Audience fit:** 6/10

## Overall Assessment

Strong narrative hook (SmartVal's MAE ≤ $40k regulatory target) and excellent "🔮 Predict first" prompts are the pedagogical highlights. The closing decision and "When to Use What" table are strong finishes. However, four of six Parts open with LaTeX as their first substantive line — the pattern most likely to alienate an engineer who self-identifies as "not a math person." The loss-landscape image is in Part 1 when it belongs in Part 2. The weight-bar chart uses ASCII characters instead of matplotlib. The most important training-loop insight (learning rate too high/too low) is never demonstrated. No learner-agency "modify and observe" moments exist.

---

## Strengths (preserve these)

- **SmartVal MAE ≤ $40k constraint** — carried through all 6 Parts and answered in the Closing Decision
- **"🔮 Predict first" prompts** in Parts 1, 3, and 6 — three concrete options each, force commitment before code runs
- **Three images with good alt text** — well-placed visual purpose
- **"When to Use What" closing table** — plain English, reusable by audience
- **Part 6 learning curve** — `sample_sizes` sweep with green $40k target line delivers the overfitting lesson without any formula
- **"District 42" anchor** — `Predicted: $Xk | Actual: $Yk | Error: $Zk` makes prediction tangible
- **Gradient descent verified against sklearn** — builds deep trust in the math
- **Pipeline comment** ("Standardise → Train on 66% → Never touch test set") — right abstraction level

---

## Gaps & Recommended Changes

### Gap 1 — Formula-before-intuition in Parts 1, 2, 4, and 5 — Priority: High

**Problem:** Parts 1, 2, 4, and 5 open with LaTeX formulas before one sentence of hook. Part 1: MSE formula as the second sentence. Part 2: gradient update rule as the second sentence. Part 4: $\lambda \sum w_i^2$ before the business motivation. Part 5: $\hat{p} = \sigma(w^T x + b)$ before explaining why a probability is needed.

**Recommendation:** In each Part header, reorder: scenario/question → intuition sentence → formula labeled "the math version of that idea."
- Part 1: "SmartVal needs to know which of the 8 features drives price. Linear regression answers with a weight per feature — a multiplier. The formula is all eight multipliers written compactly: $\hat{y} = w_1 x_1 + \cdots$"
- Part 2: "Gradient descent: stand on the loss landscape, find the steepest downhill direction, step that way. Repeat. The 'steepest downhill direction' is the gradient: $\nabla_w\text{MSE} = \frac{2}{n}X^T(Xw-y)$"

---

### Gap 2 — `regression-loss-landscape.png` belongs in Part 2, not Part 1 — Priority: High

**Problem:** The loss-landscape image (a gradient-descent visualization) sits in Part 1's cell, but Part 2 (where gradient descent runs) has no visual support at all — the update rule arrives cold.

**Recommendation:** Move the image to immediately before the Part 2 gradient descent code cell. In Part 1 where it was, add one forward-reference line: "(The loss landscape this minimises — a bowl shape — is visualised before the gradient descent code in Part 2.)"

---

### Gap 3 — Feature weights are ASCII bars, not a chart — Priority: High

**Problem:** `bar = "+" * int(abs(w)*10)` produces a text column of pluses for an audience with strong graphical memory.

**Recommendation:** After the sklearn fit cell, add a 6-line matplotlib horizontal bar chart (sorted by magnitude, color-coded positive/negative). Remove the ASCII bar generation.

```python
sorted_pairs = sorted(zip(feature_names, lr.coef_), key=lambda x: x[1])
names_s, coefs_s = zip(*sorted_pairs)
colors = ['steelblue' if c > 0 else 'coral' for c in coefs_s]
fig, ax = plt.subplots(figsize=(7, 4))
ax.barh(names_s, coefs_s, color=colors)
ax.axvline(0, color='white', lw=0.8)
ax.set_title('Which features drive California house prices?')
plt.tight_layout(); plt.show()
```

---

### Gap 4 — Part 3 title says "MSE vs. MAE" but uses HuberRegressor — Priority: Medium

**Problem:** "Loss Functions: MSE vs. MAE" — but the code uses `HuberRegressor`, not a MAE-minimising model. The predict-first prompt frames this as an MSE vs. MAE comparison.

**Recommendation:** Option A (minimal): rename to "MSE vs. Robust Loss" and add: "True MAE minimisation requires a subgradient solver. HuberRegressor approximates MAE for large errors and MSE for small ones — giving us the best of both." Option B: add a 4-line plot showing shapes of MSE, MAE, and Huber loss on a single axis.

---

### Gap 5 — Missing: learning rate too-high / too-low demonstration — Priority: Medium

**Problem:** Part 2 runs 200 epochs at `lr_rate=0.01` and converges cleanly. The crucial intuition — learning rate too high → diverges; too low → sluggish — is never shown. An engineer will encounter diverging loss in the very next notebook.

**Recommendation:** Add a ~10-line cell after the convergence plot showing three learning rates (0.001, 0.01, 0.3) on the same axes with title "Too small → slow. Too large → diverges. Just right → smooth."

---

### Gap 6 — No "modify and observe" moments — Priority: Medium

**Problem:** Every cell is "run me and read the output." Engineers learn by tinkering.

**Recommendation:** Add one `🧪 Try it` markdown cell at the end of Parts 2 and 4:
- "Change `lr_rate = 0.01` to `0.5` and re-run. Does it still converge?"
- "Add `alpha=50` to the alphas list and re-run. Which features does Lasso zero out now?"

---

### Gap 7 — Closing Decision hardcodes `α=0.1` — Priority: Low

**Problem:** The Closing Decision unconditionally prints Ridge α=0.1 regardless of what Part 4 computed. The printed recommendation may contradict the Part 4 table visible two cells above.

**Recommendation:**
```python
print(f"  → Use Ridge regularisation (α={best_alpha_r}) for the regression model")
zeroed = (l_best.coef_ == 0).sum()
print(f"  → Lasso (α={best_alpha_l}) zeroes {zeroed}/8 features")
```

---

## Do NOT Change

- SmartVal framing and MAE ≤ $40k regulatory target
- "🔮 Predict first" prompts — format, position, three-option structure
- Train/val/test split logic and `StandardScaler` fit-on-train-only pattern
- Gradient descent manual implementation and sklearn `max_diff < 0.01` check
- "District 42" concrete example
- Tier 1/2/3 coverage transparency section
- "When to Use What" closing table
- Part 6 learning curve (`sample_sizes` sweep with green $40k target line)
- All three existing image alt texts
- The `→ Next:` forward pointer
