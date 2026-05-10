# Ch.3 — Feature Engineering: Inspect → Audit → Transform → Validate

> **The story.** In **1885** Francis Galton measured the heights of 928 parents and children and asked a question that no one before him had properly quantified: "Which part of a parent's height actually predicts the child's height, and which part is just noise?" His answer — correlation — was the first formal tool for ranking how much a single variable contributes to an outcome. Pearson formalised it in **1895**. Sixty years later, Hoerl and Kennard (1970) rediscovered what happens when two predictors are *too* correlated: the weights blow up and become uninterpretable, even when predictions stay fine. Their fix — Ridge regression — lives in Ch.5. This chapter is where you learn to *see* the problem before you reach for the fix.
>
> **Where you are in the curriculum.** Ch.2 gave us all 8 features and dropped MAE from \$70k to \$55k. But we trained the model, printed the weights, and moved on. We have not yet asked: which of these 8 features is genuinely useful, which is redundant because it overlaps with another, and which is nearly inert? Before adding polynomial interactions (Ch.4) or applying regularization (Ch.5) — which will create or prune features — you need an honest diagnostic pass. This chapter is that diagnostic.
>
> **Notation in this chapter.** $\rho(x_j, y)$ — Pearson correlation of feature $j$ with the **target** (feature-target); $\rho(x_j, x_k)$ — Pearson correlation between features $j$ and $k$ (**inter-feature**); note that the same formula is used in both cases — only the second argument changes; $R^2_j$ — univariate R² of feature $j$ alone; $|w_j^{\text{std}}|$ — absolute standardised weight (partial contribution); $\text{VIF}_j = 1/(1-R^2_{j,\text{feat}})$ — Variance Inflation Factor where $R^2_{j,\text{feat}}$ is the R² from regressing feature $j$ on all other features; $\pi_j$ — permutation importance of feature $j$; $\pi_{jk}$ — joint permutation importance of the pair $(j,k)$; $\Delta_{\text{interact}}(j,k) = \pi_{jk} - \pi_j - \pi_k$ — interaction uplift.

---

## 0 · The Challenge — Where We Are

> 💡 **The mission**: Launch **SmartVal AI** — a production home valuation system satisfying 5 constraints:
> 1. **ACCURACY**: <\$40k MAE — 2. **GENERALIZATION**: Unseen districts — 3. **MULTI-TASK**: Value + Segment — 4. **INTERPRETABILITY**: Explainable — 5. **PRODUCTION**: Scale + Monitor

**What we know so far:**
- ✅ Ch.1: Single feature (MedInc) → \$70k MAE
- ✅ Ch.2: All 8 features → \$55k MAE — 21% better
- ❌ **But we don't know WHY it's better, or which features are doing the work**

**What's blocking us:**

The compliance officer at SmartVal just asked: *"Which features drive your valuations? If we remove Latitude, does the model collapse? Are AveRooms and AveBedrms really two different signals?"*

You can't answer any of those questions from Ch.2's output. You have weights, but raw weights depend on the feature scales. You have R² = 0.61, but you don't know how much came from income vs location. And if two features are measuring the same thing, adding polynomial interactions in Ch.4 will only amplify the noise — you need to understand the feature space *first*.

**What this chapter unlocks:**
- How much variance each feature explains **alone** (univariate R²)
- Which features overlap so heavily that one is redundant (VIF / multicollinearity)
- The most stable **joint** ranking of feature contributions (permutation importance)
- Which feature *pairs* are **stronger together** than the sum of their parts (joint permutation, interaction uplift)
- A **bar chart** and **heatmap** you can hand to any stakeholder

> ⚡ **Constraint #4 — Interpretability unlocked:** You can now explain to the compliance officer *which* features drive SmartVal predictions, *why* certain weight pairs are unstable (AveRooms/AveBedrms collinearity), and *which* feature is safe to drop (Population). This is the foundation for explainable valuation decisions required by lending regulations.

This does not change the MAE. It changes your understanding of *why* you're at \$55k and what levers exist to push lower.

---

## Animation

![Chapter animation](img/ch03-feature-importance-needle.gif)

---

## 1 · The Feature Engineering Workflow — Your 4-Phase Diagnostic

Before diving into the mathematical methods, understand the **practitioner workflow** you'll follow with every new dataset:

```
Phase 1: INSPECT           Phase 2: AUDIT              Phase 3: TRANSFORM        Phase 4: VALIDATE
────────────────────────────────────────────────────────────────────────────────────────────────
Look at each feature       Check for redundancy        Build transformation      Rank features by
distribution:             between feature pairs:       pipeline:                 importance:

• Compute skew, IQR        • Correlation heatmap       • Log transform skewed    • Univariate R²
• Plot histograms          • Calculate VIF             • StandardScaler all      • Standardized weights
• Identify outliers        • Flag VIF > 5              • ColumnTransformer      • Permutation importance
                                                         for mixed types

DECISION: Which scaler?    DECISION: Drop/merge?       DECISION: Pipeline order  DECISION: Keep/drop?
→ Skew > 1: log1p         → VIF > 10: Drop one        → Always fit on train     → Perm. importance
→ Heavy outliers:         → VIF 5-10: Monitor           only!                     near zero: drop
  RobustScaler            → VIF < 5: Keep both        → Transform train/test    → VIF > 5 + low perm:  
→ Symmetric: Standard                                   consistently               drop redundant one
