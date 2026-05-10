# SmartVal AI — Regression Grand Challenge

> **This is the capstone for the 01-Regression track.** Complete all 7 chapters first, then return here to integrate everything into a single production pipeline and test your understanding end-to-end.

---

## The Mission

You're the **Lead ML Engineer** at a real estate platform. Seven chapters ago the CEO asked for **SmartVal AI** — a production home valuation system that lenders, agents, and homebuyers can trust for multi-million-dollar decisions.

You've now built it. This document is the checkered flag.

---

## Constraint Scoreboard: Regression Track Final Status

| # | Constraint | Target | Status |
|---|------------|--------|--------|
| **#1** | **ACCURACY** | <$40k MAE on held-out districts | **Achieved** — Ch.7 XGBoost reaches $32k |
| **#2** | **GENERALIZATION** | No overfitting; holds on unseen districts | **Achieved** — Ridge regularization + 5-fold CV |
| **#3** | **MULTI-TASK** | Predict value AND market segment | ➡ Out of scope — continues in [02-Classification →](../02_classification) |
| **#4** | **INTERPRETABILITY** | Explainable predictions for underwriters | **Partially achieved** — VIF audit (Ch.3), SHAP feature importance (Ch.7) |
| **#5** | **PRODUCTION** | <100ms inference, scale, monitoring | **Pipeline established** — systematic tuning in Ch.7; deeper MLOps in [08-EnsembleMethods →](../08_ensemble_methods) |

---

## Journey Recap: $70k → $32k MAE

| Ch | Concept | Key Technique | MAE | Constraint |
|----|---------|---------------|-----|------------|
| Floor | Predict `median(y_train)` for every district | ~$72k MAE | Must beat this before claiming ML value |
| 1 | Linear Regression | Single-feature baseline (MedInc → MedHouseVal) | ~$70k | #1 Partial |
| 2 | Multiple Regression | All 8 features, vectorized gradient descent | ~$55k | #1 Improved |
| 3 | Feature Scaling, Importance & Multicollinearity | VIF audit, permutation importance, StandardScaler | ~$55k | **#4 Feature-level** |
| 4 | Polynomial Features | Degree-2 expansion (8 → 44 features) | ~$48k | #1 Close |
| 5 | Regularization | Ridge/Lasso/Elastic Net | **$38k** | **#1 #2 ** |
| 6 | Evaluation Metrics | Cross-validation, residual diagnostics, learning curves | $38k ± $2k | Validated |
| 7 | Hyperparameter Tuning | Optuna + XGBoost + SHAP | **$32k** | **#1 #2 #4 ** |

---

## Integration Exercises

Work through these in order. Each one requires knowledge from multiple chapters simultaneously — they cannot be solved by applying a single chapter in isolation.

---

### Exercise 1 — Reproduce the Full Pipeline from Scratch (Ch.1–5)

Without consulting chapter notebooks, build the regularized polynomial pipeline:

```python
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

data = fetch_california_housing()
X_train, X_test, y_train, y_test = train_test_split(
 data.data, data.target, test_size=0.2, random_state=42
)

# Build the pipeline: poly → scale → ridge
# Choose alpha and degree to achieve test MAE ≤ 0.40 (≤ $40k)
pipe = Pipeline([...])
```

**Success criterion:** Test MAE ≤ 0.40 (units are ×$100k, so 0.40 = $40k).

**Trap to avoid:** Fit the scaler on training data only, not the full dataset.

---

### Exercise 2 — Produce the Compliance Feature Audit (Ch.3)

The compliance officer needs a written report before launch. Build the three-view importance dashboard:

1. Univariate R² bar chart (8 features sorted descending)
2. VIF table — flag VIF > 5 with a recommendation
3. Permutation importance (30 repeats, test set)

**Deliverable:** A markdown table with a one-sentence justification for each feature's inclusion or exclusion. Which features would you drop if forced to cut to 5?

---

### Exercise 3 — Joint Hyperparameter Search (Ch.7)

Use Optuna to jointly tune `poly__degree` (1, 2, 3) and `ridge__alpha` (log-uniform [0.001, 1000]) with 5-fold CV and 100 trials.

Answer these questions:
- Does degree=3 with a tuned alpha beat degree=2?
- How much did systematic tuning improve on your hand-tuned baseline from Exercise 1?
- At what alpha does degree=3 stop overfitting? Plot the learning curve.

---

### Exercise 4 — XGBoost vs Ridge: The Final Showdown (Ch.7)

Train an XGBoost regressor using RandomizedSearchCV (50 trials, 5-fold CV). Fill in the comparison table:

| Model | CV MAE | Test MAE | Fit time | Interpretable? |
|-------|--------|----------|----------|----------------|
| Best Ridge (Exercise 3) | | | | coefficient weights |
| XGBoost (tuned) | | | | SHAP values |

Generate SHAP feature importances for the winning XGBoost model. Do the top-3 SHAP features match the permutation importance ranking from Exercise 2?

---

### Exercise 5 — Residual Stress Test (Ch.6)

Your CFO asks: "Where does the model fail — and by how much?"

1. Plot predicted vs actual. Mark any district where actual > $400k in red.
2. Plot residuals vs predicted. Does variance fan out at higher predictions?
3. Compute MAE separately for three price bands:
 - Budget: actual < $150k
 - Mid-range: $150k–$350k
 - Premium: actual > $350k

**Discussion:** The premium band almost certainly shows the highest MAE. Explain *structurally* why this is unavoidable given the California Housing dataset — and what you would do about it if you had access to uncapped sale prices.

---

### Exercise 6 — The Compliance Model Card

Write a one-page model card (≤ 400 words) that a non-technical underwriter could read:

- Model choice and brief rationale (Ridge polynomial vs XGBoost — which did you pick and why?)
- Training data: what it is, geographic scope, date of collection, any known biases
- Performance: overall test MAE, per-band MAE from Exercise 5
- Top 3 features driving predictions (name them, state the direction, cite SHAP or permutation importance)
- Known limitations: the $500k capping artifact, geographic extrapolation to non-CA markets
- Recommended monitoring: how often to retrain, what drift signal to watch

---

## Self-Assessment Rubric

| Dimension | Not Yet | Basic | Proficient | Expert |
|-----------|---------|-------|------------|--------|
| Pipeline | Can't build from memory | Adapts existing code | Builds end-to-end from scratch | Adds data validation + edge-case handling |
| Feature Audit | Skips this step | Runs one method | All three methods | Explains why rankings differ and what to do |
| Hyperparameter Tuning | Manual guessing | Grid search | Random / Bayesian | Joint search + convergence analysis |
| Evaluation | Single split MAE | 5-fold CV MAE | Per-band breakdown | Prediction intervals + residual diagnostics |
| Communication | Technical jargon only | Clear prose | Non-technical explanation | Compliance-grade model card with limitations |

---

## What Comes Next

You've satisfied **#1 ACCURACY** and **#2 GENERALIZATION** and made solid progress on **#4 INTERPRETABILITY**. Three paths forward:

**Still need classification (#3 MULTI-TASK)?**
→ [02-Classification →](../02_classification) — FaceAI on CelebA (logistic regression through SVMs)

**Want neural networks to unify regression and classification?**
→ [03-NeuralNetworks →](../03_neural_networks) — UnifiedAI ($28k MAE + 95% accuracy, same architecture)

**Want deeper ensemble and production tooling (#5)?**
→ [08-EnsembleMethods →](../08_ensemble_methods) — stacking, LightGBM, CatBoost, and production serving
