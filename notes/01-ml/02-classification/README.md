# Classification Track

> **The Mission**: Master the **Attribute Trinity** — build three specialized binary classifiers (Smiling, Eyeglasses, Bald) that achieve >85% F1-score across dramatically different class distributions (48% → 13% → 2.5%), proving you can handle balanced data, moderate imbalance, and severe imbalance with the same classical ML toolkit.

This is not an academic exercise. Every chapter builds toward a single production challenge: you're the ML Engineer at a photo-tech startup, and the product team demands classifiers that reliably detect three critical facial attributes across celebrity faces the model has never seen. Each attribute presents a different battle: **Smiling** (balanced, 48% prevalence) teaches precision fundamentals, **Eyeglasses** (moderate imbalance, 13%) demands threshold tuning, and **Bald** (severe imbalance, 2.5%) forces you to abandon accuracy as a metric entirely.

In Topic 01 (Regression) you predicted **continuous values** — house prices with a real-valued output. Classification is different: the model must choose a **category**. The same gradient-descent machinery applies, but with a new output transformation (sigmoid), a new loss (cross-entropy), and a new evaluation vocabulary (precision, recall, F1).

---

## The Grand Challenge: 5 Attribute Trinity Constraints

| # | Constraint | Target | Why It Matters |
|---|------------|--------|----------------|
| **#1** | **BALANCED PERFORMANCE** | >85% F1-score on all 3 attributes (Smiling, Eyeglasses, Bald) | Naive accuracy hides failures: 96% accuracy on Bald by always predicting "Not-Bald" is useless. F1 forces precision AND recall balance |
| **#2** | **GENERALIZATION** | Work on unseen celebrity faces | Can't memorize training faces. Must learn attribute patterns (smile curves, frame edges, scalp reflectance) that transfer to new people |
| **#3** | **CLASS IMBALANCE MASTERY** | Handle 48% → 13% → 2.5% prevalence progression | Real-world data is imbalanced. Master balanced (Smiling), moderate (Eyeglasses), and severe (Bald) scenarios with same toolkit |
| **#4** | **INTERPRETABILITY** | Visualize which features drive each prediction | Product team needs to explain *why* "Bald" was predicted for user trust. Debug edge cases (hats vs. bald) visually |
| **#5** | **EFFICIENCY** | <100ms inference per image for 3 classifiers | Photo app needs near-instant tagging. Classical ML (logistic, SVM, trees) must prove viability before investing in neural networks |

---

## Progressive Capability Unlock

| Ch | Title | What Unlocks | F1-Score | Constraints | Status |
|----|-------|--------------|----------|-------------|--------|
| **1** | [Logistic Regression](ch01_logistic_regression) | Binary baseline: Smiling (88% F1) | Smiling: 88% | #1 Partial, #5 ✅ | 🔧 |
| **2** | [Classical Classifiers](ch02_classical_classifiers) | Interpretable rules: Decision trees show "if pixel[32,40] > 128 → Smiling" | Smiling: 85% | #4 Partial | 🔧 |
| **3** | [Evaluation Metrics](ch03_metrics) | **Attribute Trinity unlocked**: measure all 3 with F1, expose Bald's 12% recall | Smiling: 88%<br>Eyeglasses: 76%<br>Bald: 12% recall! | #1 Exposed, #3 Recognized | 🔧 |
| **4** | [Support Vector Machines](ch04_svm) | Maximum-margin separation improves Eyeglasses | Smiling: 89%<br>Eyeglasses: 82% | #1 Improved | 📋 |
| **5** | [Hyperparameter Tuning](ch05_hyperparameter_tuning) | Per-attribute threshold tuning conquers Bald | **Smiling: 92%**<br>**Eyeglasses: 87%**<br>**Bald: 86%** | **#1 ✅ #2 ✅ #3 ✅** | 📋 |

> **Track scope**: This 5-chapter track achieves **all 5 constraints** with the Attribute Trinity (3 attributes). **Why 3, not 40?** Each of the 3 attributes represents a distinct class imbalance regime you'll encounter in production ML:
> - **Smiling (48%)**: Balanced data — learn fundamentals without imbalance complications
> - **Eyeglasses (13%)**: Moderate imbalance — precision-recall tradeoffs become critical
> - **Bald (2.5%)**: Severe imbalance — accuracy becomes meaningless, F1 and threshold tuning are mandatory
>
> Mastering these 3 **proves** you can handle any binary classification problem. Scaling to 40 attributes (Constraint #3 in original challenge) requires **multi-output neural networks** with shared representations, which is the natural next step in [Topic 03 — Neural Networks](../03_neural_networks/README.md).

> ✔️ **Why this progression works**: Classical ML forces you to understand the *math* (sigmoid, cross-entropy, decision boundaries) before neural networks automate it. The Trinity's escalating difficulty ensures you've debugged imbalance at small scale before tackling deep learning's complexity.

> † F1-score is used after Ch.3; before that, naive accuracy hides class imbalance (e.g., 96% accuracy on Bald by always predicting "Not-Bald").

---

## Narrative Arc: From 88% Baseline to 92% Tuned System

### Act 1: Binary Foundations (Ch.1–2)
**Build simple classifiers, understand their limits**

- **Ch.1**: Can logistic regression detect Smiling? → Yes, ~88% F1-score (decent baseline!)
  - *"88% F1 on Smiling is promising. But what about Eyeglasses (13% of faces) and Bald (2.5%)? We need all three." — Product Lead*

- **Ch.2**: Decision trees, KNN, Naive Bayes → ~85% F1 but interpretable rules
  - *"I love that the tree shows 'if pixel[32,40] > 128 → likely Smiling'. But 85% is lower than logistic regression?" — CEO*

**Status**: ❌ Need proper evaluation and better models.

---

### Act 2: Measurement & Margin (Ch.3–4)
**Learn to measure correctly, then push the boundary**

- **Ch.3**: Proper metrics → 88% F1 on Smiling was real, but **Bald recall is only 12%!**
  - *"96% accuracy on Bald by always predicting Not-Bald? That's the accuracy paradox. We need F1-score for all 3 attributes." — Data Scientist*
  - **Trinity exposed**: Smiling 88% F1, Eyeglasses 76% F1, Bald 12% recall (disaster!)

- **Ch.4**: SVM with RBF kernel → ~89% F1 on Smiling, 82% on Eyeglasses
  - *"SVM finds the widest gap between Smiling and Not-Smiling in feature space. Eyeglasses improved! But Bald is still broken." — ML Lead*

**Status**: ✅ Proper evaluation framework. SVM improves accuracy.

---

### Act 3: Optimization (Ch.5)
**Tune everything, unlock production quality**

- **Ch.5**: Grid/Random/Bayesian search + per-attribute threshold tuning → **All 3 attributes >85% F1 ✅**
  - *"Per-attribute threshold tuning pushed Bald F1 from 12% to 86%! Smiling hit 92%, Eyeglasses 87%. We conquered the Trinity." — ML Engineer*

**Status**: ✅✅✅ All 5 constraints achieved: Balanced Performance (#1), Generalization (#2), Imbalance Mastery (#3), Interpretability (#4 partial), Efficiency (#5).

---

## Chapter Structure

Each sub-chapter README uses numbered `§N` sections: `§0 · The Challenge`, `§1 · Core Idea`, `§2 · Running Example`, `§3 · Math`, and so on through `§N · Progress Check`. This mirrors the notebook cell order — read and run in sync.

---

## The Dataset: CelebA (Celebrity Faces Attributes)

Every chapter uses the same dataset: [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) (official site — if unavailable, use the Kaggle mirror `jessicali9530/celeba-dataset`) — 202,599 celebrity face images with 40 binary attribute annotations.

**The Attribute Trinity** (selected from 40 attributes):
- **`Smiling` (48%)** — Balanced target, teaches classification fundamentals (Ch.1–5)
- **`Eyeglasses` (13%)** — Moderate imbalance, introduces precision-recall tradeoffs (Ch.3–5)
- **`Bald` (2.5%)** — Severe imbalance, forces F1-score and threshold tuning mastery (Ch.3–5)

**Why these 3?** They form a natural progression in class imbalance severity, each requiring different evaluation and optimization strategies. Other attributes in CelebA (Male 42%, Young 77%, Wearing_Hat 4.8%, Mustache 4.2%) are available for exploration but not part of the core challenge.

**Image format**: 178×218 color → resized to 64×64 grayscale for classical ML (flattened to 4,096 features or HOG descriptors)

**Why CelebA is perfect for the Attribute Trinity**:
- ✅ Natural binary labels (not manufactured thresholds)
- ✅ Natural class imbalance progression (Smiling 48% → Eyeglasses 13% → Bald 2.5%)
- ✅ Visual intuition (see what the model gets right/wrong on each attribute)
- ✅ Standard research benchmark with known baselines
- ✅ Forces proper metrics (accuracy fails spectacularly on Bald)
- ✅ Prepares for multi-label (scaling 3 → 40 attributes in Neural Networks track)

---

## How to Run

Each chapter has a `notebook.ipynb_solution.ipynb` (reference) or `notebook.ipynb_exercise.ipynb` (practice) that uses a subset of CelebA (5,000 images for speed). Notebooks auto-download the dataset on first run via `torchvision.datasets.CelebA` or fall back to a synthetic proxy if download fails.

```bash
cd notes/01-ml/02_classification/ch01_logistic_regression
jupyter notebook notebook.ipynb
```
