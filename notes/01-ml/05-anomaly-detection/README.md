# Anomaly Detection Track

> **The Mission**: Build **FraudShield** — a production-grade fraud detection system that catches 80%+ of fraudulent transactions while keeping the false positive rate below 0.5%, all within 100ms inference latency.

This is not a Kaggle competition. Every chapter builds toward a single production challenge: you're the Lead ML Engineer at a payment processor, and the Head of Risk demands a system that catches fraud without blocking legitimate customers — on a dataset where only **0.17% of transactions are fraudulent**.

---

## The Grand Challenge: 5 Core Constraints

| # | Constraint | Target | Why It Matters |
|---|------------|--------|----------------|
| **#1** | **DETECTION** | >80% recall (catch most fraud) | Every missed fraud = financial loss + regulatory risk. Missing 20% is the maximum acceptable leak |
| **#2** | **FALSE POSITIVE RATE** | <0.5% FPR (don't block legit transactions) | Blocking legitimate customers = churn + support costs. 0.5% FPR on 284k transactions = ~1,420 false alarms |
| **#3** | **REAL-TIME** | <100ms inference per transaction | Transactions must be approved/declined instantly. Batch processing is not acceptable for point-of-sale |
| **#4** | **ADAPTABILITY** | Handle evolving fraud patterns (concept drift) | Fraudsters adapt. A model trained on September 2013 data degrades within weeks without drift detection |
| **#5** | **EXPLAINABILITY** | Justify flagged transactions to compliance | Regulators require reasons for declined transactions. "The model said so" doesn't satisfy auditors |

---

## Progressive Capability Unlock

| Ch | Title | What Unlocks | Recall@0.5%FPR | Constraints | Status |
|----|-------|--------------|----------------|-------------|--------|
| **1** | [Statistical Anomaly Detection](ch01_statistical_methods) | Z-score, IQR baselines on transaction amounts | ~45% | #1 Partial | 🔧 |
| **2** | [Isolation Forest](ch02_isolation_forest) | Tree-based path-length anomaly scoring | ~72% | #1 Improved | 🔧 |
| **3** | [One-Class SVM](ch04_one_class_svm) | Kernel-based novelty detection boundary | ~75% | #1 Close | 🔧 |
| **4** | [Autoencoders](ch03_autoencoders) | Neural reconstruction error as anomaly signal | ~78% | #1 Close! | 🔧 |
| **5** | [Ensemble Anomaly Detection](ch05_ensemble_anomaly) | Score fusion across all detectors | **~83%** | **#1 ✅ #2 ✅** | 🔧 |
| **6** | [Production & Real-Time](ch06_production) | Online learning, drift detection, latency optimization | 83%+ | **#1 ✅ #2 ✅ #3 ✅ #4 ✅ #5 ✅** | 🔧 |

---

## Narrative Arc: From 45% Baseline to 83%+ Production System

### 🎬 Act 1: Foundations (Ch.1-2)
**Establish baselines, understand the imbalance problem**

- **Ch.1**: Can simple statistics detect fraud? → Z-score catches 45% but misses subtle patterns
  - *"45% recall is embarrassing. We're missing more than half of all fraud!" — Head of Risk*

- **Ch.2**: Isolation Forest exploits the key insight — anomalies are easier to isolate → 72% recall
  - *"72% is promising, but we're still missing 28% of fraud. That's $2.8M in losses per quarter." — CFO*

**Status**: ❌ Detection constraint unmet. Need more sophisticated methods.

---

### ⚡ Act 2: Deep Learning & Kernel Methods (Ch.3-4)
**Learn normal transaction patterns, flag deviations**

- **Ch.3**: One-Class SVM draws a tight boundary around normal data in kernel space → 75% recall
  - *"75% recall from a kernel boundary! Simpler than deep learning and already beating the isolation forest on this data." — Data Science Lead*

- **Ch.4**: Autoencoders learn to reconstruct normal transactions; fraud = high reconstruction error → 78% recall
  - *"78%! The autoencoder is learning what normal looks like — reconstruction error catches subtle fraud patterns the SVM misses. But we still need 80%+." — VP Engineering*

**Status**: ❌ Still 2% short of the 80% target. Individual models have complementary strengths.

---

### 🏆 Act 3: Ensemble & Production (Ch.5-6)
**Combine detectors, deploy to production**

- **Ch.5**: Ensemble fuses Z-score + Isolation Forest + Autoencoder + One-Class SVM → **83% recall ✅ Target achieved!**
  - *"83% recall at 0.5% FPR! The ensemble catches fraud that no single model finds." — Head of Risk*

- **Ch.6**: Production hardening — online learning for concept drift, latency optimization, monitoring dashboards
  - *"FraudShield is live: 83% recall, <50ms latency, drift alerts, and every flag has an explanation." — CTO*

**Status**: ✅✅✅✅✅ **ALL CONSTRAINTS SATISFIED!**

---

## The Dataset: Credit Card Fraud Detection

Every chapter uses the same dataset: [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) from Kaggle (originally from ULB Machine Learning Group).

**284,807 transactions** (September 2013, European cardholders) with 31 features:

- `Time` — Seconds elapsed from first transaction in dataset
- `V1`–`V28` — PCA-transformed features (anonymized for privacy)
- `Amount` — Transaction amount (€)
- `Class` — Target: 0 = legitimate, 1 = fraud

**Target**: `Class` — binary fraud indicator

**The Central Challenge: Extreme Class Imbalance**

```
Legitimate: 284,315 transactions (99.827%)
Fraudulent:     492 transactions  (0.17%)

Ratio: 577:1
```

A model that predicts "legitimate" for every transaction gets **99.83% accuracy** — and catches **zero fraud**. This is why accuracy is meaningless here. We must optimize for **recall at a fixed false positive rate**.

**Why this dataset is perfect**:
- ✅ **Extreme imbalance**: 0.17% fraud (teaches the hardest real-world problem)
- ✅ **PCA anonymized**: V1-V28 protect cardholder privacy while remaining analytically useful
- ✅ **Temporal structure**: Time-ordered transactions enable realistic train/test splits
- ✅ **Real stakes**: Every method must prove its worth against the FPR constraint
- ✅ **Industry standard**: The most-used anomaly detection benchmark in ML

**Evaluation Protocol** (consistent across all chapters):

```python
# We fix FPR at 0.5% and measure recall at that threshold
from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_test, scores)
# Find threshold where FPR ≤ 0.005
idx = np.where(fpr <= 0.005)[0][-1]
recall_at_target_fpr = tpr[idx]
```

---

## Key Concepts Across Chapters

| Concept | Where Introduced | Why It Matters |
|---------|-----------------|----------------|
| Class imbalance | Ch.1 (everywhere) | 0.17% fraud means standard metrics are useless |
| Anomaly scoring | Ch.1 (Z-score) | Convert "normal vs. abnormal" into a continuous score |
| Contamination | Ch.2 (Isolation Forest) | Prior estimate of anomaly fraction in training data |
| Reconstruction error | Ch.3 (Autoencoders) | Learn normality; deviations = anomalies |
| Novelty detection | Ch.4 (One-Class SVM) | Train only on normal data; new patterns = novel |
| Score fusion | Ch.5 (Ensemble) | Combine complementary detector strengths |
| Concept drift | Ch.6 (Production) | Fraud patterns evolve; models must adapt |

---

## Success Criteria

Students who complete this track can:
1. ✅ Apply statistical anomaly detection (Z-score, IQR, Mahalanobis) and understand their limits
2. ✅ Deploy Isolation Forests for scalable outlier detection
3. ✅ Build autoencoders that learn normal patterns and flag reconstruction errors
4. ✅ Use One-Class SVM for novelty detection in high-dimensional space
5. ✅ Handle extreme class imbalance (0.17% positive class) with proper evaluation
6. ✅ Build ensemble anomaly detectors that exceed any single model
7. ✅ Deploy real-time fraud detection with concept drift monitoring

**Grand Challenge Completion**: FraudShield achieves 83% recall @ 0.5% FPR ✅
