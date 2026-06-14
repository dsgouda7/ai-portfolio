# Feature Engineering — Chapter Overview

> **The scenario**: You're the lead data scientist at a grocery chain predicting weekly unit sales across 1,800 SKUs and 50 stores. Raw data: store ID, product category, date, historical sales. After naive engineering you have 300+ candidate features. Validation MAPE is 34% — well above the 18% procurement target. The model is slow, overfit, and missing temporal signal entirely. This chapter is about fixing that.

## What You'll Learn

- How to encode categorical variables without poisoning your model with target leakage
- How to extract signal from time-series data without leaking the future into the past
- When TF-IDF outperforms BERT embeddings — and it happens more often than you think
- How to reduce 300 candidate features to the 60 that carry 80% of the predictive signal
- The correct pipeline order, and why getting it wrong silently corrupts your training data

## Key Concepts

| Concept | Section | Core Challenge |
|---------|---------|----------------|
| Ordinal, one-hot, and target encoding | § 2 | High-cardinality categories without leakage |
| Cross-fold target encoding with smoothing | § 2 | Preventing mean-encoding leakage |
| Lag features and rolling statistics | § 3 | Temporal signal without future leakage |
| Cyclic encoding for periodic features | § 3 | Sunday→Monday discontinuity |
| TF-IDF vs learned embeddings | § 4 | Domain-specific jargon vs. semantic similarity |
| Variance threshold + correlation filter | § 5 | Computational cost triage |
| Mutual information and Lasso selection | § 5 | Non-linear relationships + embedded sparsity |
| Permutation importance audit | § 5 | Model-faithful feature ranking |
| Full pipeline order and leakage traps | § 6 | Why ordering matters |

## The Full Chapter

[feature-engineering.md](feature-engineering.md)

---

*Companion notebooks in `playground/ml-features/`:*

- `01_feature_engineering.ipynb` — overview and pipeline walkthrough
- `02_categorical_encoding.ipynb` — ordinal, one-hot, target encoding with leakage demo
- `03_temporal_features.ipynb` — lags, rolling windows, cyclic encoding
- `04_text_tfidf.ipynb` — TF-IDF vectorization on product names
- `05_text_embeddings.ipynb` — sentence-transformer embeddings vs. TF-IDF head-to-head
- `06_feature_selection.ipynb` — all five selection methods on the grocery pipeline
