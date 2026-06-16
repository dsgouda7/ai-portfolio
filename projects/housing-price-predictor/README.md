# Housing Price Predictor

## Problem statement

**Can a regression model trained on publicly available property records predict sale prices accurately enough to be useful for buyers and sellers estimating fair market value?**

Estimating the value of a property without a formal appraisal requires either expensive professional services or crude heuristics (price per square foot, nearby comps). This project builds a model on King County, Washington housing sales data that predicts sale price from observable property attributes, exploring how far simple regression gets and where non-linearity and interaction terms make a measurable difference.

**Constraints we set for ourselves:**
- Only features observable before sale (no post-sale data)
- No paid data sources — public King County assessor records only
- Interpretability matters: a buyer should be able to understand why the model predicts a given price

**Result:** Ridge regression with polynomial features achieves R² ≈ 0.88 on a held-out test split, with median absolute error under $35k on properties spanning $75k–$7.7M.

## Dataset

| Source | What it provides | How it's used |
|---|---|---|
| [King County House Sales](https://www.kaggle.com/datasets/harlfoxem/housesalesprediction) | 21,613 residential property sales from May 2014–May 2015 with 21 features (sqft, bedrooms, grade, condition, lat/long, etc.) | Training and evaluation |

## How it works

**Target:** `price` (sale price in USD)

**Feature engineering:**
- Log-transform `price`, `sqft_living`, and `sqft_lot` to reduce right skew
- Polynomial expansion (degree 2) on the top correlated features to capture non-linear relationships between size, grade, and price
- `waterfront`, `view`, `condition`, and `grade` treated as ordinal
- Lat/long retained as raw numerics — the model learns neighbourhood effects implicitly

**Model progression:**
1. Linear regression baseline — establishes the floor
2. Ridge regression — controls multicollinearity from correlated sqft features
3. Ridge + polynomial features — best trade-off of accuracy vs interpretability

## Metrics

| Model | R² (test) | Median abs. error | Notes |
|---|---|---|---|
| Linear regression baseline | ~0.70 | ~$55k | Underfit on high-value properties |
| Ridge regression | ~0.79 | ~$45k | Collinearity controlled |
| **Ridge + polynomial features** | **~0.88** | **~$32k** | Best result |

## Limitations

The dataset covers a single county in a single year. Temporal effects (seasonality, market cycles) are not modelled. The model has no knowledge of interior condition, renovation quality, or neighbourhood amenities beyond what is captured in `grade` and `condition`. Predictions outside the training distribution (ultra-luxury, rural outliers) are unreliable.

## Run

```bash
# Requires Python 3.9+ and Jupyter
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
jupyter notebook house-price-modeling.ipynb
```

Open the notebook and run top to bottom. All data loading, EDA, feature engineering, and model evaluation steps are documented inline.
