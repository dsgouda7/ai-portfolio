# SmartVal AI — Coding Guidelines & Hints

This document provides production patterns, best practices, and implementation hints for completing the SmartVal AI exercise.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Data Pipeline Patterns](#data-pipeline-patterns)
3. [Feature Engineering Best Practices](#feature-engineering-best-practices)
4. [Model Training Patterns](#model-training-patterns)
5. [Error Handling](#error-handling)
6. [Logging Strategy](#logging-strategy)
7. [Testing Guidelines](#testing-guidelines)
8. [API Design Patterns](#api-design-patterns)
9. [Common Pitfalls](#common-pitfalls)

---

## Architecture Overview

SmartVal AI follows a **layered architecture**:

```
┌─────────────────────────────────────┐
│        API Layer (Flask)             │  ← User-facing REST API
├─────────────────────────────────────┤
│     Evaluation & Monitoring          │  ← Metrics, diagnostics
├─────────────────────────────────────┤
│       Model Registry                 │  ← Train, predict, persist
├─────────────────────────────────────┤
│     Feature Engineering              │  ← Transform raw features
├─────────────────────────────────────┤
│        Data Loading                  │  ← Load and split dataset
└─────────────────────────────────────┘
```

**Key Principle:** Each layer has a single responsibility and is independently testable.

---

## Data Pipeline Patterns

### ✅ Correct: Three-way split with validation

```python
# Split test set FIRST (completely unseen)
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2)

# Then split train/val from remaining data
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.125)
```

**Why:** Test set must never influence any decisions (feature selection, hyperparameters, stopping criteria).

### ❌ Common Mistake: Two-way split only

```python
# Don't do this - no validation set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
```

**Problem:** You'll overfit to the test set during hyperparameter tuning.

### Validation Checklist

Always validate splits for:
- ✅ No missing values (NaN, inf)
- ✅ No data leakage (overlapping indices)
- ✅ Consistent feature counts
- ✅ X and y length match

---

## Feature Engineering Best Practices

### Pattern: Fit on Train, Transform on Val/Test

```python
# ✅ Correct
engineer = FeatureEngineer(polynomial_degree=2)
X_train_transformed = engineer.fit_transform(X_train)  # Learn parameters
X_val_transformed = engineer.transform(X_val)          # Apply same transformation
X_test_transformed = engineer.transform(X_test)

# ❌ Wrong - causes data leakage
engineer.fit_transform(X_val)  # Don't fit on validation/test!
```

**Why:** Fitting on val/test leaks information from the test set into your model.

### Handling Multicollinearity (VIF Filtering)

```python
engineer = FeatureEngineer(
    polynomial_degree=2,
    vif_threshold=10.0  # Remove features with VIF > 10
)
```

**VIF interpretation:**
- VIF < 5: Low multicollinearity
- 5 < VIF < 10: Moderate (monitor)
- VIF > 10: High (consider removal)

### When to Scale

**Always scale when using:**
- Ridge/Lasso (penalty depends on feature magnitude)
- Neural networks (optimization converges faster)
- Distance-based algorithms (k-NN, SVM)

**No scaling needed for:**
- Decision trees (splits are scale-invariant)
- XGBoost/LightGBM (tree-based)

---

## Model Training Patterns

### Cross-Validation Best Practices

```python
# Use stratified k-fold for consistent folds
from sklearn.model_selection import cross_val_score

cv_scores = cross_val_score(
    model, X_train, y_train,
    cv=5,                              # 5-fold CV
    scoring='neg_mean_absolute_error'  # Consistent metric
)
mae = -cv_scores.mean()
```

**Why CV?** Single train/val split can be lucky. CV gives robust estimate.

### Model Persistence Pattern

```python
import joblib
from pathlib import Path

# Save with versioning
model_path = Path("models") / f"ridge_v{version}.pkl"
model_path.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(model, model_path)

# Load with error handling
if not model_path.exists():
    raise FileNotFoundError(f"Model not found: {model_path}")
model = joblib.load(model_path)
```

### Hyperparameter Tuning

**Quick iteration:**
```python
# Manual tuning for exploration
for alpha in [0.1, 1.0, 10.0]:
    model = Ridge(alpha=alpha)
    mae = cross_val_score(model, X_train, y_train, cv=5).mean()
    print(f"Alpha={alpha}: MAE={mae}")
```

**Production tuning:**
```python
import optuna

def objective(trial):
    alpha = trial.suggest_float("alpha", 0.01, 100.0, log=True)
    model = Ridge(alpha=alpha)
    mae = cross_val_score(model, X_train, y_train, cv=5).mean()
    return mae

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50)
```

---

## Error Handling

### Pattern: Defensive Validation

```python
def train_model(X, y, alpha=1.0):
    # Input validation
    if alpha <= 0:
        raise ValueError(f"alpha must be positive, got {alpha}")
    
    if X.isnull().any().any():
        raise ValueError("X contains missing values")
    
    try:
        model = Ridge(alpha=alpha)
        model.fit(X, y)
        return model
    except np.linalg.LinAlgError as e:
        raise RuntimeError(f"Matrix is singular or ill-conditioned: {e}")
    except Exception as e:
        logger.error(f"Unexpected training error: {e}")
        raise
```

### API Error Response Pattern

```python
try:
    prediction = model.predict(X)
    return jsonify({"prediction": float(prediction[0])}), 200
except ValidationError as e:
    return jsonify({"error": "Invalid input", "details": e.errors()}), 400
except Exception as e:
    logger.error(f"Prediction failed: {e}")
    return jsonify({"error": "Internal server error"}), 500
```

---

## Logging Strategy

### Structured Logging Pattern

```python
import logging

logger = logging.getLogger("smartval")
logger.setLevel(logging.INFO)

# Use consistent format
logger.info(f"Training Ridge model (alpha={alpha}, samples={len(X)})")
logger.warning(f"Removed {count} high-VIF features")
logger.error(f"Model training failed: {error}")
```

### What to Log

**✅ Always log:**
- Training start/end with hyperparameters
- CV scores and final metrics
- Data loading (sample counts, feature counts)
- Errors with context

**❌ Don't log:**
- Individual predictions (too verbose)
- Raw feature values (privacy concern)
- Every iteration in loops

---

## Testing Guidelines

### Unit Test Structure

```python
import pytest

@pytest.fixture
def sample_data():
    """Reusable fixture for test data"""
    return load_and_split(random_state=42)

def test_feature_engineering(sample_data):
    """Test feature engineering pipeline"""
    X_train, _, _, _ = sample_data
    
    engineer = FeatureEngineer(polynomial_degree=2)
    X_transformed = engineer.fit_transform(X_train)
    
    # Assertions
    assert X_transformed.shape[0] == X_train.shape[0]
    assert X_transformed.shape[1] > X_train.shape[1]  # More features after expansion
    assert not X_transformed.isnull().any().any()
```

### Test Coverage Goals

- **Data loading:** 90%+ (critical path)
- **Feature engineering:** 85%+ (complex transformations)
- **Model training:** 80%+ (core logic)
- **API endpoints:** 95%+ (user-facing)

### Testing Edge Cases

Always test:
- ✅ Empty inputs
- ✅ NaN/inf values
- ✅ Out-of-range values
- ✅ Wrong types
- ✅ Missing required fields

---

## API Design Patterns

### Input Validation with Pydantic

```python
from pydantic import BaseModel, Field

class PredictionRequest(BaseModel):
    MedInc: float = Field(..., ge=0, le=15)
    HouseAge: float = Field(..., ge=1, le=52)
    # ... more fields
```

**Benefits:**
- Automatic validation
- Type coercion
- Clear error messages
- OpenAPI schema generation

### Response Format

```python
# ✅ Success response
{
    "prediction": 2.5,
    "units": "$100k",
    "model": "ridge"
}

# ❌ Error response
{
    "error": "Validation failed",
    "details": [...]
}
```

### Monitoring Pattern

```python
from src.monitoring import track_prediction_latency, track_prediction_count

@track_prediction_latency
def predict(data):
    prediction = model.predict(data)
    track_prediction_count("ridge", "success")
    return prediction
```

---

## Common Pitfalls

### 1. Data Leakage

**❌ Fitting scaler on full dataset:**
```python
scaler.fit(pd.concat([X_train, X_test]))  # LEAKS test info!
```

**✅ Fit only on train:**
```python
scaler.fit(X_train)
```

### 2. Feature Engineering After Split

**❌ Create features before split:**
```python
X_with_poly = create_polynomial_features(X)
X_train, X_test = train_test_split(X_with_poly)  # Can leak if using statistics
```

**✅ Split first, then engineer:**
```python
X_train, X_test = train_test_split(X)
X_train_poly = create_polynomial_features(X_train)
```

### 3. Not Handling Convergence Failures

**❌ Assuming training always succeeds:**
```python
model.fit(X, y)  # Can fail with singular matrix
```

**✅ Catch specific errors:**
```python
try:
    model.fit(X, y)
except np.linalg.LinAlgError:
    logger.error("Matrix is singular - try regularization")
    raise
```

### 4. Hardcoding Hyperparameters

**❌ Magic numbers in code:**
```python
model = Ridge(alpha=1.0, max_iter=1000)
```

**✅ Use configuration:**
```python
alpha = config["models"]["ridge"]["alpha"]
model = Ridge(alpha=alpha)
```

---

## Next Steps

Once you've implemented the core functionality:

1. **Run tests:** `make test`
2. **Check linting:** `make lint`
3. **Train a model:** `python -m src.models`
4. **Start API:** `make serve`
5. **Test endpoint:** `curl -X POST http://localhost:5000/predict -d '{...}'`
6. **Dockerize:** `make docker-build && make docker-run`

**Good luck! 🚀**
