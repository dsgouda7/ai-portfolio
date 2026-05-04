# Exercise 01: SmartVal AI — Interactive Regression System

> **Infrastructure Note:** Docker, Docker Compose, and Makefiles have been centralized to `exercises/_infrastructure/`. Run `setup.ps1` (Windows) or `setup.sh` (Linux/Mac) for local development.

> **Learning Goal:** Implement Ridge/Lasso/XGBoost with plug-and-play experimentation and immediate feedback  
> **Prerequisites:** Completed [notes/01-ml/01-regression/](../../../notes/01-ml/01-regression/)  
> **Time Estimate:** 5-6 hours (coding) + 1 hour (deployment, optional)  
> **Difficulty:** ⭐⭐ Intermediate

---

## 🎯 **What You'll Implement**

Starting from function stubs and inline TODOs, you'll build a complete regression system with:

### **Core Implementation (5-6 hours)**

| File | What You Implement | TODOs | Time |
|------|-------------------|-------|------|
| `src/data_prep.py` | Outlier detection, imputation, SMOTE, PSI/KS tests | 12 functions | 3-4h |
| `src/models.py` | Ridge, Lasso, XGBoost training with CV | 3 classes | 2h |
| `src/models.py` | ExperimentRunner with leaderboard | 2 methods | 30min |
| `src/features.py` | Polynomial features + VIF filtering | 3 stages | 1h |
| `main.py` | Test evaluation + model saving | 2 sections | 30min |

**Interactive Experience:**
- ✅ See results immediately after each model trains
- ✅ Leaderboard shows best model automatically
- ✅ Rich console output with colors and tables
- ✅ Experiment with 9 models in one run

**Total:** 8-10 hours of focused coding (data prep + modeling)

---

### **What's Already Done (Utilities)**

These files are complete and reusable:
- ✅ `src/data.py` — Data loading and splitting
- ✅ `src/evaluate.py` — Metrics computation
- ✅ `src/utils.py` — Logging and validation
- ✅ `src/monitoring.py` — Prometheus metrics
- ✅ `src/api.py` — Flask REST API (pre-built for deployment)

**Philosophy:** Focus on ML algorithms, not boilerplate.

---

### **Optional: Production Deployment (1 hour)**

After implementing core features, deploy via Docker:
```bash
# Build container (uses shared infrastructure)
make docker-build

# Start API + Prometheus + Grafana
make docker-up

# Test API
curl -X POST http://localhost:5001/predict \
  -H "Content-Type: application/json" \
  -d '{"MedInc": 3.5, "HouseAge": 15, ...}'
```

**Infrastructure:** All Docker/Prometheus configs live in `../../_infrastructure/`.  
No need to modify — just use it!

---

## 🚀 **Quick Start**

### **1. Setup Environment**

**PowerShell (Windows):**
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\setup.ps1
.\venv\Scripts\Activate.ps1
```

**Bash (Linux/Mac/WSL):**
```bash
chmod +x setup.sh
./setup.sh
source venv/bin/activate
```

### **2. Run Interactive Training**

```bash
python main.py
```

**Expected output:**
```
📊 LOADING DATA
  ✓ Train: 16,512 samples × 8 features
  ✓ Test:  4,128 samples × 8 features

🔧 FEATURE ENGINEERING
  Polynomial features: 8 → 44
  VIF filtering: 44 → 28 features
  ✓ Features scaled (mean=0, std=1)

🤖 MODEL TRAINING

→ Training Ridge (α=0.01)...
  ✓ Ridge (α=0.01): CV MAE = $39,850 | Time: 0.2s

→ Training Ridge (α=0.1)...
  ✓ Ridge (α=0.1): CV MAE = $39,200 | Time: 0.2s

... (7 more models)

📊 LEADERBOARD
┌─────────────────────────┬──────────┬──────────┬──────┐
│ Model                   │ CV MAE   │ RMSE     │ R²   │
├─────────────────────────┼──────────┼──────────┼──────┤
│ XGBoost (d=6, n=200)    │ $31,800  │ $43,900  │ 0.86 │
│ XGBoost (d=6, n=100)    │ $32,450  │ $45,200  │ 0.85 │
│ Ridge (α=1.0)           │ $38,100  │ $51,300  │ 0.82 │
...
└─────────────────────────┴──────────┴──────────┴──────┘

🏆 Best model: XGBoost (d=6, n=200) | CV MAE: $31,800
```

---

## 📋 **Step-by-Step Guide**

### **Step 1: Implement Ridge Training (30-45 min)**

Open `src/models.py`, find `RidgeRegressor.train()`, implement:
1. Create Ridge model with alpha parameter
2. Fit on training data
3. Cross-validate (5-fold CV)
4. Compute metrics (MAE, RMSE, R²)
5. ⭐ **Print results immediately** (see console output)

**Test:**
```bash
python -c "from src.models import RidgeRegressor, ModelConfig; from src.data import load_and_split; X, _, y, _ = load_and_split(); r = RidgeRegressor(alpha=1.0); r.train(X[:100], y[:100], ModelConfig())"
```

**Expected output:**
```
✓ Ridge (α=1.0): CV MAE = $38,100 | Time: 0.2s
```

---

### **Step 2: Implement Lasso Training (20-30 min)**

Similar to Ridge, but:
- Use `Lasso` instead of `Ridge`
- Count non-zero coefficients: `np.sum(model.coef_ != 0)`
- Print non-zero count in output

**Key learning:** Lasso drives some coefficients to zero → automatic feature selection

---

### **Step 3: Implement XGBoost (30 min)**

Use `XGBRegressor` with:
- `n_estimators`: Number of trees (100-200)
- `max_depth`: Tree depth (3-6)
- `verbosity=0`: Suppress logs

**Key learning:** Tree-based models often outperform linear models

---

### **Step 4: Implement ExperimentRunner (15 min)**

Open `src/models.py`, implement:
1. `run_experiment()`: Loop through registered regressors, train each
2. `print_leaderboard()`: Sort results by CV MAE, print Rich table

**Result:** Compare 9 models in one run!

---

### **Step 5: Feature Engineering (30-40 min)**

Open `src/features.py`, implement 3 inline TODOs:
1. **Polynomial features** (10 min): Use `PolynomialFeatures(degree=2)`
2. **VIF filtering** (20 min): Iteratively remove high-VIF features
3. **Scaling** (10 min): Use `StandardScaler()`

**Each TODO has:**
- ✅ Step-by-step instructions
- ✅ Code hints
- ✅ Immediate feedback (console print)

---

### **Step 6: Final Evaluation (15 min)**

Open `main.py`, implement TODOs:
1. Predict on test set using best model
2. Compute test metrics (MAE, RMSE, R²)
3. Save model and feature engineer to `models/`

---

## ✅ **Success Criteria**

- [ ] All TODOs implemented (no `NotImplementedError`)
- [ ] `python main.py` runs without errors
- [ ] Console shows immediate feedback after each model
- [ ] Leaderboard displays 9 models sorted by CV MAE
- [ ] Best model achieves CV MAE <$40k
- [ ] Test metrics printed at end
- [ ] Models saved to `models/` directory

---

## 🧪 **Testing**

Run test suite to validate implementation:
```bash
make test
```

**Expected:** All tests pass after implementing TODOs

---

## 🎨 **Experiment Ideas**

After completing basic implementation, try:

### **Experiment 1: Polynomial Degree**
```python
# In main.py, try different degrees
fe = FeatureEngineer(polynomial_degree=3, vif_threshold=5.0)
```

**Question:** Does degree 3 beat degree 2? Or does it overfit?

### **Experiment 2: More Regularization Strengths**
```python
# Register more Ridge alphas
runner.register("Ridge (α=0.001)", RidgeRegressor(alpha=0.001))
runner.register("Ridge (α=100)", RidgeRegressor(alpha=100))
```

**Question:** What's the optimal alpha? Too small = underfit, too large = overfit?

### **Experiment 3: XGBoost Learning Rate**
```python
runner.register("XGBoost (lr=0.01)", XGBoostRegressor(learning_rate=0.01, n_estimators=200))
runner.register("XGBoost (lr=0.3)", XGBoostRegressor(learning_rate=0.3, n_estimators=200))
```

**Question:** Does slower learning rate improve accuracy?

---

## 📦 **Production Deployment (Optional)**

### **Deploy via Docker**

```bash
# Ensure ml-network exists (one-time setup)
docker network create ml-network

# Build container (uses shared Dockerfile from _infrastructure/)
make docker-build

# Start full stack (API + Prometheus + Grafana)
make docker-up

# View logs
make docker-logs

# Test API
curl -X POST http://localhost:5001/predict \
  -H "Content-Type: application/json" \
  -d '{
    "MedInc": 3.5,
    "HouseAge": 15,
    "AveRooms": 6.0,
    "AveBedrms": 1.0,
    "Population": 1500,
    "AveOccup": 3.0,
    "Latitude": 34.0,
    "Longitude": -118.0
  }'

# Response
{"prediction": 185000.0}

# Stop containers
make docker-down
```

### **View Metrics**
- **Prometheus:** http://localhost:9090
- **Grafana:** http://localhost:3000 (admin/admin)
- **API health:** http://localhost:5001/health

---

## 🗂️ **Project Structure**

```
01-regression/
├── src/
│   ├── data.py              # ✅ Complete (utility)
│   ├── features.py          # ⚠️ TODOs: Poly, VIF, scaling
│   ├── models.py            # ⚠️ TODOs: Ridge, Lasso, XGBoost, ExperimentRunner
│   ├── api.py               # ✅ Complete (Flask REST API)
│   ├── evaluate.py          # ✅ Complete (metrics)
│   └── utils.py             # ✅ Complete (logging)
├── _REFERENCE/              # 📚 Complete implementations for comparison
│   ├── models_complete.py
│   ├── features_complete.py
│   └── api_complete.py
├── tests/                   # ✅ Pre-written validation tests
├── main.py                  # ⚠️ TODOs: Test eval, model saving
├── config.yaml              # ⚠️ Tune hyperparameters here
├── requirements.txt         # ✅ Dependencies (includes rich, tabulate)
├── Makefile                 # ✅ Uses shared targets from _infrastructure/
├── docker-compose.yml       # ✅ Minimal (15 lines, extends shared base)
└── README.md                # This file
```

**Legend:**
- ✅ Complete (just use it)
- ⚠️ Has TODOs (you implement)
- 📚 Reference (compare after completing)

---

## 🎓 **Learning Objectives**

After completing this exercise, you'll be able to:

1. **Implement regularized regression:**
   - Ridge (L2 penalty) vs Lasso (L1 penalty) trade-offs
   - XGBoost gradient boosting for nonlinear relationships

2. **Engineer features effectively:**
   - Polynomial expansion to capture interactions
   - VIF filtering to remove multicollinearity
   - Feature scaling for linear models

3. **Experiment systematically:**
   - Plug-and-play registry pattern for trying multiple models
   - Cross-validation to avoid overfitting
   - Leaderboard-driven model selection

4. **Get immediate feedback:**
   - Console output shows progress in real-time
   - Rich tables for easy comparison
   - No waiting for final results

5. **Deploy ML models:**
   - Docker containerization (optional)
   - REST API design
   - Production monitoring

---

## 📚 **Reference Materials**

### **Complete Implementations**
After attempting TODOs, compare your solution to:
- `_REFERENCE/models_complete.py` — Full Ridge/Lasso/XGBoost
- `_REFERENCE/features_complete.py` — Full feature engineering
- `_REFERENCE/api_complete.py` — Full Flask API

### **Related Notes**
- [notes/01-ml/01-regression/](../../../notes/01-ml/01-regression/) — Theory and concepts
- [_infrastructure/README.md](../../_infrastructure/README.md) — Shared infrastructure guide

---

## 🐛 **Troubleshooting**

### **"NotImplementedError: Implement Ridge training"**
You need to replace the TODO stub with actual implementation. See Step-by-Step Guide above.

### **"Module 'rich' not found"**
Run `pip install -r requirements.txt` to install dependencies.

### **"Model not trained yet"**
You're calling `.predict()` before calling `.train()`. Train the model first.

### **Tests fail after implementation**
- Check that metrics dict has correct keys: `{"mae", "rmse", "r2", "cv_mae"}`
- Ensure cross-validation returns positive MAE (use `-cv_scores.mean()`)
- Verify model is stored in `self.model`

---

## 💡 **Portfolio Tips**

To maximize learning and portfolio value:

1. **Git commits showing progress:**
   ```bash
   git add src/models.py
   git commit -m "Implement Ridge training with CV"
   
   git add src/models.py
   git commit -m "Add Lasso with feature selection"
   
   git add src/models.py src/features.py
   git commit -m "Complete experiment framework + feature engineering"
   ```

2. **Document experiments in README:**
   - Which models did you try?
   - What hyperparameters worked best?
   - What surprised you? (e.g., "Lasso selected only 18/28 features!")

3. **Screenshot leaderboard:**
   - Shows you actually ran experiments
   - Demonstrates model comparison skills

4. **Add SOLUTION.md:**
   - Explain design decisions
   - Compare Ridge vs Lasso vs XGBoost trade-offs
   - Document optimal hyperparameters found

---

**Ready?** Start with Step 1: Implement Ridge Training! 🚀
