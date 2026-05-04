# Exercise 08: EnsembleAI — Interactive Ensemble Method System

> **Learning Goal:** Implement Bagging, Boosting, and Stacking with plug-and-play experimentation and feature importance analysis  
> **Prerequisites:** Completed [notes/01-ml/08-ensemble-methods/](../../../notes/01-ml/08-ensemble-methods/)  
> **Time Estimate:** 6-8 hours (coding) + 1 hour (deployment, optional)  
> **Difficulty:** ⭐⭐⭐ Advanced

> **📦 Infrastructure Note:** This exercise has been simplified to focus on core ML concepts. Docker, docker-compose, Makefile, and Prometheus configuration have been removed. The pre-built Flask API (`src/api.py`) and monitoring utilities (`src/monitoring.py`) remain available for optional local deployment.

---

## 🎯 **What You'll Implement**

Starting from function stubs and inline TODOs, you'll build a complete ensemble system with:

### **Core Implementation (6-8 hours)**

| File | What You Implement | TODOs | Time |
|------|-------------------|-------|------|
| `src/models.py` | Bagging with OOB error | 1 class | 45-60min |
| `src/models.py` | Gradient Boosting with feature importance | 1 class | 50-70min |
| `src/models.py` | Stacking with meta-learner | 1 class | 60-80min |
| `src/models.py` | ExperimentRunner + leaderboard | 2 methods | 35-40min |
| `src/features.py` | Feature selection via mutual information | 2 methods | 45-60min |
| `src/features.py` | Feature importance extraction & visualization | 3 methods | 85-95min |

**Interactive Experience:**
- ✅ See OOB scores, CV accuracy, F1 immediately after training
- ✅ Leaderboard compares all ensemble methods automatically
- ✅ Feature importance ranked across models
- ✅ Rich console output with colors and tables
- ✅ Experiment with 6 ensemble configurations in one run

**Total:** 6-8 hours of focused coding (ensembles are complex!)

---

### **What's Already Done (Utilities)**

These files are complete and reusable:
- ✅ `src/data.py` — Data loading and splitting (classification dataset)
- ✅ `src/evaluate.py` — Classification metrics computation
- ✅ `src/utils.py` — Logging and validation
- ✅ `src/monitoring.py` — Prometheus metrics
- ✅ `src/api.py` — Flask REST API (pre-built for deployment)

**Philosophy:** Focus on ensemble algorithms and feature importance, not boilerplate.

---

## 🧠 **Ensemble Methods: Theory Recap**

### **1. Bagging (Bootstrap Aggregating)**

**Goal:** Reduce variance via parallel independent models

**How it works:**
1. Create bootstrap samples (random sampling with replacement)
2. Train independent models on each sample
3. Aggregate predictions (voting/averaging)

**Key insight:** Out-of-bag (OOB) samples provide free validation!
- Each bootstrap sample uses ~63.2% of data
- Remaining ~36.8% can be used to estimate test error
- OOB score ≈ cross-validation accuracy (no need for separate CV!)

**When to use:**
- High-variance models (deep decision trees)
- Parallel training (independent base learners)
- Need unbiased error estimate without CV

---

### **2. Boosting (Sequential Error Correction)**

**Goal:** Reduce bias via sequential focus on errors

**How it works:**
1. Train first model on full data
2. Train second model on errors from first
3. Continue sequentially, each model correcting previous
4. Combine via weighted sum

**Types:**
- **AdaBoost:** Re-weights samples (focus on misclassified)
- **Gradient Boosting:** Fits residuals (errors) directly

**Key insight:** Learning rate controls contribution of each tree
- Lower learning rate = more robust but needs more trees
- Shallow trees (depth 3-5) work best (weak learners)

**When to use:**
- High-bias models (shallow trees)
- Sequential training is acceptable
- Need feature importance rankings

---

### **3. Stacking (Stacked Generalization)**

**Goal:** Leverage diversity of different model types

**How it works:**
1. Train diverse base models (e.g., RF, GBM, DT)
2. Use base predictions as features (meta-features)
3. Train meta-learner on these features
4. Meta-learner learns optimal combination

**Key insight:** Use cross-validation to create meta-features!
- Prevents overfitting (don't use training predictions directly)
- Each base model predicts on its out-of-fold samples
- Meta-learner trained on these unbiased predictions

**When to use:**
- Have diverse model types available
- Want to beat single best model
- Training time is not critical

---

### **Bias-Variance Tradeoff**

| Method | Reduces | Best Base Learner | Parallel? | Feature Importance? |
|--------|---------|-------------------|-----------|---------------------|
| **Bagging** | Variance | High variance (deep trees) | ✅ Yes | ✅ Yes (via base) |
| **Boosting** | Bias | Low variance (shallow trees) | ❌ No | ✅ Yes (built-in) |
| **Stacking** | Both | Diverse models | ✅ Yes | ⚠️ Depends on meta |

**Diversity is key!**
- Low diversity → ensemble behaves like single model
- High diversity → ensemble captures complementary patterns
- Bagging: moderate (same algorithm, different data)
- Boosting: high (sequential error focus)
- Stacking: highest (different algorithms)

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
  ✓ Train: 569 samples × 30 features
  ✓ Test:  142 samples × 30 features
  ✓ Classes: 2 ({0: 357, 1: 212})

🔧 FEATURE ENGINEERING
Selecting top 15 features...
  ✓ Features selected: 30 → 15
    Top 5: ['worst_perimeter', 'worst_concave_points', ...]
Standardizing features...
  ✓ Features scaled (mean=0, std=1)
✅ Feature engineering complete: 15 features

🎯 ENSEMBLE TRAINING
Comparing 6 ensemble configurations across 3 methods...

→ Training Bagging-30...
  ✓ Bagging (n=30, samples=0.8): CV Acc = 0.9614 | OOB = 0.9579 | Time: 1.2s

→ Training GradBoost-50-0.1...
  ✓ GradBoost (n=50, lr=0.1, d=3): CV Acc = 0.9649 | F1 = 0.9824 | Time: 2.3s
    Top features: worst_perimeter=0.287, worst_concave_points=0.198, ...

📊 ENSEMBLE LEADERBOARD
┏━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━┓
┃ Model                 ┃ CV Accuracy ┃ F1 Score ┃ Precision ┃ Recall ┃
┡━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━┩
│ Stacking-GBM          │      0.9684 │   0.9841 │    0.9850 │ 0.9832 │
│ GradBoost-100-0.05    │      0.9649 │   0.9824 │    0.9830 │ 0.9818 │
│ Bagging-50            │      0.9631 │   0.9806 │    0.9815 │ 0.9797 │
└───────────────────────┴─────────────┴──────────┴───────────┴────────┘

🏆 Best ensemble: Stacking-GBM | CV Accuracy: 0.9684
   Base Learners: 3 (higher = more diversity)

📈 FEATURE IMPORTANCE ANALYSIS
  ✓ Extracted importance from Bagging-30
    Top 3: worst_perimeter=0.245, worst_concave_points=0.189, ...

📊 FEATURE IMPORTANCE COMPARISON
┏━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┓
┃ Feature                ┃ Bagging-30  ┃ GradBoost-50-0.1  ┃ ...    ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━┩
│ worst_perimeter        │      0.2450 │           0.2870  │ ...    │
│ worst_concave_points   │      0.1890 │           0.1980  │ ...    │
└────────────────────────┴─────────────┴───────────────────┴────────┘

✅ FINAL TEST EVALUATION
🏆 Best Ensemble: Stacking-GBM
  • Test Accuracy: 0.9718
  • Precision: 0.9756
  • Recall: 0.9762
  • F1 Score: 0.9759

💾 SAVING BEST MODEL
  ✓ Saved to models/best_ensemble.pkl
```

---

## 📝 **Implementation Guide**

### **Phase 1: Bagging (45-60 minutes)**

**File:** `src/models.py` → `BaggingEnsemble.train()`

**Implement:**
1. Create `BaggingClassifier` with bootstrap sampling
2. Enable OOB score calculation (`oob_score=True`)
3. Cross-validate and compute metrics
4. Print CV accuracy and OOB score

**Success criteria:**
- ✅ OOB score ≈ CV accuracy (within 0.01)
- ✅ Training uses all CPU cores (`n_jobs=-1`)
- ✅ Console shows immediate feedback

**Theory check:**
- Why is OOB score unbiased? (samples not in bootstrap)
- When would OOB differ significantly from CV? (small dataset, low n_estimators)

---

### **Phase 2: Boosting (50-70 minutes)**

**File:** `src/models.py` → `BoostingEnsemble.train()`

**Implement:**
1. Create `GradientBoostingClassifier` with learning rate
2. Train and cross-validate
3. Extract feature importances
4. Print top 3 important features

**Success criteria:**
- ✅ Feature importances sum to ~1.0
- ✅ Shallow trees (depth 3-5) used
- ✅ Top features make domain sense
- ✅ Console shows feature rankings

**Theory check:**
- Why shallow trees for boosting? (weak learners, sequential correction)
- What happens with learning_rate=1.0? (overfitting risk)

---

### **Phase 3: Stacking (60-80 minutes)**

**File:** `src/models.py` → `StackingEnsemble.train()`

**Implement:**
1. Create diverse base learners (RF, GBM, DT)
2. Create meta-learner (logistic, RF, or GBM)
3. Use CV to generate meta-features (`cv=5`)
4. Train stacking classifier with `stack_method='predict_proba'`

**Success criteria:**
- ✅ Base learners are diverse (different algorithms)
- ✅ Meta-features use probabilities (richer than labels)
- ✅ Stacking outperforms single best base model
- ✅ Console shows number of base learners

**Theory check:**
- Why use `predict_proba` instead of `predict`? (more information)
- Why CV for meta-features? (prevents overfitting)

---

### **Phase 4: Feature Engineering (45-60 minutes)**

**File:** `src/features.py` → `FeatureEngineer.fit_transform()` and `transform()`

**Implement:**
1. Feature selection via mutual information
2. Standardization (zero mean, unit variance)
3. Print top 5 selected features

**Success criteria:**
- ✅ Features selected: 30 → 15
- ✅ Top features have high MI scores
- ✅ Scaled features have mean ≈ 0, std ≈ 1

**Theory check:**
- What is mutual information? (measures dependency between feature and target)
- Why select before scaling? (interpretability of MI scores)

---

### **Phase 5: Feature Importance (85-95 minutes)**

**File:** `src/features.py` → `FeatureImportanceAnalyzer` (3 methods)

**Implement:**
1. `add_model()`: Extract importance from tree-based models
2. `print_comparison()`: Table comparing importance across models
3. `plot_importance()`: Bar chart visualization

**Success criteria:**
- ✅ Comparison table shows top 10 features
- ✅ Models agree on most important features
- ✅ Bar chart clearly shows feature rankings

**Theory check:**
- Why might different models rank features differently? (different algorithms, biases)
- What does high agreement mean? (robust, important feature)

---

## 🧪 **Experiments to Try**

### **Experiment 1: Bagging Configuration**

```python
# Try different n_estimators
runner.register("Bagging-10", BaggingEnsemble(n_estimators=10))
runner.register("Bagging-30", BaggingEnsemble(n_estimators=30))
runner.register("Bagging-100", BaggingEnsemble(n_estimators=100))
```

**Question:** At what point do more estimators stop helping?

---

### **Experiment 2: Boosting Learning Rate**

```python
# Learning rate vs n_estimators tradeoff
runner.register("Boost-50-0.3", BoostingEnsemble(n_estimators=50, learning_rate=0.3))
runner.register("Boost-100-0.1", BoostingEnsemble(n_estimators=100, learning_rate=0.1))
runner.register("Boost-200-0.05", BoostingEnsemble(n_estimators=200, learning_rate=0.05))
```

**Question:** Is lower LR + more estimators always better?

---

### **Experiment 3: Stacking Meta-Learner**

```python
# Compare meta-learners
runner.register("Stack-Logistic", StackingEnsemble(meta_learner_type="logistic"))
runner.register("Stack-RF", StackingEnsemble(meta_learner_type="rf"))
runner.register("Stack-GBM", StackingEnsemble(meta_learner_type="gbm"))
```

**Question:** Does complex meta-learner always win?

---

## 📊 **Success Metrics**

Your implementation is complete when:

- ✅ **All TODOs implemented** (9 methods across 2 files)
- ✅ **Bagging shows OOB score** close to CV accuracy
- ✅ **Boosting shows feature importance** with top features
- ✅ **Stacking combines 3+ base learners** with meta-learner
- ✅ **ExperimentRunner runs all ensembles** with leaderboard
- ✅ **Feature importance table** compares across models
- ✅ **Test accuracy** > 0.95 on held-out test set
- ✅ **Console output is informative** (not just silent training)

---

## 🏆 **Advanced Challenges (Optional)**

### **Challenge 1: Diversity Analysis** (30-40 minutes)

**File:** `src/models.py` → `ExperimentRunner.analyze_diversity()`

Implement pairwise agreement analysis:
- Calculate % of predictions that match between base learners
- Print diversity score (1 - mean agreement)
- Verify: Stacking > Boosting > Bagging diversity

---

### **Challenge 2: Voting Ensemble** (20-30 minutes)

Add `VotingEnsemble` class:
- Combine multiple models via soft/hard voting
- Compare with Stacking (simpler but less powerful)
- Test: Does weighted voting beat equal weights?

---

### **Challenge 3: Feature Importance Plot** (already included!)

**File:** `src/features.py` → `FeatureImportanceAnalyzer.plot_importance()`

Create grouped bar chart:
- Compare feature importance across models
- Rotate x-labels for readability
- Save to `models/feature_importance.png`

---

## 🐛 **Common Issues & Solutions**

### **Issue 1: OOB score very different from CV**

**Symptoms:** OOB = 0.85, CV = 0.96  
**Cause:** Too few estimators or small dataset  
**Fix:** Increase `n_estimators` to 50+

---

### **Issue 2: Boosting overfits training data**

**Symptoms:** Train accuracy = 1.0, CV accuracy = 0.90  
**Cause:** Learning rate too high or trees too deep  
**Fix:** Lower `learning_rate` to 0.05-0.1, set `max_depth=3`

---

### **Issue 3: Stacking not improving over base models**

**Symptoms:** Stacking accuracy ≈ best base model  
**Cause:** Base models not diverse enough  
**Fix:** Use different algorithm types (RF, GBM, DT, Logistic)

---

### **Issue 4: Feature importance extraction fails**

**Symptoms:** "does not support feature importance" warning  
**Cause:** Model doesn't have `feature_importances_` attribute  
**Fix:** Only tree-based models have it (skip Stacking with logistic meta)

---

## 📚 **Next Steps**

After completing this exercise:

1. **Read documentation:**
   - [scikit-learn Ensemble Guide](https://scikit-learn.org/stable/modules/ensemble.html)
   - [XGBoost Documentation](https://xgboost.readthedocs.io/)

2. **Try advanced ensembles:**
   - XGBoost classifier (faster than sklearn GradientBoosting)
   - LightGBM (memory-efficient gradient boosting)
   - CatBoost (handles categorical features natively)

3. **Hyperparameter tuning:**
   - Use `GridSearchCV` or `RandomizedSearchCV`
   - Tune n_estimators, learning_rate, max_depth together
   - Use early stopping to prevent overfitting

4. **Deploy to production:**
   - Run the Flask API locally: `python main.py`
   - See [../../_infrastructure/](../../_infrastructure/) for additional deployment options

---

## 🤝 **Getting Help**

- **Stuck on a TODO?** Re-read the inline instructions (they have step-by-step code)
- **Ensemble theory unclear?** Review [notes/01-ml/08-ensemble-methods/](../../../notes/01-ml/08-ensemble-methods/)
- **Want to compare?** Check `_REFERENCE/models_complete.py` (but try yourself first!)

**Remember:** Ensembles are complex! Take breaks, experiment with hyperparameters, and enjoy seeing multiple models work together. 🎉
