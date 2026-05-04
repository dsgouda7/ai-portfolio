# Exercise 02: FaceAI — Interactive Classification System

> **Learning Goal:** Implement Logistic Regression/SVM/RandomForest with plug-and-play experimentation and immediate feedback  
> **Prerequisites:** Completed [notes/01-ml/02-classification/](../../../notes/01-ml/02-classification/)  
> **Time Estimate:** 4-5 hours (coding) + 1 hour (deployment, optional)  
> **Difficulty:** ⭐⭐ Intermediate

> **📦 Infrastructure Note:**  
> All Docker, Prometheus, and deployment configurations are centralized in `../../_infrastructure/`.  
> This exercise focuses on ML implementation only. Infrastructure files have been removed to reduce clutter.

---

## 🎯 **What You'll Implement**

Starting from function stubs and inline TODOs, you'll build a complete face classification system with:

### **Core Implementation (4-5 hours)**

| File | What You Implement | TODOs | Time |
|------|-------------------|-------|------|
| `src/features.py` | HOG extraction + PCA + Scaling | 3 stages | 1h |
| `src/models.py` | Logistic/SVM/RandomForest training | 3 classes | 2h |
| `src/models.py` | ExperimentRunner with leaderboard | 2 methods | 30min |
| `main.py` | Test evaluation + model saving | 2 sections | 30min |

**Interactive Experience:**
- ✅ See results immediately after each model trains
- ✅ Leaderboard shows best model automatically
- ✅ Rich console output with colors and tables
- ✅ Experiment with 9 models in one run

**Total:** 4-5 hours of focused coding

---

### **What's Already Done (Utilities)**

These files are complete and reusable:
- ✅ `src/data.py` — Olivetti faces dataset loading
- ✅ `src/evaluate.py` — Metrics computation (accuracy, precision, recall, F1)
- ✅ `src/utils.py` — Logging and validation
- ✅ `src/monitoring.py` — Prometheus metrics
- ✅ `src/api.py` — Flask REST API (pre-built for deployment)

**Philosophy:** Focus on ML algorithms, not boilerplate.

---

### **Optional: Production Deployment (1 hour)**

After implementing core features, deploy via shared infrastructure:
```bash
# Navigate to infrastructure directory
cd ../../_infrastructure

# Follow deployment instructions in _infrastructure/README.md
```

**Infrastructure:** All Docker/Prometheus/Grafana configs are centralized in `../../_infrastructure/`.  
See [_infrastructure/README.md](../../_infrastructure/README.md) for deployment details.

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
  ✓ Train: 320 samples × 4,096 pixels
  ✓ Test:  80 samples × 4,096 pixels
  ✓ Classes: 40 unique faces

🔧 FEATURE ENGINEERING
  Extracting HOG features...
    HOG features: 1,764 dimensions
  Applying PCA (target: 100 components)...
    PCA: 100 components (explained variance: 85.2%)
  ✓ Features scaled (mean=0, std=1)

🤖 MODEL TRAINING

→ Training Logistic (C=0.01)...
  ✓ Logistic (C=0.01): CV Acc = 0.892 | F1 = 0.889 | Time: 1.2s

→ Training Logistic (C=0.1)...
  ✓ Logistic (C=0.1): CV Acc = 0.918 | F1 = 0.915 | Time: 1.1s

→ Training SVM (rbf)...
  ✓ SVM (C=1.0, rbf): CV Acc = 0.945 | Support vectors: 189 | Time: 2.3s

📊 LEADERBOARD
┏━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━┓
┃ Model               ┃ CV Accuracy ┃ F1 Score ┃ Precision ┃ Recall ┃
┡━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━┩
│ SVM (rbf)           │       0.945 │    0.942 │     0.943 │  0.942 │
│ RF (n=200, d=15)    │       0.938 │    0.935 │     0.936 │  0.935 │
│ Logistic (C=1.0)    │       0.918 │    0.915 │     0.916 │  0.915 │
└─────────────────────┴─────────────┴──────────┴───────────┴────────┘

🏆 Best model: SVM (rbf) | CV Accuracy: 0.945
```

## Project Structure

```
02_classification/
├── requirements.txt          # Dependencies (includes scikit-image)
├── setup.sh / setup.ps1      # Environment setup
├── config.yaml               # Hyperparameters
├── Makefile                  # Common commands
├── README.md                 # This file
├── coding_guidelines.md      # Production patterns & hints
├── SOLUTION.md               # Reference implementation
├── src/
│   ├── data.py               # ✅ Scaffolded (Olivetti Faces loading)
│   ├── features.py           # ✅ Scaffolded (HOG + PCA)
│   ├── models.py             # ⚠️ Hints provided
│   ├── evaluate.py           # ✅ Scaffolded
│   └── api.py                # ❌ TODO
├── tests/
│   ├── test_data.py          # ✅ Complete
│   ├── test_features.py      # ⚠️ Partial
│   └── test_models.py        # ❌ TODO
└── notebooks/
    └── exploratory.ipynb     # Optional EDA
```

---

## Success Criteria

Your exercise is complete when:
- [ ] All tests pass: `pytest tests/`
- [ ] Accuracy >90% on test set
- [ ] API returns predictions in <100ms
- [ ] Code passes linting: `black . && flake8 src/`
- [ ] Confusion matrix shows balanced performance across classes
- [ ] Cross-validation confirms generalization

---

## Key Differences from Track 01 (Regression)

| Aspect | Track 01 (Regression) | Track 02 (Classification) |
|--------|----------------------|---------------------------|
| **Problem** | Predict continuous house prices | Predict discrete face classes |
| **Metrics** | MAE, RMSE, R² | Accuracy, Precision, Recall, F1 |
| **Features** | Polynomial expansion + scaling | HOG features + PCA |
| **Models** | Ridge, Lasso, XGBoost | LogisticRegression, SVM, RandomForest |
| **Evaluation** | Residual plots, learning curves | Confusion matrix, classification report |
| **Dataset** | California Housing (20k samples) | Olivetti Faces (400 samples, 40 classes) |

---

## Resources

**Concept Review:**
- [notes/01-ml/02_classification/](../../notes/01-ml/02_classification/) — Complete track
- [notes/01-ml/02_classification/grand-challenge.md](../../notes/01-ml/02_classification/grand-challenge.md) — Constraints

**Implementation Guides:**
- [coding_guidelines.md](coding_guidelines.md) — Hints & patterns
- [SOLUTION.md](SOLUTION.md) — Reference (read AFTER attempting!)

---

## Quick Start

1. **Setup environment:**
   ```bash
   ./setup.sh
   source venv/bin/activate
   ```

2. **Run tests to see what's expected:**
   ```bash
   make test
   ```

3. **Explore the data (optional):**
   ```python
   from src.data import load_and_split, load_dataset_info
   info = load_dataset_info()
   print(info)
   ```

4. **Train models:**
   ```python
   from src.data import load_and_split
   from src.features import FeatureEngineer
   from src.models import ModelRegistry
   
   # Load data
   X_train, X_val, X_test, y_train, y_val, y_test = load_and_split()
   
   # Engineer features
   engineer = FeatureEngineer(hog_orientations=9, pca_components=50)
   X_train_features = engineer.fit_transform(X_train)
   X_val_features = engineer.transform(X_val)
   
   # Train models
   registry = ModelRegistry()
   metrics = registry.train_logistic_regression(X_train_features, y_train)
   print(f"CV Accuracy: {metrics['cv_accuracy']:.3f}")
   ```

5. **Evaluate:**
   ```python
   from src.evaluate import AutoEvaluator
   
   evaluator = AutoEvaluator()
   test_metrics = evaluator.evaluate(
       registry.models["logistic_regression"],
       X_test_features,
       y_test
   )
   evaluator.plot_confusion_matrix(y_test, test_metrics["predictions"])
   ```

6. **Start API (after training):**
   ```bash
   make serve
   # Test: curl -X GET http://localhost:5000/health
   ```

---

## Production Constraints (from ml-engg-readiness.md)

This exercise addresses gaps identified in the ML engineering audit:

✅ **Addressed in scaffolding:**
- Structured logging with context
- Error handling for HOG extraction failures
- Input validation (image size, format)
- Model persistence patterns
- Configuration-driven hyperparameters

⚠️ **Your responsibility:**
- Comprehensive error handling in API
- Model comparison and selection
- Feature engineering decisions (PCA components, HOG parameters)
- Hyperparameter tuning (C for LogisticRegression, kernel for SVM)

---

## Tips & Hints

1. **Small dataset:** Olivetti Faces has only 400 samples (40 people × 10 images). Use stratified splits to ensure all classes represented.

2. **HOG features:** Already scaffolded. Extracts ~1296 features from 64×64 images. PCA recommended to reduce to 50-100 components.

3. **Model selection:** LogisticRegression often performs best for small datasets. SVM with RBF kernel can be competitive.

4. **Overfitting risk:** With 400 samples, easy to overfit. Use cross-validation and regularization.

5. **API input:** Expects flattened 64×64 grayscale image (4096 values in range [0, 1]).

---

## Challenge Extensions

Once core exercise is complete:

1. **Ensemble methods:** Stack LogisticRegression + SVM predictions
2. **Neural networks:** Add CNN classifier (requires more data augmentation)
3. **Deployment:** Deploy to AWS Lambda or Google Cloud Run
4. **Monitoring:** Build Grafana dashboard with per-class accuracy tracking

---

**Good luck! 🚀**

