# Exercise 05: FraudShield — Interactive Anomaly Detection System

> **Learning Goal:** Implement IsolationForest/LOF/Autoencoder with plug-and-play experimentation and immediate feedback  
> **Prerequisites:** Completed [notes/01-ml/05-anomaly-detection/](../../../notes/01-ml/05-anomaly-detection/)  
> **Time Estimate:** 5-6 hours (coding) + 1 hour (deployment, optional)  
> **Difficulty:** ⭐⭐⭐ Intermediate-Advanced

---

## 🎯 **What You'll Implement**

Starting from function stubs and inline TODOs, you'll build a complete anomaly detection system with:

### **Core Implementation (5-6 hours)**

| File | What You Implement | TODOs | Time |
|------|-------------------|-------|------|
| `src/models.py` | IsolationForest, LOF, Autoencoder training | 3 classes, 9 methods | 2.5h |
| `src/models.py` | ExperimentRunner with leaderboard | 2 methods | 30min |
| `src/features.py` | Standardization + optional PCA | 2 methods | 1h |
| `main.py` | Test evaluation + anomaly visualization | 3 sections | 1h |

**Interactive Experience:**
- ✅ See anomaly scores immediately after each detector trains
- ✅ Leaderboard shows best detector automatically
- ✅ Rich console output with precision/recall/F1/ROC-AUC
- ✅ Visualize detected anomalies vs. ground truth
- ✅ Experiment with 5+ detectors in one run

**Total:** 5-6 hours of focused coding

---

### **What's Already Done (Utilities)**

These files are complete and reusable:
- ✅ `src/data.py` — Synthetic imbalanced data generation
- ✅ `src/evaluate.py` — Binary metrics computation
- ✅ `src/utils.py` — Logging and validation
- ✅ `src/monitoring.py` — Prometheus metrics
- ✅ `src/api.py` — Flask REST API (pre-built for deployment)

**Philosophy:** Focus on anomaly detection algorithms, not boilerplate.

---

> **📦 Infrastructure Note:**  
> Docker/Makefile/Prometheus configs have been removed to simplify the exercise.  
> For production deployment patterns, see `../../_infrastructure/` or reference exercises.

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
  ✓ Train: 8,000 samples × 30 features
    - Normal: 7,200 | Anomalies: 800 (10.0%)
  ✓ Test:  2,000 samples × 30 features
    - Normal: 1,800 | Anomalies: 200 (10.0%)

🔧 FEATURE ENGINEERING
→ Standardizing features...
  ✓ Scaled: mean=0.000, std=1.000
✓ Feature engineering complete: 30 features

🔍 ANOMALY DETECTION TRAINING

→ Training IF-100...
  ✓ IsolationForest (n=100): F1=0.857 | Precision=0.892 | Recall=0.825 | 
    ROC-AUC=0.915 | Detected: 742/8000 | Time: 0.8s

→ Training LOF-20...
  ✓ LOF (k=20): F1=0.823 | Precision=0.845 | Recall=0.802 | 
    ROC-AUC=0.893 | Neighbors=20 | Time: 1.2s

✨ All detectors trained!

📊 LEADERBOARD (sorted by F1)
┌───────────────────┬──────────┬───────────┬─────────┬─────────┐
│ Model             │ F1 Score │ Precision │ Recall  │ ROC-AUC │
├───────────────────┼──────────┼───────────┼─────────┼─────────┤
│ IF-100            │ 0.857    │ 0.892     │ 0.825   │ 0.915   │
│ LOF-20            │ 0.823    │ 0.845     │ 0.802   │ 0.893   │
└───────────────────┴──────────┴───────────┴─────────┴─────────┘

🏆 Best detector: IF-100 | F1: 0.857 | Precision: 0.892 | Recall: 0.825
```

---

## 📋 **Step-by-Step Guide**

### **Step 1: Implement IsolationForest Training (30-45 min)**

Open `src/models.py`, find `IsolationForestDetector.train()`, implement:
1. Create IsolationForest model with contamination parameter
2. Fit on training data (unsupervised)
3. Predict and convert -1/1 to 0/1 format
4. Compute metrics (precision, recall, F1, ROC-AUC)
5. ⭐ **Print results immediately** with anomaly counts

**Test:**
```bash
python -c "from src.models import IsolationForestDetector, ModelConfig; from src.data import load_and_split; X, _, y, _ = load_and_split(); d = IsolationForestDetector(n_estimators=100); d.train(X[:100], y[:100], ModelConfig())"
```

**Expected output:**
```
✓ IsolationForest (n=100): F1=0.857 | Precision=0.892 | Recall=0.825 | ROC-AUC=0.915 | Detected: 10/100 | Time: 0.1s
```

---

### **Step 2: Implement predict() and predict_scores() (10 min)**

For IsolationForest, implement:
1. `predict()`: Convert model's -1/1 to 0/1
2. `predict_scores()`: Negate decision_function for consistent scores

**Key learning:** IsolationForest uses negative scores for anomalies, we normalize to positive

---

### **Step 3: Implement LOF Training (25-35 min)**

Similar to IsolationForest, but:
- Use `LocalOutlierFactor` with `novelty=True`
- Print neighbor count in output
- LOF is density-based (different from isolation-based)

**Key learning:** LOF measures local density deviation, works well for clustered anomalies

---

### **Step 4: Implement Autoencoder Training (45-60 min)**

Build neural network autoencoder:
1. Create encoder/decoder architecture with TensorFlow/Keras
2. **Train ONLY on normal samples** (key insight!)
3. Compute reconstruction errors on all data
4. Set threshold at contamination percentile
5. Use reconstruction error as anomaly score

**Key learning:** Autoencoders learn to reconstruct normal patterns, anomalies have high reconstruction error

---

### **Step 5: Implement ExperimentRunner (15 min)**

Open `src/models.py`, implement:
1. `run_experiment()`: Loop through registered detectors, train each
2. `print_leaderboard()`: Sort results by F1, print Rich table

**Result:** Compare 5+ detectors in one run!

---

### **Step 6: Feature Engineering (30-40 min)**

Open `src/features.py`, implement 2 TODOs:
1. **fit_transform()** (30 min): 
   - Standardization with StandardScaler
   - Optional PCA dimensionality reduction
   - Rich console feedback
2. **transform()** (10 min): Apply fitted transformations

**Each TODO has:**
- ✅ Step-by-step instructions
- ✅ Code hints
- ✅ Immediate feedback (console print)

**Key insight:** Standardization is critical for distance-based methods (IF, LOF)

---

### **Step 7: Visualization (20-30 min)**

Open `main.py`, implement TODO for visualization:
1. Anomaly score distribution (histogram)
2. Feature space scatter plot with detected anomalies
3. Confusion matrix
4. ROC curve

**Output:** `anomaly_detection_results.png` with 4-panel visualization

---

### **Step 8: Final Evaluation (10 min)**

Open `main.py`, implement TODOs:
1. Predict on test set using best detector
2. Compute test metrics (precision, recall, F1, ROC-AUC)
3. Compare to training metrics

---

## 📊 **Understanding Anomaly Detection Metrics**

### **Precision vs. Recall Trade-off**

| Metric | Definition | When to Prioritize |
|--------|-----------|-------------------|
| **Precision** | TP / (TP + FP) | Minimize false alarms (e.g., fraud alerts) |
| **Recall** | TP / (TP + FN) | Catch all real anomalies (e.g., security threats) |
| **F1 Score** | 2 × (P × R) / (P + R) | Balance both (general use case) |
| **ROC-AUC** | Area under ROC curve | Overall detector quality |

**Tuning:** Adjust `contamination` parameter to shift precision/recall trade-off

### **Example:**

```python
# Conservative (high precision, low false alarms)
ModelConfig(contamination=0.05)  # Detect top 5% most anomalous

# Aggressive (high recall, catch more anomalies)
ModelConfig(contamination=0.20)  # Detect top 20% most anomalous

# Balanced (default)
ModelConfig(contamination=0.10)  # Match expected 10% anomaly rate
```

---

## 🔬 **Experimentation Ideas**

### **1. Hyperparameter Tuning**

**IsolationForest:**
- `n_estimators`: Try 50, 100, 200 (more = more stable)
- `max_samples`: Try 'auto', 256, 512 (smaller = more diverse trees)

**LOF:**
- `n_neighbors`: Try 5, 10, 20, 50 (smaller = more local)
- `metric`: Try 'euclidean', 'manhattan', 'cosine'

**Autoencoder:**
- `encoding_dim`: Try 5, 10, 20 (smaller = more compression)
- `epochs`: Try 30, 50, 100 (more = better fit)

### **2. Feature Engineering**

```python
# No PCA (use all features)
fe = FeatureEngineer(scale_features=True, n_components_pca=None)

# PCA with 10 components (reduce noise)
fe = FeatureEngineer(scale_features=True, n_components_pca=10)

# PCA with 90% variance explained
from sklearn.decomposition import PCA
pca = PCA(n_components=0.9)  # Keep 90% variance
```

### **3. Contamination Tuning**

```python
# Conservative detection
runner.run_experiment(X, y, ModelConfig(contamination=0.05))

# Aggressive detection  
runner.run_experiment(X, y, ModelConfig(contamination=0.20))

# Data-driven (match true anomaly rate)
true_contamination = y.sum() / len(y)
runner.run_experiment(X, y, ModelConfig(contamination=true_contamination))
```

---

## 🎓 **Key Differences from Regression (Exercise 01)**

| Aspect | Regression | Anomaly Detection |
|--------|-----------|-------------------|
| **Problem** | Predict continuous values | Detect binary anomalies |
| **Data** | Balanced California Housing | Imbalanced synthetic (10% anomalies) |
| **Metrics** | MAE, RMSE, R² | Precision, Recall, F1, ROC-AUC |
| **Features** | Polynomial expansion + scaling | Scaling only (preserve anomaly patterns) |
| **Models** | Ridge, Lasso, XGBoost | IsolationForest, LOF, Autoencoder |
| **Training** | Supervised (use labels) | Unsupervised (labels only for validation) |
| **Evaluation** | Residual plots | ROC curve, confusion matrix, score distribution |
| **API** | /predict returns value | /detect returns binary + score + explanation |

---

## 🐛 **Common Issues & Solutions**

### **Issue: TensorFlow not installed**
```bash
pip install tensorflow
# Or for CPU-only (faster install):
pip install tensorflow-cpu
```

### **Issue: Low F1 score (<0.70)**
- **Solution:** Tune contamination to match true anomaly rate
- Check feature scaling is enabled
- Try different detectors (LOF may work better than IF for some datasets)

### **Issue: High false positive rate (low precision)**
- **Solution:** Decrease contamination parameter
- Use more conservative threshold
- Try Autoencoder (often more precise)

### **Issue: Missing many anomalies (low recall)**
- **Solution:** Increase contamination parameter
- Use LOF with smaller n_neighbors (more local)
- Check if anomalies are clustered or scattered

---

## 🏆 **Success Criteria**

Your exercise is complete when:
- [ ] All 3 detector classes implemented (IsolationForest, LOF, Autoencoder)
- [ ] ExperimentRunner produces sorted leaderboard
- [ ] Feature engineering with standardization working
- [ ] Test set F1 score ≥ 0.80
- [ ] Visualization shows 4 plots (scores, scatter, confusion matrix, ROC)
- [ ] Can experiment with different hyperparameters and see impact
- [ ] Console output shows immediate feedback during training

---

## 📚 **Next Steps**

After completing this exercise:

1. **Try on real datasets:**
   - Credit card fraud: [Kaggle dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud)
   - Network intrusion: KDD Cup 99
   - Sensor anomalies: NASA Bearing dataset

2. **Advanced techniques:**
   - Ensemble methods (combine multiple detectors)
   - Deep autoencoders (convolutional, variational)
   - One-class classification (SVDD, OCSVM variants)

3. **Production deployment:**
   - Deploy API with Docker
   - Add SHAP explanations for detected anomalies
   - Implement online learning (update model with new data)

---

## 🔗 **Related Exercises**

- **Exercise 01:** [Regression](../01-regression/) - Learn plug-and-play pattern
- **Exercise 02:** [Neural Networks](../02-neural-networks/) - Deep learning basics
- **Exercise 03:** [Classification](../03-classification/) - Binary classification
- **Exercise 04:** [Clustering](../04-clustering/) - Unsupervised learning

---

## 📖 **Additional Resources**

- [Isolation Forest Paper](https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf) (Liu et al., 2008)
- [LOF: Identifying Density-Based Local Outliers](https://www.dbs.ifi.lmu.de/Publikationen/Papers/LOF.pdf) (Breunig et al., 2000)
- [Autoencoders for Anomaly Detection](https://arxiv.org/abs/1712.09381) (Chalapathy & Chawla, 2019)
- [scikit-learn Anomaly Detection Guide](https://scikit-learn.org/stable/modules/outlier_detection.html)

---

**Happy coding! 🚀**
| **Monitoring** | Prediction value distribution | Anomaly rate gauge, alert thresholds |

---

## Anomaly Detection Models

### 1. Isolation Forest
- **Algorithm:** Tree-based outlier detection
- **Intuition:** Anomalies are easier to isolate (require fewer splits)
- **Hyperparameters:** `contamination`, `n_estimators`
- **Best for:** Fast training, handles high-dimensional data

### 2. One-Class SVM
- **Algorithm:** Density-based outlier detection
- **Intuition:** Learn a decision boundary around normal data
- **Hyperparameters:** `nu` (contamination upper bound), `kernel`
- **Best for:** Low-dimensional data, complex decision boundaries

### 3. Autoencoder
- **Algorithm:** Neural network reconstruction error
- **Intuition:** Normal data reconstructs well, anomalies have high error
- **Hyperparameters:** `encoding_dim`, `epochs`, `learning_rate`
- **Best for:** High-dimensional data, when normal patterns are learnable

---

## Key Metrics

### Precision@K
**What:** Fraction of top-K predictions that are true anomalies  
**Why:** In fraud detection, we investigate top suspicious cases  
**Target:** >90% precision at K=10

### ROC-AUC
**What:** Area under ROC curve (True Positive Rate vs False Positive Rate)  
**Why:** Measures model's ability to rank anomalies higher than normal  
**Target:** >0.90

### Anomaly Rate
**What:** Rolling window anomaly detection rate  
**Why:** Monitor for data drift or model degradation  
**Alert:** If rate exceeds 15% (above expected 10%)

---

## API Endpoints

### POST /detect
Detect if a single transaction is anomalous.

**Request:**
```json
{
  "features": [0.5, 1.2, -0.3, ...]  // 20 feature values
}
```

**Response:**
```json
{
  "is_anomaly": true,
  "anomaly_score": 0.85,
  "confidence": "high",
  "model": "isolation_forest",
  "current_anomaly_rate": 0.12,
  "explanation": "Strong anomaly signal detected. This transaction deviates significantly from normal patterns."
}
```

### POST /batch_detect
Detect anomalies in a batch of transactions.

**Request:**
```json
{
  "samples": [
    [0.5, 1.2, ...],
    [0.3, -0.5, ...]
  ]
}
```

### GET /health
Health check endpoint.

### GET /metrics
Prometheus metrics endpoint.

---

## Development Workflow

1. **Run tests:**
   ```bash
   pytest tests/ -v
   ```

2. **Train models:**
   ```python
   from src.data import load_and_split
   from src.features import FeatureEngineer
   from src.models import ModelRegistry
   
   X_train, X_val, X_test, y_train, y_val, y_test = load_and_split()
   engineer = FeatureEngineer()
   X_train = engineer.fit_transform(X_train)
   
   registry = ModelRegistry()
   registry.train_isolation_forest(X_train, y_train)
   registry.train_one_class_svm(X_train, y_train)
   ```

3. **Start API:**
   ```bash
   python -m src.api
   # Or with gunicorn:
   gunicorn --bind 0.0.0.0:5000 --workers 4 src.api:app
   ```

4. **Test API:**
   ```bash
   curl -X POST http://localhost:5000/detect \
     -H "Content-Type: application/json" \
     -d '{"features": [0.5, 1.2, -0.3, 0.8, 1.1, -0.5, 0.9, 1.3, 0.2, -0.4, 0.7, 1.0, -0.2, 0.6, 0.4, -0.1, 0.8, 1.2, 0.3, -0.3]}'
   ```

---

## Docker Deployment

Build and run in Docker:
```bash
docker-compose up --build
```

Services:
- FraudShield API: http://localhost:5000
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000

---

## Resources

**Concept Review:**
- [notes/01-ml/](../../notes/01-ml/) — ML fundamentals
- Anomaly detection resources (external)

**Implementation Patterns:**
- [exercises/01-ml/01_regression/](../01_regression/) — Production patterns
- [exercises/01-ml/02_classification/](../02_classification/) — Classification metrics

---

## Common Issues

### Issue: Low precision
**Solution:** Adjust `contamination` parameter or threshold

### Issue: High false positive rate
**Solution:** Use stricter threshold, ensemble multiple models

### Issue: Poor separation in ROC curve
**Solution:** Add more informative features, try different model

### Issue: Model overfits to training anomalies
**Solution:** Ensure training only on normal samples (for autoencoder)

---

## Production Checklist

- [ ] Model achieves >85% precision at 80% recall
- [ ] API latency <100ms (p99)
- [ ] All tests pass with >80% coverage
- [ ] Docker image builds successfully
- [ ] Monitoring dashboards configured
- [ ] Alert thresholds set for anomaly rate
- [ ] SHAP explanations working
- [ ] Documentation complete

---

**Ready to detect fraud? Start with `python -m src.data` to verify data generation!** 🚀

