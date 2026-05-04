# Exercises — AI Portfolio Learning Path

> **Updated:** April 28, 2026  
> **Total Exercises:** 14 (8 ML paradigms + 6 grand challenges)  
> **Infrastructure:** Shared, plug-and-play (see `_infrastructure/`)  
> **Pedagogy:** Interactive, experiment-driven, immediate feedback

---

## 🎯 **Overview**

This directory contains 14 **true learning exercises** where you implement core ML/AI algorithms with:
- ✅ **Inline TODOs** with time estimates (5-6 hours per exercise)
- ✅ **Immediate feedback** (console output after every function)
- ✅ **Plug-and-play experimentation** (try 9 models in one run)
- ✅ **Shared infrastructure** (Docker, Makefiles reused across all exercises)
- ✅ **Reference implementations** (compare your solution in `_REFERENCE/`)

**Philosophy:** Focus on ML algorithms (95%), not DevOps boilerplate (5%).

---

## 🚀 **Quick Start**

### **1. Choose an Exercise**

Start with **01-ml/01-regression** (pilot exercise with full documentation).

```bash
cd exercises/01-ml/01-regression
```

### **2. Setup Environment**

**PowerShell (Windows):**
```powershell
.\setup.ps1
.\venv\Scripts\Activate.ps1
```

**Bash (Linux/Mac/WSL):**
```bash
./setup.sh
source venv/bin/activate
```

### **3. Run Interactive Training**

```bash
python main.py
```

**Expected output:**
```
📊 LOADING DATA
  ✓ Train: 16,512 samples × 8 features

🤖 MODEL TRAINING
→ Training Ridge (α=0.01)...
  ✓ Ridge (α=0.01): CV MAE = $39,850 | Time: 0.2s

📊 LEADERBOARD
┌────────────────┬─────────┐
│ Model          │ CV MAE  │
├────────────────┼─────────┤
│ XGBoost        │ $31,800 │
│ Ridge (α=1.0)  │ $38,100 │
└────────────────┴─────────┘

🏆 Best model: XGBoost
```

---

## 📂 **Exercise Structure**

```
exercises/
├── _infrastructure/          # Shared infrastructure
│   ├── docker/               # Generic Dockerfiles (ML, DL, API)
│   ├── compose/              # Shared Prometheus + Grafana
│   ├── Makefile.include      # Common targets
│   └── README.md             # Infrastructure guide
│
├── 01-ml/                    # 8 ML Paradigm Tracks
│   ├── 01-regression/        # ⭐ PILOT
│   ├── 02-classification/
│   ├── 03-neural-networks/
│   ├── 04-recommender-systems/
│   ├── 05-anomaly-detection/
│   ├── 06-reinforcement-learning/
│   ├── 07-unsupervised-learning/
│   └── 08-ensemble-methods/
│
├── 02-advanced_deep_learning/
├── 03-ai/
├── 04-multi_agent_ai/
├── 05-multimodal_ai/
├── 06-ai_infrastructure/
└── 07-devops_fundamentals/
```

---

## 📋 **Exercise Progression**

### **🟢 Beginner Track**

| Exercise | Topics | Time |
|----------|--------|------|
| [01-regression](01-ml/01-regression/) | Ridge, Lasso, XGBoost | 5-6h |
| [02-classification](01-ml/02-classification/) | Logistic, SVM, RandomForest | 5-6h |
| [07-unsupervised](01-ml/07-unsupervised-learning/) | K-means, DBSCAN, PCA | 4-5h |

### **🟡 Intermediate Track**

| Exercise | Topics | Time |
|----------|--------|------|
| [03-neural-networks](01-ml/03-neural-networks/) | MLP, CNN, backprop | 6-8h |
| [04-recommenders](01-ml/04-recommender-systems/) | Collaborative filtering | 5-6h |
| [08-ensemble](01-ml/08-ensemble-methods/) | Bagging, boosting | 5-6h |

### **🔴 Advanced Track**

| Exercise | Topics | Time |
|----------|--------|------|
| [02-adv-dl](02-advanced_deep_learning/) | ResNet, Transformers | 10-12h |
| [03-ai](03-ai/) | LLM fine-tuning, RAG | 10-12h |
| [05-multimodal](05-multimodal_ai/) | CLIP, image-text | 10-12h |

**Total:** ~100-120 hours (2-3 weeks full-time)

---

## 🎨 **What Makes These Different?**

### **1. Immediate Feedback**

Every TODO prints results:
```python
console.print(f"✓ Ridge (α={alpha}): CV MAE = ${cv_mae:,.0f}")
```

### **2. Plug-and-Play Experiments**

Compare 9 models in one run:
```python
runner.register("Ridge (α=0.1)", RidgeRegressor(alpha=0.1))
runner.register("XGBoost", XGBoostRegressor(n_estimators=100))
runner.run_experiment(X, y)
runner.print_leaderboard()
```

### **3. Shared Infrastructure**

**Before:** 300+ lines per exercise  
**After:** 8-line Makefile + 24-line docker-compose

```makefile
PROJECT_NAME = smartval-api
DOCKERFILE = Dockerfile.ml
include ../../_infrastructure/Makefile.include
```

### **4. Reference Solutions**

```
exercises/01-ml/01-regression/
├── src/models.py              # ⚠️ Your implementation
├── _REFERENCE/models_complete.py  # ✅ Complete reference
```

---

## 🛠️ **Shared Infrastructure**

### **One-Time Setup**

```bash
docker network create ml-network
```

### **Per-Exercise Workflow**

```bash
cd exercises/01-ml/01-regression
make setup       # Venv + dependencies
make test        # Run tests
python main.py   # Train with feedback

# Optional: Docker deployment
make docker-build
make docker-up
```

### **Port Allocation**

| Exercise | Port | Container |
|----------|------|-----------|
| 01-regression | 5001 | smartval-api |
| 02-classification | 5002 | classifier-api |
| 03-neural-networks | 5003 | nn-api |

---

## ✅ **Success Criteria**

After each exercise:

1. ✅ All tests pass
2. ✅ Leaderboard shows best model
3. ✅ Git commits show iteration
4. ✅ Can explain design decisions

---

## 📚 **Documentation**

- **[_infrastructure/README.md](_infrastructure/README.md)** — Infrastructure guide
- **[IMPROVEMENT_PLAN.md](IMPROVEMENT_PLAN.md)** — Complete redesign roadmap
- **[../notes/README.md](../notes/README.md)** — Theory and concepts

---

## 🐛 **Troubleshooting**

**"NotImplementedError"**  
→ Replace TODO stub with implementation (see exercise README)

**"Module 'rich' not found"**  
→ Run `pip install -r requirements.txt`

**"Port already in use"**  
→ Check docker-compose.yml for unique port

**"Network not found"**  
→ Run `docker network create ml-network`

---

## 💡 **Tips**

1. **Start with 01-regression** (pilot with full docs)
2. **Read notes first**, then do exercise
3. **Implement before** peeking at `_REFERENCE/`
4. **Experiment freely** (try 20 alphas!)
5. **Document journey** (Git commits, screenshots)

---

## 📊 **Metrics**

**Infrastructure:**
- Before: 5,000+ duplicate lines
- After: 300 shared lines
- **Reduction: 94%**

**Time:**
- Before: 14 hours per exercise
- After: 5-6 hours per exercise
- **Saved: 126 hours across 14 exercises**

---

## 🚀 **Ready?**

```bash
cd exercises/01-ml/01-regression
python main.py
```

Good luck! 🎉
