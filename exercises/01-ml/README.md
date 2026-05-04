# Machine Learning — Exercise Track

> **Grand Challenge:** Build production ML systems across 8 independent paradigms — regression, classification, neural networks, recommender systems, anomaly detection, reinforcement learning, unsupervised learning, and ensemble methods.

**Track Status:** ✅ All 8 exercises complete

---

## 0 · Grand Challenge: SmartVal AI

> 🎯 **The mission**: Build **SmartVal AI** — a production-grade home valuation system for California real estate  
> **Target**: <$40k MAE (Mean Absolute Error) on house price predictions (regulatory appraisal accuracy requirement)

**What we know so far:**
- ✅ **Dataset available**: California Housing (20,640 districts, 8 features: income, location, house age, etc.)
- ✅ **Business context**: Real estate platform needs automated valuations for multi-million-dollar lending decisions
- ✅ **Regulatory constraint**: Appraisal regulations require ±20% accuracy ($40k on $200k median house value)
- ❌ **But we can't ship yet**: Current approaches fail to meet the <$40k MAE target

**What's blocking us:**
1. **Naive baseline failure**: Predicting the training-set mean for every house → **$68k MAE** (70% above target)
2. **Simple linear regression**: Using only median income feature → **$52k MAE** (30% above target, still unacceptable for lending decisions)
3. **Feature engineering gap**: No polynomial features, no regularization → model can't capture non-linear relationships (income-value curve flattens at high incomes)
4. **Overfitting risk**: Without cross-validation and regularization, models memorize training ZIP codes instead of learning transferable patterns

**What this exercise unlocks:**
- **Regression fundamentals** (Ch.1-2): Multiple linear regression → **$55k MAE** (↓19% from naive baseline)
- **Feature engineering** (Ch.3-4): Polynomial features + interactions → **$48k MAE** (↓29% from baseline, 80% of the way to target)
- **Regularization** (Ch.5): Ridge/Lasso/Elastic Net → **$38k MAE ✅ Target achieved!** (↓44% from baseline)
- **Production optimization** (Ch.6-7): XGBoost + hyperparameter tuning → **$32k MAE** (↓53% from baseline, 20% better than regulatory minimum)

**After completing this exercise:**
- You'll understand *why* linear models work (interpretable coefficients, fast training) and when they fail (non-linear relationships, outliers)
- You'll master *regression fundamentals* that apply to all 8 ML paradigms in this track (loss functions, evaluation metrics, overfitting detection)
- You'll know *how to measure progress* with proper cross-validation, learning curves, and residual analysis
- You'll have *production-ready skills* for deploying regression APIs with monitoring and drift detection

---

## 8 Independent Exercise Tracks

Each exercise is a complete, self-contained project with its own:
- Dataset (different domains: housing, faces, customers, movies, transactions, games, etc.)
- Grand challenge (specific business constraint to satisfy)
- Chapter sequence (progressive capability unlock)
- Implementation TODOs (scaffolded learning path)

### 🎯 Core Fundamentals (Start Here)

#### 1. [Regression](01-regression/README.md) — SmartVal AI
> **Dataset**: California Housing (20,640 districts, 8 features)  
> **Grand Challenge**: <$40k MAE (regulatory appraisal accuracy)  
> **Chapters**: 7 chapters (`ch01`–`ch07`)

Learn regression fundamentals with a real estate valuation system. Master loss functions (MSE, MAE, Huber), regularization (Ridge, Lasso), and tree ensembles.

---

#### 2. [Classification](02-classification/README.md) — FaceAI  
> **Dataset**: CelebA (202,599 celebrity faces, 40 binary attributes)  
> **Grand Challenge**: >90% average accuracy across 40 facial attributes  
> **Chapters**: 5 chapters (`ch01`–`ch05`) from logistic regression through SVMs and tuning

Master classification with natural multi-label data. Learn logistic regression, softmax, class imbalance handling, and multi-label prediction.

---

#### 7. [Unsupervised Learning](07-unsupervised-learning/README.md) — SegmentAI
> **Dataset**: UCI Wholesale Customers (440 customers, 6 spending features)  
> **Grand Challenge**: 5 actionable segments, silhouette score >0.5  
> **Chapters**: 3 chapters (`ch01`–`ch03`) covering clustering, dimensionality reduction, and unsupervised metrics

Learn clustering and dimensionality reduction. Master K-means, DBSCAN, HDBSCAN, PCA, t-SNE, UMAP, and cluster validation.

---

### 🧠 Intermediate / Deep Learning

#### 3. [Neural Networks](03-neural-networks/README.md) — UnifiedAI
> **Datasets**: California Housing (regression) + CelebA (classification)  
> **Grand Challenge**: $28k MAE + 95% accuracy with same architecture  
> **Chapters**: 10 chapters from XOR to Transformers

Discover how neural networks unify regression and classification. Same feedforward architecture + backpropagation, different output layers + loss functions.

---

### 🚀 Advanced / Domain-Specific

#### 4. [Recommender Systems](04-recommender-systems/README.md) — FlixRec AI
> **Dataset**: MovieLens (100k ratings, 1,682 movies, 943 users)  
> **Grand Challenge**: RMSE <0.85 on user-movie ratings  
> **Chapters**: 4 chapters covering collaborative filtering, matrix factorization, and hybrid systems

Build Netflix-style recommendation engines. Learn collaborative filtering, SVD, ALS, and hybrid recommender architectures.

---

#### 5. [Anomaly Detection](05-anomaly-detection/README.md) — FraudGuard AI
> **Dataset**: Credit card transactions (284,807 transactions, 492 frauds)  
> **Grand Challenge**: >80% recall, <2% FPR (fraud detection SLA)  
> **Chapters**: 3 chapters covering statistical methods, isolation forests, and autoencoders

Detect rare events in imbalanced data. Master statistical outlier detection, isolation forests, and deep learning anomaly detection.

---

#### 6. [Reinforcement Learning](06-reinforcement-learning/README.md) — GameBot AI
> **Environment**: OpenAI Gym (CartPole, LunarLander, Atari)  
> **Grand Challenge**: Average reward >195 on CartPole-v1  
> **Chapters**: 5 chapters from Q-learning to Deep RL

Learn agents that learn from trial and error. Master Q-learning, policy gradients, DQN, and actor-critic methods.

---

### 🏆 Capstone / Production

#### 8. [Ensemble Methods](08-ensemble-methods/README.md) — MetaML AI
> **Datasets**: All previous 7 exercise datasets  
> **Grand Challenge**: Beat single-model baselines on all 7 challenges  
> **Chapters**: 4 chapters covering bagging, boosting, stacking, and voting

Improve every prior grand challenge with ensemble methods. Learn Random Forests, XGBoost, Stacking, and model combination strategies.

---

## Learning Paths

### Path 1: Beginner → Production Engineer (Linear)
**Goal:** Build production ML systems from scratch

```
1. Regression → 2. Classification → 3. Neural Networks → 8. Ensemble Methods
```

### Path 2: Fast-Track to Deep Learning
**Goal:** Get to neural networks quickly

```
1. Regression → 3. Neural Networks
```

### Path 3: Domain Specialist
**Goal:** Master specific ML subfield

```
1. Regression → 2. Classification → [Pick: 4. Recommenders | 5. Anomaly | 6. RL]
```

### Path 4: Comprehensive Mastery (All 8 Tracks)
**Goal:** Complete ML education

```
1 → 2 → 7 → 3 → 4 → 5 → 6 → 8 (recommended order)
```

---

## Prerequisites

- **Python 3.9+**
- **NumPy, Pandas, Scikit-learn** (data manipulation and classical ML)
- **PyTorch or TensorFlow** (neural networks only)
- **Matplotlib, Seaborn** (visualization)

Each exercise includes `setup.ps1` (Windows) and `setup.sh` (Unix) for automatic environment setup.

---

## Success Criteria

**After completing this exercise track, you will:**

✅ **Understand** when to use regression vs classification vs clustering vs RL  
✅ **Implement** production ML pipelines with proper train/val/test splits  
✅ **Evaluate** models with appropriate metrics (MAE, accuracy, silhouette, reward)  
✅ **Debug** common issues (overfitting, class imbalance, vanishing gradients)  
✅ **Deploy** ML systems with monitoring and drift detection  

**Validation test:** Pick any real-world ML problem. Can you identify the paradigm, select appropriate algorithms, implement a baseline, and measure performance? If yes → mastery achieved.

---

## See Also

- [notes/01-ml/README.md](../../notes/01-ml/README.md) — Topic-based ML curriculum (theory and detailed explanations)
- [exercises/README.md](../README.md) — All exercise tracks across AI portfolio
- [exercises/02-advanced_deep_learning/](../02-advanced_deep_learning/) — Advanced neural architectures (GANs, VAEs, Transformers)
