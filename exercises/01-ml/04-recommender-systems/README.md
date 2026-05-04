# Exercise 04: FlixAI — Interactive Recommender System

> **Learning Goal:** Implement Collaborative Filtering and Matrix Factorization with plug-and-play experimentation and immediate feedback  
> **Prerequisites:** Completed [notes/01-ml/04-recommender-systems/](../../../notes/01-ml/04-recommender-systems/)  
> **Time Estimate:** 4-5 hours (coding) + 1 hour (evaluation)  
> **Difficulty:** ⭐⭐⭐ Intermediate-Advanced

---

## 🎯 **What You'll Implement**

Starting from function stubs and inline TODOs, you'll build a complete recommender system with:

### **Core Implementation (4-5 hours)**

| File | What You Implement | TODOs | Time |
|------|-------------------|-------|------|
| `src/features.py` | User-item matrix + rating stats | 5 functions | 1h |
| `src/models.py` | Collaborative Filtering recommender | 2 methods | 45min |
| `src/models.py` | Matrix Factorization (SVD) | 2 methods | 1h |
| `src/models.py` | ExperimentRunner + evaluation | 3 methods | 1h |
| `main.py` | Recommendation generation + display | 2 sections | 30min |

**Interactive Experience:**
- ✅ See RMSE and coverage immediately after each model trains
- ✅ Leaderboard shows best recommender automatically
- ✅ Rich console output with colors and tables
- ✅ Generate and display sample recommendations
- ✅ Experiment with 6+ models in one run

**Total:** 4-5 hours of focused coding

---

### **What's Already Done (Utilities)**

These files are complete and reusable:
- ✅ `src/data.py` — MovieLens data loading
- ✅ `src/utils.py` — Logging and validation
- ✅ `src/evaluate.py` — Ranking metrics (Precision@k, Recall@k)
- ✅ `src/monitoring.py` — Prometheus metrics
- ✅ `src/api.py` — Flask REST API (pre-built for deployment)

**Philosophy:** Focus on recommendation algorithms, not boilerplate.

---

### **Optional: Production Deployment (1 hour)**

> **Note:** Infrastructure files (Dockerfile, docker-compose.yml, Makefile) have been removed from this exercise to focus on core ML implementation. For production deployment, refer to `../../_infrastructure/` for shared Docker/Prometheus configs.

If you want to deploy the completed exercise:
1. Copy relevant infrastructure files from `../../_infrastructure/` to this directory
2. Build and deploy using the shared infrastructure setup
3. Test the API endpoint

For this exercise, focus on implementing the core recommendation algorithms in `src/`.

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
  ✓ Loaded 100,000 ratings
  Rating statistics:
    Total ratings: 100,000
    Users: 943 | Items: 1,682
    Mean rating: 3.53 ± 1.13
    Sparsity: 93.7%

✂️  SPLITTING DATA
  Split ratings: 80,000 train, 20,000 test

🔧 BUILDING USER-ITEM MATRIX
  ✓ Train matrix: 943 users × 1,682 items
  ✓ Test matrix: 943 users × 1,682 items

🤖 MODEL TRAINING

→ Training CF (k=10)...
  ✓ CF (k=10): RMSE = 1.034 | Coverage = 94.2% | Time: 12.3s

→ Training CF (k=20)...
  ✓ CF (k=20): RMSE = 0.987 | Coverage = 94.2% | Time: 13.1s

... (4 more models)

📊 LEADERBOARD
┌─────────────────────────┬──────────┬──────────┬──────────────────┐
│ Model                   │ RMSE     │ MAE      │ Coverage/Var Exp │
├─────────────────────────┼──────────┼──────────┼──────────────────┤
│ MF (k=50)               │ 0.912    │ 0.719    │ 68.3%            │
│ MF (k=100)              │ 0.928    │ 0.731    │ 72.1%            │
│ CF (k=20)               │ 0.987    │ 0.781    │ 94.2%            │
...
└─────────────────────────┴──────────┴──────────┴──────────────────┘

🏆 Best model: MF (k=50) | RMSE: 0.912

📈 RECOMMENDATION QUALITY
  Precision@10: 0.342
  Recall@10: 0.189
```

---

## 📋 **Step-by-Step Guide**

### **Step 1: Build User-Item Matrix (15 min)**

Open `src/features.py`, find `build_user_item_matrix()`, implement:
1. Create pivot table with user_id as rows, item_id as columns
2. Fill missing values with 0 (unrated items)
3. Log matrix dimensions and sparsity

**Test:**
```bash
python -c "from src.features import build_user_item_matrix; from src.data import load_movielens; ratings = load_movielens(); matrix = build_user_item_matrix(ratings); print(f'Matrix shape: {matrix.shape}')"
```

**Expected output:**
```
User-item matrix: 943 users × 1,682 items
Sparsity: 93.7% (100,000 ratings)
Matrix shape: (943, 1682)
```

---

### **Step 2: Implement Collaborative Filtering (45 min)**

Open `src/models.py`, find `CollaborativeFilteringRecommender.train()`:
1. Compute user-user cosine similarity matrix
2. For each rating, predict using k-nearest neighbors
3. Compute RMSE, MAE, and coverage
4. ⭐ **Print results immediately** (see console output)

**Test:**
```bash
python -c "from src.models import CollaborativeFilteringRecommender, ModelConfig; from src.data import load_movielens; from src.features import build_user_item_matrix, split_ratings; ratings = load_movielens(); train_ratings, _ = split_ratings(ratings); matrix = build_user_item_matrix(train_ratings); cf = CollaborativeFilteringRecommender(n_neighbors=20); metrics = cf.train(train_ratings, matrix, ModelConfig())"
```

**Expected output:**
```
✓ Collaborative Filtering (k=20): RMSE = 0.987 | Coverage = 94.2% | Time: 13.1s
```

---

### **Step 3: Implement Matrix Factorization (60 min)**

Open `src/models.py`, find `MatrixFactorizationRecommender.train()`:
1. Center user-item matrix by subtracting global mean
2. Apply SVD decomposition (scipy.sparse.linalg.svds)
3. Store user factors, item factors, and sigma matrix
4. Compute RMSE and variance explained
5. ⭐ **Print results immediately**

**Key concepts:**
- **SVD:** R ≈ U @ Σ @ V^T
- **U:** User latent factors (users × k)
- **V^T:** Item latent factors (k × items)
- **Σ:** Singular values (capture importance)

**Test:**
```bash
python -c "from src.models import MatrixFactorizationRecommender, ModelConfig; from src.data import load_movielens; from src.features import build_user_item_matrix, split_ratings; ratings = load_movielens(); train_ratings, _ = split_ratings(ratings); matrix = build_user_item_matrix(train_ratings); mf = MatrixFactorizationRecommender(n_factors=50); metrics = mf.train(train_ratings, matrix, ModelConfig())"
```

**Expected output:**
```
✓ Matrix Factorization (k=50): RMSE = 0.912 | Var Explained = 68.3% | Time: 8.4s
```

---

### **Step 4: Implement ExperimentRunner (25 min)**

Open `src/models.py`, find `ExperimentRunner.run_experiment()`:
1. Loop through registered recommenders
2. Train each model (prints immediately)
3. Store results for leaderboard

Then implement `print_leaderboard()`:
1. Sort results by RMSE (lower is better)
2. Create rich table with model comparison
3. Print winner

**Expected behavior:**
- Models train sequentially with immediate feedback
- Leaderboard shows sorted comparison
- Winner announcement at the end

---

### **Step 5: Evaluate Recommendations (25 min)**

Open `src/models.py`, find `ExperimentRunner.evaluate_recommendations()`:
1. Get best model from experiment
2. For each test user, generate top-k recommendations
3. Compute precision@k and recall@k
4. Print metrics

**Metrics explained:**
- **Precision@k:** What fraction of recommended items are relevant?
- **Recall@k:** What fraction of relevant items are recommended?
- **Relevant item:** Rating ≥ 4 stars

**Test:**
```bash
python main.py
```

---

## 📊 **Understanding the Metrics**

### **Training Metrics**

| Metric | What It Measures | Good Value |
|--------|-----------------|------------|
| **RMSE** | Average rating prediction error | < 0.90 |
| **MAE** | Mean absolute error | < 0.70 |
| **Coverage** | % of items that can be recommended | > 90% |
| **Var Explained** | How well SVD captures patterns | > 65% |

### **Recommendation Quality Metrics**

| Metric | What It Measures | Good Value |
|--------|-----------------|------------|
| **Precision@10** | Accuracy of top-10 recommendations | > 0.30 |
| **Recall@10** | Coverage of relevant items in top-10 | > 0.15 |
| **NDCG@10** | Ranking quality (order matters) | > 0.35 |

### **Why Matrix Factorization Usually Wins**

1. **Captures latent preferences:** Discovers hidden patterns (e.g., "action fans")
2. **Handles sparsity better:** SVD interpolates missing ratings
3. **Scalable:** Fast predictions (dot product)
4. **Generalizes well:** Reduces overfitting via dimensionality reduction

Collaborative filtering wins on:
- **Interpretability:** "Users like you rated this 4.5 stars"
- **Coverage:** Can recommend any item someone rated
- **Simplicity:** No training required (just similarity)

---

## 🧪 **Experimentation Ideas**

### **Easy (30 min each)**

1. **Try different k values:**
   - CF: k=5, 10, 20, 30, 50
   - MF: k=10, 20, 50, 100, 200

2. **Normalize ratings:**
   - Use `normalize_ratings()` from features.py
   - Does centering improve RMSE?

3. **Vary train/test split:**
   - Try 60/40, 70/30, 80/20, 90/10
   - How does training data affect precision@k?

### **Medium (1-2 hours each)**

4. **Item-based collaborative filtering:**
   - Compute item-item similarity instead of user-user
   - "Users who liked X also liked Y"

5. **Implicit feedback:**
   - Binarize ratings (1 if rated, 0 otherwise)
   - Use binary cross-entropy loss

6. **Cold start handling:**
   - For new users: Recommend popular items
   - For new items: Use content-based features

### **Advanced (3+ hours each)**

7. **Alternating Least Squares (ALS):**
   - Implement iterative optimization
   - Compare with SVD-based MF

8. **Neural Collaborative Filtering:**
   - Build TensorFlow model with user/item embeddings
   - Use MLP to learn interaction function

9. **Diversity-aware recommendations:**
   - Penalize similar items in top-k
   - Measure genre diversity

---

## 🐛 **Troubleshooting**

### **Issue:** "Model not trained yet"
**Fix:** Ensure you call `train()` before `recommend()`

### **Issue:** High RMSE (> 1.2)
**Fix:** 
- Check k value (try increasing for MF)
- Verify matrix isn't all zeros
- Ensure ratings are in 1-5 range

### **Issue:** Low precision@k (< 0.2)
**Fix:**
- Increase n_factors for MF
- Try normalizing ratings
- Filter out unpopular items (< 5 ratings)

### **Issue:** Memory error during SVD
**Fix:**
- Reduce n_factors (try k=20 first)
- Use sparse SVD (`scipy.sparse.linalg.svds`)
- Sample subset of users/items

### **Issue:** Slow collaborative filtering
**Fix:**
- Reduce n_neighbors (k=10 instead of 50)
- Vectorize predictions (avoid Python loops)
- Use approximate nearest neighbors (FAISS)

---

## 📚 **Resources**

### **Concept Review**
- [Matrix Factorization Tutorial](https://datajobs.com/data-science-repo/Recommender-Systems-[Netflix].pdf) — Netflix Prize winning approach
- [Collaborative Filtering Explained](https://towardsdatascience.com/various-implementations-of-collaborative-filtering-100385c6dfe0)
- [SVD for Recommenders](https://sifter.org/~simon/journal/20061211.html)

### **Metrics**
- **Precision@k:** True positives / k (accuracy of recommendations)
- **Recall@k:** True positives / total relevant (coverage of relevant items)
- **NDCG@k:** Normalized Discounted Cumulative Gain (ranking quality)

### **Advanced Topics**
- [Neural CF Paper (He et al., 2017)](https://arxiv.org/abs/1708.05031)
- [FAISS Similarity Search](https://github.com/facebookresearch/faiss)
- [Alternating Least Squares (ALS)](http://yifanhu.net/PUB/cf.pdf)

---

## 🎓 **Learning Outcomes**

After completing this exercise, you will be able to:

✅ Build user-item matrices from ratings data  
✅ Implement collaborative filtering with k-nearest neighbors  
✅ Implement matrix factorization using SVD  
✅ Compare recommender algorithms systematically  
✅ Evaluate recommendation quality with precision@k and recall@k  
✅ Generate top-k recommendations for users  
✅ Handle sparse rating matrices effectively  
✅ Explain the trade-offs between CF and MF approaches

---

## 🚀 **Next Steps**

After completing this exercise:

1. **Implement additional algorithms:**
   - Item-based CF
   - ALS matrix factorization
   - Neural collaborative filtering

2. **Add content features:**
   - Movie genres, actors, directors
   - Hybrid recommender (collaborative + content)

3. **Optimize for production:**
   - Pre-compute item similarities
   - Use approximate nearest neighbors
   - Batch recommendations

4. **Deploy and monitor:**
   - REST API for recommendations
   - A/B testing framework
   - Track click-through rate, conversion

5. **Advanced evaluation:**
   - NDCG@k for ranking quality
   - Diversity metrics (genre distribution)
   - Serendipity (surprising but relevant)

---

## 📝 **Solution Checklist**

- [ ] `build_user_item_matrix()` creates correct pivot table
- [ ] Sparsity calculation is accurate
- [ ] Collaborative filtering computes user similarity
- [ ] CF prediction uses weighted average of neighbors
- [ ] Matrix factorization applies SVD correctly
- [ ] MF recommendation reconstructs ratings
- [ ] ExperimentRunner trains all models
- [ ] Leaderboard sorts by RMSE
- [ ] Precision@k and recall@k computed correctly
- [ ] Sample recommendations displayed nicely
- [ ] All tests pass: `pytest tests/`
- [ ] Code follows style: `black . && flake8 src/`

---

**Happy recommending! 🎬**
