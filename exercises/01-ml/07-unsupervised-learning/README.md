# Exercise 07: SegmentAI — Unsupervised Learning with TODO Pattern

> **Learn unsupervised learning through hands-on implementation of KMeans, DBSCAN, and PCA with immediate feedback**

**Scaffolding Level:** 🟢 Heavy (learn the workflow with guided TODOs)

---

## Objective

Implement a complete unsupervised ML pipeline with clustering and dimensionality reduction:
- Understand distance-based clustering (KMeans)
- Implement density-based clustering (DBSCAN) with outlier detection
- Apply dimensionality reduction (PCA) for visualization
- Compare models using silhouette score and other metrics
- Find optimal K using the elbow method
- See immediate feedback after each model trains

**Learning Focus:** Understand why normalization is CRITICAL for distance-based algorithms, how clustering differs from supervised learning, and how to evaluate models without labels.

---

## What You'll Learn

### Clustering Algorithms
- **KMeans:** Centroid-based partitioning for spherical clusters
- **DBSCAN:** Density-based clustering for arbitrary shapes + outliers
- **PCA:** Dimensionality reduction preserving variance

### Unsupervised Learning Metrics
- **Silhouette Score [-1, 1]:** Measures cluster cohesion/separation
  - 1 = perfect clustering (points close to own cluster, far from others)
  - 0 = overlapping clusters
  - -1 = wrong cluster assignments
- **Calinski-Harabasz Score:** Variance ratio (higher = better)
- **Inertia:** Sum of squared distances to centroids (lower = better, but decreases with K)

### Key Concepts
1. **Why Normalization Matters:** Distance-based algorithms use Euclidean distance
   - Features with large ranges dominate (e.g., income 0-1M vs age 0-100)
   - StandardScaler ensures equal influence: (X - mean) / std
2. **Elbow Method:** Find optimal K where inertia decrease slows dramatically
3. **Noise Handling:** DBSCAN labels outliers as -1 (excluded from silhouette)
4. **Evaluation Without Labels:** Use internal metrics (silhouette, inertia) instead of accuracy

---

## Setup

**Unix/macOS/WSL:**
```bash
chmod +x setup.sh
./setup.sh
source venv/bin/activate
```

**Windows PowerShell:**
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\setup.ps1
.\venv\Scripts\Activate.ps1
```

---

## Project Structure (Simplified for Learning)

```
07-unsupervised-learning/
├── requirements.txt          # Dependencies (sklearn, rich, pandas, numpy)
├── setup.sh / setup.ps1      # Environment setup
├── config.yaml               # Hyperparameters
├── README.md                 # This file
├── main.py                   # Interactive demonstration script
├── src/
│   ├── __init__.py           
│   ├── utils.py              # Logging and timing
│   ├── data.py               # Load sample dataset
│   ├── features.py           # FeatureNormalizer (TODO: StandardScaler)
│   ├── models.py             # KMeans, DBSCAN, PCA, ExperimentRunner (TODOs)
│   └── evaluate.py           # Evaluation helpers
├── tests/                    # Unit tests
├── models/                   # Saved models (gitignored)
├── logs/                     # Logs (gitignored)
└── data/                     # Datasets (gitignored)
```

> **Note:** Infrastructure files (Dockerfile, docker-compose.yml, Makefile) have been removed to keep the focus on core ML learning. The minimal setup uses only Python virtual environments.

---

## TODO Implementation Guide

### 📝 TODOs by File (15 total, ~260 minutes)

#### `src/features.py` — Feature Normalization (3 TODOs, 35-40 min)
1. ✅ **fit_transform** (15-20 min): StandardScaler fitting and transformation
2. ✅ **transform** (10 min): Apply fitted scaler to new data
3. ✅ **_validate_input** (10 min): Check for NaN/inf values

**Key Learning:** Why normalization is CRITICAL for distance-based algorithms

#### `src/models.py` — Clustering Models (12 TODOs, 225-280 min)

**KMeansClusterer (1 TODO, 30-40 min):**
4. ✅ **fit** (30-40 min): Implement KMeans with silhouette/inertia metrics

**DBSCANClusterer (1 TODO, 35-45 min):**
5. ✅ **fit** (35-45 min): Implement DBSCAN with noise handling

**PCAReducer (1 TODO, 20-30 min):**
6. ✅ **fit** (20-30 min): Implement PCA dimensionality reduction

**ExperimentRunner (3 TODOs, 65-85 min):**
7. ✅ **run_experiment** (15-20 min): Train all registered models
8. ✅ **print_leaderboard** (20-25 min): Display results in rich table
9. ✅ **find_optimal_k** (30-40 min): Elbow method for optimal K

**Key Learning:** Plug-and-play architecture, silhouette scoring, handling outliers

---

## Running the Exercise

### Step 1: Implement TODOs
Start with `src/features.py` (easier), then move to `src/models.py`:

```bash
# Open in your editor and implement TODOs one by one
code src/features.py
code src/models.py
```

### Step 2: Run Main Script
See immediate feedback as each model trains:

```bash
python main.py
```

**Expected Output:**
```
╔═══════════════════════════════════════╗
║         SegmentAI                      ║
║ Interactive Clustering and             ║
║ Dimensionality Reduction               ║
╚═══════════════════════════════════════╝

📊 LOADING DATA
  ✓ Loaded: 150 samples × 4 features

🔧 FEATURE NORMALIZATION
  ✓ Standardized 4 features (mean=0, std=1)

📉 DIMENSIONALITY REDUCTION (PCA)
  ✓ PCA: 4 → 2 features | Variance retained = 97.8%

🤖 CLUSTERING EXPERIMENT
  ✓ KMeans (K=3): Silhouette = 0.551 | Inertia = 78.9 | Time: 0.3s
  ✓ DBSCAN (ε=0.5): Silhouette = 0.492 | Clusters = 2 | Noise = 17 (11%) | Time: 0.1s

┏━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━┓
┃ Rank┃ Model        ┃ Silhouette┃ Clusters┃ Notes      ┃
┣━━━━━╋━━━━━━━━━━━━━━╋━━━━━━━━━━━╋━━━━━━━━╋━━━━━━━━━━━━━┫
┃  1  ┃ KMeans (K=3) ┃    0.551  ┃    3   ┃ Inertia: 79┃
┃  2  ┃ DBSCAN (ε=0.5)┃    0.492  ┃    2   ┃ Noise: 11% ┃
┗━━━━━┻━━━━━━━━━━━━━━┻━━━━━━━━━━━┻━━━━━━━━┻━━━━━━━━━━━━━┛

✓ EXPERIMENT COMPLETE
```

### Step 3: Experiment with Hyperparameters
Try different configurations in `main.py`:
- KMeans: Try K=2, 4, 5, 10
- DBSCAN: Try eps=0.3, 0.7, 1.0 (observe noise ratio)
- Observe silhouette scores and cluster distributions

---

## Understanding Clustering Metrics

### Silhouette Score (Most Important)
```python
silhouette = silhouette_score(X, labels)
```
- **Range:** -1 to 1
- **Interpretation:**
  - 0.7-1.0: Strong structure (well-separated clusters)
  - 0.5-0.7: Reasonable structure
  - 0.25-0.5: Weak structure (clusters overlap)
  - <0.25: No meaningful structure

### Calinski-Harabasz Score (Variance Ratio)
```python
calinski = calinski_harabasz_score(X, labels)
```
- **Range:** 0 to ∞ (unbounded)
- **Higher is better** (between-cluster variance / within-cluster variance)
- No fixed threshold, use for comparing models on same dataset

### Inertia (KMeans-specific)
```python
inertia = kmeans.inertia_
```
- **Range:** 0 to ∞ (sum of squared distances)
- **Lower is better** BUT always decreases with more K
- Use **elbow method** to find optimal K:
  - Plot inertia vs K
  - Find "elbow" where decrease slows dramatically
  - That K is optimal

---

## Key Differences from Supervised Learning

| Aspect | Supervised | Unsupervised (Clustering) |
|--------|-----------|---------------------------|
| **Training Data** | X (features) + y (labels) | X (features) only |
| **Goal** | Predict known targets | Discover natural groupings |
| **Evaluation** | Accuracy, MAE, R² | Silhouette, Calinski-H, Inertia |
| **Metrics Need** | Ground truth labels | Internal validation |
| **Use Cases** | Regression, classification | Segmentation, anomaly detection |

---

## Troubleshooting

### "Silhouette score is -1.0"
- DBSCAN found all noise (eps too small) or <2 clusters
- Try larger eps or smaller min_samples

### "All points in one cluster"
- DBSCAN eps too large
- Try smaller eps values (e.g., 0.3 instead of 1.0)

### "Silhouette score very low (<0.2)"
- Data may not have natural clusters
- Try different algorithms (DBSCAN vs KMeans)
- Try feature engineering or different normalizations

### "Models train but leaderboard crashes"
- Check that silhouette_score exists in metrics dict
- Handle edge cases: `result.get("silhouette_score", -999)`

---

## Success Criteria

Your exercise is complete when:
- [ ] All 15 TODOs implemented and tested
- [ ] `python main.py` runs without errors
- [ ] Leaderboard displays sorted by silhouette score
- [ ] KMeans achieves >0.5 silhouette on sample dataset
- [ ] DBSCAN correctly identifies noise points
- [ ] PCA reduces dimensions while retaining >80% variance
- [ ] Console output shows immediate feedback after each model

**Bonus Challenges:**
- [ ] Implement `find_optimal_k()` with elbow method
- [ ] Add matplotlib visualization of clusters in 2D PCA space
- [ ] Compare silhouette scores across different K values
- [ ] Experiment with different datasets (Iris, Wine, Blobs)

---

## Next Steps

After completing this exercise:
1. **Compare to supervised learning (01-regression):**
   - Notice evaluation metrics differences
   - Understand when to use supervised vs unsupervised
2. **Explore advanced clustering:**
   - Hierarchical clustering (dendrograms)
   - Gaussian Mixture Models (soft clustering)
   - HDBSCAN (hierarchical density-based)
3. **Real-world applications:**
   - Customer segmentation
   - Anomaly detection
   - Image compression
   - Document clustering

---

## References

- [Scikit-learn Clustering](https://scikit-learn.org/stable/modules/clustering.html)
- [Understanding Silhouette Score](https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html)
- [KMeans Explained](https://scikit-learn.org/stable/modules/clustering.html#k-means)
- [DBSCAN Explained](https://scikit-learn.org/stable/modules/clustering.html#dbscan)
- [PCA Explained](https://scikit-learn.org/stable/modules/decomposition.html#pca)

---

## FAQ

**Q: Why is normalization so important for clustering?**  
A: Distance-based algorithms (KMeans, DBSCAN) use Euclidean distance. Features with larger ranges (e.g., income 0-1M) will dominate distance calculations over smaller-range features (e.g., age 0-100), even if age is more relevant for clustering. StandardScaler ensures each feature has equal influence by transforming to mean=0, std=1.

**Q: How do I choose between KMeans and DBSCAN?**  
A: 
- **KMeans:** Use when you know K, clusters are roughly spherical, no outliers
- **DBSCAN:** Use when you don't know K, clusters have arbitrary shapes, or you need outlier detection

**Q: What's a "good" silhouette score?**  
A: Depends on dataset, but general guidelines:
- >0.7: Excellent clustering
- 0.5-0.7: Good clustering
- 0.25-0.5: Weak but potentially useful
- <0.25: Random/no structure

**Q: Why does my DBSCAN have all noise?**  
A: `eps` (epsilon) is too small. Points must be within `eps` distance to form a cluster. Try larger values (e.g., 0.5, 0.7, 1.0) until you get reasonable clusters.

**Q: How do I find optimal K for KMeans?**  
A: Use the **elbow method**:
1. Train KMeans for K=2, 3, 4, ..., 10
2. Plot inertia vs K
3. Find "elbow" where inertia decrease slows
4. That K is optimal (diminishing returns after)
# Health check
curl http://localhost:5000/health

# Cluster a sample
curl -X POST http://localhost:5000/cluster \
  -H "Content-Type: application/json" \
  -d '{"features": [5.1, 3.5, 1.4, 0.2]}'

# View metrics
curl http://localhost:5000/metrics
```

---

## Resources

**Concept Review:**
- [notes/01-ml/07_unsupervised_learning/](../../notes/01-ml/07_unsupervised_learning/) — Complete track (when available)
- Scikit-learn clustering: https://scikit-learn.org/stable/modules/clustering.html

**Implementation Guides:**
- [exercises/01-ml/01-regression/](../01-regression/) — Reference supervised learning patterns
- UMAP documentation: https://umap-learn.readthedocs.io/
- HDBSCAN documentation: https://hdbscan.readthedocs.io/

---

## Common Commands

```bash
# Run tests
make test

# Lint code
make lint

# Format code
make format

# Run API server
make run

# Build Docker image
make docker-build

# Start services
make docker-up
```

---

## Production Patterns Applied

✅ **Configuration-driven:** All hyperparameters in `config.yaml`  
✅ **Logging:** Structured logs to `logs/api.log`  
✅ **Monitoring:** Prometheus metrics for latency, cluster distribution, errors  
✅ **Error handling:** Comprehensive validation and exception handling  
✅ **Testing:** Unit tests for all modules  
✅ **Containerization:** Multi-stage Dockerfile for optimized image size  
✅ **API validation:** Pydantic schemas for request validation  
✅ **Drift monitoring:** Track cluster distribution over time  

---

## Next Steps

After completing this exercise, try:
1. **Experiment with datasets:** Try Wine dataset or synthetic blobs with different cluster counts
2. **Tune hyperparameters:** Optimize eps/min_samples for DBSCAN, linkage method for Hierarchical
3. **Add anomaly detection:** Use DBSCAN noise points (-1) as outliers
4. **Implement elbow automation:** Automatically select optimal k from elbow plot
5. **Add dimensionality reduction:** Use UMAP for 2D visualization of high-dimensional clusters

---

## Troubleshooting

**Issue:** Silhouette score is negative  
**Fix:** Try different number of clusters (elbow method) or different algorithm (DBSCAN for non-spherical shapes)

**Issue:** All samples assigned to one cluster  
**Fix:** Check feature scaling (must standardize!), reduce eps for DBSCAN, or increase n_clusters for KMeans

**Issue:** DBSCAN finds too much noise  
**Fix:** Increase eps (neighborhood size) or decrease min_samples

**Issue:** API returns 503 (model not loaded)  
**Fix:** Train a model first and save to `models/best_model.pkl`

---

## License

MIT License - See repository root for details.


