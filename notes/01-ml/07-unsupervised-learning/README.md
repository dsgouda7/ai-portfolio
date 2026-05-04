# Unsupervised Learning Track

> **The Mission**: Build **SegmentAI** — a customer segmentation system that discovers 5 actionable customer segments with silhouette score >0.5, enabling targeted marketing without any manual labelling.

You are the Lead Data Scientist at a retail company. The CMO says: "We blast the same marketing emails to 440 customers. Our open rate is 12%. I want segments — 'Loyalists', 'Price-sensitive', 'Big spenders' — so each group gets the right message. But nobody has time to label 440 customer records by hand." No labels. No ground truth. Pure unsupervised learning.

> 💡 **Dataset choice:** This track uses the UCI Wholesale Customers dataset (440 customers, 6 spending categories) instead of California Housing. Unsupervised methods (clustering, PCA) are best illustrated with multi-dimensional non-target data where "labels" don't exist.

> ⚠️ **Track scope (3 chapters):** This track covers the three unsupervised primitives — cluster discovery, feature compression, and cluster validation — needed to deliver actionable segments in a lean curriculum. Traditional agglomerative hierarchical clustering is intentionally omitted: HDBSCAN's hierarchical density approach covers the same pedagogical ground with better practical defaults. Extensions (Gaussian Mixture Models, autoencoder-based representation learning) appear in the Neural Networks and Ensemble Methods tracks.

---

## The Grand Challenge: 5 SegmentAI Constraints

| # | Constraint | Target | Why It Matters |
|---|------------|--------|----------------|
| **#1** | **SEGMENTATION** | Discover 5 distinct customer segments | Marketing needs 4-6 segments for differentiated campaigns — too few merges unlike groups, too many is operationally impossible |
| **#2** | **INTERPRETABILITY** | Segments must be business-actionable ("Loyalists", "Price-sensitive") | A cluster labelled "Cluster 3" is useless. Each segment needs a name the sales team can act on |
| **#3** | **STABILITY** | Clusters reproducible across bootstrap samples | If segments change every time you re-run, marketing can't build campaigns around them |
| **#4** | **SCALABILITY** | Work on 10k+ customers | Pilot on 440 customers, but the algorithm must scale to the full customer base |
| **#5** | **VALIDATION** | Silhouette score >0.5 | No ground truth exists — internal metrics are the only quantitative validation available |

---

## Progressive Capability Unlock

| Ch | Title | Silhouette | Key Unlock | Constraints |
|----|-------|-----------|------------|-------------|
| **1** | [Clustering](ch01_clustering) | 0.42 | K-Means discovers 5 initial segments; DBSCAN finds noise customers; HDBSCAN auto-discovers K | #1 Partial, #4 ✅ |
| **2** | [Dimensionality Reduction](ch02_dimensionality_reduction) | 0.48 | PCA + t-SNE + UMAP compress 6D → 2D; clustering in reduced space improves separation | #1 ✅, #2 Partial |
| **3** | [Unsupervised Metrics](ch03_unsupervised_metrics) | **>0.5 ✅** | Silhouette + Davies-Bouldin + Calinski-Harabasz suite; ARI/NMI for external validation; bootstrap stability; business names assigned | **All ✅** |

---

## Narrative Arc: From Raw Data to Actionable Segments

### 🎬 Act 1: Discovery (Ch.1)
**Cluster customers, see what emerges**

- **K-Means** on 6 purchase features (Fresh, Milk, Grocery, Frozen, Detergents_Paper, Delicatessen)
- Elbow method suggests K=5; initial silhouette = 0.42 (weak but promising)
- DBSCAN reveals 23 noise customers (extreme spenders) that distort K-Means
- HDBSCAN auto-discovers 4 segments — close, but business needs 5

*"I can see the segments forming, but 0.42 silhouette means they're overlapping. The features are on wildly different scales — Fresh ranges 3–112k while Delicatessen ranges 3–48k. Can we compress this?" — Data Scientist*

**Status**: ❌ Segments overlap. Need better feature space.

---

### ⚡ Act 2: Better Feature Space (Ch.2)
**Reduce dimensions, sharpen boundaries**

- **PCA**: 6D → 2D. First 2 PCs explain 72% of variance. PC1 = "total spend", PC2 = "fresh vs grocery"
- **t-SNE**: Reveals cluster topology — 5 groups visible but distances lie
- **UMAP**: Preserves global structure, clusters tighter. Re-running K-Means on UMAP 2D → silhouette jumps to 0.48
- Curse of dimensionality was hurting us — sparse 6D space had noisy distance calculations

*"Now I can SEE the five segments! But is 0.48 good enough? And which K is actually optimal — the elbow says 3, my eyes say 5, the CMO wants 5. How do I decide?" — Data Scientist*

**Status**: ✅ Segments visible. ❌ Not yet validated quantitatively.

---

### 📊 Act 3: Validation (Ch.3)
**Measure, name, and stabilise segments**

- **Silhouette analysis**: K=5 after PCA preprocessing → silhouette = 0.52 ✅ (above 0.5 threshold!)
- **Metric disagreement**: Silhouette prefers K=3 (0.58), but business needs K=5 (0.52) — acceptable trade-off
- **Business naming**: Centroid analysis → "Loyalists", "Price-Sensitive", "Big Spenders", "Occasional Buyers", "Deli Specialists"
- **Bootstrap stability**: 95% of customers assigned to the same segment across 100 bootstrap samples

*"Five segments, silhouette 0.52, stable across resamples, and the sales team immediately recognized 'Big Spenders' vs 'Price-Sensitive'. Ship it!" — CMO*

**Status**: ✅✅✅✅✅ **ALL CONSTRAINTS SATISFIED!**

---

## The Dataset: UCI Wholesale Customers

Every chapter uses the same dataset: [Wholesale Customers](https://archive.ics.uci.edu/ml/datasets/Wholesale+customers) from the UCI ML Repository.

**440 customers** with 6 spending features (annual spend in monetary units):

| Feature | Description | Range |
|---------|-------------|-------|
| `Fresh` | Annual spending on fresh products | 3 – 112,151 |
| `Milk` | Annual spending on milk products | 55 – 73,498 |
| `Grocery` | Annual spending on grocery products | 3 – 92,780 |
| `Frozen` | Annual spending on frozen products | 25 – 60,869 |
| `Detergents_Paper` | Annual spending on detergents and paper | 3 – 40,827 |
| `Delicatessen` | Annual spending on delicatessen products | 3 – 47,943 |

Plus 2 categorical features (Channel: Hotel/Restaurant/Café vs Retail; Region: 1/2/3) used only for external validation.

**Why this dataset is perfect for unsupervised learning:**
- ✅ No ground truth labels — genuine unsupervised scenario
- ✅ Business-interpretable features (everyone understands spending)
- ✅ Small enough to visualise (440 points), representative enough to learn
- ✅ Natural cluster structure (hotel buyers vs retail buyers vs specialty shops)
- ✅ Skewed distributions — forces you to handle log-transforms and scaling

```python
import pandas as pd
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00292/Wholesale%20customers%20data.csv"
df = pd.read_csv(url)
X = df[['Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicatessen']].values
```

> ⚡ **Compute**: All examples run on CPU. No GPU required — 440 rows × 6 features is fast enough for all algorithms covered here.

---

## Five SegmentAI Segments (Final Result)

| Segment | Name | Profile | Size | Marketing Action |
|---------|------|---------|------|------------------|
| 0 | **Loyalists** | High grocery + milk + detergents; steady, predictable buyers | ~28% | Loyalty rewards, subscription offers |
| 1 | **Price-Sensitive** | Low across all categories; minimal baskets | ~22% | Discount coupons, loss leaders |
| 2 | **Big Spenders** | High fresh + frozen + deli; premium products | ~15% | Premium catalogue, early access |
| 3 | **Occasional Buyers** | Moderate spend, infrequent; grocery-focused | ~25% | Re-engagement campaigns, bundles |
| 4 | **Deli Specialists** | Disproportionately high deli + fresh; low grocery | ~10% | Specialty product recommendations |

---

## Cross-Track Connections

- **K-Means / DBSCAN fundamentals** are revisited in the Regression track (SmartVal AI neighbourhood-clustering context) and Recommender Systems (user segment cold-start priors)
- **PCA for feature compression** reappears in the Neural Networks track (Ch.8 TensorBoard projector for embedding visualisation)
- **Silhouette and Davies-Bouldin** metrics provide the evaluation vocabulary used in Ensemble Methods cluster-ensemble experiments
- **Cluster-based user segmentation** feeds directly into [04_recommender_systems](../04_recommender_systems/README.md) as a cold-start strategy for new users with no rating history
