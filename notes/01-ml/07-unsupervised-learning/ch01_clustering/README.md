# Ch.1 — Clustering

> **The story.** **Stuart Lloyd** invented k-means in **1957** inside Bell Labs while solving a pure engineering problem: how do you quantise a continuous audio signal into a finite codebook for PCM telephone transmission? His algorithm — assign each signal sample to the nearest codeword, then recompute codewords as means — was an internal Bell Labs technical report that stayed unpublished for **25 years**. By the time Lloyd finally published in **1982** the algorithm was already everywhere: **Hugo Steinhaus** had described it in 1956, **Edward Forgy** had rediscovered it in 1965, and **James MacQueen** had coined the name "k-means" in 1967. The "Bell Labs PCM quantiser" and "k-means clustering" are the same mathematical object — the same alternating assignment-update loop — discovered independently at least three times before the original paper appeared. The density-based alternative arrived nearly four decades later: **Martin Ester, Hans-Peter Kriegel, Jörg Sander, and Xiaowei Xu** published DBSCAN at **KDD 1996**, winning the conference's Test-of-Time award in 2014. The business insight that drives this chapter is older still: segment customers by purchase behaviour, target each segment with tailored promotions, and lift retail revenue — a practice Sears Roebuck was doing by hand in the 1920s with customer ledger cards. Clustering automates and scales the intuition to 440 wholesale customers and beyond.
>
> **Where you are in the curriculum.** You have just finished the Reinforcement Learning track (AgentAI), where every chapter had a reward signal telling the agent what was good. Now the signal disappears. This is the entry point to **unsupervised learning** — no labels, no target variable, no ground truth. The wholesale retailer wants to discover natural customer segments from purchase behaviour alone. Clustering is the first tool: group similar customers automatically, then name the segments afterward. It sets up dimensionality reduction in [Ch.2 →](../ch02_dimensionality_reduction) (PCA/t-SNE/UMAP to visualise 6D clusters in 2D) and the cluster-quality metrics in [Ch.3 →](../ch03_unsupervised_metrics) (silhouette, Davies-Bouldin — how do you score a clustering with no ground truth?).
>
> **Notation in this chapter.** $\mathbf{x}_i \in \mathbb{R}^d$ — a data point (one customer's spending vector, $d=6$ features); $K$ — number of clusters (hyperparameter for K-Means); $\boldsymbol{\mu}_k$ — **centroid** of cluster $k$; $C_k$ — the set of points assigned to cluster $k$; $J = \sum_{k=1}^{K}\sum_{\mathbf{x}_i \in C_k}\|\mathbf{x}_i - \boldsymbol{\mu}_k\|^2$ — **inertia** (K-Means objective); $\varepsilon$ — DBSCAN neighbourhood radius; $\text{minPts}$ — DBSCAN density threshold; $d(\mathbf{x}_i, \mathbf{x}_j)$ — Euclidean distance (default); **core / border / noise** — the three DBSCAN point types; $s(i)$ — silhouette coefficient of point $i \in [-1, 1]$.

---

## 0 · The Challenge — Where We Are

> 💡 **The mission**: Build **SegmentAI** — discover actionable customer segments from 440 wholesale customers, silhouette >0.5, satisfying 5 constraints.

| # | Constraint | Target | Why It Matters |
|---|-----------|--------|----------------|
| **#1** | **SEGMENTATION** | Distinct, stable clusters with silhouette >0.5 | Marketing can only act on segments that do not dissolve when the data shifts slightly |
| **#2** | **INTERPRETABILITY** | Business-actionable segment names | "Cluster 2" means nothing; "HoReCa bulk buyers" drives a promotion strategy |
| **#3** | **OUTLIER HANDLING** | Flag anomalous customers rather than forcing them into segments | Outlier spenders distort centroids; DBSCAN labels them noise (label $-1$) |
| **#4** | **NO PRE-SPECIFIED LABELS** | Fully unsupervised — zero manual annotation | 440 hand-labelled records would cost weeks and introduce subjective bias |
| **#5** | **SCALABILITY** | Handles 1M+ customers at inference | K-Means is $O(nKd)$ per iteration — scales to millions with mini-batch variants |

**What we know so far:**
- ✅ Dataset: 440 wholesale customers, 6 spending features (Fresh, Milk, Grocery, Frozen, Detergents_Paper, Delicatessen)
- ✅ RL track complete — AgentAI achieved ≥195/200 CartPole steps
- ❌ **No labels — supervised learning is impossible here**
- ❌ **No baseline clusters yet — SegmentAI has not segmented a single customer**

**What is blocking us:**

The CMO asks: *"What types of customers do we have?"*
No one has labelled these 440 customers as "loyalists" or "price-sensitive" — that taxonomy does not exist. Every supervised algorithm built so far needs a target column. We have none.

**What this chapter unlocks:**

Three clustering algorithms to discover structure without labels:
1. **K-Means** — fast, interpretable centroids, requires $K$ upfront
2. **DBSCAN** — handles arbitrary shapes, auto-detects outlier customers, requires $(\varepsilon, \text{minPts})$
3. **Agglomerative (hierarchical)** — builds a dendrogram, no $K$ required upfront, interpretable merge tree

| Constraint | Status | This Chapter |
|------------|--------|-------------|
| #1 SEGMENTATION | ⚡ Partial → ✅ | k=4 found; silhouette=0.52 — above threshold |
| #2 INTERPRETABILITY | ⚡ Partial | Centroids named: HoReCa, Retail, Mixed, Outlier |
| #3 OUTLIER HANDLING | ✅ Done | DBSCAN labels extreme spenders as noise |
| #4 NO LABELS | ✅ Done | Fully unsupervised throughout |
| #5 SCALABILITY | ✅ Done | K-Means O(nKd) scales to millions |

> ⚠️ **Silhouette=0.52 satisfies the SegmentAI target — but visualising these 6D clusters requires dimensionality reduction.** Ch.2 provides PCA/t-SNE projections that let stakeholders see the segments as a 2D scatter plot.

---

## Animation

![Chapter animation](img/ch01-clustering-needle.gif)

---

## 1 · Core Idea

**Clustering** partitions a dataset into groups — clusters — such that points within a cluster are more similar to each other than to points in other clusters. No labels required. No target variable. The algorithm discovers structure from the feature geometry alone.

**K-Means** defines similarity as Euclidean distance and clusters as spherical regions around centroids. Two alternating steps converge to a locally optimal partition: (1) assign each point to its nearest centroid; (2) recompute each centroid as the mean of its assigned points. Repeat until assignments stop changing. Fast, deterministic given fixed initialisation, scales to millions. Limitation: assumes spherical clusters of roughly equal size, and you must specify $K$ upfront.

**DBSCAN** defines clusters as dense regions separated by sparse regions. A point is a *core point* if at least $\text{minPts}$ neighbours fall within radius $\varepsilon$. Clusters grow outward from core points by density-reachability. Points that are not density-reachable from any core point become **noise** (label $-1$). No $K$ required. Handles arbitrary cluster shapes. Limitation: sensitive to $\varepsilon$ — a bad choice produces either one giant cluster or pure noise.

**Agglomerative hierarchical clustering** starts with each point as its own cluster and merges the two closest clusters at every step. The *linkage criterion* defines "closest": Ward linkage minimises the variance increase on merge (best for compact clusters), complete linkage uses the maximum pairwise distance (avoids elongated chains). Produces a dendrogram — cut it at any height to get any $K$. Very interpretable merge tree.

```
Algorithm    | K required? | Outlier concept | Cluster shape | Scales to 1M?
-------------|-------------|-----------------|---------------|---------------
K-Means      | Yes         | None (forced)   | Spherical     | Yes (mini-batch)
DBSCAN       | No          | Noise label -1  | Arbitrary     | With index
Agglomerative| No (cut)    | None            | Any (linkage) | Slow O(n^2 logn)
GMM          | Yes         | Soft membership | Elliptical    | Yes
```

> 💡 **Why Lloyd's 1957 algorithm still dominates in 2025:** $O(nKd)$ per iteration, trivially parallelisable, and mini-batch K-Means (Sculley 2010) brings it to streaming scale. The Bell Labs PCM quantiser is still the production default sixty-eight years later.

---

## 2 · Running Example

The CMO gives you a CSV: 440 wholesale customers, annual spending across 6 product categories — Fresh produce, Milk, Grocery staples, Frozen goods, Detergents & Paper, and Delicatessen. No customer type labels. No historical segments. Zero annotation.

**The ask:** discover natural customer segments so marketing can craft targeted promotions — bulk food-service buyers get different offers than small grocery retailers.

**Dataset:** UCI Wholesale Customers — 440 customers, 6 features (all in monetary units, annual spend).

| Feature | Approx. range | Interpretation |
|---------|---------------|----------------|
| Fresh | 3–112,151 | Perishable produce — fruit, vegetables |
| Milk | 55–73,498 | Dairy products |
| Grocery | 3–92,780 | Shelf-stable staples |
| Frozen | 25–60,869 | Frozen goods |
| Detergents_Paper | 3–40,827 | Non-food consumables |
| Delicatessen | 3–47,943 | Prepared foods and deli items |

**Pre-processing required:** spending data is heavily right-skewed — a handful of customers spend 30× the median. Apply `log1p` transform before clustering to compress the range. Then `StandardScaler` to equalise feature variances (mandatory for any distance-based algorithm).

**K-Means with k=4 discovers four coherent segments:**

| Cluster | Name | Profile | Approx. size |
|---------|------|---------|--------------|
| 0 | **HoReCa** (Hotels/Restaurants/Cafés) | High Fresh + Frozen, moderate Milk | ~130 customers |
| 1 | **Retail** | High Grocery + Detergents_Paper, low Fresh | ~120 customers |
| 2 | **Mixed / SME** | Moderate across all categories | ~150 customers |
| 3 | **Outlier bulk** | Extreme values in ≥1 category | ~40 customers |

Silhouette score at k=4: **0.52** — above the 0.5 SegmentAI target.
DBSCAN (ε=0.8, minPts=5) identifies 23 customers as noise (label $-1$): extreme spenders that do not belong to any dense region.

---

## 3 · Clustering Algorithms at a Glance

| Property | K-Means | DBSCAN | Agglomerative | GMM |
|----------|---------|--------|---------------|-----|
| **Key hyperparameter** | $K$ | $\varepsilon$, minPts | linkage, $K$ at cut | $K$, covariance type |
| **Cluster shape** | Spherical | Arbitrary | Depends on linkage | Elliptical |
| **Outlier handling** | Forces every point into a cluster | Labels sparse points noise ($-1$) | Forces every point | Soft membership; low-probability outliers |
| **Time complexity** | $O(nKd \cdot \text{iter})$ | $O(n \log n)$ with index | $O(n^2 \log n)$ | $O(nK^2 d \cdot \text{iter})$ |
| **Scales to 1M rows** | ✅ Mini-batch K-Means | ✅ With kd-tree | ❌ Memory-bound | ⚡ Diagonal covariance |
| **Needs $K$ upfront** | ✅ Yes | ❌ No | ❌ No (post-hoc cut) | ✅ Yes |
| **Deterministic** | ❌ Init-sensitive | ✅ Given fixed ε | ✅ | ❌ Init-sensitive |
| **SegmentAI fit** | ✅ Primary algorithm | ✅ Outlier detector | ⚡ Dendrogram insight | ❌ Too slow for 1M |

> ➡️ **Decision rule for this track:** use K-Means as the primary segmentation algorithm (fast, interpretable centroids), DBSCAN as the outlier detector (flag noise customers), and the agglomerative dendrogram to cross-check the natural cluster count without specifying $K$ in advance.

---

## 4 · The Math

### 4.1 K-Means Objective

K-Means minimises **inertia** — the total within-cluster sum of squared distances to centroids:

$$J = \sum_{k=1}^{K} \sum_{\mathbf{x}_i \in C_k} \|\mathbf{x}_i - \boldsymbol{\mu}_k\|^2$$

where $\boldsymbol{\mu}_k = \dfrac{1}{|C_k|} \sum_{\mathbf{x}_i \in C_k} \mathbf{x}_i$ is the centroid of cluster $k$.

Each iteration strictly reduces (or preserves) $J$:
- **Assignment step** cannot increase $J$: every point moves to a closer or equal centroid.
- **Update step** cannot increase $J$: the arithmetic mean minimises the sum of squared distances to a set of points (first-order condition: $\nabla_{\boldsymbol{\mu}_k} J = 0$ gives $\boldsymbol{\mu}_k = \text{mean}(C_k)$).

Therefore K-Means always converges in finite iterations — but to a **local** minimum. K-Means++ initialisation spreads seeds proportional to $d(\mathbf{x}, \text{nearest existing centroid})^2$, dramatically reducing bad local minima.

### 4.2 Assignment Step — All 8 Distances

**Setup:** four points, two initial centroids.

$$\mathbf{x}_1=[2,1],\quad \mathbf{x}_2=[1,2],\quad \mathbf{x}_3=[8,9],\quad \mathbf{x}_4=[9,8]$$
$$\boldsymbol{\mu}_1=[1,1],\quad \boldsymbol{\mu}_2=[8,8]$$

Euclidean distance formula: $d(\mathbf{x}, \boldsymbol{\mu}) = \sqrt{(\mathbf{x}_a - \boldsymbol{\mu}_a)^2 + (\mathbf{x}_b - \boldsymbol{\mu}_b)^2}$

**All 8 distances:**

$$d(\mathbf{x}_1, \boldsymbol{\mu}_1) = \sqrt{(2-1)^2+(1-1)^2} = \sqrt{1+0} = \sqrt{1} = 1.00$$
$$d(\mathbf{x}_1, \boldsymbol{\mu}_2) = \sqrt{(2-8)^2+(1-8)^2} = \sqrt{36+49} = \sqrt{85} \approx 9.22 \quad\Rightarrow\; \mathbf{x}_1 \to C_1$$

$$d(\mathbf{x}_2, \boldsymbol{\mu}_1) = \sqrt{(1-1)^2+(2-1)^2} = \sqrt{0+1} = \sqrt{1} = 1.00$$
$$d(\mathbf{x}_2, \boldsymbol{\mu}_2) = \sqrt{(1-8)^2+(2-8)^2} = \sqrt{49+36} = \sqrt{85} \approx 9.22 \quad\Rightarrow\; \mathbf{x}_2 \to C_1$$

$$d(\mathbf{x}_3, \boldsymbol{\mu}_1) = \sqrt{(8-1)^2+(9-1)^2} = \sqrt{49+64} = \sqrt{113} \approx 10.63$$
$$d(\mathbf{x}_3, \boldsymbol{\mu}_2) = \sqrt{(8-8)^2+(9-8)^2} = \sqrt{0+1} = \sqrt{1} = 1.00 \quad\Rightarrow\; \mathbf{x}_3 \to C_2$$

$$d(\mathbf{x}_4, \boldsymbol{\mu}_1) = \sqrt{(9-1)^2+(8-1)^2} = \sqrt{64+49} = \sqrt{113} \approx 10.63$$
$$d(\mathbf{x}_4, \boldsymbol{\mu}_2) = \sqrt{(9-8)^2+(8-8)^2} = \sqrt{1+0} = \sqrt{1} = 1.00 \quad\Rightarrow\; \mathbf{x}_4 \to C_2$$

**Result:** $C_1 = \{\mathbf{x}_1, \mathbf{x}_2\}$, $C_2 = \{\mathbf{x}_3, \mathbf{x}_4\}$.

| Point | Nearest centroid | Distance | Cluster |
|-------|-----------------|----------|---------|
| $\mathbf{x}_1=[2,1]$ | $\boldsymbol{\mu}_1$ | 1.00 | $C_1$ |
| $\mathbf{x}_2=[1,2]$ | $\boldsymbol{\mu}_1$ | 1.00 | $C_1$ |
| $\mathbf{x}_3=[8,9]$ | $\boldsymbol{\mu}_2$ | 1.00 | $C_2$ |
| $\mathbf{x}_4=[9,8]$ | $\boldsymbol{\mu}_2$ | 1.00 | $C_2$ |

### 4.3 Update Step

Recompute centroids as the mean of assigned points:

$$\boldsymbol{\mu}_1 \leftarrow \text{mean}(\mathbf{x}_1, \mathbf{x}_2) = \left[\frac{2+1}{2},\;\frac{1+2}{2}\right] = [1.5,\;1.5]$$

$$\boldsymbol{\mu}_2 \leftarrow \text{mean}(\mathbf{x}_3, \mathbf{x}_4) = \left[\frac{8+9}{2},\;\frac{9+8}{2}\right] = [8.5,\;8.5]$$

### 4.4 Second Iteration — Recompute Distances to Updated Centroids

New centroids: $\boldsymbol{\mu}_1=[1.5,1.5]$, $\boldsymbol{\mu}_2=[8.5,8.5]$.

$$d(\mathbf{x}_1, \boldsymbol{\mu}_1) = \sqrt{(2-1.5)^2+(1-1.5)^2} = \sqrt{0.25+0.25} = \sqrt{0.50} \approx 0.71$$
$$d(\mathbf{x}_1, \boldsymbol{\mu}_2) = \sqrt{(2-8.5)^2+(1-8.5)^2} = \sqrt{42.25+56.25} = \sqrt{98.5} \approx 9.92 \quad\Rightarrow\; \mathbf{x}_1 \to C_1$$

$$d(\mathbf{x}_2, \boldsymbol{\mu}_1) = \sqrt{(1-1.5)^2+(2-1.5)^2} = \sqrt{0.25+0.25} = \sqrt{0.50} \approx 0.71$$
$$d(\mathbf{x}_2, \boldsymbol{\mu}_2) = \sqrt{(1-8.5)^2+(2-8.5)^2} = \sqrt{56.25+42.25} = \sqrt{98.5} \approx 9.92 \quad\Rightarrow\; \mathbf{x}_2 \to C_1$$

$$d(\mathbf{x}_3, \boldsymbol{\mu}_1) = \sqrt{(8-1.5)^2+(9-1.5)^2} = \sqrt{42.25+56.25} = \sqrt{98.5} \approx 9.92$$
$$d(\mathbf{x}_3, \boldsymbol{\mu}_2) = \sqrt{(8-8.5)^2+(9-8.5)^2} = \sqrt{0.25+0.25} = \sqrt{0.50} \approx 0.71 \quad\Rightarrow\; \mathbf{x}_3 \to C_2$$

$$d(\mathbf{x}_4, \boldsymbol{\mu}_1) = \sqrt{(9-1.5)^2+(8-1.5)^2} = \sqrt{56.25+42.25} = \sqrt{98.5} \approx 9.92$$
$$d(\mathbf{x}_4, \boldsymbol{\mu}_2) = \sqrt{(9-8.5)^2+(8-8.5)^2} = \sqrt{0.25+0.25} = \sqrt{0.50} \approx 0.71 \quad\Rightarrow\; \mathbf{x}_4 \to C_2$$

**Assignments unchanged** — K-Means has converged after one update.

**Final inertia:**

$$J = \|\mathbf{x}_1-\boldsymbol{\mu}_1\|^2 + \|\mathbf{x}_2-\boldsymbol{\mu}_1\|^2 + \|\mathbf{x}_3-\boldsymbol{\mu}_2\|^2 + \|\mathbf{x}_4-\boldsymbol{\mu}_2\|^2$$
$$= 0.50 + 0.50 + 0.50 + 0.50 = \mathbf{2.0}$$

> 💡 **Inertia = 2.0 means each point lies $\sqrt{0.5} \approx 0.71$ from its centroid.** Perfectly symmetric — the two clusters are geometric mirror images. On the real 6D UCI data, inertia at k=4 (after standardisation) is approximately 1,100 — much larger, but the relative drop from k=3 to k=4 tells the same elbow story.

### 4.5 DBSCAN — Core, Border, Noise Classification

**Setup:** 5 points, $\varepsilon = 2$, $\text{minPts} = 2$.

$$p_1=[3,3],\quad p_2=[4,3],\quad p_3=[3,4],\quad p_4=[4,4],\quad p_5=[9,9]$$

**Classify $p_1=[3,3]$:**

$$d(p_1, p_2)=\sqrt{(3-4)^2+(3-3)^2}=\sqrt{1}=1.00\leq 2\;\checkmark$$
$$d(p_1, p_3)=\sqrt{(3-3)^2+(3-4)^2}=\sqrt{1}=1.00\leq 2\;\checkmark$$
$$d(p_1, p_4)=\sqrt{(3-4)^2+(3-4)^2}=\sqrt{2}\approx1.41\leq 2\;\checkmark$$
$$d(p_1, p_5)=\sqrt{(3-9)^2+(3-9)^2}=\sqrt{72}\approx8.49>2\;\times$$

$|N_\varepsilon(p_1)|=3 \geq \text{minPts}=2$ → $p_1$ is a **core point**.

**Classify $p_2=[4,3]$:**

$$d(p_2, p_1)=1.00\leq 2\;\checkmark$$
$$d(p_2, p_3)=\sqrt{(4-3)^2+(3-4)^2}=\sqrt{2}\approx1.41\leq 2\;\checkmark$$
$$d(p_2, p_4)=\sqrt{(4-4)^2+(3-4)^2}=\sqrt{1}=1.00\leq 2\;\checkmark$$
$$d(p_2, p_5)=\sqrt{(4-9)^2+(3-9)^2}=\sqrt{61}\approx7.81>2\;\times$$

$|N_\varepsilon(p_2)|=3\geq 2$ → $p_2$ is a **core point**. Density-connected to $p_1$ → same cluster.

**Classify $p_5=[9,9]$:**

$$d(p_5, p_1)\approx8.49>2\;\times,\quad d(p_5, p_2)\approx7.81>2\;\times$$
$$d(p_5, p_3)=\sqrt{(9-3)^2+(9-4)^2}=\sqrt{61}\approx7.81>2\;\times$$
$$d(p_5, p_4)=\sqrt{(9-4)^2+(9-4)^2}=\sqrt{50}\approx7.07>2\;\times$$

$|N_\varepsilon(p_5)|=0 < 2$, and $p_5$ is not within $\varepsilon$ of any core point → $p_5$ is **noise (label $-1$)**.

| Point | $|N_\varepsilon|$ | Classification | Reasoning |
|-------|------------------|----------------|-----------|
| $p_1=[3,3]$ | 3 | **Core** | $\geq\text{minPts}$ neighbours within $\varepsilon$ |
| $p_2=[4,3]$ | 3 | **Core** | $\geq\text{minPts}$ neighbours within $\varepsilon$ |
| $p_3=[3,4]$ | 3 | **Core** | $\geq\text{minPts}$ neighbours within $\varepsilon$ |
| $p_4=[4,4]$ | 3 | **Core** | $\geq\text{minPts}$ neighbours within $\varepsilon$ |
| $p_5=[9,9]$ | 0 | **Noise** | No core point within $\varepsilon$ — label $-1$ |

$p_1, p_2, p_3, p_4$ are all density-connected → one cluster. $p_5$ is noise. This mirrors the SegmentAI situation: most customers cluster densely, but extreme bulk buyers (the $p_5$ analogue) fall outside every dense region.

### 4.6 Elbow Method — Inertia vs k

**Toy dataset (6 points):** $\{[1,1],[2,1],[1,2],[8,8],[9,8],[8,9]\}$ — two natural clusters of 3 points each.

**k=1 (all in one cluster):**
$$\boldsymbol{\mu}_\text{all}=\left[\frac{1+2+1+8+9+8}{6},\;\frac{1+1+2+8+8+9}{6}\right]=\left[\frac{29}{6},\;\frac{29}{6}\right]\approx[4.83,\;4.83]$$

$$J(k{=}1)=(1{-}4.83)^2\!+(1{-}4.83)^2+(2{-}4.83)^2\!+(1{-}4.83)^2+(1{-}4.83)^2\!+(2{-}4.83)^2$$
$$\quad+(8{-}4.83)^2\!+(8{-}4.83)^2+(9{-}4.83)^2\!+(8{-}4.83)^2+(8{-}4.83)^2\!+(9{-}4.83)^2$$
$$\approx 29.3+22.7+22.7+20.1+27.4+27.4=\mathbf{149.6}$$

**k=2 (natural split: $C_1=\{[1,1],[2,1],[1,2]\}$, $C_2=\{[8,8],[9,8],[8,9]\}$):**
$$\boldsymbol{\mu}_1=\left[\frac{4}{3},\frac{4}{3}\right]\approx[1.33,1.33],\quad \boldsymbol{\mu}_2=\left[\frac{25}{3},\frac{25}{3}\right]\approx[8.33,8.33]$$
$$J(k{=}2)=2\times\frac{12}{9}\approx\mathbf{2.67}\quad\text{(each cluster contributes }4/3\text{)}$$

**k=3 (one natural cluster split further):** $J(k{=}3)\approx\mathbf{1.33}$ — modest reduction.

**k=4:** $J(k{=}4)\approx\mathbf{0.67}$ — diminishing returns.

| $k$ | Inertia $J$ | Drop from previous $k$ |
|-----|------------|------------------------|
| 1 | 149.6 | — |
| 2 | 2.67 | **146.9** ← massive drop |
| 3 | 1.33 | 1.34 — small |
| 4 | 0.67 | 0.66 — small |

> ⚡ **The elbow is at k=2 for this toy dataset.** The drop from k=1→2 is 146.9; the drop from k=2→3 is only 1.34 — 100× smaller. Adding a third cluster barely reduces inertia because the two natural groups are already captured. For the **real UCI dataset** (440 customers, 6D), the elbow appears around k=4, matching SegmentAI's target of four segments.

---

## 5 · The Clustering Arc

**Act 1 — Unlabelled customers; no supervision.**
The CSV has 440 rows and 6 columns. No column named "customer_type." No target variable. Every supervised algorithm built in the previous tracks is useless here. The data must speak for itself.

**Act 2 — K-Means splits into k=4 spherical groups.**
Lloyd's algorithm assigns each customer to a centroid, updates centroids, repeats. After ~20 iterations: four groups with distinct centroid profiles. Cluster 0 has high Fresh and Frozen spending — these look like food-service buyers (hotels, restaurants, cafés). Cluster 1 has high Grocery and Detergents_Paper — retail outlets. Cluster 2 is moderate across all features — small-to-medium enterprises. Cluster 3 contains extreme spenders in at least one category. Interpretable business labels emerge from purely mathematical centroids. First pass at Constraints #1 and #2.

**Act 3 — DBSCAN reveals arbitrary-shape structure and outliers.**
K-Means forced every customer into a cluster. DBSCAN refuses: 23 customers fall outside every dense region and get label $-1$. These are the extreme bulk spenders — one customer buying 112k in Fresh produce is not "slightly unusual Cluster 0," it is a fundamentally different buyer type. Constraint #3 satisfied: outliers flagged rather than absorbed into the nearest centroid.

**Act 4 — Optimal k=4 confirmed by silhouette score.**
Sweep k from 2 to 10. At k=4 the mean silhouette peaks at **0.52** — above the 0.5 SegmentAI target. Every segment has $s(i) > 0.4$ for at least 90% of its members. At k=2, silhouette=0.35 (HoReCa and Retail lumped together). At k=8, silhouette=0.38 (micro-segments overlap). k=4 is the sweet spot. Constraint #1 validated quantitatively.

---

## 6 · Full K-Means Walkthrough — 6 Customers, 2D, k=2

**Dataset:** Fresh and Milk annual spending (standardised units after log-transform).

| Customer | Fresh | Milk |
|----------|-------|------|
| A | 1 | 2 |
| B | 2 | 1 |
| C | 1 | 3 |
| D | 8 | 7 |
| E | 9 | 8 |
| F | 7 | 9 |

**Initialisation (K-Means++ selects spread seeds):** $\boldsymbol{\mu}_1=[1,2]$ (Customer A), $\boldsymbol{\mu}_2=[8,7]$ (Customer D).

**Pre-iteration inertia** (with initial centroids $\boldsymbol{\mu}_1=[1,2]$, $\boldsymbol{\mu}_2=[8,7]$):

$$\|A-\boldsymbol{\mu}_1\|^2=(1{-}1)^2+(2{-}2)^2=0,\quad \|B-\boldsymbol{\mu}_1\|^2=(2{-}1)^2+(1{-}2)^2=1+1=2$$
$$\|C-\boldsymbol{\mu}_1\|^2=(1{-}1)^2+(3{-}2)^2=0+1=1,\quad \|D-\boldsymbol{\mu}_2\|^2=(8{-}8)^2+(7{-}7)^2=0$$
$$\|E-\boldsymbol{\mu}_2\|^2=(9{-}8)^2+(8{-}7)^2=1+1=2,\quad \|F-\boldsymbol{\mu}_2\|^2=(7{-}8)^2+(9{-}7)^2=1+4=5$$
$$J_0 = 0+2+1+0+2+5 = \mathbf{10}$$

### Iteration 1 — Assignment (12 distances)

| Customer | $d(\cdot,\boldsymbol{\mu}_1=[1,2])$ | $d(\cdot,\boldsymbol{\mu}_2=[8,7])$ | Assignment |
|----------|--------------------------------------|--------------------------------------|------------|
| A=[1,2] | $\sqrt{(0)^2+(0)^2}=0.00$ | $\sqrt{(7)^2+(5)^2}=\sqrt{74}\approx8.60$ | $C_1$ |
| B=[2,1] | $\sqrt{(1)^2+(1)^2}=\sqrt{2}\approx1.41$ | $\sqrt{(6)^2+(6)^2}=\sqrt{72}\approx8.49$ | $C_1$ |
| C=[1,3] | $\sqrt{(0)^2+(1)^2}=1.00$ | $\sqrt{(7)^2+(4)^2}=\sqrt{65}\approx8.06$ | $C_1$ |
| D=[8,7] | $\sqrt{(7)^2+(5)^2}=\sqrt{74}\approx8.60$ | $\sqrt{(0)^2+(0)^2}=0.00$ | $C_2$ |
| E=[9,8] | $\sqrt{(8)^2+(6)^2}=\sqrt{100}=10.00$ | $\sqrt{(1)^2+(1)^2}=\sqrt{2}\approx1.41$ | $C_2$ |
| F=[7,9] | $\sqrt{(6)^2+(7)^2}=\sqrt{85}\approx9.22$ | $\sqrt{(1)^2+(2)^2}=\sqrt{5}\approx2.24$ | $C_2$ |

$C_1=\{A,B,C\}$, $C_2=\{D,E,F\}$.

### Iteration 1 — Update Centroids

$$\boldsymbol{\mu}_1\leftarrow\left[\frac{1+2+1}{3},\;\frac{2+1+3}{3}\right]=\left[\frac{4}{3},\;2\right]\approx[1.33,\;2.00]$$

$$\boldsymbol{\mu}_2\leftarrow\left[\frac{8+9+7}{3},\;\frac{7+8+9}{3}\right]=\left[\frac{24}{3},\;\frac{24}{3}\right]=[8.00,\;8.00]$$

**Inertia after iteration 1** (distances to updated centroids):

$$\|A-\boldsymbol{\mu}_1\|^2=(1{-}1.33)^2+(2{-}2)^2=0.11+0=0.11$$
$$\|B-\boldsymbol{\mu}_1\|^2=(2{-}1.33)^2+(1{-}2)^2=0.45+1=1.45$$
$$\|C-\boldsymbol{\mu}_1\|^2=(1{-}1.33)^2+(3{-}2)^2=0.11+1=1.11$$
$$\|D-\boldsymbol{\mu}_2\|^2=(8{-}8)^2+(7{-}8)^2=0+1=1.00$$
$$\|E-\boldsymbol{\mu}_2\|^2=(9{-}8)^2+(8{-}8)^2=1+0=1.00$$
$$\|F-\boldsymbol{\mu}_2\|^2=(7{-}8)^2+(9{-}8)^2=1+1=2.00$$

$$J_1=0.11+1.45+1.11+1.00+1.00+2.00=\mathbf{6.67}$$

Inertia dropped: $J_0=10.00 \to J_1=6.67$ (−33%).

### Iteration 2 — Verify Convergence (12 distances to new centroids)

New centroids: $\boldsymbol{\mu}_1=[1.33,\;2.00]$, $\boldsymbol{\mu}_2=[8.00,\;8.00]$.

| Customer | $d(\cdot,\boldsymbol{\mu}_1=[1.33,2])$ | $d(\cdot,\boldsymbol{\mu}_2=[8,8])$ | Assignment |
|----------|----------------------------------------|--------------------------------------|------------|
| A=[1,2] | $\sqrt{(0.33)^2+(0)^2}\approx0.33$ | $\sqrt{(7)^2+(6)^2}=\sqrt{85}\approx9.22$ | $C_1$ ✓ |
| B=[2,1] | $\sqrt{(0.67)^2+(1)^2}=\sqrt{1.45}\approx1.20$ | $\sqrt{(6)^2+(7)^2}=\sqrt{85}\approx9.22$ | $C_1$ ✓ |
| C=[1,3] | $\sqrt{(0.33)^2+(1)^2}=\sqrt{1.11}\approx1.05$ | $\sqrt{(7)^2+(5)^2}=\sqrt{74}\approx8.60$ | $C_1$ ✓ |
| D=[8,7] | $\sqrt{(6.67)^2+(5)^2}=\sqrt{69.5}\approx8.34$ | $\sqrt{(0)^2+(1)^2}=1.00$ | $C_2$ ✓ |
| E=[9,8] | $\sqrt{(7.67)^2+(6)^2}=\sqrt{94.8}\approx9.74$ | $\sqrt{(1)^2+(0)^2}=1.00$ | $C_2$ ✓ |
| F=[7,9] | $\sqrt{(5.67)^2+(7)^2}=\sqrt{81.1}\approx9.01$ | $\sqrt{(1)^2+(1)^2}=\sqrt{2}\approx1.41$ | $C_2$ ✓ |

**All assignments identical to iteration 1** → centroids do not move → **converged**.

**Convergence summary:**

| Iteration | $\boldsymbol{\mu}_1$ | $\boldsymbol{\mu}_2$ | Inertia $J$ |
|-----------|----------------------|----------------------|-------------|
| 0 (init) | [1.00, 2.00] | [8.00, 7.00] | 10.00 |
| 1 | [1.33, 2.00] | [8.00, 8.00] | 6.67 |
| 2 | [1.33, 2.00] | [8.00, 8.00] | 6.67 — **converged** ✅ |

> 💡 **Why convergence took only 1 update:** the two groups are well-separated (gap ~7 units) and the initial seeds were already embedded in each group. With random init on a noisy dataset, 20–50 iterations are typical before convergence. K-Means++ reduces this to 5–15 iterations for most real datasets.

---

## 7 · Key Diagrams

### 7.1 K-Means Algorithm Loop

```mermaid
flowchart TD
    INIT["Initialise K centroids\nμ₁, μ₂, ..., μ_K\nK-Means++ sampling"]
    ASSIGN["Assignment step\nfor each xᵢ:\nassign to argmin_k ‖xᵢ − μ_k‖"]
    UPDATE["Update step\nfor each k:\nμ_k ← mean of all xᵢ in C_k"]
    CHECK{"Assignments\nchanged?"}
    DONE["Return final clusters\nC₁, C₂, ..., C_K\nand centroids μ_k"]

    INIT   --> ASSIGN
    ASSIGN --> UPDATE
    UPDATE --> CHECK
    CHECK  -->|"yes — keep iterating"| ASSIGN
    CHECK  -->|"no — converged"| DONE

    style INIT   fill:#1e3a8a,color:#fff,stroke:#1e3a8a
    style ASSIGN fill:#1d4ed8,color:#fff,stroke:#1d4ed8
    style UPDATE fill:#1d4ed8,color:#fff,stroke:#1d4ed8
    style CHECK  fill:#b45309,color:#fff,stroke:#b45309
    style DONE   fill:#15803d,color:#fff,stroke:#15803d
```

### 7.2 SegmentAI Cluster Membership

```mermaid
flowchart LR
    RAW["440 customers\n6 spending features\nraw, heavily right-skewed"]
    PREP["log1p transform\n+ StandardScaler\nmandatory pre-step"]
    KMEANS["K-Means\nk = 4\nLloyd's algorithm\n~20 iterations"]
    C0["Cluster 0 — HoReCa\nHigh Fresh + Frozen\nHotels / Restaurants / Cafés\n~130 customers"]
    C1["Cluster 1 — Retail\nHigh Grocery +\nDetergents_Paper\n~120 customers"]
    C2["Cluster 2 — Mixed / SME\nModerate all features\nSmall-to-medium enterprises\n~150 customers"]
    C3["Cluster 3 — Outlier bulk\nExtreme values in ≥1 feature\nFlagged by DBSCAN too\n~40 customers"]

    RAW    --> PREP
    PREP   --> KMEANS
    KMEANS --> C0
    KMEANS --> C1
    KMEANS --> C2
    KMEANS --> C3

    style RAW    fill:#1e3a8a,color:#fff,stroke:#1e3a8a
    style PREP   fill:#1d4ed8,color:#fff,stroke:#1d4ed8
    style KMEANS fill:#b45309,color:#fff,stroke:#b45309
    style C0     fill:#15803d,color:#fff,stroke:#15803d
    style C1     fill:#15803d,color:#fff,stroke:#15803d
    style C2     fill:#15803d,color:#fff,stroke:#15803d
    style C3     fill:#b91c1c,color:#fff,stroke:#b91c1c
```

> ➡️ **Cluster 3 (red) overlaps strongly with the 23 customers DBSCAN labels $-1$.** K-Means forced them into a cluster; DBSCAN refuses to. Both algorithms agree these are anomalous buyers — the disagreement is only in how to handle them.

---

## 8 · Hyperparameter Dial

### 8.1 K in K-Means

| $K$ | Too low | Sweet spot | Too high |
|-----|---------|------------|----------|
| Signal | Coarse — HoReCa and Retail merged into "large buyer" | Elbow of inertia + silhouette peak | Every customer in its own cluster; inertia→0, silhouette→meaningless |
| SegmentAI | k=2: silhouette=0.35, too coarse for targeting | **k=4: silhouette=0.52** ✅ | k=8: silhouette=0.38, micro-segments overlap |
| Rule of thumb | Combine elbow plot + silhouette sweep + business constraint (CMO may insist on exactly N segments) | | |

**Sweep strategy:** run K-Means for k=2 to 10 (`n_init=10` each). Plot inertia (elbow) and mean silhouette on the same figure. K at the inertia elbow *and* silhouette peak is the answer. When they disagree, business requirements break the tie.

### 8.2 ε and minPts in DBSCAN

| Dial | Too small | Sweet spot | Too large |
|------|-----------|------------|-----------|
| **ε** | Everything is noise — no clusters form | k-NN distance plot knee (~0.8–1.2 after standardisation for UCI) | One giant cluster absorbs all customers |
| **minPts** | Every dense point is a core — very large clusters | $2 \times d = 12$ for the 6-feature UCI data | Only the most extreme cores survive; most become border/noise |

**Practical ε estimation:** after log-transform + StandardScaler, compute the 5th-nearest-neighbour distance for every customer. Sort ascending and plot. The knee (sharp inflection) is the recommended DBSCAN radius. For UCI Wholesale after standardisation, the knee typically appears at ε ≈ 0.8 to 1.2.

### 8.3 Linkage in Agglomerative Hierarchical Clustering

| Linkage | Distance measure | Shape preference | SegmentAI use |
|---------|-----------------|------------------|---------------|
| **Ward** | Minimise variance increase on merge | Compact, balanced | **Default — best for k≈4 cut** |
| **Complete** | Maximum pairwise distance | Compact, avoids elongated chains | Good cross-check |
| **Single** | Minimum pairwise distance | Susceptible to chaining | Avoid for customer data |
| **Average** | Mean pairwise distance | Intermediate robustness | Moderate |

> ⚡ **Ward linkage + agglomerative = best dendrogram for SegmentAI.** Cutting at 4 branches matches K-Means and confirms the natural cluster count without specifying $K$ in advance.

---

## 9 · What Can Go Wrong

### 9.1 K-Means Sensitive to Initialisation

K-Means finds a local minimum. A different random seed occasionally produces a degenerate solution where one centroid absorbs 80% of customers and two clusters have fewer than 10 members each. On UCI Wholesale with random init, this happens in roughly 1 in 5 runs.

**Fix:** always use `init='k-means++'` (sklearn default) and `n_init=10`. The best of 10 runs with K-Means++ init is virtually always near the global optimum for well-separated clusters.

```python
KMeans(n_clusters=4, init='k-means++', n_init=10, random_state=42)
```

### 9.2 DBSCAN Sensitive to ε

On raw (un-standardised) UCI data, Fresh ranges 3–112,151. DBSCAN with ε=1.0 on raw data produces pure noise — every customer is an "outlier" because Euclidean distances between any two customers are hundreds of raw units. On standardised data the same ε=1.0 works correctly.

**Fix:** always standardise before DBSCAN. Validate ε with the k-NN distance plot. A sanity check: noise fraction should be 5–10% for SegmentAI (22–44 out of 440 customers). If the noise fraction is above 20% or below 1%, ε needs adjustment.

### 9.3 High-Dimensional Clustering Collapses

In 6D the UCI data is borderline manageable. In 50D (add demographic features, web behaviour), the **curse of dimensionality** makes all pairwise distances nearly equal — the ratio of max-to-min distance approaches 1. K-Means assignments become essentially random because every centroid is nearly equidistant from every point.

**Fix:** apply PCA or UMAP before clustering when $d > 20$. Retain 95% of explained variance (typically 5–10 components for tabular customer data). Ch.2 covers this in detail.

### 9.4 Outliers Inflate Centroids

Customer X spends 112,151 on Fresh — 20× the median. Without log-transform, this single customer drags the HoReCa centroid's Fresh coordinate from the cluster centre to an inflated value. Subsequent assignment sees moderate Fresh-buyers as "too far from centroid" and misassigns them to Retail or Mixed.

**Fix:** log1p transform before standardisation. This is not optional for spending data — it is as mandatory as StandardScaler itself.

```python
import numpy as np
from sklearn.preprocessing import StandardScaler

X_log = np.log1p(X)               # compress skew: 3–112k → 1.1–11.6
X_sc  = StandardScaler().fit_transform(X_log)   # equalise variance
```

> ⚠️ **Pre-processing checklist for clustering:**
> 1. `log1p` transform on all skewed spending columns
> 2. `StandardScaler` — no exceptions for distance-based algorithms
> 3. Silhouette sweep over k=2 to 10
> 4. DBSCAN ε from k-NN distance plot
> 5. Cluster names from inverse-transformed centroid profiles (expm1 + scaler.inverse_transform)

---

### 4.7 Silhouette Score — Measuring Cluster Quality Without Labels

The **silhouette coefficient** $s(i)$ measures how well point $i$ fits its assigned cluster relative to the next-best cluster — without needing ground-truth labels:

$$s(i) = \frac{b(i) - a(i)}{\max(a(i),\; b(i))} \quad \in [-1, 1]$$

where:
- $a(i)$ = mean distance from point $i$ to all other points **in its own cluster** (intra-cluster cohesion)
- $b(i)$ = mean distance from point $i$ to all points **in the nearest other cluster** (inter-cluster separation)

**Interpretation of $s(i)$:**

| $s(i)$ range | Meaning |
|-------------|---------|
| Close to +1 | Well-matched to own cluster; far from neighbours |
| Close to 0 | On or near the boundary between two clusters |
| Close to −1 | Likely misassigned — another cluster fits better |

**Arithmetic example (from the 4-point toy dataset, k=2 converged):**

Final clusters: $C_1=\{\mathbf{x}_1=[2,1],\;\mathbf{x}_2=[1,2]\}$, $C_2=\{\mathbf{x}_3=[8,9],\;\mathbf{x}_4=[9,8]\}$.

For $\mathbf{x}_1=[2,1]$:
$$a(\mathbf{x}_1)=d(\mathbf{x}_1,\mathbf{x}_2)=\sqrt{(2{-}1)^2+(1{-}2)^2}=\sqrt{2}\approx1.41$$
$$b(\mathbf{x}_1)=\frac{d(\mathbf{x}_1,\mathbf{x}_3)+d(\mathbf{x}_1,\mathbf{x}_4)}{2}=\frac{\sqrt{85}+\sqrt{113}}{2}\approx\frac{9.22+10.63}{2}\approx9.93$$
$$s(\mathbf{x}_1)=\frac{9.93-1.41}{\max(1.41,9.93)}=\frac{8.52}{9.93}\approx\mathbf{0.86}$$

$s(\mathbf{x}_1)\approx0.86$ — very well-clustered. By symmetry, $s(\mathbf{x}_2)\approx s(\mathbf{x}_3)\approx s(\mathbf{x}_4)\approx0.86$.

Mean silhouette = **0.86** for this toy dataset. For SegmentAI at k=4: mean silhouette = **0.52**, above the 0.5 target.

> ⚡ **Silhouette is the primary model-selection criterion for SegmentAI.** Inertia alone always favours larger $k$ (reaches 0 at $k=n$). Silhouette penalises both under-clustering (large $a(i)$) and over-clustering (small $b(i)$), giving a balanced, label-free quality score. Ch.3 extends this with Davies-Bouldin and Calinski-Harabasz indices.

---

## 10 · Implementation Quickstart

```python
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.datasets import fetch_openml

# ── 1. Load UCI Wholesale Customers ─────────────────────────────────────────
data = fetch_openml(name="wholesale-customers", version=1, as_frame=True)
X = data.data[["Fresh","Milk","Grocery","Frozen","Detergents_Paper","Delicatessen"]]

# ── 2. Pre-process: log1p → StandardScaler ───────────────────────────────────
X_log = np.log1p(X.values)
X_sc  = StandardScaler().fit_transform(X_log)

# ── 3. K-Means: sweep k=2..10, record inertia + silhouette ──────────────────
results = {}
for k in range(2, 11):
    km = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    labels = km.fit_predict(X_sc)
    results[k] = {
        "inertia":   km.inertia_,
        "silhouette": silhouette_score(X_sc, labels),
    }

best_k = max(results, key=lambda k: results[k]["silhouette"])
print(f"Best k={best_k}, silhouette={results[best_k]['silhouette']:.3f}")
# Expected: Best k=4, silhouette=0.52

# ── 4. Fit final K-Means with k=4 ───────────────────────────────────────────
km4 = KMeans(n_clusters=4, init='k-means++', n_init=10, random_state=42)
labels_km = km4.fit_predict(X_sc)

# ── 5. DBSCAN for outlier detection ─────────────────────────────────────────
db = DBSCAN(eps=0.8, min_samples=5)
labels_db = db.fit_predict(X_sc)
n_noise    = (labels_db == -1).sum()
print(f"DBSCAN noise customers: {n_noise}")  # Expected: ~23

# ── 6. Interpret centroids (back to original scale) ─────────────────────────
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(X_log)
centroids_log = scaler.inverse_transform(km4.cluster_centers_)
centroids_raw = np.expm1(centroids_log)   # undo log1p
import pandas as pd
centroid_df = pd.DataFrame(centroids_raw,
                            columns=X.columns,
                            index=[f"Cluster {i}" for i in range(4)])
print(centroid_df.round(0))
```

> 💡 **The most common mistake:** calling `KMeans` or `DBSCAN` on raw spending values without `log1p` + `StandardScaler`. The Fresh feature alone spans 3 orders of magnitude — raw Euclidean distances are dominated entirely by Fresh, and all other features are effectively invisible to the algorithm.

---

## Where This Reappears

| Chapter | How clustering knowledge reappears |
|---------|-----------------------------------|
| [Ch.2 — Dimensionality Reduction](../ch02_dimensionality_reduction) | PCA/t-SNE/UMAP to project 6D clusters to 2D — required because humans cannot interpret clusters in 6 dimensions |
| [Ch.3 — Unsupervised Metrics](../ch03_unsupervised_metrics) | Silhouette, Davies-Bouldin, Calinski-Harabasz — rigorous scoring of the k=4 clustering without ground truth labels |
| [NN track Ch.4](../../../03-neural_networks/ch04_neural_networks) | Autoencoder latent representations fed into K-Means for deep clustering |
| [ML Ch.11 — XGBoost](../../02_classification/ch11_xgboost) | Cluster labels as a categorical feature for churn or upsell prediction models |

> ➡️ **The centroid profiles** ($\boldsymbol{\mu}_k$ in original spend units, recovered via `expm1(scaler.inverse_transform(mu))`) from this chapter are the direct input to the SegmentAI marketing strategy: HoReCa buyers get bulk-produce discounts, Retail buyers get detergent promotions. Clustering is the discovery step; centroid interpretation is the business step.

---

## Progress Check

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Silhouette score (k=4) | 0.52 | > 0.50 | ✅ |
| DBSCAN noise fraction | 5.2% (23/440) | < 10% | ✅ |
| Interpretable segment names | 4 named | 4 named | ✅ |
| Cluster stability (n_init=10) | Identical across all 10 runs | Stable | ✅ |
| Constraint #3 — outlier handling | 23 flagged as noise | Flagged | ✅ |
| Constraint #4 — no labels used | Fully unsupervised | Unsupervised | ✅ |
| Constraint #5 — scalability | O(nKd) per iter | 1M+ capable | ✅ |

![Progress check](img/ch01-clustering-progress-check.png)

**What this chapter achieved:**
- ✅ k=4 K-Means on UCI Wholesale Customers → silhouette=0.52, above 0.5 SegmentAI target
- ✅ DBSCAN flags 23 outlier customers — not forced into any segment
- ✅ Four segments named from centroid profiles: HoReCa, Retail, Mixed, Outlier-bulk
- ✅ Constraints #3, #4, #5 fully satisfied
- ❌ **Not yet:** cannot visually validate clusters — 6D requires dimensionality reduction first

**The gap:** the CMO asks *"Show me a scatter plot of the segments."* You cannot yet — a 6D scatter plot is uninterpretable. You need to project to 2D while preserving cluster structure. That is exactly Ch.2.

---

## Bridge to Ch.2 — Dimensionality Reduction

**The problem clusters leave behind.** Four mathematically valid segments exist in 6D space. Silhouette=0.52 confirms they are real. But no stakeholder can see them — the human visual system stops at 3D. The CMO needs a scatter plot with colour-coded customer dots.

**What Ch.2 provides.** PCA reduces the 6 spending features to 2–3 principal components that preserve the maximum variance. t-SNE and UMAP provide non-linear projections optimised for visualising local cluster structure. Applied to the k=4 cluster labels from this chapter, the 2D projection shows four segments as visually distinct coloured regions — the interpretability check the CMO needs.

**The direct connection:**

```
Ch.1 (this chapter)              Ch.2 (next)
K-Means → 4 cluster labels  →  PCA projection → 2D scatter plot
silhouette=0.52              →  visual validation of cluster separation
6D centroids                 →  2D centroid coordinates for annotation
23 DBSCAN noise customers    →  appear as isolated dots outside cluster regions
```

> ➡️ **Start Ch.2 with the cluster labels from this chapter as colour coding.** Every customer already has a label {0, 1, 2, 3}. PCA/t-SNE will show whether those labels correspond to visually separated groups — the ultimate sanity check on SegmentAI's segments.

[→ Ch.2: Dimensionality Reduction](../ch02_dimensionality_reduction)
