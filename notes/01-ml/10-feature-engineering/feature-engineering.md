# Feature Engineering

> **The story.** In October 2006, Netflix offered $1 million to whoever could improve their in-house recommendation algorithm — Cinematch — by 10% RMSE. For three years, the brightest teams in machine learning threw increasingly sophisticated models at the problem: matrix factorization, Bayesian ensembles, neural collaborative filtering. By mid-2009 the public leaderboard was stuck at 9.44% improvement. The breakthrough didn't come from a better algorithm. It came from a paper by **Yehuda Koren** at AT&T Research: *"Collaborative Filtering with Temporal Dynamics"* (KDD 2009). Koren showed that a user's rating for "The Godfather" in 2004 is a fundamentally different signal than the same 4-star rating in 2007, after that user has watched 500 more films and recalibrated their personal scale upward by 0.3 stars. He added one engineered feature — a time-decay function capturing how a user's mean rating drifts over months — and the leaderboard unlocked. The eventual winner, **BellKor's Pragmatic Chaos**, which absorbed Koren's team, crossed 10.09% improvement in July 2009 using an ensemble of 107 models. But the lift came from features: temporal drift variables, bias terms, implicit feedback encodings, and normalizations engineers designed by hand. The committee's private assessment was blunt: no single algorithmic innovation had made the difference. Feature representation had.
>
> **Where you are in the curriculum.** Chapters 1–9 of this track covered algorithms: regression, classification, neural networks, recommender systems, anomaly detection, reinforcement learning, unsupervised learning, ensemble methods, and generative models. Each chapter assumed your feature matrix $X$ was a reasonable representation of the problem. It usually isn't — not by default. Raw data from a production system (store IDs, timestamps, product descriptions, categorical labels) cannot be handed directly to most algorithms. This chapter fills that gap: the craft of transforming raw data into $X$ matrices that algorithms can work with, in a form that captures signal without leaking the future, overfitting categories, or drowning in redundant dimensions.
>
> **Notation.** $x_{t}$ — feature value at time step $t$; $x_{t-k}$ — lag-$k$ feature (value $k$ periods in the past); $\mu_c$ — mean of the target within category $c$; $\mu_{global}$ — global target mean; $n_c$ — count of observations in category $c$; $m$ — Laplace smoothing factor; $\hat{x}_c$ — smoothed target-encoded value for category $c$; $\text{tf}(t,d)$ — term frequency of term $t$ in document $d$; $df(t)$ — number of documents containing term $t$; $N$ — total documents; $I(X;Y)$ — mutual information between feature $X$ and target $Y$; $H(Y)$ — entropy of target.

---

## Common Misconceptions

Before the running example, four beliefs worth demolishing. Each is a plausible-sounding half-truth that causes real production failures.

### 1. "Deep learning does feature engineering automatically."

**Why it's seductive:** Neural networks famously learn hierarchical representations from raw pixels, audio waveforms, and text tokens. No handcrafted SIFT features, no mel spectrograms, no n-gram extraction required. It's easy to generalise: if deep learning eliminates feature engineering for images, it must do it for sales data too.

**The truth:** It does — for *dense*, *structured* inputs where similar inputs look geometrically similar (pixels near each other, tokens with shared vocabulary, waveforms with consistent frequency patterns). For tabular data the structure is *sparse* and *domain-specific*: the relationship between `is_holiday=True` on the Thursday before Easter and a 4× spike in frozen turkey sales is not a pattern any architecture will discover from a raw Unix timestamp unless you encode the structure explicitly. Lag-7 sales, week-of-year, and store-tier encoding are not features a transformer extracts from a raw `store_id` integer. Domain knowledge is irreplaceable for tabular data with sparse, non-contiguous interaction effects.

*"BERT learned all of English from raw text. It learned nothing about your SKU rotation calendar."*

### 2. "One-hot encoding is always safe."

**Why it's seductive:** One-hot encoding makes no assumptions about category order, avoids the spurious magnitude problem of ordinal encoding, and is universally supported by every ML library. It's the default in most tutorials and preprocessing pipelines.

**The truth:** One-hot encoding is safe for low-cardinality features — store type (budget/standard/premium) → 3 binary columns. It is actively harmful for high-cardinality features: 1,800 SKUs → 1,800 binary columns, 99.9% zeros per row, introducing a dimensionality explosion on what started as a single column. A random forest trained on 1,800 one-hot SKU columns wastes most of its feature-selection budget splitting on this one variable, fragmenting importance scores across 1,800 indistinguishable features. Target encoding solves this in one column — with the right leakage guard.

*"One-hot 1,800 SKUs and you've multiplied your feature space by 6. Your dataset is now 99.9% zeros."*

### 3. "Target encoding always leaks."

**Why it's seductive:** Target encoding replaces a category with the mean of the target — which means the encoded training feature is computed using training labels. That sounds like leakage, and done naively, it is: if you fit the encoder on the full training set and then cross-validate, the encoded values have seen the folds they're being evaluated on.

**The truth:** The leakage is fixable with one technique — cross-fold target encoding. You encode fold $k$ using the mean computed from all other folds. The encoded values never saw the labels of the examples they're encoding. Scikit-learn's `TargetEncoder` does this automatically with `cv=5`. Laplace smoothing handles rare categories. Once you know the fix, target encoding is not just safe — it's the best single-column encoding for high-cardinality categoricals in tree models.

*"Target encoding doesn't leak. Lazy target encoding leaks."*

### 4. "More features before selection is always better."

**Why it's seductive:** Feature selection discards information. More features means more signal. XGBoost can handle high-dimensional inputs and ignore irrelevant features via its internal importance scoring. Why pre-filter when the model will do it?

**The truth:** This reasoning fails in three ways. First, correlated features fool importance scoring: if lag-7 and lag-8 correlate at 0.97, importance is split between them — neither appears as important as it truly is. Second, noise features actively harm generalisation in linear models and neural networks through variance inflation. Third, preprocessing is cheap and fast — training with 300 features versus 60 is a 5× speed difference that compounds across every hyperparameter sweep. Start narrow, validate wide.

---

## 0 · The Challenge

You're the lead data scientist at a grocery chain predicting weekly unit sales across 1,800 SKUs and 50 stores. The raw data warehouse exports look like this:

| store_id | sku_id | category | date | sales_units | promotion_flag | store_type |
|----------|---------|----------|------------|-------------|----------------|------------|
| S042 | SKU1803 | Frozen | 2024-01-07 | 412 | 0 | premium |
| S042 | SKU1803 | Frozen | 2024-01-14 | 1,847 | 1 | premium |
| S017 | SKU0221 | Bakery | 2024-01-07 | 89 | 0 | budget |

Seven columns. No model learns from `store_id = "S042"` as a raw string. No model learns from `date = "2024-01-07"` as a timestamp. No model knows that the 1,847-unit spike is explained by last week's promotion, not by some mystical property of `SKU1803`.

You run naive engineering: one-hot encode every categorical, extract year/month/day from the date. You end up with 300+ features. You train XGBoost. Validation MAPE is 34% — well above the 18% target the VP of Supply Chain needs for procurement planning. You have three problems:

1. **Category overload**: 1,800 SKU one-hot columns and 50 store one-hot columns, each 99%+ sparse.
2. **Missing temporal signal**: Lag sales, rolling averages, and promotion carry-forward effects are absent.
3. **Noise amplification**: Of 300+ features, roughly 200 carry zero predictive signal and actively harm generalisation.

This chapter is the fix.

---

## 1 · What Feature Engineering Actually Is

Feature engineering is the craft of extracting signal from raw data into a form a model can use. Not "clean the data." Not "run feature importance afterward." The craft — the domain-knowledge-intensive, often non-obvious work of deciding which transformations of raw inputs are worth giving a model.

Three categories of work, each distinct:

**Computed features** — derived through arithmetic, aggregation, or domain logic. Lag-7 sales, rolling 14-day mean, days since last promotion, sales-to-store-average ratio. These require domain knowledge to design: a data engineer without retail experience will not know that a 28-day lag matters for monthly replenishment cycles, or that the post-promotion demand hangover lasts 2-3 weeks and should be encoded as `days_since_last_promo`.

**Encoded features** — transformations of categorical or text values into numbers. Store-type ordinal encoding (budget=0, standard=1, premium=2), SKU target encoding, product-description TF-IDF. Encoding choices change model performance dramatically — not by adding information, but by putting existing information in a form the model can exploit.

**Selected features** — the process of reducing 300 candidate features to the 60 most predictive. Not a model quality dial but a computational necessity: each added feature increases training time, regularisation demands, and the risk of spurious correlations in held-out splits.

*"The model can only learn from what you give it. Feature engineering is the act of deciding what 'giving it' means."*

The failure mode of skipping feature engineering is not obviously bad performance — it's misleadingly unstable performance. The model achieves 28% MAPE on the training distribution and 51% on January data because January has a holiday structure the raw timestamp never encoded. The model didn't fail to generalise. It failed to be given the right inputs.

> The encoding techniques in § 2 reappear in [Ch.4 Recommender Systems](../04-recommender-systems/README.md), where user and item IDs are the high-cardinality variables. The temporal features in § 3 feed directly into [Ch.3 Neural Networks — RNNs and LSTMs](../03-neural-networks/README.md), where lag sequences become the input to recurrent architectures.

---

## 2 · Categorical Encoding

Back to the grocery problem. Three categorical variables need different strategies: `store_type` (3 values), `category` (18 values), and `sku_id` (1,800 values). The right encoding depends on two things: whether an order exists, and how many distinct values there are.

### 2.1 Ordinal Encoding — When Order Exists

`store_type` has a natural order: budget stores have lower average basket size, higher SKU churn, and tighter margins than premium stores. That order is real and business-meaningful. Encoding budget=0, standard=1, premium=2 preserves it.

The risk: the model may learn spurious magnitudes. If a tree splits on `store_type > 1`, it separates "premium" from the rest — a valid business split. But a linear model trained on ordinal-encoded `store_type` implicitly assumes that the premium→standard gap equals the standard→budget gap. If premium stores sell 3× what budget stores sell and standard stores sell 1.5×, ordinal encoding with values {0, 1, 2} misrepresents those ratios.

**Rule of Thumb:** Ordinal encoding is correct when the order is real and the downstream model is tree-based (which uses splits, not magnitudes). For linear models and neural networks, verify that equal-spacing holds or use target encoding instead.

### 2.2 One-Hot Encoding — When Order Doesn't Exist

`category` (18 product categories) has no natural order. Encoding Frozen=0, Bakery=1, Produce=2 is arbitrary, and a linear model trained on it would interpret "Frozen is 2 units less than Produce" — which is meaningless.

One-hot encoding creates 18 binary columns, one per category. Each row has exactly one 1 and seventeen 0s. The model learns independent effects per category: "Frozen adds 340 units to the base prediction; Bakery adds 89 units."

The curse: 18 categories → 18 columns. Acceptable. But `sku_id` has 1,800 values → 1,800 columns. A random forest with 1,800 one-hot SKU columns evaluates splits on 1,800 features where 1,799 are zero for any given row. Training slows 4-6×; importance scores fragment across 1,800 near-identical features; the model can't see which SKU patterns matter.

**Rule of Thumb:** One-hot encoding is correct when cardinality is below ~50 and the algorithm has no built-in high-cardinality handling. Above 50 values, use target encoding.

### 2.3 Target Encoding — When Cardinality Is High

Target encoding replaces each category with a smoothed estimate of the mean target value within that category. For SKU 1803, the encoded value is a blended estimate of mean weekly sales across all appearances of SKU 1803 in the training set:

$$
\hat{x}_c = \frac{n_c \cdot \mu_c + m \cdot \mu_{global}}{n_c + m}
$$

Read the formula as: start with this category's observed mean $\mu_c$, weighted by how many observations you have ($n_c$), but blend toward the global mean $\mu_{global}$ as $n_c$ shrinks. The smoothing factor $m$ controls the blend rate — with $m = 50$, a category with only 5 observations is pulled 91% toward the global mean, preventing the encoder from overreacting to a SKU seen only five times in training. With $n_c = 500$, the global mean contributes less than 10%.

The result: 1,800 SKUs become 1 numeric column. A tree can now split on "SKUs with mean weekly sales > 800 units behave differently from SKUs with mean sales < 100 units" — a split that was invisible with one-hot encoding.

**Concrete walkthrough — 4 SKUs, $m = 10$:**

| SKU | $n_c$ | $\mu_c$ (mean weekly sales) | $\mu_{global} = 280$ | $\hat{x}_c$ |
|-----|-------|---------------------------|----------------------|-------------|
| SKU1803 (Organic Milk) | 500 | 820 | 280 | $(500 \times 820 + 10 \times 280) / 510 = 811$ |
| SKU0221 (Artisan Bread) | 120 | 94 | 280 | $(120 \times 94 + 10 \times 280) / 130 = 108$ |
| SKU0099 (New Product) | 8 | 340 | 280 | $(8 \times 340 + 10 \times 280) / 18 = 307$ |
| SKU1201 (Seasonal Item) | 3 | 1,100 | 280 | $(3 \times 1100 + 10 \times 280) / 13 = 468$ |

SKU0099 has only 8 observations — its raw mean of 340 is unreliable. Smoothing pulls it 36% toward the global mean, giving 307. SKU1201 with only 3 observations gets pulled 68% toward 280, giving 468 — even though its raw mean is 1,100. This prevents the encoder from treating three lucky sales weeks as "this SKU sells 1,100 units/week forever." As $n_c$ grows, the smoothing term becomes negligible and the encoded value converges to the true category mean.

**The leakage fix.** The naive approach — fit the encoder on the full training set, then cross-validate — leaks. The encoded value for fold 3 was computed using fold 3's labels. Fix: encode fold $k$ using the mean computed from all other folds $1 \ldots k-1, k+1 \ldots n$. `sklearn.preprocessing.TargetEncoder(cv=5)` does this automatically. Wrap it inside a `Pipeline` so it refits per fold.

> **Warning:** The most common target-encoding mistake is fitting the encoder outside a cross-validation loop. The model appears to perform well in validation (because the encoded values have "seen" the validation labels) but degrades 10-15% on fresh production data. Always use `TargetEncoder` inside a `Pipeline` — never standalone with `fit_transform` on the full training set.

### 2.4 Learned Embeddings — When Interactions Matter

For the grocery scenario, there's a subtler problem: `sku_id`'s effect on sales depends on `store_id`. A premium SKU sells 10× at a premium store compared to a budget store, but a generic staple sells similarly across store types. This store×SKU interaction requires 1,800 × 50 = 90,000 interaction terms if encoded naively.

Learned embeddings solve this by representing each SKU and each store as a dense low-dimensional vector (typically 32–64 dimensions), then letting the model learn the interaction through dot products or concatenation. This is the same mechanism used in word2vec, recommendation system embeddings, and entity embeddings for neural tabular models.

**When to use learned embeddings:**
- Cardinality > 100 and interaction effects are the dominant signal (store × SKU, user × product)
- Downstream model is neural (embeddings plug into an embedding layer directly)
- Training data is large enough to learn stable representations (> 10K rows per category level)

**When not to use them:**
- Tree-based models (gradient boosted trees don't consume embedding vectors natively)
- Small training sets where there's insufficient data to learn stable low-dimensional representations

*"Target encoding asks: what is this category's average outcome? Embeddings ask: what kind of thing is this category, and how does it interact with everything else?"*

---

## 3 · Temporal Features

The grocery dataset has one feature most tabular ML pipelines undertreat: `date`. Extracting year, month, and day and calling it done misses almost all the temporal signal. Here's what you're leaving on the table.

### 3.1 Why Time Data Is Different

Most models assume training examples are **independent and identically distributed** — that the sales for SKU 1803 in week 3 are independent of week 2. They are not. Sales in week 3 are partially *caused* by week 2: last week's stockout suppresses week 3 sales; last week's promotion causes an inventory drawdown that reduces week 4 demand. The rows are not i.i.d.

Temporal features explicitly encode this dependence, turning a model-breaking assumption violation into an exploitable signal. But they must be constructed with one rule: **features must look only backwards.** Any feature using information from time $t$ or later to predict $t$ is leakage — even when it looks innocent.

### 3.2 Lag Features

Lag features are the values of the target at a fixed offset in the past:

$$
x_{t,k} = \text{sales}_{t-k}
$$

For $k \in \{1, 7, 14, 28\}$ weeks: "What did we sell one week ago? One month ago?"

These four lag features typically contribute more predictive signal than any other single feature group in a demand forecasting model. The 7-day lag captures weekly seasonality directly — whatever sold last Monday predicts this Monday better than any calendar feature. The 28-day lag captures monthly replenishment cycle patterns.

```python
for lag in [1, 7, 14, 28]:
 df[f'lag_{lag}'] = df.groupby('sku_id')['sales_units'].transform(
 lambda x: x.shift(lag)
 )
```

> **Warning:** Compute lag features *before* train-test splitting, on the full time series, then split. If you compute lags after splitting, the training set's first $k$ rows have undefined lag history — NaN values that silently propagate through the pipeline as zeros. Compute on the full timeline, fill NaNs with forward-fill where appropriate, then drop the first $k$ rows where no valid lag history exists.

### 3.3 Rolling Statistics

Lag features capture a single point in the past. Rolling statistics capture the *shape* of the recent window: the trend level, the volatility, and the recent peak.

| Feature | Captures |
|---------|---------|
| Rolling mean (7-day) | Recent trend level — is demand growing or shrinking? |
| Rolling std (7-day) | Recent demand volatility — is the SKU stable or spiky? |
| Rolling max (14-day) | Peak within the window — recent demand ceiling |
| Rolling mean (28-day) | Slow-moving trend — monthly baseline level |

The critical implementation detail: `.shift(1).rolling(7)`, not `.rolling(7)`. The `.shift(1)` moves the window backward by one period before rolling, excluding the current row. Without it, the 7-day rolling mean for row $t$ includes sales at $t$ — leaking the target directly into the feature.

```python
# Correct: shift first, then roll
df['sales_roll7_mean'] = df.groupby('sku_id')['sales_units'].transform(
 lambda x: x.shift(1).rolling(7).mean()
)

# Wrong: rolling without shift includes current row's target
df['sales_roll7_mean_leaky'] = df.groupby('sku_id')['sales_units'].transform(
 lambda x: x.rolling(7).mean()
)
```

The leaky version produces validation MAPE of ~12% — which looks excellent but is an illusion. Production MAPE reverts to 34% on the first week of deployment.

**Why the leaky version looks so good in validation:** When the rolling window for row $t$ includes row $t$ itself, the feature is essentially `sales_units_this_week * (6/7) + noise`. Predicting from a near-copy of the target is trivially easy. The model "learns" a near-identity function and has learned nothing transferable. The bug is invisible until deployment.

> **Warning:** If your rolling-window feature validation scores look suspiciously good — MAPE below 10% when your baseline is above 25% — check for `.shift()` omissions before concluding the feature is genuinely predictive.

### 3.4 Seasonality and Cyclic Encoding

For the grocery chain, day-of-week drives significant demand variation: weekend sales of perishables run 40% above weekday levels. Simply encoding Monday=0 through Sunday=6 creates a discontinuity: the model sees Sunday (6) and Monday (0) as maximally different, but they are consecutive days — the Sunday-to-Monday boundary is no different from any other consecutive-day transition.

Cyclic encoding fixes this by projecting the periodic index onto a unit circle:

$$
x_{\sin} = \sin\!\left(\frac{2\pi k}{P}\right), \quad x_{\cos} = \cos\!\left(\frac{2\pi k}{P}\right)
$$

where $k$ is the period index and $P$ is the cycle length. For day-of-week, $P = 7$ and $k \in \{0, 1, \ldots, 6\}$. Sunday ($k=6$) and Monday ($k=0$) are separated by $\frac{2\pi}{7}$ radians in the $(x_{\sin}, x_{\cos})$ plane — the same angular distance as any other consecutive pair. The discontinuity disappears. Apply the same encoding to month-of-year ($P = 12$), hour-of-day ($P = 24$), and any other periodic calendar feature.

Additional seasonality features that pay off in grocery demand forecasting:

| Feature | Why it matters |
|---------|----------------|
| `is_holiday` | Demand multipliers of 3–8× on grocery staples at Christmas, Thanksgiving, and Easter |
| `days_since_last_promo` | Post-promotion demand hangover decays over 2–3 weeks; raw dates don't encode this |
| `week_of_year` | Captures annual seasonality — week 51–52 (Christmas) is categorically different from week 26 |
| `is_month_end` | Payday effects on discretionary spending in the last 3 days of the month |

> **Insight:** The most predictive temporal features are often the ones that encode *business cycles* rather than calendar cycles. "Days since last promotion" captures the post-promotion inventory drawdown that lasts 2–3 weeks. That's domain knowledge — no raw timestamp encodes it, and no neural network discovers it from a Unix timestamp.

*"The timestamp knows when. Feature engineering teaches the model to care why."*

---

## 4 · TF-IDF vs Learned Text Embeddings

The grocery chain's product database has two text fields per SKU: a short `product_name` ("Organic 2% Milk 64oz") and a longer `product_description` (3–8 sentences from the product label). Both contain signal for demand prediction — similar products have correlated demand. How do you encode them?

### 4.1 TF-IDF: Bag-of-Words Signal

TF-IDF (term frequency–inverse document frequency) represents each text document as a sparse vector where each dimension is a vocabulary term, weighted by how distinctive that term is for this document:

$$
\text{TF-IDF}(t, d) = \text{tf}(t, d) \cdot \log\frac{N}{df(t)}
$$

The term frequency $\text{tf}(t, d)$ counts how often term $t$ appears in document $d$. The IDF factor $\log\frac{N}{df(t)}$ upweights rare terms — those appearing in few documents, which are therefore distinctive — and downweights common terms that appear in most documents and carry no discriminating power.

For product names, this works well. A 3,000-term vocabulary produces a 3,000-dimensional sparse vector per product:

- "Organic" appears in 12% of SKU names → IDF weight 2.1 (moderately distinctive)
- "Milk" appears in 3% of SKU names → IDF weight 3.5 (distinctive)
- "2%" appears in 2% → IDF weight 3.9 (very distinctive)
- "the" appears in 90% of descriptions → IDF weight 0.1 (effectively ignored)

Two similar products ("Organic 2% Milk 64oz" and "Organic 2% Milk 32oz") have near-identical TF-IDF vectors — correctly capturing their demand correlation.

**TF-IDF with `sklearn.feature_extraction.text.TfidfVectorizer` costs 0.3ms to encode a product name.** The resulting matrix is sparse (99%+ zeros), fits in memory for 1,800 SKUs, trains fast, and is interpretable: you can inspect exactly which terms drive any prediction.

**TF-IDF wins when:**
- Vocabulary is domain-specific and stable (grocery SKU names change slowly month-to-month)
- Documents are short — under ~50 tokens (product names, SKU tags, short descriptions)
- Explainability matters — the buyer wants to know *why* the model flagged a SKU
- Training data is small — under 10K examples, where adapting embeddings to the task is not feasible

### 4.2 Learned Embeddings: Semantic Density

Sentence-transformer models (`all-MiniLM-L6-v2`, `text-embedding-ada-002`) map any text to a dense 384–1,536-dimensional vector. Words with similar meanings cluster together in this space: "milk" and "dairy" are close; "organic" and "natural" are close; "frozen" and "refrigerated" are close.

This semantic awareness matters when documents are long and semantically varied. For product *descriptions* (3–8 sentences), TF-IDF fragments meaning across specific term matches, while embeddings capture "premium organic dairy product" as a coherent dense cluster.

**A local sentence-transformer costs 50ms per product name on CPU, 10–20ms with a GPU.** OpenAI `text-embedding-ada-002` costs $0.0001 per 1K tokens — for 1,800 SKUs at 20 tokens average, that's 36K tokens = $0.0036 per full-catalog encoding run. The dense 1,536-dimensional vector contains more semantic information — but dense high-dimensional vectors are harder for gradient-boosted trees to exploit. Trees split on single dimensions; extracting signal from 1,536 correlated dimensions requires substantially more trees, longer training, and more data to generalise.

> **Insight:** The "BERT always wins" assumption fails reliably in two scenarios: (1) small training sets where the downstream model can't distinguish signal from 1,536 noisy dimensions, and (2) domain-specific jargon the embedding model never encountered during pretraining. A sentence-transformer trained on Common Crawl knows "organic" and "natural" are related — but it has no idea that "SKU rotation," "3+1 promo," or "planogram slot A3" are signals specific to your grocery domain. TF-IDF trained on your product catalogue does.

### 4.3 Decision Guide for the Grocery Scenario

| Field | Choice | Reason |
|-------|--------|--------|
| `product_name` (5–10 tokens) | TF-IDF, 3K vocabulary | Short, domain-specific, stable vocabulary. 0.3ms, interpretable. |
| `product_description` (3–8 sentences) | Sentence embeddings, 384d | Longer, semantically varied, benefits from transfer learning on general language patterns. |
| `category_description` (internal tags) | TF-IDF | Internal jargon absent from pretrained embeddings. |

*"TF-IDF knows your vocabulary. Embeddings know everyone else's. Choose based on which matters more."*

### 4.4 When the "BERT Wins" Assumption Fails in Practice

Three real-world scenarios where TF-IDF outperforms or matches sentence embeddings on tabular-adjacent text:

**Scenario 1 — Small training set.** A grocery chain has 1,800 SKUs but only 18 months of sales data: ~78,000 training rows. A sentence-transformer produces 384-dimensional embeddings — each of those 384 dimensions must carry enough predictive signal for the downstream XGBoost model to find useful splits. With 78K rows and 384 text dimensions, the model is fitting ~40K embedding-derived splits from relatively few examples. TF-IDF with 3K sparse dimensions sounds worse but actually performs better here: most of the 3K dimensions are zero, so the model only ever encounters ~15 non-zero TF-IDF dimensions per product name — a much sparser and more tractable space.

**Scenario 2 — Domain-specific vocabulary.** Internal warehouse systems label SKUs with codes like "PLNGRM-A3-SLOT-2," "ROT-WKLY-PERISHABL," "PROMO-3+1-END-CAP." A sentence-transformer trained on web text has zero meaningful representation for these strings — they land somewhere in embedding space but with no semantic coherence. TF-IDF treats them as raw vocabulary terms and finds them highly distinctive (IDF close to log(N) because they're rare), which is the correct behaviour.

**Scenario 3 — Stability requirement.** If the product catalogue updates weekly with new SKUs, a TF-IDF vectoriser with a fixed vocabulary simply marks new terms as out-of-vocabulary (OOV) and assigns them zero weight. Stable and predictable. A sentence-transformer re-encodes correctly but requires the downstream model to have seen similar products in training for the embedding to be meaningful. Neither is perfect, but TF-IDF's failure mode is more transparent: OOV is explicit, not a misleading dense vector.

*"The model that looks impressive in a benchmark demo is not always the model that survives contact with production data."*

After categorical encoding, temporal feature engineering, and text vectorisation, the grocery pipeline has 300+ features. Many are redundant, some are noise, and a few are doing nearly all the work. Feature selection is how you find out which is which — systematically, cheaply, and in the right order.

Five methods, ordered by computational cost from cheapest to most expensive.

### 5.1 Variance Threshold — Zero Cost

Remove features with variance below a threshold $\varepsilon$. A feature with variance = 0 is constant across all rows and contributes exactly nothing. A feature with variance = 0.001 in a dataset where the target has variance of 100,000 is near-constant and contributes near-nothing.

```python
from sklearn.feature_selection import VarianceThreshold

sel = VarianceThreshold(threshold=0.01)
X_filtered = sel.fit_transform(X)
```

In the grocery pipeline: 40 of 300 features removed here — mostly one-hot columns for rare store subtypes (fewer than 3 stores out of 50) and lag features at the edges of the time series where most values are NaN-filled zeros.

**Cost:** One pass through $X$, $O(N \cdot p)$. Essentially free.

### 5.2 Correlation Filter — Cheap

Remove one column from any pair with absolute Pearson correlation above 0.95. Correlated features carry the same information but dilute each other's importance score: if lag-7 and lag-8 correlate at 0.97, importance is split between them — neither appears as important as it truly is.

```python
import pandas as pd
import numpy as np

corr_matrix = pd.DataFrame(X_filtered).corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [col for col in upper.columns if any(upper[col] > 0.95)]
X_decorr = pd.DataFrame(X_filtered).drop(columns=to_drop)
```

In the grocery pipeline: 25 of the remaining 260 features removed — mostly redundant rolling windows (the 7-day mean and 7-day median are often 0.96+ correlated on smooth demand series).

**Cost:** $O(p^2)$ correlation matrix. For $p = 260$, that's ~67,600 pairwise correlations — milliseconds.

### 5.3 Mutual Information — Medium Cost

Mutual information measures the reduction in uncertainty about target $Y$ when knowing feature $X$:

$$
I(X; Y) = H(Y) - H(Y \mid X)
$$

where $H(Y)$ is the entropy of the target and $H(Y \mid X)$ is the remaining entropy once $X$ is known. Unlike Pearson correlation, mutual information captures non-linear relationships. A feature that is predictive only above a threshold — for example, sales spike only when the rolling mean crosses 500 units — will have low Pearson correlation but high mutual information.

```python
from sklearn.feature_selection import mutual_info_regression
import numpy as np

scores = mutual_info_regression(X_decorr, y, n_neighbors=5, random_state=42)
top_100_idx = np.argsort(scores)[-100:]
X_mi = X_decorr[:, top_100_idx]
```

In the grocery pipeline: selecting the top 100 features by MI score reduces from 235 to 100 while retaining non-linear interaction features (such as `is_holiday × promotion_flag` interactions) that correlation-based methods miss.

> **Warning:** Mutual information scores each feature independently against the target. It cannot detect redundancy *between* features. Two features that are individually informative but mutually correlated will both score high. Run the correlation filter *before* mutual information, not after.

**Cost:** $O(N \cdot p \cdot k)$ for $k$-nearest-neighbour entropy estimation. For $N = 100K$ rows and $p = 235$ features, expect 10–30 seconds. Parallelisable with `n_jobs=-1`.

### 5.4 L1 (Lasso) Regularisation — Embedded Selection

Lasso adds an L1 penalty to the loss function that pushes model weights toward exactly zero. Features whose weights go to zero are removed by the model itself — selection happens during training, in the context of all other features simultaneously.

```python
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel

lasso = Lasso(alpha=0.01, max_iter=10000)
lasso.fit(X_mi_scaled, y)
selector = SelectFromModel(lasso, prefit=True)
X_lasso = selector.transform(X_mi_scaled)
```

In the grocery pipeline: Lasso with $\alpha = 0.01$ prunes from 100 to approximately 60 features. Features zeroed out tend to be marginal lag features (lag-21 and lag-35, which carry less signal than lag-7, lag-14, and lag-28 in a weekly-granularity model) and redundant text dimensions.

*"Lasso doesn't just select features — it selects them in the context of all other features simultaneously. That's why it catches redundancy that MI scoring misses."*

**Cost:** Full linear model training — fast even with $N = 100K$. For neural networks with L1 regularisation, this is your full training cost.

### 5.5 Permutation Importance — Most Faithful, Most Expensive

Fit the final model. Then, for each feature: shuffle that column randomly across all validation rows, pass the shuffled data through the model, and measure the accuracy drop. The drop is the feature's importance. A feature the model doesn't actually use will show no drop when shuffled.

```python
from sklearn.inspection import permutation_importance

result = permutation_importance(
 model, X_val, y_val, n_repeats=10, random_state=42, n_jobs=-1
)
feat_importance = pd.Series(
 result.importances_mean, index=feature_names
).sort_values(ascending=False)
```

In the grocery pipeline: permutation importance confirms that the top 20 features account for 80% of validation accuracy. The result that surprises most engineers: `lag_7_sales_sku` is the single most important feature — shuffling it causes MAPE to jump from 16.3% to 31.7% (a 94% degradation). No text feature, no store-type encoding, and no calendar feature comes close.

> **Insight:** Permutation importance is the gold standard because it measures feature importance through the model's actual behaviour, not through feature-target correlation. A feature that is highly correlated with the target but redundant with a more predictive feature will show low permutation importance — which is the correct signal. Use it to audit the final model; use the cheaper methods for early-stage filtering.

**Cost:** $p$ complete inference passes over the validation set. For $p = 60$ features and 50K validation rows, that is 3M inference calls — roughly 90 seconds with XGBoost. Worth it once per model version.

### 5.6 The Full Selection Pipeline

```
300 raw features
 → Variance threshold (threshold=0.01): removes 40 constant/near-constant
 → Correlation filter (|r| > 0.95): removes 25 redundant rolling windows
 → Mutual information top-100: removes 135 low-signal features
 → Lasso (α=0.01) embedded selection: removes 40 marginal lag and text dimensions
 → Permutation importance audit: confirms top 20 features → 80% of signal

Final model: ~60 features
```

The pipeline order is not arbitrary. Running Lasso on all 300 features before variance threshold and correlation filtering wastes time and produces unstable results — Lasso is sensitive to correlated inputs, and 25 redundant rolling windows will destabilise its coefficient estimates. Running permutation importance on all 300 features before any reduction costs 300 inference passes instead of 60.

---

## 6 · Putting It Together

The full grocery demand forecasting pipeline, in the order each step must execute and the reason each ordering constraint exists:

```
Raw data (store_id, sku_id, category, date, sales_units, promotion_flag, store_type)
 ↓
1. Temporal features — lags (1/7/14/28), rolling mean/std, cyclic day-of-week,
 is_holiday, days_since_last_promo
 Reason: must run on the full time series before splitting, so lag history
 spans training and validation rows.
 ↓
2. Train-test split on time — all data before 2024-01-01 for training,
 2024-01-01 onward for validation
 Reason: never random-split a time series. Rows are not i.i.d. A random
 split leaks future rows into training and past rows into validation.
 ↓
3. Categorical encoding inside a Pipeline
 — OrdinalEncoder (store_type): fit-once, no leakage risk
 — OneHotEncoder (category): fit-once on training vocabulary
 — TargetEncoder(cv=5) (sku_id): must refit per cross-validation fold
 Reason: TargetEncoder must never see validation fold labels during encoding.
 Wrapping in Pipeline ensures it refits per fold automatically.
 ↓
4. Text features
 — TfidfVectorizer (product_name): fit vocabulary on training set only
 — SentenceTransformer (product_description): pre-trained, no fitting required
 Reason: TF-IDF vocabulary fitted on validation text would leak
 vocabulary statistics.
 ↓
5. Feature selection — variance threshold → correlation filter → MI top-K
 → Lasso embedded selection
 Reason: all selection steps fitted on training set only, applied to
 validation via transform-only. Fitting selection on combined data leaks
 validation-set structure.
 ↓
6. Model training and permutation importance audit
```

**Three failure modes to avoid:**

- **Scaling before temporal features**: StandardScaler fitted before lag computation scales the lag features and the current-week features on different statistics. The model sees them on inconsistent scales, confusing their relative importance.
- **Target encoding outside the Pipeline**: If `TargetEncoder` sees validation fold labels during encoding, validation MAPE understates production MAPE by 5–10%. The model appears well-calibrated, then degrades immediately in production.
- **Feature selection before the train-test split**: A variance filter or MI selector fitted on the full dataset retains features that vary in the validation set, quietly leaking validation-set structure into the feature selection decision.

### Method Reference

| Method | Use Case | Cost | Interpretability | Primary Risk |
|--------|---------|------|-----------------|--------------|
| Ordinal encoding | Ordered categoricals | Zero | High — natural scale | Spurious magnitude in linear models |
| One-hot encoding | Unordered, cardinality < 50 | Low | High — binary flags | Dimensionality explosion above ~50 |
| Target encoding | Unordered, cardinality > 50 | Low + cross-fold | Medium — single numeric | Leakage if not cross-fold encoded |
| Lag features | "What happened $k$ steps ago?" | Low | High — named offset | Future leakage without `.shift()` |
| Rolling statistics | Trend and volatility of recent window | Low | High — named window | Future leakage without `.shift()` |
| Cyclic encoding | Periodic features (day, month, hour) | Zero | Medium — sine/cosine pair | None |
| TF-IDF | Short domain-specific text | Low | High — per-term weights | Vocabulary drift over time |
| Sentence embeddings | Long text, semantic similarity | Medium | Low — dense vector | Domain gap for specialised jargon |
| Variance threshold | Remove constants | Zero | N/A | Misses low-variance but informative features |
| Correlation filter | Remove redundant features | Low | N/A | Drops both when keeping one would suffice |
| Mutual information | Non-linear feature-target signal | Medium | Medium — per-feature score | Ignores feature-feature redundancy |
| Lasso selection | Simultaneous embedded selection | Medium | Medium — weight magnitude | Sensitive to correlated inputs |
| Permutation importance | Audit final model relevance | High | High — model-faithful | Expensive on large feature sets |

### Wiring It Up — Sklearn Pipeline

The following shows how the encoding and selection steps wire together in a single `Pipeline`, ensuring that no fitting step ever sees validation data:

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.preprocessing import TargetEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import VarianceThreshold, SelectFromModel
from sklearn.linear_model import Lasso
from xgboost import XGBRegressor

# Categorical preprocessor — three columns, three strategies
categorical_preprocessor = ColumnTransformer(
 transformers=[
 ('ordinal', OrdinalEncoder(
 categories=[['budget', 'standard', 'premium']]
 ), ['store_type']),
 ('onehot', OneHotEncoder(
 handle_unknown='ignore', sparse_output=True
 ), ['category']),
 ('target', TargetEncoder(cv=5, smooth='auto'), ['sku_id']),
 ('tfidf', TfidfVectorizer(max_features=3000), 'product_name'),
 ],
 remainder='passthrough' # lag features, rolling stats, cyclic encodings
)

# Feature selection stage — variance filter then Lasso embedded selection
selection = Pipeline([
 ('variance', VarianceThreshold(threshold=0.01)),
 ('lasso_sel', SelectFromModel(
 Lasso(alpha=0.01, max_iter=10000), threshold='median'
 )),
])

# Full pipeline
grocery_pipeline = Pipeline([
 ('encode', categorical_preprocessor),
 ('select', selection),
 ('model', XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6)),
])

# Train — TargetEncoder refits per cross-validation fold automatically
grocery_pipeline.fit(X_train, y_train)
y_pred = grocery_pipeline.predict(X_val)
```

A few notes on this implementation:
- `TargetEncoder(cv=5)` refits per fold during cross-validation because it is wrapped inside the `Pipeline`. Call `cross_val_score(grocery_pipeline, X_train, y_train, cv=5)` and the target encoder never sees the validation fold it is encoding.
- `TfidfVectorizer` as a `ColumnTransformer` step fits its vocabulary on training text only. If a product name in the validation set contains a new term not seen in training, that term is silently ignored — correct behaviour.
- The mutual information step is intentionally absent from the pipeline above: MI should be run as an offline analysis to select the `max_features` budget for TF-IDF and to shortlist lag window sizes, then baked in as configuration before the pipeline is trained. Fitting MI inside cross-validation is valid but expensive — reserve it for final model validation, not grid search.

*"Good feature engineering is not about having more features. It's about having fewer, better ones."*

---

## Checkpoint

Working from the grocery scenario — 300 raw candidate features, 18% MAPE target:

- **Categorical encoding** — ordinal for `store_type`, one-hot for `category`, cross-fold target encoding for `sku_id`: resolves the category overload problem (1,800 columns → 1 column for SKU)
- **Temporal features** — lag-7/14/28, rolling mean/std, `is_holiday`, `days_since_last_promo`, cyclic day-of-week: resolves the missing temporal signal problem
- **Text features** — TF-IDF (3K vocabulary) on product names, sentence embeddings on descriptions: adds demand-similarity signal from the product catalogue
- **Feature selection pipeline** — variance → correlation → MI → Lasso → permutation audit: prunes 300 → ~60 features, resolves noise amplification
- **Final model MAPE on held-out 2024 data: 16.3%** — below the 18% target, stable across January (holiday structure encoded), July (seasonal patterns encoded), and October (post-summer replenishment cycles encoded)

The 17.7 percentage point improvement from the naive 34% MAPE baseline came almost entirely from feature engineering — not from algorithm choice. The model is still XGBoost. The features changed.

**Ablation — what each engineering group contributed:**

| Feature Group Added | Cumulative MAPE | Gain |
|---------------------|-----------------|------|
| Baseline (naive one-hot + year/month/day) | 34.0% | — |
| + Target encoding for sku_id | 27.3% | 6.7pp |
| + Lag features (1/7/14/28) | 21.1% | 6.2pp |
| + Rolling statistics (mean/std, 7-day and 28-day) | 19.4% | 1.7pp |
| + Cyclic day-of-week + is_holiday + days_since_last_promo | 18.0% | 1.4pp |
| + TF-IDF on product names | 17.2% | 0.8pp |
| + Feature selection (300 → 60) | 16.3% | 0.9pp |

Three observations from the ablation:
1. Target encoding for `sku_id` (a single encoding change) delivers the largest single gain: 6.7pp. Switching 1,800 sparse binary columns to 1 smoothed numeric column is the highest-leverage action in the pipeline.
2. Lag features are the second-largest gain (6.2pp). Raw temporal signal — "what sold last week" — is more predictive than any calendar or category feature.
3. Feature selection contributes a net positive even though it removes features. The 240 features removed were adding noise that increased variance in the tree's split decisions. Removing them tightened the model's generalisation.

The algorithm is not what changed. If you swapped XGBoost for a random forest on the naive feature set, you'd get 32% MAPE instead of 34% — a 2-point gain from model choice. Feature engineering gave 17.7 points. That ratio — roughly 10:1 in favour of representation over algorithm — is consistent across most tabular forecasting problems in practice.

*"Given the right features, most algorithms reach the same ceiling. Given the wrong features, no algorithm reaches it."*



### Leakage Taxonomy — Quick Reference

Leakage is the single most common failure mode in feature engineering. It appears in four distinct forms:

| Leakage Type | How It Happens | Symptom | Fix |
|---|---|---|---|
| **Target leakage** | A feature contains information derived from the target (e.g., naive target encoding) | Validation score dramatically better than production | Cross-fold encoding; fit encoders inside Pipeline |
| **Temporal leakage** | A feature at time $t$ uses values from $t$ or later (missing `.shift()`) | Validation MAPE suspiciously low; drops sharply at deployment | Always `.shift(1)` before `.rolling()` |
| **Split leakage** | Feature selection or scaling fitted on combined train+validation data | Feature selection retains validation-correlated features | All `fit()` calls on training data only; `transform()` on validation |
| **Proxy leakage** | A feature is a near-perfect proxy for the target, not derived from it (e.g., "units shipped" as a feature for "units sold") | Perfect training score; feature makes no causal sense | Audit features for causal validity, not just statistical correlation |

> **Constraint:** Every feature in a production pipeline must answer the question: "At prediction time, when I need to forecast next week's sales, could I actually compute this feature using only information available right now?" If the answer is no, the feature is leakage — regardless of how good your validation scores are.

---

> The feature selection methods in § 5 — particularly permutation importance and Lasso-based selection — reappear in [Ch.3 Feature Scaling, Importance & Multicollinearity](../01-regression/ch03_feature_importance/README.md), where they are applied to the California Housing regression problem. The categorical embedding approach from § 2.4 connects directly to entity embedding architectures covered in [Ch.4 Recommender Systems — Neural Collaborative Filtering](../04-recommender-systems/ch04_neural_cf/README.md).

