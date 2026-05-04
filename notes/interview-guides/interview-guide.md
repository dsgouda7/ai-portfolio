# Machine Learning Interview Guide

← Back to learning chapter: [Notes Overview](../README.md)

> **Purpose**: Consolidated interview preparation content for all 19 ML chapters. Each section contains must-know facts, likely questions, and common traps organized by topic.

---

## How to Use This Guide

- **Must know**: Core facts expected in every interview
- **Likely asked**: Common follow-up questions or derivations
- **Trap to avoid**: Common misconceptions or mistakes that signal incomplete understanding

This guide mirrors the progression of the ML track chapters. For deeper context on any topic, refer to the corresponding chapter README.

---

> 💡 **How to use the junior/senior answer comparisons** — Each question below includes a junior-level answer and a senior-level answer. Junior answers are technically correct but surface-level. Senior answers demonstrate production experience, failure awareness, and trade-off reasoning. Hiring managers at FAANG and growth-stage AI companies distinguish these instantly. Study the DIFFERENCE between the two, not just the senior answer.

## Ch.1 — Linear Regression

| Must know | Likely asked | Trap to avoid |
|---|---|---|
| The MSE formula and why squaring matters | Derive the gradient descent update rule for MSE | "R² = 0.9 means the model is good" — not if you have 50 features and should be using Adjusted R² |
| Difference between loss (MSE) and metric (R²) | When would you choose MAE over MSE? | Forgetting to scale features before gradient descent |
| What the bias term does | Explain the difference between batch, mini-batch, and SGD | Claiming linear regression can't overfit — it can on small datasets with many features |
| How learning rate affects convergence | What happens if features are left unscaled? | Confusing R² on training set (always high) with test set R² |
| The four assumptions of linear regression: **linearity** (relationship between X and y is linear), **independence** of errors, **homoscedasticity** (constant error variance across all X values), and **normality of errors** (residuals are normally distributed — required for valid confidence intervals) | "What assumptions does linear regression make?" — all four are expected | Stating only "linearity"; interviewers penalise missing homoscedasticity in particular, as it is the most commonly violated assumption in real data |
| Normal equations: $\hat{W} = (X^TX)^{-1}X^Ty$ — closed-form exact solution, $O(d^3)$ due to matrix inversion; prefer gradient descent when $d > 10{,}000$ because $O(d^3)$ becomes prohibitive and $(X^TX)$ can be near-singular | "When would you use normal equations vs gradient descent?" | "Normal equations are always better because they're exact" — they are numerically unstable when $X^TX$ is near-singular (collinear features) and prohibitively slow for large feature spaces |

---

## Ch.2 — Logistic Regression

| Must know | Likely asked | Trap to avoid |
|---|---|---|
| Sigmoid formula and why it maps $\mathbb{R}$ → $(0,1)$ | Derive the gradient of BCE with respect to $z$ | "Logistic regression is a classification algorithm" — it outputs probabilities; the threshold makes it a classifier |
| BCE loss formula and what each term does | Why is MSE a bad loss for classification? | Reporting accuracy without asking about class balance |
| Precision vs Recall trade-off | When would you prefer precision over recall? | Confusing AUC-ROC (ranking quality) with accuracy (at a single threshold) |
| What decision threshold $\tau$ does | How do you pick the optimal threshold? | Using `predict` instead of `predict_proba` and then trying to threshold |
| Confusion matrix: TP, TN, FP, FN | What is AUC-ROC and when does it fail? | AUC-ROC can be misleadingly high on imbalanced datasets — use AUC-PR instead |
| Multiclass extension — **One-vs-Rest (OvR):** train $K$ binary classifiers, pick the one with the highest score; **Softmax (Multinomial):** single model with $K$ output logits, `multi_class='multinomial'` in sklearn | "How would you extend logistic regression to 10 classes?" — both approaches expected | "OvR is always worse than Softmax" — for well-separated classes OvR is faster to train and each classifier is individually interpretable |
| `class_weight='balanced'` in sklearn weights each sample by the inverse class frequency, counteracting imbalance during training | "How do you handle a 1:100 class imbalance in logistic regression?" | Using `class_weight='balanced'` without re-calibrating the decision threshold — the model's predicted probabilities are skewed and the default 0.5 threshold is now wrong; use a PR curve or F1-vs-threshold sweep to find the new optimal $\tau$ |

---

## Ch.3 — The XOR Problem

| Must know | Likely asked | Trap to avoid |
|---|---|---|
| Why a single perceptron cannot solve XOR (draw the four XOR points) | Explain what the Universal Approximation Theorem says and what it does NOT say | "Adding more hidden units always improves performance" — more units = more overfitting risk without regularisation |
| What a hidden layer does geometrically (transforms the space) | Why do we use ReLU instead of Sigmoid for hidden layers? | Confusing depth (layers) with width (units) — for XOR specifically, width is what matters |
| Why non-linear activations are necessary (without them, stacking layers collapses to one linear layer) | What is the symmetry problem with zero initialisation? | Claiming one hidden layer is always best because of the UAT — in practice, depth is more parameter-efficient |
| The role of the activation function at each layer | What happens if you use a linear activation throughout? | Forgetting that the UAT is an existence result, not a learning guarantee |
| **Depth vs width in practice:** depth adds representational hierarchy (each layer composes features from the previous); width adds capacity at one level. For most structured-data problems, 2–3 deep narrow layers beats 1 wide layer of equivalent parameters | "When would you add depth vs width to a network?" | "The UAT says one hidden layer is sufficient, so depth doesn't matter" — the UAT is an existence proof; the number of units required may be exponential; deep networks learn the hierarchy explicitly and are more parameter-efficient |

---

## Ch.4 — Neural Networks

| Must know | Likely asked | Trap to avoid |
|---|---|---|
| Why not stack linear layers? | Composition of linear maps is still linear — you get no extra expressiveness | "Just add more neurons to the single layer" — same problem |
| When to use ReLU vs Tanh? | ReLU for hidden layers (fast, no saturation for positives); Tanh when zero-centring matters (e.g., RNN hidden states) | Dying ReLU: if all inputs to a neuron are negative, gradient is permanently zero |
| What does Xavier init solve? | Keeps activation variance constant across layers so signal neither explodes nor vanishes during the forward pass | He is better for ReLU because ReLU kills half the variance |
| Describe the forward pass | $\mathbf{h} = g(\mathbf{W}^\top\mathbf{x}+\mathbf{b})$ repeated per layer; final layer linear for regression | Forgetting to transpose / mismatching shapes is the most common bug |
| Why scale inputs? | Gradient descent treats all weight dimensions equally; un-scaled features force one weight to be orders of magnitude smaller than others | BatchNorm can compensate, but input scaling is cheaper and faster |
| **Batch Normalisation:** normalise each feature in a mini-batch to zero mean / unit variance, then apply learnable scale $\gamma$ and shift $\beta$; placed **before** the activation. Reduces internal covariate shift, enables higher learning rates, acts as a mild regulariser, and reduces sensitivity to weight initialisation | "What does Batch Normalisation do and where do you place it?" | "BatchNorm is always helpful" — for small batch sizes (<8) the batch statistics are too noisy; use LayerNorm (NLP) or GroupNorm (CV with small batches) instead |
| **He initialisation:** $W \sim \mathcal{N}(0, \sqrt{2/n_\text{in}})$ — preferred over Xavier for ReLU activations because ReLU zeroes roughly half of activations, halving the variance; He compensates by doubling the scale relative to Xavier | "Why use He init instead of Xavier with ReLU?" | Using Xavier initialisation with ReLU — activations shrink with depth because Xavier assumes a symmetric activation with full pass-through gain (gain = 1) |

---

## Ch.5 — Backprop & Optimisers

| Must know | Likely asked | Trap to avoid |
|---|---|---|
| What is backprop? | Efficient application of chain rule backwards through the computation graph — reuses stored forward-pass values | It's not "gradient descent" — backprop computes gradients; GD uses them |
| Why does ReLU make backprop easy? | Derivative is 1 for $z>0$, 0 otherwise — a single comparison with no exp/log | Dying ReLU: if $z<0$ for all inputs, gradient is zero forever |
| Why use Adam over SGD? | Adam adapts step size per parameter using gradient history; converges faster with less LR tuning| Adam can converge to sharper minima than SGD; SGD often generalises better |
| What is bias correction in Adam? | Divides $m_t, v_t$ by $(1-\beta^t)$ to compensate for zero-initialisation at step 1 | Without it, first few steps are severely under-sized |
| What does a learning rate schedule do? | Allows large steps early (fast progress) and small steps later (fine-grained convergence) | Cosine annealing without warmup can be unstable at the start |
| **Gradient clipping:** if $\|\nabla\| > \text{clip\_value}$, rescale the gradient vector so its norm equals `clip_value`; direction is preserved. Fixes exploding gradients without affecting the update direction | "How do you handle exploding gradients in an RNN?" | "Gradient clipping fixes vanishing gradients too" — clipping only limits large gradients; vanishing gradients (near-zero norms) are unaffected and require architecture changes (residual connections, LSTM gates, careful initialisation) |
| **Weight decay vs L2 regularisation:** identical in SGD ($W_{t+1} = W_t(1-\eta\lambda) - \eta\nabla L$) but *different* in Adam — standard L2 adds $\lambda W$ to the gradient before adaptive scaling, so high-variance parameters get less regularisation; **AdamW** applies weight decay directly to the parameter decoupled from the gradient update, giving consistent regularisation regardless of gradient magnitude | "What is the difference between Adam+L2 and AdamW?" | "Adam with L2 regularisation is the same as AdamW" — it is not; AdamW is the correct decoupled implementation and is the default in most modern training recipes |

---

## Ch.6 — Regularisation

| Must know | Likely asked | Trap to avoid |
|---|---|---|
| What does L2 regularisation do to weights? | Adds a penalty $\lambda\|\mathbf{W}\|^2$ to the loss, producing a constant shrinkage factor $(1-2\eta\lambda)$ each step | L2 shrinks toward zero but never reaches it — L1 can push weights to exactly zero |
| Why does Dropout work? | Each step trains a random sub-network; the full network is an implicit ensemble of $2^n$ sub-networks | At test time, dropout is disabled — forgetting this is a common bug |
| When does early stopping outperform L2? | When the optimum number of epochs is highly dataset-dependent and hard to predict a priori | Patience must be set relative to how noisily the val curve moves |
| L1 vs L2 in neural networks? | L1 is rarely used directly on neural net weights (non-smooth gradient at 0); L2 / weight decay is standard | L1 is very common in linear models (Lasso) but in neural nets Dropout fills the sparsity role |
| What is the generalisation gap? | $\mathcal{L}_\text{val} - \mathcal{L}_\text{train}$ — a rising gap is the early-warning signal for overfitting | A small gap with high loss on both sets is underfitting, not good generalisation |
| **Batch Normalisation as regulariser:** the noise from estimating mean and variance from a mini-batch provides implicit regularisation similar to Dropout; using both BatchNorm and Dropout together often underperforms — BatchNorm's internal normalisation is disrupted by Dropout's random zeroing | "Can you use BatchNorm and Dropout together?" | Stacking BatchNorm + Dropout in every layer — in practice one or the other is sufficient; prefer BatchNorm for CNNs, Dropout for fully connected layers |
| **Data augmentation as regularisation:** random crops, flips, colour jitter, or Mixup applied during training increase the effective dataset size, directly reducing variance without touching the model architecture | "How does data augmentation relate to regularisation?" | "Data augmentation only helps with image data" — text augmentation (synonym replacement, back-translation), audio augmentation (SpecAugment), and tabular augmentation (Mixup on features) are all well-established |

---

## Ch.7 — Convolutional Neural Networks

| Must know | Likely asked | Trap to avoid |
|---|---|---|
| Why do CNNs need fewer parameters than dense nets? | Filters are shared across all spatial positions (weight sharing) — a 3×3 filter has just 9 weights regardless of image size | The dense equivalent of a conv layer grows quadratically with image resolution |
| What is translation equivariance? | The same filter response occurs wherever the pattern appears in the image | Pooling adds translation *invariance* — the response is similar regardless of exact position |
| Max pool vs average pool? | Max pool retains the strongest activation (was pattern present?); average pool retains overall energy | For classification, max pool is standard; for regression on spatial outputs, average pool can destroy signal |
| What does increasing filter depth (32→64→128) achieve? | More filters = more distinct patterns detected at that scale | Doubling filters doubles computation at that layer — computational cost grows linearly with depth but quadratically with resolution |
| What is the receptive field? | The region of the original input that influences a given neuron — grows with depth without increasing per-layer parameters | A neuron in layer 5 (all 3×3 kernels, stride 1) has receptive field $1 + 5 \times 2 = 11×11$ |
| **Depthwise separable convolution (MobileNet):** splits a standard conv into a **depthwise conv** (one filter per input channel, captures spatial patterns) + a **1×1 pointwise conv** (mixes channels); total compute $\approx \frac{1}{N} + \frac{1}{D_k^2}$ of a standard conv — roughly 8–9× cheaper for 3×3 kernels with 64 channels | "What makes MobileNet efficient?" | "Faster convolutions always sacrifice accuracy" — MobileNetV3 matches ResNet-50 on ImageNet at 4× fewer parameters; efficiency and accuracy are not fundamentally at odds |
| **Residual (skip) connections (ResNet):** $y = F(x) + x$ — the identity shortcut allows gradients to flow directly to early layers during backprop, making 50–200 layer networks trainable; the network learns the *residual* $F(x) = y - x$ rather than the full mapping, making each layer's task as small as possible | "Why do residual connections help with very deep networks?" | "Residual connections eliminate vanishing gradients" — they mitigate gradient shrinkage; very deep networks still benefit from careful initialisation and BatchNorm |

---

## Ch.8 — RNNs & LSTMs

| Must know | Likely asked | Trap to avoid |
|---|---|---|
| RNN recurrence equation: $\mathbf{h}_t = \tanh(\mathbf{W}_{hh}\mathbf{h}_{t-1} + \mathbf{W}_{xh}\mathbf{x}_t + \mathbf{b})$ | Why does the vanilla RNN suffer from vanishing gradient? (product of Jacobians decays exponentially in $T$) | "LSTM solves vanishing gradient by using attention" — wrong; it uses a gated cell state |
| What the four LSTM gates do (forget, input, candidate, output) | Why is the cell state $\mathbf{c}_t$ the key to long-term memory? (additive update avoids repeated multiplication) | Claiming `return_sequences=True` is always needed — it depends on whether you want step-by-step or a single final prediction |
| GRU uses two gates (reset, update) and no separate cell state | When would you choose GRU over LSTM? (training speed / parameter budget) | Shuffling time-series validation data — always split chronologically |
| `shuffle=False` in `train_test_split` for time series | How does gradient clipping fix exploding gradients? (caps the norm; doesn't fix vanishing) | Confusing sequence length $T$ with hidden size $H$ when asked "how do you increase model capacity?" |
| **Bidirectional RNN:** runs two RNNs in opposite directions over the sequence and concatenates hidden states at each step; doubles parameters and compute, but gives each position access to both past and future context — critical for NER and classification tasks | "When would you use a bidirectional RNN?" | "Bidirectional RNNs are always better" — they cannot be used autoregressively (generation, online inference) because future tokens are unavailable at the time of prediction; transformer encoder attention achieves the same effect more efficiently |
| **Teacher forcing:** during training, feed the ground-truth token from step $t-1$ as input at step $t$ instead of the model's own prediction; speeds convergence but creates **exposure bias** — the model never learns to recover from its own mistakes. **Scheduled sampling** gradually replaces teacher tokens with model predictions during training | "What is exposure bias and how do you mitigate it?" | "Teacher forcing makes the model more accurate" — it makes training faster but widens the train-inference distribution gap; inference errors compound because the model trained on gold inputs, not its own |

---

## Ch.9 — Metrics

| Must know | Likely asked | Trap to avoid |
|---|---|---|
| Confusion matrix: TP, TN, FP, FN — derive Precision and Recall from it | "When would you prefer a high-recall model over high-precision?" (FN is more costly: medical diagnosis, fraud detection) | "AUC = 0.9 means 90% accuracy" — AUC is a ranking probability, not accuracy |
| F1 = harmonic mean of P and R; why harmonic mean punishes extreme imbalance | Explain AUC-ROC in one sentence: P(model ranks positive above negative) | Reporting accuracy on imbalanced data — always check class distribution first |
| AUC-PR is preferred to AUC-ROC under class imbalance | What does a PR curve at the random baseline look like? (horizontal line at y = class prior) | Picking the decision threshold on the test set — threshold selection must use a separate validation split |
| Adjusted R² penalises adding useless features; plain R² always increases | Derive why quadratic errors penalise outliers more than MAE | Confusing MAPE's undefined behaviour at $y=0$ and asymmetric penalty with "MAPE is more robust than RMSE" |
| **Multi-class F1 averaging:** `macro` = mean per-class F1 (treats all classes equally, surfaces poor rare-class performance); `micro` = compute TP/FP/FN globally (dominated by frequent classes); `weighted` = macro weighted by class size. For imbalanced multi-class, `macro` is the most honest signal | "What's the difference between macro and weighted F1?" | Reporting `weighted F1 = 0.95` on a 10-class dataset where one class has 90% of samples — the score is almost entirely driven by the majority class; `macro F1` would expose the failure on minority classes |
| **Calibration:** a model is well-calibrated if among predictions at 80% confidence, 80% are actually correct. Measured by ECE (Expected Calibration Error) or reliability diagrams. Post-hoc calibration: Platt scaling (logistic regression on logits) or isotonic regression | "What is model calibration and why does it matter?" | "High accuracy implies good calibration" — neural networks are notoriously overconfident (high ECE) even at high accuracy; calibration is a separate property from discriminative performance |

---

## Ch.10 — Classical Classifiers

| Must know | Likely asked | Trap to avoid |
|---|---|---|
| Gini impurity formula: $1 - \sum p_c^2$ ; Gini=0 = pure, Gini=0.5 = maximally mixed (binary) | How does a decision tree choose the best split? (maximise information gain = impurity reduction weighted by node sizes) | "Decision trees can't overfit" — they are one of the easiest models to overfit; `max_depth` is essential |
| KNN stores the training data and votes at query time — no explicit training | Why does large $k$ increase bias? (predictions smooth over more training points, losing local structure) | "KNN doesn't need feature scaling" — it absolutely does; distance is scale-dependent |
| Decision Tree feature importance = total impurity reduction weighted by sample count at each split | When would you choose a Decision Tree over Logistic Regression? (need interpretable rules, non-linear decision boundary without feature engineering, mixed feature types) | Comparing training accuracy between a fully-grown tree and logistic regression — always compare on the test set |
| DT is a high-variance model; KNN at $k=1$ is also high-variance; increasing $k$ adds bias | What is the time complexity of KNN at inference? ($O(nd)$ brute force; use KD-Tree for large $n$) | Reporting only accuracy on an imbalanced test set (always use F1 or AUC — Ch.9) |
| **Curse of dimensionality in KNN:** in high dimensions, all pairwise distances converge toward the same value — the max-to-min distance ratio approaches 1, making nearest neighbours meaningless. Rule of thumb: KNN degrades past ~20–30 raw features without PCA/UMAP first | "How does the curse of dimensionality affect KNN?" | "Just add more training data to fix high-d KNN" — the required data grows exponentially with dimension; this is the *definition* of the curse |
| **Decision tree inference is $O(\text{depth})$ per sample** regardless of training set size — a key practical advantage over KNN's $O(nd)$ per query; however `max_depth` is essential for both speed and generalisation | "Compare the inference complexity of KNN and Decision Tree" | "Shallow trees are always interpretable" — a tree with `max_depth=3` is interpretable; a tree with `max_depth=30` and thousands of leaves is not, even though it is still technically a decision tree |

---

## Ch.11 — SVM & Ensembles

| Must know | Likely asked | Trap to avoid |
|---|---|---|
| SVM finds the max-margin hyperplane; support vectors are the only training points that determine the boundary | Why does the kernel trick work? (SVM dual only needs dot products — kernel computes them in a higher-dimensional space without constructing it) | "SVM is always better than logistic regression on high-dimensional data" — not true; LR is often competitive and faster |
| Bagging reduces variance by averaging independent models; boosting reduces bias by fitting residuals sequentially | Derive the ensemble variance formula: $\rho\sigma^2 + (1-\rho)\sigma^2/N$ — why does decorrelation matter? (the $\rho\sigma^2$ floor) | "Random Forest always needs scaling" — tree splits are based on rank/thresholds, not distances; scaling has no effect |
| XGBoost adds second-order Taylor expansion and regularisation on leaf weights; early stopping is essential | What is the difference between a learning rate in XGBoost and in neural networks? (in XGBoost, it shrinks each tree's contribution to prevent over-reliance on early trees) | Tuning XGBoost without early stopping — n_estimators becomes meaningless without a validation signal |
| OOB score in Random Forest is a free cross-validation estimate (built from the ~37% unsampled examples per tree) | When would you choose SVM over Random Forest? (high-dimensional, sparse data; clear margin structure; small datasets where kernel SVM scales OK) | Comparing untuned XGBoost against tuned Random Forest and concluding Random Forest is better |
| **SVM `C` parameter:** high $C$ → small margin, low bias, high variance (overfits); low $C$ → large soft margin, high bias, low variance (regularised). With RBF kernel, `C` and `gamma` interact strongly — always cross-validate both | "What does the C parameter control in SVM?" | "Higher C always better because it correctly classifies more training points" — training accuracy is not the goal; the soft margin exists precisely to accept some misclassifications for better generalisation |
| **Stacking vs bagging vs boosting:** bagging trains base models in parallel on bootstrap samples (Random Forest); boosting trains sequentially fixing residuals (XGBoost, AdaBoost); stacking trains a meta-learner on out-of-fold predictions from diverse base models | "When does stacking fail to beat bagging?" | "Stacking always beats bagging" — if the base models are highly correlated, the meta-learner gains nothing; the power of stacking comes from model diversity (e.g., RF + XGB + LR → LR meta-learner) |

---

## Ch.12 — Clustering

| Must know | Likely asked | Trap to avoid |
|---|---|---|
| K-Means objective: minimise within-cluster sum of squared distances | What are the two steps of Lloyd's algorithm? (assignment: each point → nearest centroid; update: each centroid → cluster mean) | "K-Means always finds the optimal clustering" — Lloyd's is a local optimiser; K-Means++ reduces but doesn't eliminate bad initialisation |
| DBSCAN core / border / noise definitions; cluster = maximal density-connected set | How do you choose ε for DBSCAN? (k-NN distance plot — sort points by distance to k-th neighbour, pick the knee) | "DBSCAN requires specifying k" — no, it requires ε and min_samples; k is only for the k-NN distance plot |
| HDBSCAN uses mutual reachability distance + MST + stability to extract clusters at variable density; only needs min_cluster_size | When would you prefer HDBSCAN over DBSCAN? (when clusters have varying density; when ε is hard to tune) | Comparing cluster label integers across K-Means runs — labels are arbitrary; compare by centroid features |
| K-Means assumes spherical clusters; DBSCAN handles arbitrary shapes but a single density | What happens to silhouette score at K=n (one point per cluster)? (undefined or 0 — every point is its own cluster, silhouette is meaningless) | Skipping feature scaling before K-Means or DBSCAN — both use Euclidean distance |
| **Hierarchical (agglomerative) clustering:** builds a dendrogram bottom-up by merging the two closest clusters at each step; linkage variants: single (minimum distance, prone to chaining), complete (maximum, compact clusters), **Ward** (minimises within-cluster variance — default choice). Does not require specifying $K$ upfront | "Compare single, complete, and Ward linkage" | "Hierarchical clustering scales to large datasets" — naive implementation is $O(n^3)$ in time and $O(n^2)$ in memory; only feasible for $n < 10{,}000$ without approximation |
| **Gaussian Mixture Models (GMM):** models clusters as $K$ Gaussians with learnable means, covariances, and mixing weights; trained by EM. Unlike K-Means, GMM gives **soft cluster membership** (probabilities) and handles elliptical clusters. Use BIC/AIC to select $K$ | "How does GMM differ from K-Means?" | "GMM is always better than K-Means because it's probabilistic" — GMM has $O(Kd^2)$ parameters for full covariance; with $d=100$ and $K=10$, fitting full covariance matrices requires enormous data; use diagonal covariance as a default |

---

## Ch.13 — Dimensionality Reduction

| Must know | Likely asked | Trap to avoid |
|---|---|---|
| PCA finds orthogonal directions of maximum variance; uses eigendecomposition (or SVD) of the covariance matrix | What is the explained variance ratio? ($\lambda_i / \sum \lambda_j$ — fraction of total variance explained by component $i$) | "PCA preserves distances" — PCA preserves variance, not pairwise distances; reconstruction error = dropped variance |
| t-SNE minimises KL divergence between high-d Gaussian similarities and low-d Student-t similarities; perplexity ≈ effective neighbourhood size | Why does t-SNE use a Student-t (heavy-tailed) distribution in low-d space? (to avoid the crowding problem — points would collapse if Gaussian were used in low-d) | "Cluster A is far from cluster B in t-SNE means they're very different" — distances between clusters are NOT meaningful in t-SNE; only topology is |
| UMAP constructs a fuzzy topological graph in high-d and finds a low-d embedding that minimises cross-entropy between the two graphs; has `transform()` for new data | Why is UMAP generally preferred over t-SNE for production pipelines? (faster, scales to millions, has transform(), better global structure, can be supervised) | Forgetting to standardise before PCA or UMAP — both use distances/variances and are scale-sensitive |
| **Feature selection vs feature extraction:** selection picks a subset of original features (Lasso, mutual information, RFE) — no transformation applied; extraction projects to new lower-dimensional space (PCA, UMAP). Use selection when original feature interpretability matters; use extraction when compactness matters more than interpretability | "When would you choose feature selection over PCA?" | "PCA is sufficient for all dimensionality reduction" — PCA is linear; nonlinear manifolds (image patches, molecular structure) require UMAP or kernel methods |
| **Kernel PCA:** applies PCA in a kernel-induced feature space without explicitly computing the high-dimensional transformation; RBF kernel maps effectively to infinite dimensions. Requires an $n \times n$ kernel matrix ($O(n^2)$ memory) and no clean `transform()` for unseen data | "How does kernel PCA differ from standard PCA?" | "Kernel PCA scales like PCA" — standard PCA is $O(nd^2)$ in the number of features; kernel PCA is $O(n^2)$ in the number of samples; prohibitive for $n > 10{,}000$ |

---

## Ch.14 — Unsupervised Metrics

| Must know | Likely asked | Trap to avoid |
|---|---|---|
| Silhouette: $s(i) = (b(i) - a(i)) / \max(a(i), b(i))$; range [−1, 1]; higher = better; $a(i)$ = mean intra-cluster distance, $b(i)$ = mean distance to nearest other cluster | What is a good silhouette score? (>0.5 is often cited, but highly dataset-dependent; always compare relative to other K values) | "Silhouette of 0 means bad clustering" — 0 means a point is on the boundary; the mean score across all K values is what matters |
| Davies-Bouldin: mean over clusters of best (intra / inter) ratio; lower is better; 0 is theoretical perfect | How does DBI differ from silhouette? (DBI uses centroids and mean intra-cluster distance to centroid; silhouette uses pairwise distances — DBI is faster, silhouette is more robust to cluster shape) | Applying DBI to DBSCAN — sklearn computes centroids from whatever cluster members it receives, so DBI technically runs; but DBI's centroid-based scatter calculations are semantically inappropriate for non-convex DBSCAN clusters and will produce misleading scores |
| ARI: corrected-for-chance fraction of concordant pairs; range [−1, 1]; requires ground truth | When would you use ARI in an unsupervised problem? (semi-supervised: you have partial labels or a known grouping to validate against; e.g., geographic regions, known customer segments) | Comparing ARI across datasets — it is relative; only ARI > ~0.3 is considered meaningful overlap |
| **Calinski-Harabasz Index (Variance Ratio Criterion):** ratio of between-cluster dispersion to within-cluster dispersion; higher is better; computed from centroids, so $O(Knd)$ — fast. Tends to favour convex, compact clusters; biased toward K-Means-style outputs | "Name three internal clustering metrics and how they differ" | "CH index and silhouette always agree on the optimal $K$" — they often disagree; CH penalises diffuse clusters more heavily, silhouette penalises misassignment; use both and triangulate |
| **Normalised Mutual Information (NMI):** measures how much knowing the cluster label reduces uncertainty about the true class label; range 0–1; AMI (Adjusted MI) corrects for chance like ARI does. Works even when cluster count differs from class count | "When would you use NMI vs ARI for cluster validation?" | "NMI > ARI means the clustering is better" — they capture different aspects (information content vs pair concordance); high NMI with low ARI can occur when cluster sizes are very unequal |

---

## Ch.15 — MLE & Loss Functions

| Must know | Likely asked | Trap to avoid |
|---|---|---|
| MSE derives from MLE under Gaussian noise: $\hat{\theta} = \arg\min\sum(y_i-\hat{y}_i)^2$ | Why is MSE equivalent to MLE for regression? (The $-\frac{(y-\hat{y})^2}{2\sigma^2}$ term in the Gaussian log-likelihood gives MSE when you drop constants and minimise) | "MSE is used for regression because it's differentiable" — the real reason is that it corresponds to the Gaussian noise model; MAE is also differentiable (almost everywhere) |
| Binary cross-entropy derives from Bernoulli MLE: $L=-\frac{1}{n}\sum[y\log\hat{p}+(1-y)\log(1-\hat{p})]$ | Why does MSE fail for classification near $\hat{p}=0.99$? (gradient $(2\hat{p}-2y)=−0.02$ is tiny; BCE gradient $-y/\hat{p}=-1.01$ is large — better training signal) | "MSE for classification is just less accurate" — it is a modelling error: the Gaussian noise assumption is wrong for a Bernoulli-distributed target |
| Huber loss: MSE for $|r|\leq\delta$, linear ($\delta|r|-\delta^2/2$) for $|r|>\delta$ — robust to outliers | When would you use Huber over MSE? (when the target has outliers that you don't want to penalise quadratically — e.g., luxury properties in a housing dataset) | "MAE is always more robust than MSE" — MAE is non-differentiable at 0; Huber combines robustness with differentiability |
| **KL divergence and cross-entropy:** $\text{CE}(p, q) = H(p) + D_\text{KL}(p \| q)$. Minimising cross-entropy against fixed ground-truth labels (constant $H(p)$) is equivalent to minimising KL divergence — which is why "minimise cross-entropy" and "maximise log-likelihood" are the same objective | "Show why minimising cross-entropy is equivalent to minimising KL divergence" | "KL divergence is symmetric" — $D_\text{KL}(p \| q) \ne D_\text{KL}(q \| p)$; the asymmetry matters in variational inference (forward KL is mean-seeking, reverse KL is mode-seeking) |
| **Focal loss:** $\text{FL}(p_t) = -(1-p_t)^\gamma \log(p_t)$. Down-weights the loss contribution of easy examples (where $p_t \to 1$) so training focuses on hard or rare examples. Introduced for object detection (RetinaNet) to address foreground/background imbalance | "When would you use focal loss instead of cross-entropy?" | "Focal loss replaces class weighting" — they address different problems; class weights correct for *frequency* imbalance, focal loss corrects for *difficulty* imbalance; they can and often should be combined |

---

## Ch.16 — TensorBoard

| Must know | Likely asked | Trap to avoid |
|---|---|---|
| TensorBoard callback: `histogram_freq` logs weight/gradient histograms; `update_freq='epoch'` logs Scalars each epoch; `write_graph=True` captures the computational graph | What does a weight histogram that stops changing across epochs tell you? (Early layers have vanishing gradients — the backward pass information is not reaching them; check for sigmoid/tanh activations in a deep network or missing skip connections) | "TensorBoard is only for TensorFlow" — PyTorch has `torch.utils.tensorboard.SummaryWriter`; same log format, same browser UI |
| Projector tab: logs a tensor $\mathbf{Z} \in \mathbb{R}^{n \times d}$ + optional metadata; renders via PCA or t-SNE in the browser; useful for validating that learned embeddings separate classes | When would you use the Projector tab? (When training embeddings — word2vec, sentence encoders, or any model where a hidden layer is meant to represent meaningful features; validate that same-class examples cluster) | `update_freq='batch'` — this logs a scalar per batch step, which can produce millions of events and make the UI unusable; always use `'epoch'` for long runs |
| Dead neurons: ReLU outputs 0 for all inputs when pre-activation is always negative; gradient is 0, weights never update; TensorBoard histograms show a spike at 0 for that layer's gradients | How do you fix dying ReLUs noticed in TensorBoard? (Switch to LeakyReLU or ELU which have non-zero gradient for negative inputs; use He initialisation; reduce learning rate to prevent large negative pre-activations early in training) | Confusing `write_graph=True` overhead with `histogram_freq` overhead — graph logging runs once at epoch 1; histograms run every `histogram_freq` epochs and scale with model size |
| **W&B vs TensorBoard vs MLflow:** TensorBoard is local, framework-native, zero setup; **Weights & Biases** logs to the cloud, adds team sharing, Bayesian sweep scheduling, and artifact versioning; **MLflow** is self-hosted and adds a model registry and experiment tracking database | "When would you choose W&B over TensorBoard?" | "TensorBoard is deprecated with modern ML tooling" — TensorBoard is still the default in TensorFlow/Keras and natively supported in PyTorch; W&B is additive and integrates with TensorBoard logs |
| **Diagnosing training with loss curves:** train ↓ val ↑ → overfitting; both plateau early → underfitting or LR too low; train loss unstable/spiky → LR too high or batch too small; val loss improves then suddenly jumps → data distribution shift, label noise, or checkpoint corruption | "What does it mean if validation loss is lower than training loss?" | "If training loss decreases, the model is improving" — always track both; a widening gap between train and val loss is the overfitting signal, not the training curve alone |

---

## Ch.17 — Sequences to Attention

| Must know | Likely asked | Trap to avoid |
|---|---|---|
| Attention is a soft dictionary lookup: softmax-weighted sum of values over all keys | "Explain attention without writing QKᵀ/√d_k" — you should be able to do this in one sentence | Jumping to "it's like a database query" without explaining that the output is a **blend**, not a single value |
| Dot product measures similarity; softmax makes argmax differentiable | "Why softmax instead of argmax?" | Saying "softmax normalises" without naming differentiability — which is the whole reason it's used |
| Attention is permutation-equivariant and therefore order-free by default | "Why do transformers need positional encoding?" | Saying "because of the architecture" — the correct answer is a one-line math fact about permutation equivariance |
| $Q$, $K$, $V$ are three *different* projections of the same input that play three different roles | "Why three projections and not one?" | Conflating Q and K; "the same vector is used three times" is wrong even in the simplest encoder |
| The attention matrix is $T \times T$, each row is a probability distribution | "What is the computational cost of attention?" — $O(T^2 \cdot d)$ | Answering with flop counts without naming the $T^2$ term — the reason long contexts are expensive |

---

## Ch.18 — Transformers

| Must know | Likely asked | Trap to avoid |
|---|---|---|
| Scaled dot-product attention formula and why $\sqrt{d_k}$ | Why not use L2 distance instead of dot product for similarity? | Saying attention is O(n) — it is O(n²) in sequence length |
| What Q, K, V represent and what each projection does | How multi-head attention differs from running attention once | Confusing `d_model` with `d_k` — they are different when H > 1 |
| Why positional encoding is necessary | Sinusoidal vs learned PE — tradeoffs | Saying the encoder has a causal mask — it does not |
| Encoder vs decoder — the one mask difference | What is RoPE and why is it preferred in modern LLMs? | Saying transformers have no vanishing gradient problem — they can still struggle with very deep stacks without ResNet-style residuals |
| Residual connections and LayerNorm — where and why | Pre-LN vs Post-LN — which is more stable and why? | Treating `MultiHeadAttention` as a black box without knowing its parameter count |
| **Flash Attention:** reorders the attention computation to tile Q/K/V into SRAM blocks, avoiding materialising the full $n \times n$ attention matrix in HBM (GPU DRAM); memory complexity drops from $O(n^2)$ to $O(n)$; wall-clock 2–4× faster on long sequences. Produces **exact** output, not an approximation | "How does Flash Attention speed up the transformer?" | "Flash Attention approximates attention to be faster" — it is IO-aware tiling of the *exact* computation; no approximation is made |
| **KV cache at inference:** during autoregressive decoding, keys and values for all prior tokens are stored and reused; only the new token's Q/K/V projections are computed each step. Memory cost grows with sequence length and batch size — at seq_len=8k, batch=32 on Llama-3-8B: ~16 GB, comparable to model weights | "What is the KV cache and why does it matter for serving?" | "KV cache has no cost" — it dominates VRAM at long sequences and large batch sizes; PagedAttention (vLLM) exists specifically to manage KV cache fragmentation |
| **Encoder-only vs decoder-only vs encoder-decoder:** encoder-only (BERT) — bidirectional attention, ideal for classification and embeddings; decoder-only (GPT, LLaMA) — causal attention, ideal for generation; encoder-decoder (T5, BART) — encoder processes input fully, decoder generates autoregressively with cross-attention to encoder outputs, ideal for seq2seq | "What architecture would you use for translation vs. classification vs. open-ended generation?" | "GPT-style models can't do classification" — any decoder-only model can do generative classification by predicting the class label token; it's less parameter-efficient than a BERT-style head but works |

---

## Ch.19 — Hyperparameter Tuning

| Must know | Likely asked | Trap to avoid |
|---|---|---|
| Difference between a parameter and a hyperparameter | "What hyperparameters would you tune first?" | Naming regularisation strength $\lambda$ as a parameter |
| Why learning rate is the single most important dial | "How do you pick a learning rate?" | Saying "I use whatever the paper uses" — run an LR-range test |
| How batch size trades off noise vs speed vs generalisation | "Why does large batch hurt test accuracy?" | Claiming large batch is always faster — yes per epoch, but fewer updates and sharper minima |
| Why you should initialise with He (ReLU) or Xavier (tanh) | "Can you initialise a deep net with all zeros?" | No — every neuron becomes identical; no learning |
| Dropout is train-only; turned off at inference | "What does dropout do at test time?" | Saying "it zeros activations" at test — it does **not** |
| Loss choice must match the target's noise model | "Why not use MSE for classification?" | Saying "because the outputs aren't continuous" — the real reason is vanishing gradients of MSE+sigmoid |
| When more data will vs won't help | "You have 10k more labels in budget — should you buy them?" | Always saying yes — it only helps in the high-variance regime |
| Random search > grid search in high dimensions | "How would you tune 8 hyperparameters efficiently?" | Grid over all of them — exponential cost |
| Early stopping + weight decay are the default regularisers | "You are overfitting — what's your first move?" | Jumping to dropout before looking at epochs / weight decay |
| AdamW ≠ Adam + L2 | "Why AdamW?" | Conflating them — see Ch.5 |

---

## Next Steps

This guide covers the core ML fundamentals. For advanced topics:
- **LLM fine-tuning**: See `notes/03-ai/FineTuning/`
- **Prompt engineering**: See `notes/03-ai/PromptEngineering/`
- **Multi-agent systems**: See `notes/MultiAgentAI/`
- **Diffusion models**: See `notes/MultimodalAI/DiffusionModels/`

These topics build on the foundations above — the same interview principles (know the math, anticipate follow-ups, avoid common traps) apply at scale.
