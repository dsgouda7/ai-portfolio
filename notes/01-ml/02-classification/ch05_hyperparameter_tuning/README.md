# Ch.5 — Hyperparameter Tuning

> **The story.** The idea that you should hold out data to honestly evaluate a model is old. In **1974** Mervyn Stone published the first rigorous treatment of **cross-validation**, showing that the leave-one-out estimate converges to the true generalisation error — the mathematical licence to trust your validation score. Practitioners in the **HPC era of the 1980s–90s** turned cross-validation into **grid search**: enumerate a finite set of hyperparameter values, fit the model at every combination, pick the winner. It was methodical, reproducible, and *exponentially expensive* — doubling the number of values per axis squares the cost. Then in **2012** James Bergstra and Yoshua Bengio published "Random Search for Hyper-Parameter Optimization" and shocked the community: random sampling of the same budget is almost always *better* than the grid. The intuition is clean — if only a handful of dimensions actually drive performance, random search spreads its budget across all of them while grid search wastes most of its budget on the irrelevant axes. That same year **Jasper Snoek, Hugo Larochelle, and Ryan Adams** operationalised **Bayesian optimisation** in Spearmint, building a Gaussian Process surrogate of the validation-score surface and directing each new trial to the region of highest *expected improvement*. Today Optuna, Ray Tune, and HuggingFace Trainer wrap all three strategies in a few lines of Python. Every time you see a data scientist staring at a validation curve and tweaking `C` by hand, they are doing **grad-student descent** — the painfully slow, manual version of what this chapter automates.
>
> **Where you are in the curriculum.** Ch.1–4 built FaceAI classifiers with hand-picked hyperparameters. Logistic Regression reached **88%** on Smiling, Random Forest hit **91%**, and SVM with an RBF kernel achieved **90%**. Averaged across all 40 CelebA attributes the best single model sits around **89%** — tantalizingly close to the 90% target, but 39 of 40 attributes are still individually below 90%. "Close" doesn't ship. This chapter provides the systematic search machinery to cross the line on every attribute.
>
> **Notation in this chapter.** $\lambda$ — generic hyperparameter (or hyperparameter vector); $\mathcal{H}$ — hyperparameter space (the grid, distribution, or search domain); $f(\lambda)$ — validation score as a function of hyperparameters; $k$ — number of folds in $k$-fold cross-validation; $\varepsilon_i$ — error on fold $i$; $\widehat{\text{CV}}_k$ — $k$-fold cross-validation error estimate; $|\mathcal{H}_j|$ — number of distinct values for the $j$-th hyperparameter; $n$ — number of random or Bayesian trials; $p$ — fraction of hyperparameter space that materially affects performance; $\text{EI}(\lambda)$ — Expected Improvement acquisition function; $\mu(\lambda), \sigma(\lambda)$ — Gaussian Process posterior mean and standard deviation at point $\lambda$; $f^+$ — best validation score observed so far.

---

## 0 · The Challenge — Where We Are

> 💡 **FaceAI Mission**: >90% avg accuracy across all 40 binary CelebA attributes
>
> | # | Constraint | Ch.1–4 Status | This Chapter |
> |---|-----------|---------------|-------------|
> | 1 | ACCURACY: >90% avg across 40 attrs | 89% avg — 39 attrs below 90% | Systematic search pushes every attr above 90% |
> | 2 | GENERALIZATION: unseen celebrity faces | Cross-validated per chapter | Nested CV locks in generalisation guarantees |
> | 3 | MULTI-LABEL: 40 simultaneous predictions | Separate model per attribute | Per-attribute threshold and hyperparameter tuning |
> | 4 | INTERPRETABILITY: explainable predictions | Feature importance (RF, SVM weights) | Retained — search doesn't change model class |
> | 5 | PRODUCTION: <200ms inference, reproducible | sklearn pipeline, <10ms | Tuning is offline; production model is unchanged |

**What we know so far:**
- ✅ Ch.1 — Logistic Regression: **88%** on Smiling; solid linear baseline on all 40 attributes
- ✅ Ch.2 — Random Forest: **91%** on Smiling; **87% average** across 40 attributes (best so far)
- ✅ Ch.3 — Evaluation Framework: F1-macro, ROC-AUC, confusion matrices, per-attribute breakdowns
- ✅ Ch.4 — SVM with RBF kernel: **90%** on Smiling; **89% average** across 40 attributes
- ❌ **But 39 of 40 attributes are still individually below 90%.** The goal requires ALL of them.

**What's blocking us:**

Every classifier in Ch.1–4 used hand-picked hyperparameters. Random Forest used `max_depth=10, n_estimators=100` because those are reasonable defaults. SVM used `C=10, γ=0.01` because a few manual trials looked good. These were guesses — educated guesses, but guesses. The model that reaches 91% on Smiling with those parameters might reach **93%** with `max_depth=15, n_estimators=200`. The model stuck at 79% on Attractive might reach 91% with `max_depth=12, n_estimators=150`.

> ⚠️ **Production crisis:** The product analytics team reports that FaceAI's **Attractive** attribute is at 79% — the lowest of all 40. Users who identify as attractive are being systematically mis-labelled. The app is getting negative reviews.
>
> **Root cause:** The Random Forest for Attractive was never tuned. It runs with the same defaults as the Smiling classifier, even though Attractive has different class balance and different feature importance patterns.
>
> **The fix:** Run systematic hyperparameter search per attribute. Different attributes have different optimal hyperparameters.

**What this chapter unlocks:**
- **$k$-fold cross-validation** as a principled error estimator — the mathematical foundation for all search strategies
- **Grid search**: exhaustive, reproducible, but exponential in cost
- **Random search** (Bergstra & Bengio 2012): the same budget, far better coverage of dimensions that actually matter
- **Bayesian optimisation** (Snoek et al. 2012): learn from past trials, direct the next trial to the highest-expected-improvement region
- **Constraint #1 ✅**: Push average accuracy from 89% → **>91%** and every individual attribute above 90%

```mermaid
graph LR
    A["Ch.1 LogReg\n88% Smiling\n85% avg"] --> B["Ch.2 RF\n91% Smiling\n87% avg"]
    B --> C["Ch.4 SVM\n90% Smiling\n89% avg"]
    C --> D["Ch.5 Tuned\n>91% avg\nAll 40 ≥ 90% ✅"]
    style D fill:#15803d,stroke:#e2e8f0,stroke-width:2px,color:#ffffff
```

---

## Animation

![animation](img/ch05-hyperparameter-tuning-needle.gif)

---

## 1 · Core Idea

**Hyperparameters are chosen; parameters are learned.** During training, gradient descent or impurity minimisation finds the model's weights and splits automatically. But the *structure* of the search — how deep the tree can grow, how many trees to average, how much to penalise complexity — is set before training starts. These are hyperparameters, and no learning algorithm sets them for you.

**The default is a compromise.** Scikit-learn's defaults (`max_depth=None`, `n_estimators=100`, `C=1.0`) are chosen to work reasonably on a wide range of tabular problems. CelebA face attributes are not a wide range of tabular problems — they are 40 specific binary classification tasks on 40-dimensional HOG feature vectors extracted from 64×64 face crops. The optimal hyperparameters differ by attribute. Tuning them per-attribute is not over-engineering; it is engineering.

**Three strategies, increasing in sophistication:**

1. **Grid search** — define a finite grid $\mathcal{H} = \mathcal{H}_1 \times \mathcal{H}_2 \times \ldots \times \mathcal{H}_d$ and evaluate every combination. Guaranteed to find the global best *on the grid*. Cost grows multiplicatively: add one more hyperparameter axis with 5 values and you 5× the compute.

2. **Random search** — sample $n$ configurations uniformly from $\mathcal{H}$. Bergstra & Bengio (2012) showed that if only a small fraction of the hyperparameter dimensions drive performance, random search has *exponentially better* expected performance than grid search for the same budget. Their bound: with $n = \lceil \log(0.05) / \log(1-p) \rceil$ trials, you have 95% probability of hitting the $p$-fraction of space that matters.

3. **Bayesian optimisation** — maintain a probabilistic surrogate $\hat{f}(\lambda)$ of the validation score surface and use it to compute an *acquisition function* (Expected Improvement) that trades off exploitation (try near the current best) against exploration (try where uncertainty is high). Each new trial makes the surrogate sharper. After 30–50 trials Bayesian opt typically matches what grid search needs 200+ trials to find.

---

## 2 · Running Example — Pushing Attractive from 79% to >90%

The **Attractive** attribute is FaceAI's weakest link: 79% accuracy with default Random Forest. Let us walk through every tuning step, from the first cross-validation fold to the winning hyperparameter combination.

**Dataset slice:** 202,599 CelebA images. Attractive is positive for 83,985 (41.4%) — slightly imbalanced but workable. Feature vector: 4,096-dimensional HOG descriptor. Model: Random Forest.

**Baseline:** `RandomForestClassifier(max_depth=10, n_estimators=100, random_state=42)` — 5-fold CV accuracy = **79.1%**.

**What we suspect:** The forest is shallow and small. Attractive is a nuanced label — it correlates with many subtle features simultaneously (facial symmetry, skin tone, hair style). A deeper forest with more trees should capture more of those interactions.

**The plan:**
- Grid: `max_depth ∈ {3, 5, 10}`, `n_estimators ∈ {50, 100, 200}` — 9 combinations, each evaluated with 5-fold CV (45 total fits)
- Random: 60 trials sampling `max_depth ~ Uniform(3, 20)`, `n_estimators ~ Uniform(50, 500)` — same compute budget as grid but denser coverage
- Bayesian: 40 trials guided by Expected Improvement on the same search space

**Target:** 5-fold CV accuracy > 90% — at which point we retrain on the full dataset and deploy.

---

## 3 · Tuning Pipeline at a Glance

Before the math, here is the full pipeline you will be running. Each stage has a deeper treatment in the sections below.

```
STAGE 1 — TRAIN / VALIDATION / TEST SPLIT
──────────────────────────────────────────
Hold out 10% as a final test set (never touched during tuning).
Remaining 90% is split k=5 ways for all CV.

STAGE 2 — CHOOSE SEARCH STRATEGY
──────────────────────────────────
   Option A: Grid search  → enumerate all λ ∈ H1 × H2 × … × Hd
   Option B: Random search → sample n configs from H
   Option C: Bayesian opt  → propose λ_next = argmax EI(λ)

STAGE 3 — CROSS-VALIDATION FOR EACH CANDIDATE λ
──────────────────────────────────────────────────
   For each fold i = 1 … k:
      Train on folds {1…k} \ {i}
      Evaluate on fold i → record ε_i
   CV_error = (1/k) Σ ε_i    [see §4.1 for explicit numbers]

STAGE 4 — SELECT BEST λ* = argmin CV_error(λ)

STAGE 5 — FINAL EVALUATION ON HELD-OUT TEST SET
──────────────────────────────────────────────────
   Retrain on all 90% training data with λ*
   Evaluate on the 10% test set → unbiased accuracy estimate
```

The key discipline: the test set is untouched until the very end. Evaluating candidate hyperparameters on the test set is one of the most common and most costly mistakes in applied ML (§9).

**Why the train/val/test split is non-negotiable.** A model evaluated on the test set during tuning will appear to perform better than it actually does — because the tuning process implicitly learns the noise in the test set. If you tune 200 hyperparameter combinations and pick the one with the best test score, you have effectively overfit to 200 independent "decisions" about the test set, even though no gradient steps touched it. The resulting model's test accuracy is a biased-upward estimate of real-world performance. The fix is simple: reserve the test set for one final evaluation only. Use $k$-fold CV for all intermediate selection decisions.

**The compute arithmetic.** For 40 CelebA attributes with $n=60$ random trials each, 5-fold CV, and 12-second fit time per RF: $40 \times 60 \times 5 \times 12\text{s} = 144,000\text{s} \approx$ **40 hours** of serial compute. With `n_jobs=-1` (all 8 cores), this becomes **~5 hours** — a practical overnight run. This is why tuning is typically done offline (Constraint #5 satisfied: tuning is not on the inference path).

---

## 4 · The Math

### 4.1 · Cross-Validation — Estimating Error Honestly

$k$-fold cross-validation partitions the training data into $k$ equal folds and rotates the validation role through each fold in turn. For $k=5$ and a dataset of $N$ rows:

$$\text{Fold size} = \lfloor N/5 \rfloor, \quad \text{Training size per fold} = 4 \cdot \lfloor N/5 \rfloor$$

**Which fold is validation in each round:**

| Round | Validation Fold | Training Folds |
|------:|----------------|----------------|
| 1 | Fold **1** | 2, 3, 4, 5 |
| 2 | Fold **2** | 1, 3, 4, 5 |
| 3 | Fold **3** | 1, 2, 4, 5 |
| 4 | Fold **4** | 1, 2, 3, 5 |
| 5 | Fold **5** | 1, 2, 3, 4 |

Every row is used exactly once for validation and four times for training — no data is wasted, and no row influences its own error estimate.

**Minimal worked example — $k=5$, $N=100$, fold assignment:**

> 💡 **This is pedagogical scaffolding** — the toy 100-row fold assignment lets you verify CV logic by hand (20 rows per fold, 80 for training, 20 for validation) before running on 182k rows. Once you understand the rotation on 100 rows, scaling to 182k is just arithmetic.

For exactly 100 training samples and $k=5$ folds, each fold contains $\lfloor 100/5 \rfloor = 20$ rows:

| Fold | Sample indices | Role |
|-----:|----------------|------|
| 1 | rows 1–20 | Validation in Round 1 · Training in Rounds 2–5 |
| 2 | rows 21–40 | Validation in Round 2 · Training in Rounds 1, 3–5 |
| 3 | rows 41–60 | Validation in Round 3 · Training in Rounds 1–2, 4–5 |
| 4 | rows 61–80 | Validation in Round 4 · Training in Rounds 1–3, 5 |
| 5 | rows 81–100 | Validation in Round 5 · Training in Rounds 1–4 |

Training set size per round = 80 rows (80% of data). Every row contributes to exactly 4 training rounds and exactly 1 validation round.

**The error formula:**

$$\widehat{\text{CV}}_k = \frac{1}{k} \sum_{i=1}^{k} \varepsilon_i$$

where $\varepsilon_i$ is the classification error (1 − accuracy) on fold $i$.

**Toy arithmetic — fold errors $[0.12,\; 0.14,\; 0.11,\; 0.15,\; 0.13]$:**

$$\widehat{\text{CV}}_5 = \frac{0.12 + 0.14 + 0.11 + 0.15 + 0.13}{5} = \frac{0.65}{5} = \mathbf{0.13}$$

Cross-validated accuracy $= 1 - 0.13 = \mathbf{87\%}$. Fold std $= 0.015$, giving a 95% CI of roughly $87\% \pm 1.7\%$. Two configurations whose CV scores differ by less than this margin are statistically indistinguishable.

**Worked example — Attractive attribute, baseline RF:**

| Fold | Validation accuracy | $\varepsilon_i$ |
|-----:|--------------------:|----------------:|
| 1 | 78.8% | 0.212 |
| 2 | 79.4% | 0.206 |
| 3 | 80.1% | 0.199 |
| 4 | 78.5% | 0.215 |
| 5 | 78.7% | 0.213 |

$$\widehat{\text{CV}}_5 = \frac{0.212 + 0.206 + 0.199 + 0.215 + 0.213}{5} = \frac{1.045}{5} = \mathbf{0.209}$$

Cross-validated error = 20.9%, meaning **5-fold CV accuracy = 79.1%** — consistent with what you would see by training on all data and evaluating on a fresh test set, but computed without touching the test set at all.

> ⚡ **Priority: Intuition over calculation.** The test: Can you explain why every row must be used exactly once for validation and exactly four times for training? If yes, you understand cross-validation. The arithmetic (0.212 + 0.206 + … = 1.045) is secondary.

**Why $k=5$ or $k=10$?** There is a bias–variance trade-off in $k$ itself. Small $k$ (e.g., $k=2$) uses less data for training each fold → the error estimate is pessimistic (high bias). Large $k$ (e.g., $k=N$, leave-one-out) uses nearly all data for training → low bias but high variance across folds because each fold's test set is a single row. $k=5$ or $k=10$ is the empirical sweet spot for most datasets; Stone (1974) proved leave-one-out is asymptotically equivalent to AIC, but for practical sample sizes 5-fold is more stable.

**Confidence interval on the CV estimate.** The $k$ fold errors $\varepsilon_1, \ldots, \varepsilon_k$ are not independent (they share training data), so the standard error $\hat{\sigma} / \sqrt{k}$ slightly underestimates uncertainty. In practice, the corrected 95% confidence interval is:

$$\widehat{\text{CV}}_k \pm t_{k-1, 0.025} \cdot \hat{\sigma}_\varepsilon \cdot \sqrt{\frac{1}{k} + \frac{n_\text{val}}{n_\text{train}}}$$

where $\hat{\sigma}_\varepsilon = \text{std}(\varepsilon_1, \ldots, \varepsilon_k)$ and the Nadeau–Bengio correction factor $\sqrt{1/k + n_\text{val}/n_\text{train}}$ accounts for the overlap in training sets. For the Attractive example: $\hat{\sigma}_\varepsilon = 0.006$, correction factor $\approx 0.50$, $t_{4, 0.025} = 2.78$, so the 95% CI is $0.209 \pm 2.78 \times 0.006 \times 0.50 = 0.209 \pm 0.008$. In other words: **CV accuracy = 79.1% ± 0.8%**. Two configurations whose CV scores differ by less than this margin should be considered statistically equivalent.

---

### 4.2 · Grid Search — Exhaustive but Exponential

Grid search defines a finite candidate set for each hyperparameter and evaluates every combination:

$$\lambda^* = \underset{\lambda \in \mathcal{H}_1 \times \ldots \times \mathcal{H}_d}{\arg\max}\; \widehat{\text{CV}}_k(\lambda)$$

**Complexity formula:**

$$\text{Total model fits} = \left(\prod_{j=1}^{d} |\mathcal{H}_j|\right) \times k$$

**Worked example — RF on Attractive, 2 hyperparameters, $k=5$:**

$$\mathcal{H}_1 = \{\texttt{max\_depth}: 3, 5, 10\} \quad |\mathcal{H}_1| = 3$$
$$\mathcal{H}_2 = \{\texttt{n\_estimators}: 50, 100, 200\} \quad |\mathcal{H}_2| = 3$$

$$\text{Total fits} = 3 \times 3 \times 5 = \mathbf{45}$$

At approximately 12 seconds per fit (RF on 5,000 CelebA samples), grid search costs **~9 minutes** — very manageable for this 2-axis grid. Add a third axis (`min_samples_leaf ∈ {1, 2, 5}`) and the cost triples to 27 minutes. Add a fourth axis and you are at 81 minutes. This multiplicative blowup is the defining limitation of grid search.

---

### 4.3 · Random Search — The Bergstra & Bengio Result

Instead of evaluating every grid point, sample $n$ configurations independently from the search space:

$$\lambda_1, \lambda_2, \ldots, \lambda_n \overset{\text{i.i.d.}}{\sim} \text{Uniform}(\mathcal{H})$$

**The key probability result.** Suppose only a fraction $p$ of the hyperparameter space produces near-optimal performance (i.e., within, say, 2% of the best achievable accuracy). What is the probability that at least one of $n$ random trials lands in that region?

$$P(\text{at least one trial in good region}) = 1 - (1-p)^n$$

**Solve for $n$ to achieve 95% confidence with $p = 0.05$:**

$$1 - (1 - 0.05)^n \geq 0.95$$
$$(0.95)^n \leq 0.05$$
$$n \geq \frac{\log 0.05}{\log 0.95} = \frac{-2.996}{-0.05129} \approx 58.4$$

So **$n = 59$ trials** is enough for 95% confidence. Rounding to $n = 60$ (as Bergstra & Bengio stated) gives:

$$P = 1 - (0.95)^{60} = 1 - e^{60 \ln 0.95} = 1 - e^{-3.077} \approx 1 - 0.0461 = \mathbf{0.954}$$

**Probability table — how $n$ trials affect coverage ($p = 0.05$, i.e., top 5% of the space is near-optimal):**

| $n$ (trials) | $(0.95)^n$ | $P = 1-(0.95)^n$ | Interpretation |
|-------------:|-----------:|------------------:|----------------|
| 5 | 0.7738 | **22.6%** | Weak — 4 in 5 budgets miss the good region |
| 10 | 0.5987 | **40.1%** | Coin-flip odds; unreliable for production |
| 20 | 0.3585 | **64.2%** | Better than even, but still fails 1 in 3 runs |
| 60 | 0.0461 | **95.4%** | ✅ Bergstra & Bengio's 95% guarantee |

Verification for $n=5$: $(0.95)^5 = 0.95^2 \cdot 0.95^2 \cdot 0.95 = 0.9025 \times 0.9025 \times 0.95 = 0.8145 \times 0.95 = 0.7738$, so $P = 1 - 0.7738 = 0.2262$. The jump from $n=20$ (64%) to $n=60$ (95%) is why practitioners use 60 as the default rather than stopping at 20.

> 💡 **The critical insight.** This result holds regardless of how many hyperparameter dimensions you have. A 10-dimensional search space with $p = 5\%$ effective volume still requires only 60 trials to find a good region with 95% probability. Grid search on 10 dimensions with 5 values each requires $5^{10} = 9.7$ million trials. Random search beats grid search by a factor of $9.7 \times 10^6 / 60 \approx 160,000\times$ — with the same performance guarantee — whenever only a few dimensions matter.

> ⚡ **Priority: Intuition over calculation.** Can you explain why random search works without memorizing $(1-p)^n$? The intuition: if only 2 of 10 hyperparameters actually matter, random search spreads its budget across both important axes while grid search wastes most trials on the 8 irrelevant ones. That dimension-independence is the whole game.

**Why only a few dimensions matter.** For the Attractive attribute, experiments show that `max_depth` and `n_estimators` together explain ~85% of the variance in CV accuracy. `min_samples_leaf`, `max_features`, and `bootstrap` contribute almost nothing above their defaults. Grid search wastes its budget testing all combinations of the irrelevant dimensions; random search automatically covers the important ones.

---

### 4.4 · Bayesian Optimisation — Learning as You Search

Grid and random search treat each trial as independent. Bayesian optimisation treats the trials as observations of an unknown function $f(\lambda)$ and builds a probabilistic model (the *surrogate*) of the entire landscape.

**The Gaussian Process surrogate.** A Gaussian Process (GP) defines a distribution over functions. Given $t$ observed trials $\{(\lambda_i, f_i)\}_{i=1}^{t}$, the GP posterior gives at any un-tried point $\lambda$:
- $\mu(\lambda)$ — the expected validation score
- $\sigma(\lambda)$ — the uncertainty in that expectation

Points near previously observed high-scoring regions have high $\mu$ (exploitation). Points far from any observation have high $\sigma$ (exploration). The acquisition function balances both.

**Expected Improvement (EI) acquisition function:**

**Intuition first — what "expected improvement" means in plain English.**

EI balances **exploitation** (pick where $\mu$ is high = known-good regions) vs **exploration** (pick where $\sigma$ is high = uncertain/promising-unknown). High EI means either:
- The surrogate expects a high score there ($\mu$ is near or above current best $f^+$), OR
- The surrogate is very uncertain there ($\sigma$ is large), so there's a real chance of discovering something better

At a point where $\mu(\lambda) = 88\%$ and $\sigma(\lambda) = 3\%$, the GP says: "I expect this configuration to score 88% but I'm uncertain by ±3%." If $f^+ = 87\%$, then EI is positive and meaningful — there's a real probability this configuration beats the current best. At a point where $\mu(\lambda) = 82\%$ and $\sigma(\lambda) = 0.5\%$ (near a previously tested bad region), EI is essentially zero — the model is confident this configuration is worse than the current best.

**Now the closed-form expression** (you don't need to memorize this):

$$\text{EI}(\lambda) = \mathbb{E}\!\left[\max\!\left(f(\lambda) - f^+,\, 0\right)\right]$$

where $f^+$ is the best validation score observed so far. Under the GP posterior, this expectation has a closed form:

$$\text{EI}(\lambda) = (\mu(\lambda) - f^+)\,\Phi(Z) + \sigma(\lambda)\,\phi(Z), \quad Z = \frac{\mu(\lambda) - f^+}{\sigma(\lambda)}$$

where $\Phi$ and $\phi$ are the standard normal CDF and PDF respectively.

**The Bayesian search loop:**

```
1.  Run a few random initial trials (typically 10–20) to seed the surrogate.
2.  Fit the GP (or TPE in Optuna) to all observed (λ, f(λ)) pairs.
3.  Compute EI(λ) across a dense grid of candidate points (fast — evaluating
    the GP is cheap; training the actual model is expensive).
4.  Select λ_next = argmax EI(λ).
5.  Train and evaluate the model at λ_next — get f(λ_next).
6.  Add (λ_next, f(λ_next)) to the history.
7.  Go to step 2 and repeat until budget exhausted.
```

After 30–50 trials, Bayesian opt reliably finds hyperparameter configurations that grid search would need 200+ trials to match.

---

### 4.5 · Nested CV — Unbiased Evaluation After Tuning

A single loop of $k$-fold CV selects the best hyperparameters. But if you then report "our CV accuracy is 90.3%" as the final metric, you are reporting the score on the *same* data that selected the winner — an optimistic estimate. Nested cross-validation fixes this by separating the selection decision from the evaluation decision with two concentric loops:

```
Outer loop (k_outer = 5 folds):   evaluates the *tuning procedure*
└── Inner loop (k_inner = 5 folds):  selects λ* within the outer training set

For each outer fold i:
    inner_train  = outer_train \ outer_val_i          (80% of data)
    inner_val    = outer_val_i                         (20% of data — never seen during tuning)
    λ*_i = argmax CV_inner(λ)                          (inner loop picks best config)
    ε_i  = error(model(λ*_i), inner_val)               (outer fold gives unbiased estimate)

Final estimate = (1/k_outer) Σ ε_i
```

**Why it matters for FaceAI.** If we run grid search on all 40 attributes and pick the winning configuration *per attribute* using the same CV fold that produced the winning score, the reported 90.3% already reflects the best fold. Nested CV reports the accuracy you would expect on truly unseen faces — typically 0.5–1.5% lower than the inner-loop optimum. For a production system, the nested estimate is what you promise to users.

**Practical cost.** Nested CV with $k_\text{outer}=5$, $k_\text{inner}=5$, and a 9-point grid multiplies total fits by $k_\text{outer}$: $9 \times 5 \times 5 = 225$ fits per attribute. For 40 attributes at 12 seconds each: $225 \times 40 \times 12\text{s} \approx$ **30 hours** serial compute. With `n_jobs=-1`: **~4 hours**. Worth it for the final reported metric; overkill during early exploratory tuning.

**sklearn implementation:**

```python
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

inner_cv = KFold(n_splits=5, shuffle=True, random_state=42)
outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)

clf = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid={"max_depth": [3, 5, 10], "n_estimators": [50, 100, 200]},
    cv=inner_cv, scoring="accuracy", n_jobs=-1
)
# nested estimate — unbiased
nested_scores = cross_val_score(clf, X_attractive, y_attractive, cv=outer_cv, scoring="accuracy")
print(f"Nested CV accuracy: {nested_scores.mean():.3f} ± {nested_scores.std():.3f}")
# → Nested CV accuracy: 0.895 ± 0.011   (vs inner-loop 0.903 — 0.8pp optimism penalty)
```

---

### 4.6 · Strategy Comparison — Choosing the Right Search for Your Budget

The three strategies occupy different points in the cost–quality trade-off:

| Property | Grid Search | Random Search | Bayesian Opt |
|----------|------------|---------------|-------------|
| **Best configuration found** | Best on grid (guaranteed) | Best random sample | Near-global (highest EI) |
| **Scales with dimensionality** | Exponential ✗ | $O(n)$ ✓ | $O(n \log n)$ ✓ |
| **Learns from previous trials** | No | No | Yes ✓ |
| **Reproducible** | Yes (grid is fixed) | Yes (with seed) | Yes (with seed + sampler) |
| **Compute per trial** | 1 model fit | 1 model fit | 1 model fit + surrogate update |
| **Implementation complexity** | Low | Low | Medium (Optuna / scikit-optimize) |
| **When to use** | ≤ 3 axes, ≤ 5 vals each | > 3 axes, fast eval | > 3 axes, expensive eval |

**Decision flowchart for FaceAI:**

```
How many hyperparameter axes?
├─ 1–2 axes, ≤ 5 values each: → GRID SEARCH
│    (9 combinations × 5-fold = 45 fits → ~9 min for RF)
├─ 3–5 axes OR expensive evaluation: → RANDOM SEARCH (n=60)
│    (60 trials × 5-fold = 300 fits → ~60 min for RF)
└─ Any axes, very expensive evaluation: → BAYESIAN OPT (n=30)
     (30 trials × 5-fold = 150 fits + ~1s surrogate → ~30 min for RF)
```

**Practical rule for this chapter:** The Attractive attribute RF uses only 2 axes → grid search. When extending to SVM (4+ axes: $C, \gamma$, kernel, class\_weight) → switch to random search with $n = 60$.

---

## 5 · Search Strategy Arc

### Act 1 — Manual Tuning: Expert Intuition, $O(1/\text{expert})$ Scaling

You hand-pick `max_depth=10, n_estimators=100` based on past experience. For Smiling (a clear binary signal) this works: 91%. For Attractive (a nuanced, culturally mediated label) this fails: 79%. Manual tuning is $O(1/\text{expert})$ — it scales only as well as the practitioner's intuition, which is attribute-specific and non-transferable.

**When to use it:** Never as your only strategy. Useful to narrow the search space before running automated search.

### Act 2 — Grid Search: Exhaustive but Exponential

Define a $3 \times 3$ grid, evaluate all 9 × 5-fold = 45 fits. You are guaranteed to find the best configuration *on the grid*. The limitation is not correctness but scale: a $3^{10}$ grid for 10 hyperparameters requires 59,049 × 5 = 295,245 fits. Grid search is the right tool when the search space is small (≤ 2–3 axes, ≤ 5 values each) and compute is not the bottleneck.

**The exponential wall:** Each additional axis with $m$ values multiplies the cost by $m$. At $d=5$ axes and $|\mathcal{H}_j|=5$, you need $5^5 = 3,125$ fits. At $d=10$, that is $5^{10} \approx 10^7$ fits. Grid search hits the wall fast.

### Act 3 — Random Search: Surprisingly Competitive

60 random trials on the same 2D space for Attractive. No guarantee of finding the exact best — but statistical assurance (95% confidence) of landing in the 5% of space that contains near-optimal configurations. For 3 or more axes, random search dominates grid search. The Bergstra & Bengio bound makes this rigorous.

**The dimension-independence property:** Adding a third irrelevant hyperparameter axis does not change the number of trials needed ($n = 60$) to find a good region. It only changes the size of the unimportant dimensions — which random search ignores for free because it never enumerates them.

### Act 4 — Bayesian Optimisation: Best Sample Efficiency

30 trials guided by Expected Improvement. The surrogate model learns the shape of the validation-score landscape after ~10 trials and starts directing new trials to regions of high expected improvement. Sample efficiency is 2–5× better than random search in practice — it requires 30 trials to match what random search finds in 60.

**When to use it:** Expensive evaluations (deep neural network training, large datasets) where even 60 trials is too many. For fast-evaluating models like RF on a 5,000-row dataset, random search is sufficient and simpler to implement.

### The Four-Act Arc — Summary

| Act | Strategy | Cost | Sample efficiency | Best for |
|----:|---------|------|------------------|---------|
| 1 | Manual tuning | O(1/expert) | None | Establishing rough search ranges |
| 2 | Grid search | $\prod |\mathcal{H}_j| \times k$ | Low on high-$d$ spaces | ≤ 3 axes, small grids |
| 3 | Random search | $n \times k$ | High when few dims matter | > 3 axes, fast evaluation |
| 4 | Bayesian opt | $n_\text{seed} \times k$ + $n_\text{guided} \times k$ | Highest | > 3 axes, expensive evaluation |

> 💡 **Rule of thumb for FaceAI:** Start with Act 2 (grid, 2 axes per attribute). If any attribute is still below 90% after grid search, escalate to Act 3 (random, 60 trials). Only escalate to Act 4 if per-trial compute exceeds 5 minutes — which happens in the Neural Networks track, not here.

---

## 6 · Full Grid Search Run — Attractive Attribute, RF, 3×3 Grid

**Setup:** `RandomForestClassifier(random_state=42)`, 5-fold CV, metric = accuracy.
Search space: `max_depth ∈ {3, 5, 10}`, `n_estimators ∈ {50, 100, 200}`.

**Results — all 9 combinations, mean 5-fold CV accuracy:**

| `max_depth` \ `n_estimators` | 50 | 100 | 200 |
|-----------------------------:|---:|----:|----:|
| **3** | 79.4% | 80.1% | 80.5% |
| **5** | 82.7% | 83.8% | 84.6% |
| **10** | 86.2% | 88.1% | **90.3%** |

**Winner:** `max_depth=10, n_estimators=200` — **90.3% CV accuracy**. That is a +11.2% improvement over the baseline (`max_depth=10, n_estimators=100`) which scored 79.1%.

**Reading the table:**

- *Depth drives accuracy more than tree count.* Moving from `max_depth=3` to `max_depth=10` (same `n_estimators=100`) gains **8 percentage points**: 80.1% → 88.1%. Moving from 50 to 200 trees (same `max_depth=10`) gains only **4 points**: 86.2% → 90.3%. The feature interactions in Attractive require depth, not breadth.
- *Diminishing returns on trees.* The 100→200 tree gain is +2.2% at depth 10; the 50→100 gain is also +1.9%. Doubling from 200 to 400 would likely yield < 1% additional gain.
- *Shallow trees plateau fast.* At `max_depth=3`, going from 50 to 200 trees gains only +1.1%. Averaging more weak trees doesn't compensate for each tree being too shallow to capture the signal.

> ⚡ **What this demonstrates — intuition over calculation.** The test: Can you explain why depth matters more than tree count without referring to the specific 90.3% or 80.1% numbers? If yes, you understand the bias-variance trade-off for this attribute. The arithmetic is evidence, not the concept.

**Deploying the winner:**

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

best_rf = RandomForestClassifier(
    max_depth=10,
    n_estimators=200,
    random_state=42,
)
cv_scores = cross_val_score(best_rf, X_attractive, y_attractive, cv=5, scoring='accuracy')
print(f"CV accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
# → CV accuracy: 0.903 ± 0.008
```

**Random search finds the same winner with more coverage:**

Running 60 random trials (sampling `max_depth ~ Uniform(3, 20)`, `n_estimators ~ Uniform(50, 500)`) produces a best configuration of approximately `max_depth=12, n_estimators=230` with **90.8% CV accuracy** — slightly better than the grid's 90.3%, because the continuous search space includes values between the grid points (e.g., `max_depth=11`, `n_estimators=175`). The grid search was constrained to 3 discrete values per axis; random search is not.

**Bayesian optimisation finds a near-optimal configuration fastest:**

After 10 random seed trials + 20 EI-guided trials, Bayesian opt converges to approximately `max_depth=13, n_estimators=220` — **90.9% CV accuracy** in only 30 total trials (vs 45 grid fits or 60 random trials). The surrogate model identified after trial 15 that the high-accuracy region is at `max_depth ≥ 10` and `n_estimators ≥ 180`, and directed all subsequent trials there.

**Summary — Attractive attribute, all three strategies:**

| Search Strategy | Best `max_depth` | Best `n_estimators` | CV Accuracy | Total fits |
|----------------|-----------------|---------------------|------------|-----------|
| Default (no search) | 10 | 100 | 79.1% | 5 |
| Grid search | 10 | 200 | 90.3% | 45 |
| Random search ($n=60$) | 12 | 230 | 90.8% | 300 |
| Bayesian opt ($n=30$) | 13 | 220 | 90.9% | 150 |

All three strategies beat the 90% target. For a 2-axis grid, the difference is small — grid search is fine here. The advantage of random and Bayesian search emerges clearly when extending to 4+ axes (adding `min_samples_leaf`, `max_features`, `class_weight`).

---

## 7 · Key Diagrams

### Diagram 1 — Three Search Strategies on the Same Space

```mermaid
flowchart TD
    SPACE["Hyperparameter Space\nmax_depth × n_estimators\n(continuous, not just 3×3)"]

    SPACE --> GRID["Grid Search\n9 fixed combinations\nExhaustive on the grid\n45 model fits"]
    SPACE --> RAND["Random Search\n60 independent samples\nCovers more of each axis\n60 model fits"]
    SPACE --> BAYES["Bayesian Optimisation\n10 random seed + 20 EI-guided\nLearns the landscape\n30 model fits"]

    GRID --> WINNER["Best on grid:\nmax_depth=10, n_estimators=200\n→ 90.3% CV accuracy"]
    RAND --> WINNER
    BAYES --> WINNER2["Near-best after 30 trials:\nmax_depth=11, n_estimators=185\n→ 90.5% CV accuracy"]

    WINNER --> DEPLOY["Retrain on full training set\nEval on held-out test set\n→ FaceAI Attractive ≥ 90% ✅"]
    WINNER2 --> DEPLOY

    style DEPLOY fill:#15803d,color:#ffffff,stroke:#15803d
    style BAYES  fill:#1e3a8a,color:#ffffff,stroke:#1e3a8a
    style WINNER fill:#b45309,color:#ffffff,stroke:#b45309
    style WINNER2 fill:#b45309,color:#ffffff,stroke:#b45309
```

### Diagram 2 — Strategy Comparison Arc

```mermaid
flowchart TD
    START(["Same search space\nmax_depth × n_estimators"])

    START --> A1["Act 1 — Manual\nHand-pick defaults\n79.1% on Attractive\nO(1/expert) scaling"]
    A1 --> A2["Act 2 — Grid Search\n9 combos × 5 folds = 45 fits\nExhaustive on 3×3 grid\n90.3% found"]
    A2 --> A3["Act 3 — Random Search\n60 trials × 5 folds = 300 fits\n95% coverage guarantee\n90.8% found"]
    A3 --> A4["Act 4 — Bayesian Opt\n30 trials (10 seed + 20 EI)\nLearns the landscape\n90.9% found"]

    A4 --> VERDICT["All three exceed 90%\nDifference = sample efficiency\nnot final quality\n\nFor FaceAI RF (2 axes) → grid\nFor SVM (4+ axes) → random\nFor deep nets (8+ axes) → Bayesian"]

    style START fill:#1e3a8a,color:#ffffff,stroke:#1e3a8a
    style A1 fill:#b91c1c,color:#ffffff,stroke:#b91c1c
    style A2 fill:#b45309,color:#ffffff,stroke:#b45309
    style A3 fill:#1d4ed8,color:#ffffff,stroke:#1d4ed8
    style A4 fill:#15803d,color:#ffffff,stroke:#15803d
    style VERDICT fill:#15803d,color:#ffffff,stroke:#15803d
```

### Diagram 3 — $k$-Fold Cross-Validation as a Rotation

```mermaid
flowchart LR
    DATA["Full Training Set\n90% of CelebA\nN = 182,340 rows"]

    DATA --> F1["Fold 1\n36,468 rows"]
    DATA --> F2["Fold 2\n36,468 rows"]
    DATA --> F3["Fold 3\n36,468 rows"]
    DATA --> F4["Fold 4\n36,468 rows"]
    DATA --> F5["Fold 5\n36,468 rows"]

    F1 --> R1["Round 1\n🔴 Val: Fold 1\n🟢 Train: 2,3,4,5\n→ ε₁"]
    F2 --> R2["Round 2\n🔴 Val: Fold 2\n🟢 Train: 1,3,4,5\n→ ε₂"]
    F3 --> R3["Round 3\n🔴 Val: Fold 3\n🟢 Train: 1,2,4,5\n→ ε₃"]
    F4 --> R4["Round 4\n🔴 Val: Fold 4\n🟢 Train: 1,2,3,5\n→ ε₄"]
    F5 --> R5["Round 5\n🔴 Val: Fold 5\n🟢 Train: 1,2,3,4\n→ ε₅"]

    R1 --> CV["CV_error = (ε₁+ε₂+ε₃+ε₄+ε₅) / 5\nUnbiased generalisation estimate"]
    R2 --> CV
    R3 --> CV
    R4 --> CV
    R5 --> CV

    style CV fill:#1d4ed8,color:#ffffff,stroke:#1d4ed8
```

---

## 8 · Hyperparameter Dial

Three dials matter most in a tuning workflow. Getting them right is the difference between a fast, well-calibrated search and one that wastes hours.

### Dial 1 — $k$ in $k$-Fold Cross-Validation

| $k$ | Bias | Variance | Compute cost | Use when |
|----:|------|----------|--------------|----------|
| 2 | High (pessimistic; only 50% data for training) | Low | Cheap | Never — too biased |
| **5** | **Moderate** | **Moderate** | **Moderate** | **Default for most tasks** |
| 10 | Low | Moderate | 2× vs $k=5$ | Large datasets where 2× compute is affordable |
| $N$ (LOO) | Near-zero | High (one test point per fold) | Very expensive | Tiny datasets ($N < 50$) |

**Rule of thumb:** Use $k=5$ unless you have a reason to change it. For the final model selection step (outer CV in nested CV), $k=5$ is standard.

### Dial 2 — Search Budget (Number of Trials)

| Strategy | Minimum useful | Recommended | Diminishing returns after |
|----------|---------------|-------------|--------------------------|
| Grid search | All combinations (by definition) | — | — |
| Random search | 20 | **60** | 150 |
| Bayesian opt | 15 (10 seed + 5 guided) | **30–50** | 80 |

The Bergstra & Bengio 95% guarantee requires $n = 60$ for $p = 0.05$. For $p = 0.10$ (a larger effective region), $n = 45$ suffices. For $p = 0.01$ (a very narrow optimum), $n = 298$. Know your problem's effective volume before setting $n$.

### Dial 3 — Early Stopping (in Neural Network Tuning)

Not applicable to RF or SVM. But when you carry this pipeline into the Neural Networks track, add early stopping to each trial: stop training after $P$ epochs without validation improvement. This turns a potentially 100-epoch training run into a 15-epoch run, reducing per-trial cost by 6× and making Bayesian opt viable on deep networks.

**Standard patience values:** $P = 5$ for fast experiments; $P = 10$ for final tuning; $P = 20$ for models prone to slow convergence.

---

## 9 · What Can Go Wrong

| Mistake | Symptom | Fix |
|---------|---------|-----|
| **Data leakage in CV** — preprocessing (scaling, PCA) done on all data before splitting | CV accuracy appears high; test accuracy drops sharply | Put all preprocessing inside the CV pipeline using `sklearn.pipeline.Pipeline` |
| **Tuning on the test set** — evaluating candidates on the final test set and picking the best | Test accuracy is optimistic; real-world accuracy is lower | The test set is touched exactly once: at the very end. Use CV for all selection decisions. |
| **Overfitting to the validation set** — running thousands of trials against a fixed validation split | Model appears well-tuned but generalises poorly | Use $k$-fold CV (not a single split) for all hyperparameter evaluation; use nested CV for unbiased reporting |
| **Ignoring compute budget** — using grid search on a 6-axis space | Compute bill explodes; experiment never finishes | Switch to random or Bayesian search for > 3 hyperparameter axes |
| **Single scoring metric for all 40 attributes** — using accuracy on imbalanced attributes | Rare attributes (Bald 2.5%) appear well-tuned while recall is catastrophically low | Tune with `f1_macro`; report per-attribute F1 alongside accuracy |
| **Not fixing `random_state`** | Results are not reproducible; different runs give different "best" hyperparameters | Set `random_state=42` in the model, in `RandomizedSearchCV`, and in the Optuna sampler |

---

## 10 · Where This Reappears

**Hyperparameter tuning is not confined to classical ML.** The three strategies — grid, random, Bayesian — recur throughout the curriculum each time applied to a more complex search space:

| Track | Where tuning reappears | Search space | Recommended strategy |
|-------|----------------------|--------------|----------------------|
| **Neural Networks Ch.4–6** | Learning rate, batch size, width, depth, regularisation | 5–8 continuous axes | Bayesian + early stopping |
| **Neural Networks Ch.19** | Full sweep on the housing network | $\alpha$ + batch + depth + width | Random search, $n=60$ |
| **Advanced Deep Learning** | Architecture search (NAS) — heads, FFN width, dropout | Discrete + continuous mixed | Random + population-based training |
| **Transformers** | Warmup steps, head count, positional encoding | 4–6 axes, expensive eval | Bayesian with Optuna / Ray Tune |

**Bayesian optimisation gets deeper.** The GP surrogate in Spearmint (Snoek et al. 2012) was the first production-grade implementation. Modern successors:

- **Optuna** (Tree-structured Parzen Estimator — TPE): replaces the GP with a density-ratio estimator that scales to high-dimensional discrete/continuous mixed spaces. Used in the Neural Networks track.
- **Ray Tune**: distributed search wrapping Optuna, Hyperband, and Population-Based Training behind a unified API — the production choice when each trial takes minutes.
- **HuggingFace Trainer**: built-in Optuna and Ray Tune integration for fine-tuning language models.

**Cross-validation in unsupervised tracks.** $k$-fold CV as a model-selection criterion travels to:

- **Clustering (SegmentAI)**: choosing $k$ in $k$-Means via held-out within-cluster SSE stabilisation
- **Dimensionality reduction (SegmentAI)**: choosing number of PCA components via reconstruction error on held-out rows
- **Anomaly detection (FraudShield)**: threshold-selection cross-validation on imbalanced fraud labels

> ⚡ **The key insight that carries forward:** Any time you are choosing something *before* training — number of clusters, number of components, regularisation coefficient, architecture width — you are doing hyperparameter tuning. The $k$-fold CV + search framework from this chapter is the universal tool.

**The same three questions always apply:**
1. *How many axes does the search space have?* (≤3 → grid; >3 → random or Bayesian)
2. *How expensive is one trial?* (<1 min → random; >5 min → Bayesian with early stopping)
3. *Do you need a statistically honest final estimate?* (Yes → nested CV; Exploratory → single-level CV)

These three questions are the decision interface for every hyperparameter problem you will encounter across all remaining tracks.

---

## 11 · Progress Check

![Progress check](img/ch05-hyperparameter-tuning-progress-check.png)

### FaceAI COMPLETE ✅

> **Mission:** >90% average accuracy across all 40 CelebA binary attributes.
> **Result:** **91.2% average accuracy — all 40 attributes individually ≥ 90%.**

**Chapter-by-chapter accuracy progression:**

```
Ch.1 Logistic Regression   →  85% avg   (88% on Smiling)
Ch.2 Random Forest         →  87% avg   (91% on Smiling)
Ch.4 SVM with RBF kernel   →  89% avg   (90% on Smiling)
Ch.5 Hyperparameter Tuning → 91.2% avg  (all 40 ≥ 90%) ✅
```

**All 5 FaceAI constraints — final status:**

| # | Constraint | Target | Final status | How achieved |
|---|-----------|--------|-------------|--------------|
| **1** | ACCURACY | >90% avg, all 40 attrs | ✅ **91.2% avg — all 40 ≥ 90%** | Grid search per attribute; random search for SVM (4+ axes) |
| **2** | GENERALIZATION | Unseen celebrity faces | ✅ **Nested CV confirmed** | Outer 5-fold CV on all 40 attributes; no test-set leakage at any stage |
| **3** | MULTI-LABEL | 40 simultaneous predictions | ✅ **40 tuned models deployed** | Per-attribute hyperparameter search + per-attribute decision threshold |
| **4** | INTERPRETABILITY | Explainable predictions | ✅ **RF feature importances retained** | Search does not change model class — RF importance ranking intact |
| **5** | PRODUCTION | <200ms inference, reproducible | ✅ **<10ms sklearn pipeline, seed=42** | Tuning is fully offline; production inference path unchanged |

✅ **What we can now do:**

- Systematically tune any sklearn classifier across any finite hyperparameter space using `GridSearchCV` or `RandomizedSearchCV`
- Estimate generalisation error honestly with $k$-fold cross-validation, with a valid confidence interval (Nadeau–Bengio correction)
- Choose between grid, random, and Bayesian search based on number of axes and per-trial compute budget
- Reproduce all results end-to-end: `random_state=42` in the model, the search, and the CV splitter
- Deploy with confidence: the 90% target is validated on a held-out test set that was never touched during tuning

❌ **What classical ML cannot do:**

- Learn shared representations across all 40 attributes simultaneously (each model is trained independently)
- Scale to raw pixel inputs without hand-engineered HOG features
- Beat ~93% on the hardest attributes (symmetry, age, hair texture) without non-linear feature learning

**FaceAI is production-ready.** The pipeline from raw CelebA face image → HOG descriptor → per-attribute tuned classifier → binary prediction satisfies all five constraints. It runs in <10ms per image on commodity hardware and is fully reproducible from a single `random_state=42` seed.

---

## 12 · Bridge to Neural Networks Track

This chapter closes the **02-Classification** track. FaceAI has achieved >90% average accuracy across all 40 CelebA attributes using classical ML — logistic regression, random forests, SVM, and systematic hyperparameter tuning. The full toolkit is in hand.

**What the Classification track established:**

- Four model families with distinct bias–variance profiles (logistic, RF, SVM, tuned ensembles)
- A principled evaluation framework: cross-validation, F1-macro, ROC-AUC, per-attribute confusion matrices
- The hyperparameter tuning toolkit (grid, random, Bayesian) applicable to any sklearn-compatible model
- Production-ready inference: <10ms, fully reproducible, interpretable via feature importances

**What the Neural Networks track adds:**

The FaceAI result required *four separate model types* tuned *per attribute* — 40 separate classifiers, each with its own hyperparameters. The next question: **can a single neural network architecture solve both the regression task (California housing prices) and the classification task (CelebA face attributes) simultaneously, and beat the classical baselines on both?**

The **UnifiedAI mission** (Neural Networks track) answers this:

- **Target**: ≤$28k MAE on California Housing *and* ≥95% accuracy on CelebA — from the **same architecture**
- **Method**: shared backbone → task-specific heads → joint training with a multi-task loss function
- **New tools unlocked**: backpropagation, SGD/Adam, regularisation (L1/L2/Dropout), CNNs, and hyperparameter tuning with early stopping

> ➡️ The hyperparameter tuning toolkit you built in this chapter reappears in full force in the Neural Networks track. Learning rate, batch size, width, depth, dropout rate — all swept with the same grid/random/Bayesian framework, but now using Optuna and early stopping to manage 100× more expensive per-trial evaluations (minutes per trial instead of seconds).

**The conceptual bridge:**

```
Ch.5 (Classification) established:
   systematic search → best λ* per attribute, per model class
            ↓
Ch.19 (Neural Networks) extends to:
   systematic search over architecture space (width, depth, activation)
   + early stopping to manage per-trial compute
   + multi-task objective (housing MAE + face accuracy simultaneously)
   + Bayesian opt via Optuna TPE instead of GP (scales to 8+ axes)
```

**Next up:** Ch.1 of the Neural Networks track — **The XOR Problem** — where you discover why every single classifier in this track fails at a task a child can solve instantly, and why that failure is the key that unlocks non-linear models and the entire deep learning curriculum.

---

*Classification track complete. FaceAI delivered.*
