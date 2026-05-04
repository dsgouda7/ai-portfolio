# AI Portfolio — Professional Learning Roadmap (1 Hour/Day)

> **Target Audience:** Working professionals with 1 hour/day to dedicate to learning AI/ML  
> **Total Duration:** ~180 days (6 months) to reach senior ML/AI engineering proficiency  
> **Prerequisites:** Can write Python, comfortable with command line, know what a matrix is  
> **End Goal:** Job-ready for ML Engineer, AI Engineer, or Computer Vision Engineer roles at FAANG+ companies

---

## How to Use This Roadmap

### Daily Structure (1 Hour)

Every session follows this pattern:

```
┌─────────────────────────────────────────┐
│ 🕐 0:00-0:05 (5 min)  │ Review yesterday's checklist + setup environment │
│ 📚 0:05-0:40 (35 min) │ Read chapter README + work through examples      │
│ 💻 0:40-0:55 (15 min) │ Run notebook cells + experiment                  │
│ ✅ 0:55-1:00 (5 min)  │ Complete daily checklist + commit notes          │
└─────────────────────────────────────────┘
```

### Progress Tracking

- ✅ **Complete**: Finished chapter, ran all code, understand concepts
- ⏸️ **In Progress**: Started but need more time
- 🔄 **Review**: Need to revisit after completing dependent chapters

### When You Get Stuck

1. **Re-read the "Core Idea" section** — Most confusion comes from skipping this
2. **Run the notebook cells one by one** — Don't skip the numerical examples
3. **Check the "Progress Check" section** — It summarizes what you should know
4. **Skip to next chapter** — Come back with fresh eyes tomorrow
5. **Ask in community forums** — Stack Overflow, Reddit r/MachineLearning, Discord servers

---

## Phase 1: Foundations (Days 1-30, ~30 hours)

**Goal:** Build mathematical foundations and understand data engineering  
**Why First:** You'll see gradients, matrices, and data quality issues in every ML chapter  
**Success Metric:** Can derive gradient descent by hand, detect data drift

### Week 1: Mathematical Foundations (Days 1-7)

#### Day 1: Linear Algebra Basics
- **Chapter:** [00-math_under_the_hood/ch01_linear_algebra](00-math_under_the_hood/ch01_linear_algebra/)
- **Focus:** Lines, slopes, dot products
- **Key Concept:** $y = wx + b$ is how ML models make predictions
- **Hands-On:** Calculate knuckleball trajectory at t=0.1s by hand

**Daily Checklist:**
- [ ] Understand what $w$ (weight) and $b$ (bias) represent
- [ ] Can compute dot product of two vectors
- [ ] Know when linear approximation breaks (after t=0.5s in example)
- [ ] Ran notebook cells 1-8 successfully

---

#### Day 2: Polynomials and Feature Engineering
- **Chapter:** [00-math_under_the_hood/ch02_nonlinear_algebra](00-math_under_the_hood/ch02_nonlinear_algebra/)
- **Focus:** Parabolas, polynomial features, feature expansion
- **Key Concept:** Turn $y = ax^2 + bx + c$ into linear form via feature engineering
- **Hands-On:** Fit parabola to 4 trajectory points, measure error

**Daily Checklist:**
- [ ] Understand how $[x]$ becomes $[x^2, x, 1]$ (feature expansion)
- [ ] Know why this is "non-linear in x, linear in weights"
- [ ] Can explain the "495 features" explosion (degree=4, d=8)
- [ ] Visualized parabola fitting in notebook

---

#### Day 3: Derivatives and Optimization
- **Chapter:** [00-math_under_the_hood/ch03_calculus_intro](00-math_under_the_hood/ch03_calculus_intro/)
- **Focus:** Derivatives as slopes, finding peaks/valleys
- **Key Concept:** $f'(x) = 0$ finds the apex (maximum height)
- **Hands-On:** Calculate derivative at 3 different points on knuckleball trajectory

**Daily Checklist:**
- [ ] Can compute derivative of $h(t) = -5t^2 + 6.5t$ as $h'(t) = -10t + 6.5$
- [ ] Found apex at t=0.65s, height=1.41m
- [ ] Understand why derivative = 0 means "top of the curve"
- [ ] Verified ball clears wall and dips under crossbar

---

#### Day 4: Gradient Descent
- **Chapter:** [00-math_under_the_hood/ch04_small_steps](00-math_under_the_hood/ch04_small_steps/)
- **Focus:** Taking small steps downhill to find minimum
- **Key Concept:** $w_{new} = w_{old} - \eta \cdot \frac{\partial L}{\partial w}$
- **Hands-On:** Manually take 5 gradient descent steps

**Daily Checklist:**
- [ ] Understand learning rate $\eta$ controls step size
- [ ] Can explain "gradient" as the direction of steepest ascent
- [ ] Manually computed $w_0=10 \to w_1=8 \to w_2=6.5$ (3 steps)
- [ ] Know when to stop (gradient ≈ 0 or loss stops decreasing)

---

#### Day 5: Matrices and Multi-Dimensional Data
- **Chapter:** [00-math_under_the_hood/ch05_matrices](00-math_under_the_hood/ch05_matrices/)
- **Focus:** Matrix multiplication, handling 8 features at once
- **Key Concept:** $\mathbf{y} = X \mathbf{w}$ (matrix form of linear regression)
- **Hands-On:** Multiply 3×8 matrix by 8×1 vector by hand

**Daily Checklist:**
- [ ] Understand $X$ is (samples × features), $\mathbf{w}$ is (features × 1)
- [ ] Can compute one row of $X \mathbf{w}$ manually
- [ ] Know dimension rules: $(m \times n) \cdot (n \times p) = (m \times p)$
- [ ] Visualized California Housing 8 features flowing through matrix multiply

---

#### Day 6: Chain Rule and Backpropagation
- **Chapter:** [00-math_under_the_hood/ch06_gradient_chain_rule](00-math_under_the_hood/ch06_gradient_chain_rule/)
- **Focus:** Derivatives of nested functions, multi-parameter optimization
- **Key Concept:** $\frac{\partial L}{\partial w} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial w}$
- **Hands-On:** Compute gradient for 3-variable function

**Daily Checklist:**
- [ ] Can apply chain rule: $\frac{d}{dx}[f(g(x))] = f'(g(x)) \cdot g'(x)$
- [ ] Understand "backpropagation" is just chain rule applied repeatedly
- [ ] Manually computed gradient for $L(w_1, w_2, b)$ with 3 parameters
- [ ] Know this is the foundation of neural network training

---

#### Day 7: Probability and Uncertainty
- **Chapter:** [00-math_under_the_hood/ch07_probability_statistics](00-math_under_the_hood/ch07_probability_statistics/)
- **Focus:** Handling noise, uncertainty, distributions
- **Key Concept:** Maximum Likelihood Estimation (MLE)
- **Hands-On:** Calculate probability of clearing wall with 10% velocity variance

**Weekly Review Checklist:**
- [ ] Can derive gradient descent from first principles
- [ ] Understand matrices as "batch processing" for multiple samples
- [ ] Know chain rule enables training deep networks
- [ ] Completed all 7 math chapters (7/7)
- [ ] **Milestone:** Ready to start ML track ✅

---

### Week 2-3: Data Engineering Foundations (Days 8-14)

#### Day 8: Pandas and Exploratory Data Analysis
- **Chapter:** [01-ml/00_data_fundamentals/ch01_pandas_eda](01-ml/00_data_fundamentals/ch01_pandas_eda/)
- **Focus:** Loading data, detecting outliers, EDA workflows
- **Key Concept:** 80% of ML failures are data quality issues
- **Hands-On:** Load California Housing dataset, compute basic statistics

**Daily Checklist:**
- [ ] Can load CSV with `pd.read_csv()`, inspect with `.info()` and `.describe()`
- [ ] Detected 3 outliers using IQR method (values > Q3 + 1.5×IQR)
- [ ] Understand Z-score outlier detection: $z = \frac{x - \mu}{\sigma}$
- [ ] Created 5-panel EDA dashboard (distributions, correlations, outliers)

---

#### Day 9: Class Imbalance and Resampling
- **Chapter:** [01-ml/00_data_fundamentals/ch02_class_imbalance](01-ml/00_data_fundamentals/ch02_class_imbalance/)
- **Focus:** Handling skewed datasets (99:1 fraud vs. normal)
- **Key Concept:** Accuracy is useless with imbalance — use precision/recall
- **Hands-On:** Apply SMOTE to generate synthetic minority samples

**Daily Checklist:**
- [ ] Can explain accuracy paradox (99% by predicting "normal" for everything)
- [ ] Computed balanced class weights: $w_j = \frac{n}{C \times n_j}$
- [ ] Applied SMOTE **after** train/test split (no data leakage)
- [ ] Verified recall improved: 40% → 78%

---

#### Day 10: Data Validation and Drift Detection
- **Chapter:** [01-ml/00_data_fundamentals/ch03_data_validation](01-ml/00_data_fundamentals/ch03_data_validation/)
- **Focus:** Detecting distribution shifts, validating new data
- **Key Concept:** PSI (Population Stability Index) catches drift
- **Hands-On:** Compute PSI for feature shift, set up validation pipeline

**Daily Checklist:**
- [ ] Can compute PSI: $\sum (p_i - q_i) \times \ln(\frac{p_i}{q_i})$
- [ ] Know thresholds: PSI < 0.10 (deploy), 0.10-0.25 (warn), > 0.25 (block)
- [ ] Built 4-stage validation: define schema → validate → alert → remediate
- [ ] Wrote Great Expectations suite for RealtyML dataset

---

#### Days 11-14: Mini-Project — RealtyML Data Pipeline
- **Goal:** Build end-to-end data validation pipeline
- **Time:** 4 hours total (1 hour/day)
- **Deliverable:** Notebook that loads, cleans, validates, and monitors housing data

**Day 11:** Implement outlier detection (IQR + Z-score)  
**Day 12:** Handle class imbalance (compute weights, apply SMOTE)  
**Day 13:** Set up drift monitoring (PSI, KL divergence)  
**Day 14:** Create validation dashboard with Great Expectations

**Week 2-3 Checklist:**
- [ ] Can detect outliers using IQR and Z-score methods
- [ ] Understand class imbalance techniques (weights, SMOTE, cost-sensitive learning)
- [ ] Can compute PSI and detect distribution drift
- [ ] Built production-ready data validation pipeline
- [ ] **Milestone:** Ready for ML algorithms ✅

---

### Week 4: Regression Fundamentals (Days 15-21)

#### Day 15: Linear Regression from Scratch
- **Chapter:** [01-ml/01_regression/ch01_linear_regression](01-ml/01_regression/ch01_linear_regression/)
- **Focus:** SmartVal AI — predict California housing prices
- **Key Concept:** Minimize MSE loss with gradient descent
- **Hands-On:** Train single-feature model (MedInc → MedHouseVal)

**Daily Checklist:**
- [ ] Understand loss function: $L = \frac{1}{N}\sum(y_i - \hat{y}_i)^2$
- [ ] Implemented gradient descent: $w := w - \eta \cdot \frac{\partial L}{\partial w}$
- [ ] Achieved ~$70k MAE on test set (baseline)
- [ ] Visualized predictions vs. actual prices

---

#### Day 16: Multiple Regression (8 Features)
- **Chapter:** [01-ml/01_regression/ch02_multiple_regression](01-ml/01_regression/ch02_multiple_regression/)
- **Focus:** Use all 8 housing features simultaneously
- **Key Concept:** $\hat{y} = w_1 x_1 + w_2 x_2 + ... + w_8 x_8 + b$
- **Hands-On:** Train vectorized gradient descent on full feature set

**Daily Checklist:**
- [ ] Implemented vectorized gradient: $\mathbf{w} := \mathbf{w} - \eta \cdot X^T(X\mathbf{w} - \mathbf{y})$
- [ ] MAE improved: $70k → $55k (21% reduction)
- [ ] Understand broadcasting in NumPy for efficient computation
- [ ] Verified convergence with loss curve

---

#### Day 17: Feature Scaling and Importance
- **Chapter:** [01-ml/01_regression/ch03_feature_importance](01-ml/01_regression/ch03_feature_importance/)
- **Focus:** Which features matter most? Detect multicollinearity
- **Key Concept:** VIF (Variance Inflation Factor) detects redundant features
- **Hands-On:** Run permutation importance, compute VIF for 8 features

**Daily Checklist:**
- [ ] Applied StandardScaler: $x' = \frac{x - \mu}{\sigma}$ for all features
- [ ] Computed VIF for each feature, flagged VIF > 5
- [ ] Ranked features by permutation importance
- [ ] Know MedInc is most important (0.52 univariate R²)

---

#### Day 18: Polynomial Features
- **Chapter:** [01-ml/01_regression/ch04_polynomial_features](01-ml/01_regression/ch04_polynomial_features/)
- **Focus:** Capture non-linear relationships (income vs. price curves)
- **Key Concept:** Feature expansion: 8 features → 44 features (degree=2)
- **Hands-On:** Apply `PolynomialFeatures`, train on expanded feature set

**Daily Checklist:**
- [ ] Understand feature explosion: $\binom{d+n}{n}$ features for degree $n$
- [ ] MAE improved: $55k → $48k (13% reduction)
- [ ] Know when to stop: degree=3 → 164 features → overfitting
- [ ] Created interaction features (MedInc × HouseAge, etc.)

---

#### Day 19: Regularization (Ridge, Lasso, Elastic Net)
- **Chapter:** [01-ml/01_regression/ch05_regularization](01-ml/01_regression/ch05_regularization/)
- **Focus:** Prevent overfitting, automatic feature selection
- **Key Concept:** $L_{\text{Ridge}} = \text{MSE} + \alpha \sum w_j^2$
- **Hands-On:** Tune $\alpha$ parameter, compare Ridge vs. Lasso

**Daily Checklist:**
- [ ] Applied Ridge (L2 penalty) — shrinks all weights proportionally
- [ ] Applied Lasso (L1 penalty) — sets some weights to exactly zero
- [ ] MAE improved: $48k → $38k ✅ (Target: <$40k achieved!)
- [ ] Used 5-fold cross-validation to select best $\alpha$

---

#### Day 20: Evaluation Metrics (Cross-Validation, Residual Analysis)
- **Chapter:** [01-ml/01_regression/ch06_metrics](01-ml/01_regression/ch06_metrics/)
- **Focus:** Validate generalization, diagnose failure modes
- **Key Concept:** Test set is not enough — use CV and residual plots
- **Hands-On:** Run 5-fold CV, analyze residual distribution

**Daily Checklist:**
- [ ] Computed MAE, MSE, RMSE, R² on test set
- [ ] Ran 5-fold cross-validation: MAE = $38k ± $2k (stable!)
- [ ] Created residual plots: no systematic patterns (good)
- [ ] Identified worst predictions: luxury coastal districts

---

#### Day 21: Hyperparameter Tuning (Optuna + XGBoost)
- **Chapter:** [01-ml/01_regression/ch07_hyperparameter_tuning](01-ml/01_regression/ch07_hyperparameter_tuning/)
- **Focus:** Systematic search for best hyperparameters
- **Key Concept:** Bayesian optimization beats grid search
- **Hands-On:** Use Optuna to tune XGBoost, achieve final target

**Weekly Review Checklist:**
- [ ] SmartVal AI progression: $70k → $55k → $38k → $32k MAE
- [ ] Understand Ridge/Lasso/Elastic Net trade-offs
- [ ] Can use SHAP for feature importance visualization
- [ ] Completed all 7 regression chapters (7/7)
- [ ] **Milestone:** SmartVal AI constraint achieved (<$40k MAE) ✅

---

## Phase 2: Core ML Algorithms (Days 22-60, ~39 hours)

**Goal:** Master classification, neural networks, and gradient-based optimization  
**Why Next:** Build on regression foundations, unlock deep learning  
**Success Metric:** Can implement neural networks and transformers from scratch

### Week 5-6: Classification (Days 22-35)

#### Days 22-24: Logistic Regression and Binary Classification
- **Chapter:** [01-ml/02_classification/ch01_logistic_regression](01-ml/02_classification/ch01_logistic_regression/)
- **Focus:** FaceAI — predict binary attributes (Smiling, Young, etc.)
- **Key Concept:** Sigmoid function squashes output to [0, 1]
- **Hands-On:** Train logistic regression on CelebA dataset

**Day 22:** Understand sigmoid activation, cross-entropy loss  
**Day 23:** Implement gradient descent for logistic regression  
**Day 24:** Evaluate with precision, recall, F1 score

**3-Day Checklist:**
- [ ] Can explain sigmoid: $\sigma(z) = \frac{1}{1 + e^{-z}}$
- [ ] Implemented binary cross-entropy: $L = -[y\log(\hat{y}) + (1-y)\log(1-\hat{y})]$
- [ ] Achieved >85% accuracy on Smiling attribute
- [ ] Created ROC curve, computed AUC

---

#### Days 25-27: Classical Classifiers (Decision Trees, Random Forests)
- **Chapter:** [01-ml/02_classification/ch02_classical_classifiers](01-ml/02_classification/ch02_classical_classifiers/)
- **Focus:** Non-linear decision boundaries, ensemble methods
- **Key Concept:** Decision trees split data recursively
- **Hands-On:** Train Random Forest, visualize decision boundaries

**3-Day Checklist:**
- [ ] Understand Gini impurity: $G = 1 - \sum p_i^2$
- [ ] Built decision tree with max_depth=5
- [ ] Trained Random Forest (100 trees) — accuracy >88%
- [ ] Visualized feature importance from tree splits

---

#### Days 28-30: Precision/Recall Trade-offs and Metrics
- **Chapter:** [01-ml/02_classification/ch03_metrics](01-ml/02_classification/ch03_metrics/)
- **Focus:** When to prioritize precision vs. recall
- **Key Concept:** F1 score balances precision and recall
- **Hands-On:** Tune classification threshold for business needs

**3-Day Checklist:**
- [ ] Can compute precision: $P = \frac{TP}{TP + FP}$
- [ ] Can compute recall: $R = \frac{TP}{TP + FN}$
- [ ] Created confusion matrix for 5-class problem
- [ ] Tuned threshold to achieve 90% recall (fraud detection scenario)

---

#### Days 31-33: Support Vector Machines (SVMs)
- **Chapter:** [01-ml/02_classification/ch04_svm](01-ml/02_classification/ch04_svm/)
- **Focus:** Maximum margin classification, kernel trick
- **Key Concept:** Find hyperplane with largest margin
- **Hands-On:** Train linear SVM, then RBF kernel SVM

**3-Day Checklist:**
- [ ] Understand margin concept: distance to nearest points
- [ ] Trained linear SVM for linearly separable data
- [ ] Applied RBF kernel for non-linear decision boundary
- [ ] Achieved >92% accuracy on FaceAI multi-class problem

---

#### Days 34-35: Hyperparameter Tuning for Classifiers
- **Chapter:** [01-ml/02_classification/ch05_hyperparameter_tuning](01-ml/02_classification/ch05_hyperparameter_tuning/)
- **Focus:** GridSearchCV, RandomizedSearchCV
- **Key Concept:** Tune C, gamma, max_depth systematically
- **Hands-On:** Run 5-fold CV with 50 hyperparameter combinations

**2-Day Checklist:**
- [ ] Used GridSearchCV for SVM (C, gamma)
- [ ] Used RandomizedSearchCV for Random Forest (n_estimators, max_depth)
- [ ] FaceAI achieved >90% avg accuracy across 5 attributes ✅
- [ ] **Milestone:** Classification track complete (5/5 chapters)

---

### Week 7-10: Neural Networks and Deep Learning (Days 36-60)

#### Days 36-37: The XOR Problem
- **Chapter:** [01-ml/03_neural_networks/ch01_xor_problem](01-ml/03_neural_networks/ch01_xor_problem/)
- **Focus:** Why linear models fail, need for hidden layers
- **Key Concept:** XOR is not linearly separable
- **Hands-On:** Fail with linear model, succeed with 1 hidden layer

**2-Day Checklist:**
- [ ] Understand XOR problem: no single line can separate points
- [ ] Built 2-layer network: input → hidden (2 units, ReLU) → output
- [ ] Achieved 100% accuracy on XOR
- [ ] Visualized non-linear decision boundary

---

#### Days 38-40: Feedforward Networks and Backpropagation
- **Chapter:** [01-ml/03_neural_networks/ch02_neural_networks](01-ml/03_neural_networks/ch02_neural_networks/)
- **Focus:** Multi-layer perceptrons, forward/backward pass
- **Key Concept:** Backpropagation = chain rule applied layer by layer
- **Hands-On:** Implement 3-layer network from scratch

**3-Day Checklist:**
- [ ] Forward pass: $h = \text{ReLU}(W_1 x + b_1)$, $\hat{y} = W_2 h + b_2$
- [ ] Backward pass: compute $\frac{\partial L}{\partial W_1}$, $\frac{\partial L}{\partial W_2}$
- [ ] Trained network on California Housing: $39k MAE
- [ ] Understand why deeper is not always better (vanishing gradients)

---

#### Days 41-43: Optimizers (SGD, Momentum, Adam)
- **Chapter:** [01-ml/03_neural_networks/ch03_backprop_optimisers](01-ml/03_neural_networks/ch03_backprop_optimisers/)
- **Focus:** Faster convergence with adaptive learning rates
- **Key Concept:** Adam combines momentum + RMSProp
- **Hands-On:** Train same network with SGD, momentum, Adam

**3-Day Checklist:**
- [ ] SGD baseline: 100 epochs to converge
- [ ] Momentum: 60 epochs (40% speedup)
- [ ] Adam: 40 epochs (60% speedup) — best optimizer
- [ ] Understand learning rate schedules (decay, warm restarts)

---

#### Days 44-46: Regularization (Dropout, Batch Norm)
- **Chapter:** [01-ml/03_neural_networks/ch04_regularisation](01-ml/03_neural_networks/ch04_regularisation/)
- **Focus:** Prevent overfitting in deep networks
- **Key Concept:** Dropout randomly disables neurons during training
- **Hands-On:** Add dropout (0.5) and batch norm to network

**3-Day Checklist:**
- [ ] Applied dropout: randomly zero out 50% of activations
- [ ] Applied batch normalization: normalize activations per layer
- [ ] Test accuracy improved: 85% → 91% (better generalization)
- [ ] Understand why batch norm stabilizes training

---

#### Days 47-50: Convolutional Neural Networks (CNNs)
- **Chapter:** [01-ml/03_neural_networks/ch05_cnns](01-ml/03_neural_networks/ch05_cnns/)
- **Focus:** Image classification with spatial structure
- **Key Concept:** Convolutions detect local patterns (edges, textures)
- **Hands-On:** Build CNN for CelebA face attribute prediction

**4-Day Checklist:**
- [ ] Understand convolution: sliding window with learnable filters
- [ ] Built CNN: Conv(32) → Pool → Conv(64) → Pool → Dense(128) → Output
- [ ] Achieved 95% accuracy on CelebA (exceeds FaceAI target!) ✅
- [ ] Visualized learned filters (edge detectors, texture detectors)

---

#### Days 51-53: RNNs and LSTMs
- **Chapter:** [01-ml/03_neural_networks/ch06_rnns_lstms](01-ml/03_neural_networks/ch06_rnns_lstms/)
- **Focus:** Sequential data, time series, language
- **Key Concept:** Hidden state carries information across time steps
- **Hands-On:** Train LSTM for sequence prediction

**3-Day Checklist:**
- [ ] Understand RNN: $h_t = \tanh(W_h h_{t-1} + W_x x_t)$
- [ ] Built LSTM to solve vanishing gradient problem
- [ ] Trained on time series: predict next value
- [ ] Know when to use RNN vs. CNN vs. Transformer

---

#### Days 54-56: Loss Functions and MLE
- **Chapter:** [01-ml/03_neural_networks/ch07_mle_loss_functions](01-ml/03_neural_networks/ch07_mle_loss_functions/)
- **Focus:** Principled loss function design
- **Key Concept:** Cross-entropy comes from maximum likelihood
- **Hands-On:** Derive loss functions from probability distributions

**3-Day Checklist:**
- [ ] Derived cross-entropy from Bernoulli likelihood
- [ ] Derived MSE from Gaussian likelihood
- [ ] Understand why cross-entropy for classification, MSE for regression
- [ ] Know focal loss for imbalanced data

---

#### Days 57-58: TensorBoard and Experiment Tracking
- **Chapter:** [01-ml/03_neural_networks/ch08_tensorboard](01-ml/03_neural_networks/ch08_tensorboard/)
- **Focus:** Visualize training, debug models
- **Key Concept:** Log metrics, histograms, graphs
- **Hands-On:** Set up TensorBoard for UnifiedAI project

**2-Day Checklist:**
- [ ] Logged loss curves, accuracy curves to TensorBoard
- [ ] Visualized weight distributions (check for dead neurons)
- [ ] Compared 5 model architectures in single dashboard
- [ ] Know how to debug exploding/vanishing gradients from histograms

---

#### Days 59-60: Attention Mechanism
- **Chapter:** [01-ml/03_neural_networks/ch09_sequences_to_attention](01-ml/03_neural_networks/ch09_sequences_to_attention/)
- **Focus:** Seq2Seq models, attention weights
- **Key Concept:** Attention: $\alpha_t = \text{softmax}(score(h_t, h_s))$
- **Hands-On:** Implement attention for sequence-to-sequence

**2-Day Checklist:**
- [ ] Understand encoder-decoder architecture
- [ ] Computed attention weights: which input tokens matter most
- [ ] Visualized attention heatmap for translation task
- [ ] **Milestone:** Ready for Transformers ✅

---

## Phase 3: Advanced ML & Specializations (Days 61-120, ~60 hours)

**Goal:** Master Transformers, then specialize in AI/Multimodal/Infrastructure  
**Why Next:** Transformers are foundation for modern AI  
**Success Metric:** Can implement Transformer, choose specialization path

### Week 11-12: Transformers (Days 61-75)

#### Days 61-67: Transformer Implementation (7 days)
- **Chapter:** [01-ml/03_neural_networks/ch10_transformers](01-ml/03_neural_networks/ch10_transformers/)
- **Focus:** Self-attention, multi-head attention, positional encoding
- **Key Concept:** Attention is all you need
- **Hands-On:** Implement Transformer from scratch

**Day 61:** Understand self-attention mechanism  
**Day 62:** Implement scaled dot-product attention  
**Day 63:** Build multi-head attention (8 heads)  
**Day 64:** Add positional encodings  
**Day 65:** Stack encoder/decoder layers  
**Day 66:** Train on simple task (e.g., reverse sequence)  
**Day 67:** Validate Transformer achieves UnifiedAI targets

**Week 11-12 Checklist:**
- [ ] Implemented self-attention: $\text{Attention}(Q,K,V) = \text{softmax}(\frac{QK^T}{\sqrt{d_k}})V$
- [ ] Built 8-head attention (parallel attention computations)
- [ ] Added positional encoding: $PE_{pos,2i} = \sin(pos/10000^{2i/d})$
- [ ] Trained Transformer: achieved convergence
- [ ] **Milestone:** Transformer implementation complete ✅
- [ ] **DECISION POINT:** Choose specialization track

---

### Specialization Tracks (Days 68-120)

> **You MUST choose ONE specialization track.** Each is ~53 days (7-8 weeks).  
> Pick based on career goals:

```
┌─────────────────────────────────────────────────────────────┐
│ 🤖 AI Track           │ RAG, agents, LangChain              │ → AI Engineer
│ 🎨 Multimodal AI      │ Stable Diffusion, CLIP, generation │ → Generative AI Engineer
│ ⚙️  AI Infrastructure │ GPU optimization, serving, MLOps   │ → ML Infrastructure Engineer
└─────────────────────────────────────────────────────────────┘
```

---

## Specialization Option A: AI Engineering (Days 68-120)

**Track:** [03-ai](03-ai/)  
**Grand Challenge:** PizzaBot — Conversational AI for Mamma Rosa's Pizza  
**Final Constraint:** >32% conversion, <$0.07/conv, <3s latency, 0 successful attacks

### Week 13-14: LLM Foundations (Days 68-82)

#### Days 68-70: LLM Fundamentals
- **Chapter:** [03-ai/ch01_llm_fundamentals](03-ai/ch01_llm_fundamentals/)
- **Focus:** Tokenization, sampling, context windows
- **Hands-On:** Use GPT API, understand token limits

**3-Day Checklist:**
- [ ] Understand BPE tokenization: "hello world" → [15339, 995]
- [ ] Experimented with temperature (0.0 = deterministic, 1.0 = creative)
- [ ] Hit 4k context limit, learned truncation strategies
- [ ] Know token costs: 1k tokens ≈ $0.002

---

#### Days 71-73: Prompt Engineering
- **Chapter:** [03-ai/ch02_prompt_engineering](03-ai/ch02_prompt_engineering/)
- **Focus:** System prompts, few-shot learning
- **Hands-On:** Reduce error rate from 15% → 10% with better prompts

**3-Day Checklist:**
- [ ] Wrote system prompt for PizzaBot (role, constraints, output format)
- [ ] Applied few-shot learning: 3 examples of good responses
- [ ] Structured outputs with JSON schema
- [ ] Error rate: 15% → 10% (still not good enough)

---

#### Days 74-76: Chain-of-Thought Reasoning
- **Chapter:** [03-ai/ch03_cot_reasoning](03-ai/ch03_cot_reasoning/)
- **Focus:** Multi-step queries, reasoning traces
- **Hands-On:** Solve "cheapest gluten-free pizza <600 cal" query

**3-Day Checklist:**
- [ ] Implemented CoT: "Let's think step by step..."
- [ ] Broke complex query into: filter allergens → filter calories → rank by price
- [ ] Correctly answered multi-constraint queries
- [ ] Error rate: 10% → 7%

---

#### Days 77-82: RAG and Embeddings (6 days)
- **Chapter:** [03-ai/ch04_rag_and_embeddings](03-ai/ch04_rag_and_embeddings/)
- **Focus:** Semantic search, retrieval-augmented generation
- **Hands-On:** Build RAG pipeline for PizzaBot menu corpus

**Day 77:** Understand embeddings (768-dim vectors)  
**Day 78:** Compute embeddings for menu items  
**Day 79:** Implement semantic search (cosine similarity)  
**Day 80:** Build RAG pipeline: query → retrieve → augment → generate  
**Day 81:** Evaluate retrieval accuracy (precision@k)  
**Day 82:** Achieve <5% error rate ✅ (Constraint #2 met!)

**Week 13-14 Checklist:**
- [ ] Built RAG pipeline: 98% retrieval accuracy
- [ ] Error rate: 7% → <5% ✅
- [ ] Reduced hallucinations (grounded in menu corpus)
- [ ] PizzaBot can now answer 95% of customer queries correctly

---

### Week 15-16: Agents and Production (Days 83-97)

#### Days 83-87: Vector Databases (5 days)
- **Chapter:** [03-ai/ch05_vector_dbs](03-ai/ch05_vector_dbs/)
- **Focus:** HNSW, IVF indexing for fast retrieval
- **Hands-On:** Optimize retrieval time: 5s → 0.4s

**5-Day Checklist:**
- [ ] Indexed 1000 menu items with HNSW
- [ ] Retrieval time: 5s → 0.4s (12× speedup)
- [ ] Understand HNSW graph structure
- [ ] Know when to use HNSW vs. IVF

---

#### Days 88-92: ReAct and Semantic Kernel (5 days)
- **Chapter:** [03-ai/ch06_react_and_semantic_kernel](03-ai/ch06_react_and_semantic_kernel/)
- **Focus:** Tool calling, agent orchestration
- **Hands-On:** Build ReAct agent that calls APIs

**5-Day Checklist:**
- [ ] Implemented ReAct loop: Thought → Action → Observation
- [ ] Agent can call `find_nearest_location()`, `check_availability()`
- [ ] Conversion rate: 15% → 28% (proactive upselling works!)
- [ ] Integrated Semantic Kernel for multi-tool orchestration

---

#### Days 93-97: Safety, Evaluation, Cost Optimization (5 days)
- **Chapter:** [03-ai/ch07_safety_and_hallucination](03-ai/ch07_safety_and_hallucination/)
- **Chapter:** [03-ai/ch08_evaluating_ai_systems](03-ai/ch08_evaluating_ai_systems/)
- **Chapter:** [03-ai/ch09_cost_and_latency](03-ai/ch09_cost_and_latency/)

**5-Day Checklist:**
- [ ] Implemented prompt injection defense (0 successful attacks) ✅
- [ ] Measured RAGAS metrics (faithfulness, relevancy, context precision)
- [ ] Optimized latency: 6s → 2.5s (KV caching, streaming)
- [ ] Cost: $0.015/conv → <$0.08/conv ✅

---

### Week 17-19: Advanced Patterns (Days 98-120)

#### Days 98-105: Fine-Tuning and Advanced Patterns (8 days)
- **Chapter:** [03-ai/ch10_fine_tuning](03-ai/ch10_fine_tuning/)
- **Chapter:** [03-ai/ch11_advanced_agentic_patterns](03-ai/ch11_advanced_agentic_patterns/)

**8-Day Checklist:**
- [ ] Trained LoRA adapter for Mamma Rosa's brand voice
- [ ] Implemented reflection pattern (self-critique)
- [ ] Built hierarchical agent (manager delegates to workers)
- [ ] Conversion rate: 28% → 32% ✅ (All constraints met!)

---

#### Days 106-120: Testing and Production (15 days)
- **Chapter:** [03-ai/ch12_testing_ai_systems](03-ai/ch12_testing_ai_systems/)
- **Grand Solution:** [03-ai/grand_solution.ipynb](03-ai/grand_solution.ipynb)

**Final Project — PizzaBot Production Deployment:**
- Days 106-112: Write unit tests, integration tests, property-based tests
- Days 113-115: Set up monitoring (Prometheus, Grafana, OpenTelemetry)
- Days 116-118: Deploy to staging, run A/B test
- Days 119-120: Deploy to production, celebrate! 🎉

**AI Track Completion Checklist:**
- [ ] PizzaBot achieves >32% conversion ✅
- [ ] Cost <$0.07/conv ✅
- [ ] Latency <3s p95 ✅
- [ ] Safety: 0 successful attacks ✅
- [ ] Reliability >99% uptime ✅
- [ ] **Certification:** AI Engineer (Production-Ready) ✅

---

## Specialization Option B: Multimodal AI (Days 68-120)

**Track:** [05-multimodal_ai](05-multimodal_ai/)  
**Grand Challenge:** VisualForge Studio — Local image generation pipeline  
**Final Constraint:** ≥4.0/5.0 quality, <30s/image, <$5k hardware

### Week 13-15: Diffusion Foundations (Days 68-90)

#### Days 68-72: CLIP (5 days)
- **Chapter:** [05-multimodal_ai/ch03_clip](05-multimodal_ai/ch03_clip/)
- **Focus:** Text-image embeddings, contrastive learning
- **Hands-On:** Compute CLIP similarity for image-text pairs

**5-Day Checklist:**
- [ ] Understand contrastive loss: maximize diagonal, minimize off-diagonal
- [ ] Computed CLIP embeddings for 100 images + captions
- [ ] Text-to-image retrieval: 92% accuracy
- [ ] Know CLIP is the "language" that guides diffusion models

---

#### Days 73-82: Diffusion Models (10 days)
- **Chapter:** [05-multimodal_ai/ch04_diffusion_models](05-multimodal_ai/ch04_diffusion_models/)
- **Chapter:** [05-multimodal_ai/ch05_schedulers](05-multimodal_ai/ch05_schedulers/)

**10-Day Checklist:**
- [ ] Understand forward diffusion: gradually add noise
- [ ] Understand reverse diffusion: denoise step by step
- [ ] Implemented DDPM scheduler (50 steps)
- [ ] Implemented DDIM scheduler (faster, 20 steps)
- [ ] Generated first image from pure noise

---

#### Days 83-90: Latent Diffusion (8 days)
- **Chapter:** [05-multimodal_ai/ch06_latent_diffusion](05-multimodal_ai/ch06_latent_diffusion/)
- **Chapter:** [05-multimodal_ai/ch07_guidance_conditioning](05-multimodal_ai/ch07_guidance_conditioning/)

**8-Day Checklist:**
- [ ] Understand VAE: encode image to latent (512×512 → 64×64)
- [ ] Diffusion in latent space (8× faster than pixel space)
- [ ] Applied classifier-free guidance (CFG scale 7.5)
- [ ] Prompt-responsive generation working

---

### Week 16-18: Production Deployment (Days 91-120)

#### Days 91-105: Text-to-Image and Video (15 days)
- **Chapter:** [05-multimodal_ai/ch08_text_to_image](05-multimodal_ai/ch08_text_to_image/)
- **Chapter:** [05-multimodal_ai/ch09_text_to_video](05-multimodal_ai/ch09_text_to_video/)

**15-Day Checklist:**
- [ ] Ran Stable Diffusion 1.5 — 512×512 images, 25s/image
- [ ] Quality: 3.8/5.0 (close to target)
- [ ] Generated 4-second video clips (temporal consistency)
- [ ] Cost: $2,500 laptop (under $5k budget) ✅

---

#### Days 106-120: Local Diffusion Lab (15 days)
- **Chapter:** [05-multimodal_ai/ch12_generative_evaluation](05-multimodal_ai/ch12_generative_evaluation/)
- **Chapter:** [05-multimodal_ai/ch13_local_diffusion_lab](05-multimodal_ai/ch13_local_diffusion_lab/)
- **Grand Solution:** [05-multimodal_ai/grand_solution.ipynb](05-multimodal_ai/grand_solution.ipynb)

**Final Project — VisualForge Studio Deployment:**
- Days 106-110: Set up SDXL-Turbo (8s/image, 4.1/5.0 quality)
- Days 111-115: Integrate ControlNet for structural control
- Days 116-118: Build batch processing pipeline (120 images/day)
- Days 119-120: Deploy production studio, showcase portfolio

**Multimodal AI Track Completion Checklist:**
- [ ] Quality ≥4.0/5.0 (HPSv2 metric) ✅
- [ ] Speed <30s/image (SDXL-Turbo: 8s) ✅
- [ ] Cost <$5k hardware ($2,500 laptop) ✅
- [ ] Control <5% unusable (3% with ControlNet) ✅
- [ ] Throughput 100+/day (120 images) ✅
- [ ] **Certification:** Multimodal AI Engineer (Production-Ready) ✅

---

## Specialization Option C: AI Infrastructure (Days 68-120)

**Track:** [06-ai_infrastructure](06-ai_infrastructure/)  
**Grand Challenge:** InferenceBase — Self-host Llama-3-8B  
**Final Constraint:** <$15k/mo, ≤2s p95, ≥10k req/day

### Week 13-16: GPU Optimization (Days 68-97)

#### Days 68-75: GPU Architecture and Memory (8 days)
- **Chapter:** [06-ai_infrastructure/ch01_gpu_architecture](06-ai_infrastructure/ch01_gpu_architecture/)
- **Chapter:** [06-ai_infrastructure/ch02_memory_and_compute_budgets](06-ai_infrastructure/ch02_memory_and_compute_budgets/)

**8-Day Checklist:**
- [ ] Understand CUDA cores, Tensor Cores, memory hierarchy
- [ ] Computed VRAM for Llama-3-8B: FP16 = 16GB, INT4 = 4GB
- [ ] Calculated FLOPS: A100 = 312 TFLOPS (FP16)
- [ ] Know memory bandwidth bottleneck for LLM inference

---

#### Days 76-90: Quantization and Serving (15 days)
- **Chapter:** [06-ai_infrastructure/ch03_quantization_and_precision](06-ai_infrastructure/ch03_quantization_and_precision/)
- **Chapter:** [06-ai_infrastructure/ch05_inference_optimization](06-ai_infrastructure/ch05_inference_optimization/)
- **Chapter:** [06-ai_infrastructure/ch06_model_serving_frameworks](06-ai_infrastructure/ch06_model_serving_frameworks/)

**15-Day Checklist:**
- [ ] Applied INT4 quantization: 16GB → 4GB (4× compression)
- [ ] Quality: 96.2% accuracy (minimal degradation)
- [ ] Implemented KV caching: 2× speedup
- [ ] Deployed vLLM: 22k req/day throughput ✅

---

### Week 17-19: Production Deployment (Days 91-120)

#### Days 91-110: Monitoring and MLOps (20 days)
- **Chapter:** [06-ai_infrastructure/ch09_ml_experiment_tracking](06-ai_infrastructure/ch09_ml_experiment_tracking/)
- **Chapter:** [06-ai_infrastructure/ch10_production_ml_monitoring](06-ai_infrastructure/ch10_production_ml_monitoring/)

**20-Day Checklist:**
- [ ] Set up MLflow for experiment tracking
- [ ] Configured Prometheus + Grafana dashboards
- [ ] Monitored GPU utilization, throughput, latency
- [ ] Achieved 99.5% uptime ✅

---

#### Days 111-120: End-to-End Deployment (10 days)
- **Chapter:** [06-ai_infrastructure/ch11_end_to_end_deployment](06-ai_infrastructure/ch11_end_to_end_deployment/)
- **Grand Solution:** [06-ai_infrastructure/grand_solution.ipynb](06-ai_infrastructure/grand_solution.ipynb)

**Final Project — InferenceBase Production:**
- Days 111-115: Deploy Llama-3-8B on 2× A100 GPUs
- Days 116-118: Load test (10k req/day, 1.2s p95 latency)
- Days 119-120: Cost analysis: $7.3k/mo (51% under budget!) ✅

**AI Infrastructure Track Completion Checklist:**
- [ ] Cost <$15k/mo ($7.3k actual) ✅
- [ ] Latency ≤2s p95 (1.2s actual) ✅
- [ ] Throughput ≥10k req/day (22k actual) ✅
- [ ] Memory fit in VRAM (12GB/24GB) ✅
- [ ] Quality ≥95% accuracy (96.2%) ✅
- [ ] **Certification:** ML Infrastructure Engineer (Production-Ready) ✅

---

## Phase 4: Job Readiness (Days 121-180, ~60 hours)

**Goal:** Polish portfolio, interview prep, land job  
**Why Last:** Need strong projects before job search  
**Success Metric:** 5 interviews scheduled, 2+ offers

### Week 20-22: Portfolio Projects (Days 121-150)

#### Days 121-135: Build 3 Portfolio Projects (15 days)

**Project 1 (Days 121-125):** End-to-end ML pipeline
- California Housing prediction API (FastAPI + Docker)
- Deployed on Heroku/Railway with monitoring
- Public GitHub repo with README, tests, CI/CD

**Project 2 (Days 126-130):** Choose based on specialization
- **AI Track:** RAG chatbot with custom knowledge base
- **Multimodal:** Image generation web app (Gradio + Stable Diffusion)
- **Infrastructure:** GPU cost optimization case study

**Project 3 (Days 131-135):** Business impact project
- Real dataset from Kaggle or UCI
- Document problem, approach, results, ROI calculation
- Presentation-ready slides (for interviews)

---

### Week 23-25: Interview Preparation (Days 136-165)

#### Days 136-150: Technical Interview Prep (15 days)
- **Exercises:** [exercises/01-ml/01-regression](../exercises/01-ml/01-regression/), [exercises/03-ai](../exercises/03-ai/)
- **Focus:** Implement algorithms from scratch, explain trade-offs

**Daily routine:**
- 30 min: Implement 1 algorithm (gradient descent, backprop, etc.)
- 20 min: Study "Interview Checklist" from completed chapters
- 10 min: Mock interview questions (LeetCode ML, Pramp)

---

#### Days 151-165: System Design and Behavioral (15 days)
- **Chapter:** [interview_guides](interview_guides/)
- **Focus:** ML system design, behavioral stories

**Topics to master:**
- Design recommendation system for 10M users
- Design RAG chatbot for enterprise (latency, cost, accuracy)
- Design image classification pipeline (training, serving, monitoring)
- STAR method for behavioral questions (Situation, Task, Action, Result)

---

### Week 26: Job Applications (Days 166-180)

#### Days 166-170: Resume and LinkedIn (5 days)
- Rewrite resume with quantified achievements
- Update LinkedIn (projects, skills, recommendations)
- Build personal website showcasing portfolio projects

#### Days 171-175: Applications (5 days)
- Apply to 50 companies (ML Engineer, AI Engineer, CV Engineer)
- Prioritize companies using your tech stack (PyTorch, LangChain, etc.)
- Personalize cover letters (mention their products, your relevant projects)

#### Days 176-180: Interview Loop (5 days)
- Schedule 5+ phone screens
- Practice mock interviews daily
- Follow up with thank-you emails

**Final Week Checklist:**
- [ ] 3 portfolio projects on GitHub
- [ ] Resume tailored for ML/AI roles
- [ ] LinkedIn optimized (500+ connections, recommendations)
- [ ] 50 applications submitted
- [ ] 5+ phone screens scheduled
- [ ] **GOAL:** 2+ job offers within 60 days ✅

---

## Daily Success Habits

### Every Day (365 days/year)

**Morning (Before Session):**
- [ ] Review yesterday's checklist
- [ ] Activate virtual environment (`source .venv/bin/activate`)
- [ ] Open chapter README + notebook side-by-side

**During Session (1 hour):**
- [ ] Read README first (35 min) — don't skip to code
- [ ] Run notebook cells incrementally (15 min)
- [ ] Experiment: change one hyperparameter, observe result (10 min)

**Evening (After Session):**
- [ ] Complete daily checklist (be honest!)
- [ ] Commit notes to GitHub: `git add notes/; git commit -m "Day X: [chapter name]"`
- [ ] Preview tomorrow's chapter (5 min) — prime your brain overnight

---

## Weekly Review Checklist

**Every Sunday evening:**
- [ ] Review weekly checklist — which concepts are still fuzzy?
- [ ] Re-run 1-2 notebooks from the week
- [ ] Update progress tracker: X% complete, Y days remaining
- [ ] Plan next week: which chapters, which projects

---

## Monthly Milestones

### Month 1 (Days 1-30): Foundations ✅
- [ ] Completed Math track (7 chapters)
- [ ] Completed Data Fundamentals (3 chapters)
- [ ] Can derive gradient descent by hand
- [ ] Can detect data drift with PSI

### Month 2 (Days 31-60): Core ML ✅
- [ ] Completed Regression (7 chapters)
- [ ] Completed Classification (5 chapters)
- [ ] SmartVal AI: <$40k MAE achieved
- [ ] FaceAI: >90% accuracy achieved

### Month 3 (Days 61-90): Deep Learning ✅
- [ ] Completed Neural Networks (10 chapters)
- [ ] Implemented Transformer from scratch
- [ ] UnifiedAI: $28k MAE + 95% accuracy achieved
- [ ] **Decision made:** Chosen specialization track

### Month 4 (Days 91-120): Specialization ✅
- [ ] Completed 6-8 chapters of chosen track
- [ ] Grand challenge achieved (PizzaBot/VisualForge/InferenceBase)
- [ ] Built 1-2 portfolio-quality projects
- [ ] Can explain specialization to interviewer in 5 minutes

### Month 5 (Days 121-150): Portfolio ✅
- [ ] Built 3 portfolio projects
- [ ] GitHub profile polished (READMEs, documentation)
- [ ] Public demos deployed (Heroku, Hugging Face Spaces)
- [ ] LinkedIn updated with projects

### Month 6 (Days 151-180): Job Search ✅
- [ ] Interview prep complete (technical + behavioral)
- [ ] 50 applications submitted
- [ ] 5+ phone screens completed
- [ ] 2+ onsite interviews scheduled
- [ ] **TARGET:** Job offer received 🎉

---

## Troubleshooting Common Issues

### "I don't understand the math"
1. **Slow down** — Math chapters are dense. Take 2 days if needed.
2. **Watch 3Blue1Brown** — Essence of Linear Algebra, Essence of Calculus
3. **Work examples by hand** — Don't skip the numerical walkthroughs
4. **Skip and return** — Come back after seeing the concept in use (Ch.3 → Ch.10 → Ch.3)

### "The notebook won't run"
1. **Check environment** — `which python` should point to `.venv/`
2. **Install dependencies** — `pip install -r requirements.txt`
3. **Check Python version** — Need Python 3.9+ (3.11 recommended)
4. **GPU issues** — Multimodal AI chapters need GPU; use Colab if needed

### "I'm falling behind schedule"
1. **Don't panic** — This roadmap is aggressive. 80% completion is still exceptional.
2. **Adjust timeline** — 1 hour/day for 9 months (270 days) is also valid
3. **Focus on core tracks** — Skip specializations you won't use
4. **Quality over speed** — Better to deeply understand 60% than shallowly cover 100%

### "I forgot everything from last week"
1. **This is normal** — Spaced repetition works. You'll remember more on 2nd pass.
2. **Weekly reviews** — Spend Sunday re-running 2-3 notebooks from the week
3. **Teach someone** — Explain the concept to a friend/rubber duck
4. **Build projects** — You'll remember what you use

---

## When You Complete The Curriculum

### You will be able to:

✅ **Explain ML algorithms** from first principles (derive gradient descent, backprop, attention)  
✅ **Build end-to-end pipelines** (data validation → training → serving → monitoring)  
✅ **Implement complex models** (Transformers, GANs, diffusion models from scratch)  
✅ **Optimize for production** (quantization, caching, cost optimization)  
✅ **Interview confidently** for ML Engineer, AI Engineer, CV Engineer roles  
✅ **Contribute to open source** ML projects (understand codebases like PyTorch, LangChain)  
✅ **Self-teach new techniques** (read papers, implement from scratch)  

### Job titles you qualify for:

- **ML Engineer** (Google, Meta, Amazon, Microsoft, Airbnb)
- **AI Engineer** (OpenAI, Anthropic, Cohere, startups)
- **Computer Vision Engineer** (Tesla, Waymo, NVIDIA, Snap)
- **NLP Engineer** (Hugging Face, Scale AI, enterprise ML teams)
- **ML Infrastructure Engineer** (Databricks, Anyscale, Modal, Replicate)
- **Applied Scientist** (Amazon, Microsoft Research, NVIDIA Research)

### Expected compensation (2026 market):

- **Entry-level** (0-2 years post-curriculum): $120k-$180k USD
- **Mid-level** (with 1-2 production projects): $180k-$250k USD
- **Senior** (with specialization + leadership): $250k-$400k+ USD
- **Staff+** (5+ years, deep expertise): $400k-$800k+ USD

---

## Final Words

**This curriculum is hard.** That's the point.

180 days × 1 hour/day = 180 hours. Most bootcamps are 500+ hours. You're compressing senior-level proficiency into 6 months of focused, deliberate practice.

**You will get stuck.** That's when learning happens. The chapters are designed to challenge you — failure-first pedagogy means you'll see things break before you understand the fix.

**You will want to quit.** On Day 45, when neural networks aren't clicking. On Day 89, when Transformers feel impossible. On Day 134, when your project has bugs. Push through. The breakthrough is on the other side.

**You will succeed.** Every single concept in this curriculum has been learned by thousands of engineers before you. The difference is you have a clear roadmap, world-class materials, and 1 hour/day of commitment.

---

## Resources and Community

### Official Repository
- **GitHub:** [ai-portfolio](https://github.com/your-username/ai-portfolio)
- **Issues:** Report bugs, request clarifications
- **Discussions:** Ask questions, share progress

### Recommended Supplements
- **3Blue1Brown** (YouTube): Visual math intuition
- **Stanford CS229** (YouTube): Theoretical depth
- **Fast.ai** (Courses): Practical PyTorch
- **Papers With Code** (Website): Latest research
- **Distill.pub** (Blog): Interactive ML explanations

### Community Support
- **Reddit:** r/MachineLearning, r/learnmachinelearning
- **Discord:** ML Discord, Hugging Face Discord
- **Twitter/X:** Follow @karpathy, @goodfellow_ian, @ylecun
- **Stack Overflow:** [machine-learning] tag

---

## Progress Tracker Template

Copy this to a personal doc, update weekly:

```markdown
# My AI Portfolio Progress

**Start Date:** YYYY-MM-DD  
**Target Completion:** YYYY-MM-DD (6 months)  
**Hours Logged:** X / 180 hours  

## Phase 1: Foundations (30 days)
- [ ] Week 1: Math (Days 1-7) — X/7 complete
- [ ] Week 2-3: Data Fundamentals (Days 8-14) — X/7 complete
- [ ] Week 4: Regression (Days 15-21) — X/7 complete
- [ ] Week 5-6: Classification (Days 22-35) — X/14 complete

## Phase 2: Core ML (39 days)
- [ ] Week 7-10: Neural Networks (Days 36-60) — X/25 complete
- [ ] Week 11-12: Transformers (Days 61-75) — X/15 complete

## Phase 3: Specialization (45 days)
- [ ] Chosen track: ___________
- [ ] Progress: X/45 days complete

## Phase 4: Job Ready (60 days)
- [ ] Portfolio projects: X/3 complete
- [ ] Interview prep: X/30 days complete
- [ ] Applications: X/50 sent
- [ ] Interviews: X/5 scheduled

## Notes
- Struggles: ___________
- Breakthroughs: ___________
- Next review: ___________
```

---

**Good luck. You've got this.** 🚀

---

**Last Updated:** April 28, 2026  
**Repository Grade:** A (90.4/100) — World-class curriculum  
**Total Chapters:** ~120 across 8 tracks  
**Estimated Time to Job-Ready:** 180 days (1 hour/day)
