# Ch.9 — ML Experiment Tracking & Model Registry

> **The story.** In **2015** the machine learning engineering community faced a reproducibility crisis — research papers claimed 95% accuracy on benchmark datasets, but nobody could reproduce the results six months later. The hyperparameters were buried in email threads, the training data had been overwritten, and the model weights lived on someone's laptop that was now wiped. That year, **Databricks** open-sourced **MLflow**, the first unified experiment tracking system that logged every parameter, metric, and artifact in one place. At the same time, **DVC** (Data Version Control) launched to version datasets the way Git versions code. Together they established the modern experiment tracking workflow: every run is a **tuple (code version, data version, hyperparameters, metrics, artifacts)** — and if you can't reproduce it, you didn't track it right. The idea was so obvious in hindsight that within three years, every major ML platform (Azure ML, AWS SageMaker, Google Vertex AI, Weights & Biases) shipped equivalent tracking APIs.
>
> **Where you are in the curriculum.** You've just finished [Ch.8: Feature Stores & Data Infrastructure](../ch08_feature_stores) where you built a reusable feature pipeline. Now you have **clean data** but **100+ experiments to run**: 3 learning rates × 2 batch sizes × 2 warmup schedules × 10 random seeds = 120 training runs. Without tracking, you'll lose track of which run gave 94% accuracy by Tuesday morning. This chapter teaches the discipline that separates research prototypes from production ML: **log everything, version everything, reproduce everything**. You'll track a BERT fine-tuning sweep from first baseline to deployed model using MLflow (free, local) and DVC (free, Git-based data versioning).
>
> **Notation in this chapter.** `run_id` — unique identifier for one training run (UUID); `params` — hyperparameters logged at run start (learning rate, batch size, epochs); `metrics` — time-series outputs logged during training (loss, accuracy per epoch); `artifacts` — files logged at run end (model weights, plots, confusion matrix); `stage` — model registry lifecycle (None → Staging → Production → Archived); `git_commit` — code version (SHA hash); `dvc.lock` — DVC-tracked data version (SHA hash of dataset).
<!-- notation: key variables defined here -->

---

## 0 · The Challenge — Where We Are

> **The mission**: You're a research engineer at **InferenceBase** (the AI startup from Ch.1-5). The product team wants to improve document classification accuracy from 89% to 94%. You have:
> - A clean feature pipeline from Ch.8
> - 5 GPUs for parallel training
> - 3 team members running experiments
> - **NO experiment tracking system** — just Jupyter notebooks, scattered CSV logs, and model weights in `/tmp/`

**What's blocking us:**
After **2 weeks of hyperparameter tuning** (100+ training runs), your teammate Slack's you:
> "Hey, which run got 94% accuracy? I remember it was learning rate 2e-5 with batch size 16… or was it 3e-5 with batch size 32? Also, which dataset version did we use? The one from last Monday or the cleaned one from Wednesday?"

You realize:
- **Lost experiments** — No record of which hyperparameters were tested
- **Irreproducible results** — Can't recreate the 94% run even with the right hyperparameters (was the random seed logged?)
- **No version control for data** — The training dataset was overwritten twice this week
- **Manual comparison** — Comparing 100 runs by scrolling through Jupyter notebooks takes hours

**What this chapter unlocks:**
The **ML experiment tracking & model registry** discipline:
1. **Log every experiment** — MLflow tracks params, metrics, artifacts automatically
2. **Version every dataset** — DVC versions data the way Git versions code
3. **Compare runs visually** — MLflow UI shows 100 experiments in one table
4. **Promote best models** — Model registry lifecycle (Staging → Production)
5. **Reproduce any experiment** — Given `run_id`, reconstruct exact environment
**After this chapter**: You can answer "which hyperparameters gave 94%?" in 10 seconds, reproduce any experiment from 3 months ago, and deploy models with one CLI command.

---

## Animation

![Chapter animation](img/ch09-experiment-tracking-needle.gif)

*Debugging time: 2 days → 30 minutes with tracking*

---

## 1 · The Core Idea — Every Experiment is a Versioned Tuple

Experiment tracking solves one problem: **make every training run reproducible**. The solution is conceptually simple:

### Every Experiment = (Code Version, Data Version, Hyperparameters, Metrics, Artifacts)

| Component | What It Tracks | Tool | Example |
|---|---|---|---|
| **Code version** | Git commit SHA | Git | `7a3f9b2` |
| **Data version** | Dataset checksum | DVC | `data.csv.dvc` (SHA: `a8d3c1f`) |
| **Hyperparameters** | All config values at run start | MLflow | `lr=2e-5, batch_size=16, epochs=3` |
| **Metrics** | Time-series outputs during training | MLflow | `loss=[0.8, 0.5, 0.3], acc=[0.85, 0.91, 0.94]` |
| **Artifacts** | Files generated after training | MLflow | `model.pt, confusion_matrix.png, vocab.json` |

**Why this tuple solves reproducibility:**
- Change the **code** (bug fix) → new Git commit → new run
- Change the **data** (add more labels) → DVC detects hash mismatch → new run
- Change **hyperparameters** (tune learning rate) → MLflow logs new params → new run
- Same tuple → **exact same results** (if random seed is logged)

**The two layers of tracking:**
1. **Experiment tracking (MLflow)** — Logs runs *during training* (metrics per epoch, hyperparameters, artifacts)
2. **Data versioning (DVC)** — Logs datasets *before training* (tracks which CSV/Parquet file version was used)

Both are **local-first tools** — no cloud account required until you want team collaboration.

> 📖 **Industry Standard: MLflow vs Alternatives**
>
> **MLflow** (Databricks, 2018) — Free, open-source, local-first. Install with `pip install mlflow`, works offline, stores everything in `./mlruns/`. **Use when:** Small team (<10), want full control, zero cloud costs.
>
> **Weights & Biases** (W&B, 2018) — Hosted SaaS, free tier (100 GB), real-time dashboards, team collaboration. **Use when:** Need live experiment sharing, automatic hyperparameter importance plots, hosted infrastructure.
>
> **Neptune.ai** (2016) — Enterprise focus, advanced query language, 200+ integrations. **Use when:** Large team (>20), need audit trails and compliance, budget >$50/user/month.
>
> **Comet.ml** (2017) — Similar to W&B, strong visualization features, model registry included. **Use when:** Need advanced experiment comparison (diff view, metric correlations).
>
> **Rule of thumb:** Start with MLflow (free, no lock-in). Switch to W&B/Neptune if team size or collaboration needs exceed local setup. All four track the same tuple: (code, data, params, metrics, artifacts).

---

## 1.5 · The Practitioner Workflow — Your 4-Phase Tracking Discipline

> **Warning — Two ways to read this chapter:**
> - **Theory-first (recommended for learning):** Read §0→§3 sequentially to understand the concepts, then use this workflow as your reference
> - **Workflow-first (practitioners with existing knowledge):** Use this diagram as a jump-to guide when working with real data
>
> **Note:** Section numbers don't follow phase order because the chapter teaches concepts pedagogically (theory before application). The workflow below shows how to APPLY those concepts.

**What you'll build by the end:** A reproducible ML experiment pipeline with MLflow tracking (params, metrics, artifacts), DVC data versioning, and a model registry workflow (Staging → Production). This is the system that answers "which hyperparameters gave 94%?" in 10 seconds and "can I reproduce run #47 from 3 months ago?" with confidence.

```
Phase 1: SETUP Phase 2: TRACK Phase 3: COMPARE Phase 4: PROMOTE
─────────────────────────────────────────────────────────────────────────────────────────────────
Initialize tracking: Log during training: Analyze experiments: Deploy best model:

• Start MLflow server • mlflow.log_params() • mlflow ui --port 5000 • mlflow.register_model()
• Configure backend • mlflow.log_metrics() • Sort by test_accuracy • Transition to Staging
• Set experiment name • mlflow.log_artifact() • Compare training curves • A/B test in staging
• Init DVC for data • Track Git commit • Filter by metric > X • Promote to Production
 • Version data with DVC • Download best artifacts • Version with DVC

→ DECISION: → DECISION: → DECISION: → DECISION:
 Local vs cloud storage? What to log? Which run to deploy? When to promote v2?
 • Local: ./mlruns/ • ALWAYS: lr, batch, • Best test_accuracy • v2 beats v1 by >2%
 • Team: S3 backend epochs, random_seed • Best F1 if imbalanced in staging A/B test
 • DVC remote: S3/Azure • ALWAYS: final metrics • Speed/accuracy tradeoff • Monitor for 24h
 • NEVER: intermediate • Rollback plan ready
 checkpoints (use TB)
```

> **How to use this workflow:** Phase 1 is one-time setup per project. Phase 2→3→4 is your daily loop: train experiments (Phase 2), compare results (Phase 3), deploy winners (Phase 4). The sections below teach WHY each phase works; refer back here for WHAT to do.

> **Tracking verdict:** MLflow registry promoted INT4-AWQ checkpoint — experiment audit trail confirmed 96.2% accuracy before production registration .
> ➡ Ch.10 monitors this registered model in production and detects when accuracy drifts below threshold.

---

### Phase 1: SETUP — Initialize Tracking Infrastructure

**Goal:** Set up MLflow tracking server (local or cloud), configure artifact storage, initialize DVC for data versioning. This is **one-time setup per project** — once configured, all team members log to the same backend.

**Setup checklist:**

```bash
# 1. Install dependencies
pip install mlflow dvc[s3] # Add [azure] or [gs] for other cloud providers

# 2. Initialize MLflow tracking
mlflow server \
 --backend-store-uri sqlite:///mlflow.db \
 --default-artifact-root ./mlruns \
 --host 0.0.0.0 \
 --port 5000
# For production: replace sqlite with PostgreSQL, ./mlruns with S3

# 3. Initialize DVC for data versioning
dvc init
dvc remote add -d myremote s3://my-bucket/dvc-cache # Optional: cloud storage
git add .dvc .dvcignore
git commit -m "Initialize DVC"

# 4. Set MLflow tracking URI (in your training script or environment)
export MLFLOW_TRACKING_URI=http://localhost:5000
```

**Decision checkpoint #1 — Storage backend selection:**

> **Run #0 (setup):** Team size = 3, dataset = 2GB IMDB reviews, budget = $0/month
> → **Decision:** Local SQLite + local filesystem (./mlruns/) → zero cost, sufficient for 3 people
> → **Alternative:** If team > 5 or need remote access → PostgreSQL + S3 backend (~$15/month)
> → **Result:** All experiments logged to shared ./mlruns/, Git tracks code, DVC tracks data

**Why this phase matters:** Without Phase 1, every team member logs to their laptop — experiments are siloed, comparisons require manual CSV merging, and "which hyperparameters gave 94%?" takes 2 hours instead of 10 seconds.

> 📖 **Industry Standard:** MLflow for experiment tracking, DVC for data versioning. Alternatives: Weights & Biases (hosted, $0–50/user/month), Neptune.ai (team collaboration features), Comet.ml (enterprise focus). MLflow + DVC is the **free, local-first baseline** — switch to hosted platforms when team size or storage exceeds local capacity.

---

### Phase 2: TRACK — Log Everything During Training

**Goal:** Wrap your training loop with MLflow logging calls. **Every experiment becomes a versioned tuple:** (code commit, data hash, hyperparameters, metrics, artifacts).

**Minimal production-ready training script:**

```python
import mlflow
import torch
from transformers import BertForSequenceClassification, Trainer, TrainingArguments

# Phase 1: Connect to tracking server (set in environment or here)
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("bert-sentiment-classification")

# Phase 2: Log this run
with mlflow.start_run(run_name="bert-lr2e-5-bs16-warmup100"):
 # Log hyperparameters FIRST (before training starts)
 mlflow.log_params({
 "learning_rate": 2e-5,
 "batch_size": 16,
 "warmup_steps": 100,
 "epochs": 3,
 "model": "bert-base-uncased",
 "dataset": "imdb",
 "random_seed": 42,
 "optimizer": "AdamW",
 "weight_decay": 0.01
 })

 # Set random seeds (CRITICAL for reproducibility)
 torch.manual_seed(42)
 torch.cuda.manual_seed_all(42)

 # Train model (standard Hugging Face Trainer)
 model = BertForSequenceClassification.from_pretrained("bert-base-uncased")
 training_args = TrainingArguments(
 output_dir="./results",
 learning_rate=2e-5,
 per_device_train_batch_size=16,
 num_train_epochs=3,
 warmup_steps=100,
 logging_steps=50,
 evaluation_strategy="epoch",
 save_strategy="epoch"
 )
 trainer = Trainer(model=model, args=training_args, train_dataset=train_ds, eval_dataset=val_ds)
 trainer.train()

 # Log final metrics (after training completes)
 eval_results = trainer.evaluate(test_ds)
 mlflow.log_metrics({
 "test_accuracy": eval_results["eval_accuracy"],
 "test_f1": eval_results["eval_f1"],
 "test_loss": eval_results["eval_loss"]
 })

 # Log artifacts (model weights, plots, confusion matrix)
 trainer.save_model("./final_model")
 mlflow.log_artifact("./final_model/pytorch_model.bin", artifact_path="model")
 mlflow.log_artifact("./confusion_matrix.png")

 # Log data version (DVC hash from .dvc file)
 import yaml
 with open("data/imdb_train.csv.dvc") as f:
 dvc_meta = yaml.safe_load(f)
 mlflow.set_tag("dvc_data_hash", dvc_meta["outs"][0]["md5"])

# MLflow auto-logs: start time, end time, duration, Git commit (if in repo)
```

**Decision checkpoint #2 — What to log (logging hygiene):**

> **Run #23:** val_loss = 0.35 (epoch 1) → 0.28 (epoch 2) → 0.31 (epoch 3) — overfitting detected
> → **Decision:** Log final test metrics only + best val_loss → keep MLflow DB small (<100 MB)
> → **Alternative:** Log per-epoch metrics to TensorBoard (separate tool for training curves)
> → **Result:** MLflow UI stays fast, can filter 100 runs by test_accuracy in <2s

**What to log (production checklist):**

| Category | ALWAYS log | NEVER log | OPTIONAL (log if debugging) |
|---|---|---|---|
| **Hyperparameters** | lr, batch_size, epochs, warmup, random_seed, optimizer | Unchanged defaults (tokenizer=bert-base) | weight_decay, dropout, max_seq_len |
| **Metrics** | Final test accuracy/F1/loss | Per-batch metrics (floods DB) | Best val_loss, epoch of best checkpoint |
| **Artifacts** | Final model weights, confusion matrix | Intermediate checkpoints (use model_checkpoint callback) | Training curves plot, attention weights viz |
| **Metadata** | Git commit, DVC data hash | Code diffs (use Git) | GPU utilization, peak VRAM |

> **Warning — Common mistake:** Logging 1000 metrics per epoch (train_loss at every step) → MLflow UI becomes unusably slow. **Fix:** Log summary metrics only (final test_accuracy, best val_loss). Use TensorBoard for detailed training curves.

---

### Phase 3: COMPARE — Analyze 100 Experiments Visually

**Goal:** Use MLflow UI (or programmatic API) to compare all logged runs, identify the best hyperparameters, and spot patterns (e.g., "batch_size=16 always beats 32").

**Start MLflow UI:**

```bash
mlflow ui --port 5000
# Open http://localhost:5000 in browser
```

**MLflow UI walkthrough:**

1. **Table view** — All runs in one table with sortable columns
 - Click `test_accuracy` header → sort by best accuracy
 - Click column header + `Filter` → show only runs where `test_accuracy > 0.93`

2. **Compare runs** — Select 2+ runs → Click `Compare` button
 - Side-by-side hyperparameters (which changed?)
 - Overlaid training curves (which converged faster?)
 - Metric deltas (how much did accuracy improve?)

3. **Parallel coordinates plot** — See all hyperparameters + metrics as colored lines
 - X-axis: learning_rate, batch_size, warmup_steps, test_accuracy
 - Y-axis: Normalized values [0, 1]
 - Pattern: "All green lines (high accuracy) cluster at lr=2e-5"

4. **Scatter plot** — X = one hyperparameter, Y = one metric, color = another hyperparameter
 - X = learning_rate, Y = test_accuracy, color = batch_size
 - Pattern: "lr=2e-5 beats 5e-5 regardless of batch size"

**Programmatic search (for automation):**

```python
import mlflow

# Find best run by test accuracy
best_runs = mlflow.search_runs(
 experiment_names=["bert-sentiment-classification"],
 filter_string="metrics.test_accuracy > 0.93",
 order_by=["metrics.test_accuracy DESC"],
 max_results=5
)

print("Top 5 runs:")
for idx, row in best_runs.iterrows():
 print(f"{row['run_id'][:8]} lr={row['params.learning_rate']} "
 f"acc={row['metrics.test_accuracy']:.3f} "
 f"duration={row['end_time'] - row['start_time']}")

# Download artifacts from best run
best_run_id = best_runs.iloc[0]['run_id']
mlflow.artifacts.download_artifacts(run_id=best_run_id, artifact_path="model", dst_path="./best_model")
```

**Decision checkpoint #3 — Model selection criteria:**

> **Run #47:** test_accuracy = 0.94, test_f1 = 0.93, train_time = 22 min, lr = 2e-5, batch_size = 16
> → **Decision:** Register as candidate → A/B test in staging (serve 10% traffic, monitor for 24h)
> → **Alternative:** Run #51 has same accuracy but 20 min train time → pick if retraining is frequent
> → **Result:** Deployed to staging, staging accuracy = 0.91 (3% drop), investigate data drift

**Selection criteria (in order):**

1. **Test accuracy** — Primary metric (must meet product requirement, e.g., >94%)
2. **F1 score** — Secondary metric (accounts for class imbalance)
3. **Training time** — Tiebreaker (if retraining frequently, prefer faster model)
4. **Inference latency** — Critical for production (measure separately, not in MLflow)

> 📖 **Industry Standard:** MLflow UI for visual exploration (parallel coordinates, scatter plots). Programmatic API for CI/CD automation (deploy best run from last night's sweep). Weights & Biases adds real-time collaboration (team members see each other's experiments live).

---

### Phase 4: PROMOTE — Deploy to Staging → Production

**Goal:** Register the best model in MLflow Model Registry, transition through lifecycle stages (Staging → Production), version data with DVC for reproducibility.

**Model registry workflow:**

```python
import mlflow
from mlflow.tracking import MlflowClient

client = MlflowClient()

# Step 1: Register model from best run
best_run_id = "8b4c3d1a..." # From Phase 3 selection
model_uri = f"runs:/{best_run_id}/model"
model_name = "bert-sentiment-classifier"

model_version = mlflow.register_model(model_uri, model_name)
print(f"Registered model version {model_version.version}")

# Step 2: Transition to Staging
client.transition_model_version_stage(
 name=model_name,
 version=model_version.version,
 stage="Staging",
 archive_existing_versions=False # Keep old staging versions for rollback
)

# Step 3: A/B test in staging (manual step — deploy to staging server, monitor metrics)
# - Serve 10% of staging traffic to new model
# - Monitor for 24-48 hours: accuracy, latency, error rate
# - If metrics are stable or improved → proceed to Step 4

# Step 4: Promote to Production (after staging validation)
client.transition_model_version_stage(
 name=model_name,
 version=model_version.version,
 stage="Production",
 archive_existing_versions=True # Archive old production model (enables rollback)
)

# Step 5: Version dataset with DVC
# (Run in terminal, then commit .dvc file)
# dvc add data/imdb_train.csv
# git add data/imdb_train.csv.dvc
# git commit -m "Version dataset for model v{model_version.version}"
# git tag model-v{model_version.version} # Tag Git commit for easy checkout
```

**Production deployment script (loads model by stage, not run_id):**

```python
import mlflow

# Load Production model (no hardcoded run_id!)
model = mlflow.pyfunc.load_model("models:/bert-sentiment-classifier/Production")

# Inference
predictions = model.predict(test_inputs)
```

**Decision checkpoint #4 — Promotion criteria:**

> **Model v2 (staging):** accuracy = 0.95 (vs v1 prod = 0.94), latency = 180ms (vs v1 = 150ms), 24h monitoring = stable
> → **Decision:** Promote to Production → latency increase acceptable (30ms = 20% slower but 1% accuracy gain = $50k/year revenue)
> → **Alternative:** If latency > 200ms → reject v2 → optimize (quantization, distillation) → re-test in staging
> → **Result:** v2 in Production, v1 archived (but kept for rollback), monitor for 7 days

**Promotion decision tree:**

```
Staging A/B test results:
├─ Accuracy improved by >2% AND latency <200ms
│ └─ PROMOTE to Production
├─ Accuracy improved by <2% OR latency >200ms
│ ├─ Revenue impact = +$10k/year
│ │ └─ PROMOTE (business case justified)
│ └─ Revenue impact = +$1k/year
│ └─ REJECT (not worth operational risk)
└─ Accuracy degraded OR error rate spiked
 └─ REJECT → investigate (data drift? bug in preprocessing?)
```

> **Warning — Rollback plan:** Always keep previous Production model in Registry with `stage="Archived"`. If new model fails in production, transition archived model back to Production in <5 min.

**Reproducibility verification (6 months later):**

```bash
# 1. Checkout Git tag for model version
git checkout model-v2

# 2. Restore dataset version from that commit
dvc pull # Downloads data/imdb_train.csv as it existed at model-v2 tag

# 3. Get hyperparameters from MLflow
mlflow runs describe <run_id> # Shows all logged params

# 4. Retrain with exact same config
python train.py --config configs/bert-lr2e-5-bs16.yaml

# 5. Verify metrics match (within random seed variance)
# Original: test_accuracy = 0.94
# Reproduced: test_accuracy = 0.939 (0.1% difference = acceptable variance)
```

> 📖 **Industry Standard:** MLflow Model Registry for lifecycle management (Staging → Production). Alternatives: AWS SageMaker Model Registry (tight AWS integration), Azure ML Model Registry (tight Azure integration), Kubernetes-native registries (Seldon, KServe) for containerized deployments. MLflow is **cloud-agnostic** — works with any deployment target (Flask API, AWS Lambda, Docker, Kubernetes).

---

### Workflow Summary — 4 Phases in Practice

**Time breakdown (for 12-run hyperparameter sweep):**

| Phase | Duration | Tools | Output |
|---|---|---|---|
| **Setup** | 15 min (one-time) | MLflow server, DVC init | Tracking infrastructure ready |
| **Track** | 4 hours (parallel on 5 GPUs) | MLflow logging in training loop | 12 runs logged with params/metrics/artifacts |
| **Compare** | 10 min | MLflow UI, programmatic search | Best run identified (run_id, hyperparameters) |
| **Promote** | 30 min | Model Registry, DVC, Git tags | Model in Production, dataset versioned |

**Before tracking (manual process):**
- Compare 12 runs: 2 hours (scroll through notebooks, copy metrics to spreadsheet)
- Reproduce experiment: Impossible (hyperparameters lost, dataset overwritten)
- Deploy model: 30 min (manually copy weights, no versioning)

**After tracking (automated process):**
- Compare 12 runs: 10 seconds (sort by test_accuracy in MLflow UI)
- Reproduce experiment: 5 min (git checkout + dvc pull + python train.py)
- Deploy model: 1 min (mlflow.pyfunc.load_model by stage)

> **The 100× speedup:** Tracking doesn't make training faster — it makes **everything around training** 10–100× faster. Finding the best hyperparameters (2 hours → 10s), reproducing experiments (impossible → 5 min), deploying models (30 min → 1 min). The ROI is immediate for teams running >10 experiments/week.

---

## 2 · Running Example — BERT Fine-Tuning Hyperparameter Sweep

You're tuning **BERT-base-uncased** for sentiment classification on the IMDB review dataset (25k train, 25k test). The product team needs **94% test accuracy**. You'll run a grid search over:

| Hyperparameter | Values to try |
|---|---|
| Learning rate | `[2e-5, 3e-5, 5e-5]` |
| Batch size | `[16, 32]` |
| Warmup steps | `[100, 500]` |

**Total experiments:** 3 LR × 2 batch sizes × 2 warmup = **12 runs** (each run = 3 epochs, ~20 min on one GPU)

**Constraint:** 3 team members share 5 GPUs. Without tracking, you'd be manually logging results in a Google Sheet. With MLflow, every run is logged automatically.

### [Phase 1: SETUP] Step 1: Initialize Tracking & Log First Experiment

**Before training:**
```python
import mlflow

mlflow.start_run(run_name="bert-baseline")
mlflow.log_params({
 "learning_rate": 5e-5,
 "batch_size": 32,
 "epochs": 3,
 "warmup_steps": 500,
 "model": "bert-base-uncased",
 "dataset": "imdb",
 "random_seed": 42
})
```

**During training (called each epoch):**
```python
mlflow.log_metrics({
 "train_loss": 0.45,
 "train_accuracy": 0.87,
 "val_accuracy": 0.89
}, step=epoch)
```

**After training:**
```python
mlflow.log_artifact("model.pt")
mlflow.log_artifact("confusion_matrix.png")
mlflow.end_run()
```

**What you get:** A unique `run_id` (UUID like `7a3f9b2e-4c8d-4f9a-8b2c-1e5d7f9a3c6b`) that points to:
- Logged hyperparameters (7 key-value pairs)
- Logged metrics (3 epochs × 3 metrics = 9 data points)
- Logged artifacts (2 files: `model.pt` + `confusion_matrix.png`)
- Automatic metadata (start time, duration, Git commit if inside a repo)

### [Phase 2: TRACK] Step 2: Log Experiments During Training Loop

```python
for lr in [2e-5, 3e-5, 5e-5]:
 for batch_size in [16, 32]:
 for warmup in [100, 500]:
 mlflow.start_run(run_name=f"bert-lr{lr}-bs{batch_size}-warmup{warmup}")
 mlflow.log_params({
 "learning_rate": lr,
 "batch_size": batch_size,
 "warmup_steps": warmup,
 "epochs": 3,
 "random_seed": 42
 })

 # Train model (pseudocode)
 model = train_bert(lr=lr, batch_size=batch_size, warmup=warmup)

 # Log final metrics
 mlflow.log_metrics({
 "test_accuracy": model.evaluate(test_data),
 "test_f1": model.f1_score(test_data)
 })

 # Save artifacts
 mlflow.log_artifact("model.pt")
 mlflow.end_run()
```

**What you get:** 12 runs logged in ~4 hours. Now you can compare them visually.

### [Phase 3: COMPARE] Step 3: Analyze Experiments in MLflow UI

Start the MLflow tracking server:
```bash
mlflow ui --port 5000
```

Open `http://localhost:5000` in your browser. You see:

| run_id | learning_rate | batch_size | warmup_steps | test_accuracy | test_f1 | duration |
|---|---|---|---|---|---|---|
| `7a3f9b2e...` | 5e-5 | 32 | 500 | 0.89 | 0.88 | 18 min |
| `8b4c3d1a...` | 2e-5 | 16 | 100 | **0.94** | **0.93** | 22 min |
| `9c5d2e3b...` | 3e-5 | 32 | 500 | 0.92 | 0.91 | 19 min |
| ... | ... | ... | ... | ... | ... | ... |

**Visual tools:**
- **Parallel coordinates plot** — See all 12 runs as colored lines across hyperparameters and metrics
- **Scatter plot** — X = learning rate, Y = accuracy, color = batch size
- **Metrics timeline** — Compare training curves (loss per epoch) across runs

**Find best run programmatically:**
```python
best_run = mlflow.search_runs(
 filter_string="metrics.test_accuracy > 0.93",
 order_by=["metrics.test_accuracy DESC"],
 max_results=1
)
print(f"Best run: {best_run['run_id'][0]} with {best_run['metrics.test_accuracy'][0]:.2%} accuracy")
```

> **What this MLflow UI enables:** Instead of manually comparing 12 runs in spreadsheets or Jupyter notebooks, you now:
> - **Answer "which hyperparameters gave 94%?" in 10 seconds** (not 30 minutes of scrolling)
> - **Visualize 50 experiments at once** in parallel coordinates plot (spot patterns like "batch_size=16 always beats 32")
> - **Compare training curves side-by-side** (see if learning rate 2e-5 converges faster than 5e-5)
> - **Filter runs programmatically** with SQL-like queries ("show me all runs where test_accuracy > 0.93 AND duration < 20 min")
> - **Share results with teammates** via one URL (http://localhost:5000) instead of emailing CSV files

### [Phase 4: PROMOTE] Step 4: Register & Deploy Best Model

Once you've identified the best run, promote it to the **model registry**:

```python
# Register model from best run
mlflow.register_model(
 model_uri=f"runs:/{best_run_id}/model",
 name="bert-sentiment-classifier"
)

# Transition to Staging
client = mlflow.tracking.MlflowClient()
client.transition_model_version_stage(
 name="bert-sentiment-classifier",
 version=1,
 stage="Staging"
)
```

**Model registry lifecycle:**

```
None → Staging → Production → Archived
```

- **None** — Model just registered, not deployed anywhere
- **Staging** — Model deployed to staging environment for QA testing
- **Production** — Model deployed to production serving live traffic
- **Archived** — Old model, replaced by newer version

**Why this matters:** Your deployment script can pull models by stage:
```python
# Load model from Production stage (no hardcoded run_id!)
model = mlflow.pyfunc.load_model("models:/bert-sentiment-classifier/Production")
```

When you want to deploy a new model, just transition it to Production — the deployment code doesn't change.

### [Phase 4: PROMOTE] Step 5: Version Dataset for Reproducibility

**Problem:** You've tracked hyperparameters and metrics, but the training dataset (`imdb_train.csv`) has been modified 3 times this week. Which version produced the 94% run?

**Solution:** Track the dataset with DVC (like Git for data):

```bash
# Initialize DVC in your repo
dvc init

# Track the training dataset
dvc add data/imdb_train.csv

# This creates data/imdb_train.csv.dvc (metadata file)
# Commit both the .dvc file and the actual data hash
git add data/imdb_train.csv.dvc data/.gitignore
git commit -m "Track training dataset v1"
```

**What DVC does:**
- Computes SHA256 hash of `imdb_train.csv` → `a8d3c1f7b2e4...`
- Stores the hash in `imdb_train.csv.dvc` (tiny metadata file tracked by Git)
- Moves the actual CSV to `.dvc/cache/` (ignored by Git)
- When you `dvc push`, uploads the CSV to remote storage (S3, Azure Blob, local NAS)

**Reproducing an experiment 3 months later:**
```bash
# Checkout the Git commit from the experiment
git checkout 7a3f9b2

# Restore the dataset version from that commit
dvc pull

# Now data/imdb_train.csv is the exact version from 3 months ago
# Run training again with the same hyperparameters
python train.py --run-id 7a3f9b2e-4c8d-4f9a-8b2c-1e5d7f9a3c6b
```

**The complete versioning story:**
- **Code version** — Git commit `7a3f9b2`
- **Data version** — DVC hash `a8d3c1f` (stored in `imdb_train.csv.dvc`)
- **Hyperparameters** — MLflow logged params (lr=2e-5, batch_size=16, ...)
- **Results** — MLflow logged metrics (accuracy=0.94)

Change any one component → you get a different result. Keep all components the same → you get **exact reproducibility**.

> 📖 **Industry Standard: Data Versioning Tools**
>
> **DVC (Data Version Control)** — Git-like CLI (`dvc add`, `dvc push`, `dvc pull`), supports S3/Azure/GCS/SSH remotes, integrates with CI/CD. **Use when:** Need Git-like workflow, want to version datasets alongside code.
>
> **Git LFS (Large File Storage)** — Native Git extension, transparent to users (files auto-push/pull). **Use when:** Datasets <2 GB, already using GitHub/GitLab (both support LFS).
>
> **LakeFS** — S3-compatible Git for data lakes, branch/merge/revert datasets like code. **Use when:** Managing petabyte-scale data lakes, need multi-user branching.
>
> **Delta Lake** — ACID transactions on Parquet files, time travel queries. **Use when:** Using Databricks/Spark, need SQL-queryable version history.
>
> **Key tradeoff:** DVC is **file-centric** (version CSV/Parquet files), Delta Lake is **table-centric** (version SQL tables). DVC works with any storage, Delta Lake requires Spark. For ML experiments tracking a single CSV/NPZ file, DVC is the simplest choice.

---

## 3 · Mental Model — The Experiment Tracking Stack

```
┌─────────────────────────────────────────────────────────────────────┐
│ EXPERIMENT TRACKING STACK │
│ │
│ USER INTERFACE LAYER │
│ ┌────────────────────────────────────────────────────────────────┐ │
│ │ MLflow UI (localhost:5000) W&B Dashboard (cloud) │ │
│ │ • Compare runs (table, plots) • Team collaboration │ │
│ │ • Search by metrics/params • Real-time sync │ │
│ │ • Download artifacts • Hosted storage │ │
│ └───────────────────────────┬────────────────────────────────────┘ │
│ │ │
│ TRACKING LAYER │
│ ┌────────────────────────────▼──────────────────────────────────┐ │
│ │ MLflow Tracking API │ │
│ │ • mlflow.log_params({"lr": 2e-5, "batch_size": 16}) │ │
│ │ • mlflow.log_metrics({"accuracy": 0.94}, step=epoch) │ │
│ │ • mlflow.log_artifact("model.pt") │ │
│ │ • mlflow.start_run() / mlflow.end_run() │ │
│ └───────────────────────────┬────────────────────────────────────┘ │
│ │ │
│ STORAGE LAYER │
│ ┌────────────────────────────▼──────────────────────────────────┐ │
│ │ Local Filesystem (./mlruns/) Cloud Storage (optional) │ │
│ │ • mlruns/0/7a3f9b2e.../ • S3 / Azure Blob │ │
│ │ ├── params/ • Centralized team store │ │
│ │ ├── metrics/ │ │
│ │ └── artifacts/model.pt │ │
│ └────────────────────────────────────────────────────────────────┘ │
│ │
│ DATA VERSIONING LAYER │
│ ┌────────────────────────────────────────────────────────────────┐ │
│ │ DVC (Data Version Control) │ │
│ │ • dvc add data/train.csv → .dvc/cache/a8d3c1f7... │ │
│ │ • dvc push → Upload to remote (S3, GCS, Azure, SSH) │ │
│ │ • dvc pull → Download specific version by Git commit │ │
│ └────────────────────────────────────────────────────────────────┘ │
│ │
│ MODEL REGISTRY LAYER │
│ ┌────────────────────────────────────────────────────────────────┐ │
│ │ MLflow Model Registry │ │
│ │ • Register model from best run │ │
│ │ • Lifecycle: None → Staging → Production → Archived │ │
│ │ • Load by stage: models:/bert-classifier/Production │ │
│ └────────────────────────────────────────────────────────────────┘ │
└───────────────────────────────────────────────────────────────────┘
```

**Key insight:** Tracking is **local-first** (works with no internet), **Git-integrated** (code and metadata versioned together), and **artifact-agnostic** (logs any file: models, plots, configs, datasets).

**When to use cloud storage:**
- **MLflow** — Switch from local `./mlruns/` to S3 backend when you need team collaboration (everyone queries the same tracking server)
- **DVC** — Switch from local cache to cloud remote when datasets are too large for laptop storage (push/pull like Git LFS)
- **W&B** — Use the hosted free tier (100 GB storage) if you want real-time dashboards without managing infrastructure

> 📖 **Industry Standard: Model Registry Lifecycle**
>
> **MLflow Model Registry** — Built into MLflow, lifecycle stages (None → Staging → Production → Archived), load by stage (`models:/name/Production`). **Use when:** Using MLflow for tracking, need simple stage-based deployment.
>
> **AWS SageMaker Model Registry** — Tight integration with SageMaker endpoints, auto-scaling, A/B testing built-in. **Use when:** Already on AWS, need managed inference infrastructure.
>
> **Azure ML Model Registry** — Tight integration with Azure ML pipelines, approval workflows, RBAC. **Use when:** Already on Azure, need enterprise governance.
>
> **Kubernetes-native registries (Seldon, KServe)** — Container-based model serving, declarative YAML configs. **Use when:** Deploying to Kubernetes, need multi-framework support (ONNX, TensorFlow, PyTorch).
>
> **Key difference:** MLflow is **cloud-agnostic** (works with Flask, Docker, AWS, Azure, GCP). SageMaker/Azure ML are **cloud-specific** (lock-in). Seldon/KServe are **Kubernetes-specific** (infrastructure dependency).

---

## Key Diagrams

<!-- TODO: add key diagrams -->

---

## 4 · What Can Go Wrong — Tracking Pitfalls and Fixes

| Pitfall | Symptom | Fix |
|---|---|---|
| **Forgot to log random seed** | "I got 94% yesterday but 91% today with the same hyperparameters!" | Always log `random_seed` as a parameter. Set seeds for Python, NumPy, PyTorch, and CUDA. |
| **Logged metrics but not hyperparameters** | "This run got 94% accuracy but I don't know which learning rate I used." | Call `mlflow.log_params()` at the start of every run. Log *all* config values (lr, batch size, optimizer, model architecture). |
| **Deleted local model files** | "The registry points to `model.pt` but the file is gone." | Use MLflow's model logging (`mlflow.pytorch.log_model()`) instead of manually saving files. MLflow copies artifacts to `./mlruns/` storage. |
| **Dataset changed but not versioned** | "Results from last week aren't reproducible — I think we cleaned the dataset since then." | Track datasets with DVC. Every time you modify a dataset, run `dvc add data/train.csv` and commit the `.dvc` file. |
| **Didn't log Git commit** | "I can't find the code version that produced this run." | MLflow auto-logs Git commit if you run inside a Git repo. If not, manually log: `mlflow.set_tag("git_commit", <commit_hash>)`. |
| **Multiple team members overwrite each other's runs** | "My experiment from this morning disappeared." | Use a shared MLflow tracking server (remote backend) or unique experiment names per person. |
| **Logged too many metrics (1000+ per epoch)** | "MLflow UI is slow and the database is 10 GB." | Log summary metrics only (final test accuracy, best val loss). Use TensorBoard for detailed training curves. |
| **No experiment naming convention** | "There are 500 runs named 'test', 'final', 'final_v2', 'final_final'." | Use descriptive run names: `bert-lr2e-5-bs16-warmup100` or auto-generate: `f"{model}-{datetime.now()}"`. |

---

## 5 · Progress Check — Can You Identify the Best Run?

**Scenario:** You're looking at the MLflow UI after running a hyperparameter sweep. The table shows:

| run_id | learning_rate | batch_size | warmup_steps | test_accuracy | test_f1 | train_time |
|---|---|---|---|---|---|---|
| `a1b2c3d4...` | 5e-5 | 32 | 500 | 0.89 | 0.88 | 18 min |
| `b2c3d4e5...` | 2e-5 | 16 | 100 | 0.94 | 0.93 | 22 min |
| `c3d4e5f6...` | 3e-5 | 32 | 500 | 0.92 | 0.91 | 19 min |
| `d4e5f6g7...` | 2e-5 | 32 | 100 | 0.94 | 0.92 | 20 min |
| `e5f6g7h8...` | 5e-5 | 16 | 500 | 0.91 | 0.90 | 21 min |

**Questions:**

1. **Which run should you deploy to production?**
 - *Answer:* `b2c3d4e5...` (highest test_accuracy **and** test_f1)
 - *Why not `d4e5f6g7...`?* Same accuracy but lower F1 — likely worse precision/recall balance
 - *Why not just pick max accuracy?* F1 score accounts for class imbalance (important for sentiment where "negative" class might be rare)

2. **Which hyperparameter had the biggest impact on accuracy?**
 - *Answer:* Learning rate (2e-5 consistently better than 5e-5)
 - *How to verify:* Sort by `learning_rate`, compare mean accuracy per group

3. **If training time is critical, which run is the best speed/accuracy tradeoff?**
 - *Answer:* `a1b2c3d4...` (18 min, 89% accuracy) — if 89% is acceptable
 - *But if 94% is required:* `d4e5f6g7...` (20 min, 94% accuracy) — 2 min faster than the best run

4. **How would you reproduce run `b2c3d4e5...` in 6 months?**
 ```python
 # 1. Get run metadata
 run = mlflow.get_run("b2c3d4e5...")
 git_commit = run.data.tags["mlflow.source.git.commit"]

 # 2. Checkout code version
 git checkout <git_commit>

 # 3. Pull data version
 dvc pull # Restores data/imdb_train.csv to the version from that commit

 # 4. Retrain with logged hyperparameters
 python train.py \
 --learning-rate 2e-5 \
 --batch-size 16 \
 --warmup-steps 100 \
 --random-seed 42 # Must be logged in the original run!
 ```

---

## Where This Reappears

<!-- TODO: add forward pointer table -->

---

## 6 · Bridge to Ch.10 — Production ML Monitoring & A/B Testing

You've just logged 100 experiments, registered the best model, and versioned your dataset. **Congratulations — your model is reproducible.** But the moment you deploy it to production, new questions emerge:

### What Ch.10 Answers

**Question 1: "The model got 94% accuracy in training, but users are complaining it's wrong. What happened?"**
→ Ch.10 teaches **data drift detection** (Evidently AI): monitor if production input distribution shifted (e.g., users are now submitting movie reviews instead of product reviews)

**Question 2: "We deployed model v2 (95% accuracy) but revenue dropped. Should we roll back to v1 (94%)?"**
→ Ch.10 teaches **A/B testing**: serve 50% of traffic to v1, 50% to v2, measure business metrics (CTR, revenue), keep the winner

**Question 3: "How do we know if the model is still working correctly after 3 months in production?"**
→ Ch.10 teaches **model performance monitoring**: log predictions + ground truth (when available), track accuracy/precision/recall over time, alert when metrics degrade

**Question 4: "The model crashes every Tuesday at 3 PM. How do we debug?"**
→ Ch.10 teaches **inference logging**: log every prediction (input, output, latency, errors) to a time-series database, correlate crashes with input patterns

**The operational workflow:**

```
Ch.9 (Experiment Tracking) Ch.10 (Production Monitoring)
┌─────────────────────────┐ ┌──────────────────────────┐
│ 1. Train 100 models │ │ 5. Monitor predictions │
│ 2. Log all experiments │ → │ 6. Detect drift │
│ 3. Pick best model │ │ 7. A/B test v1 vs v2 │
│ 4. Register in registry │ │ 8. Roll back if worse │
└─────────────────────────┘ └──────────────────────────┘
 ↑ │
 └─────────────────────────────────────┘
 Retrain with new data (Ch.9 loop)
```

**What you bring from Ch.9:**
- Model registry (load Production model by stage, not hardcoded path)
- Experiment provenance (when Production model fails, find the run_id and retrain with different hyperparameters)
- Data versioning (compare training data distribution vs production data distribution to diagnose drift)

**What you'll gain in Ch.10:**
- Real-time monitoring dashboards (latency, throughput, error rate)
- Automated drift alerts (email when input distribution shifts)
- A/B testing framework (split traffic, measure impact, promote winner)
- Rollback procedures (deploy v1 → v2 → detect regression → rollback to v1 in <5 min)

---

## Navigation

**AI Infrastructure Track:**
- [← Ch.8: Feature Stores & Data Infrastructure](../ch08_feature_stores)
- [→ Ch.10: Production ML Monitoring & A/B Testing](../production_monitoring) *(planned)*
- [↑ Track README](../README.md)

**Related Content:**
- [DevOps Fundamentals](../../devops_fundamentals) — Git, CI/CD, Docker (prerequisites for reproducibility)
- [ML Track: Hyperparameter Tuning](../../ml/01_regression/ch07_hyperparameter_tuning) — Grid search, random search, Bayesian optimization
- [AI Track: Evaluating AI Systems](../../ai/evaluating_ai_systems) — Metrics, benchmarks, human evaluation

**Tools Documentation:**
- [MLflow Official Docs](https://mlflow.org/docs/latest/index.html)
- [DVC Official Docs](https://dvc.org/doc)
- [Weights & Biases Quickstart](https://docs.wandb.ai/quickstart)

---

**Last updated:** April 26, 2026
**Status:** README complete — notebook + assets in progress
