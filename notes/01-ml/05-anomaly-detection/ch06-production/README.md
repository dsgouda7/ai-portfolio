# Ch.6 — Production & Real-Time Inference

> **The story.** In **1999**, PayPal launched with a fraud problem that nearly killed the company. Their early rule-based system was blocking legitimate users while fraudsters adapted their patterns within days. By 2001, they deployed one of the first real-time ML fraud detection systems in fintech — a logistic regression model scoring every transaction in under 200ms — and cut fraud losses by 60%. A decade later, **Stripe** (founded 2010) went further: by 2012 they had a fully automated ML pipeline scoring every payment in under 50ms, retraining every 24 hours as fraud patterns evolved. The lesson both companies learned the hard way: a research model with 82% recall deployed on day one will drift to 60% recall within six months if left unattended. Fraudsters probe defences systematically, share tactics on dark-web forums, and rotate card numbers faster than static models can track. **The production gap** is real — your offline evaluation is an upper bound, not a guarantee. This chapter closes that gap.
>
> **Where you are in the curriculum.** Ch.5 built an ensemble of four detectors (Z-score, Isolation Forest, Autoencoder, One-Class SVM) that achieved **82% recall @ 0.5% FPR** in offline evaluation — the Detection constraint is met *in the lab*. This final FraudShield chapter transforms the lab prototype into a system that handles 1,000 transactions per second, scores each in under 100ms, monitors for drift, and automatically retrains when recall begins to slide. When you finish, all five FraudShield constraints are satisfied.
>
> **Notation in this chapter.** $\text{PSI}$ — Population Stability Index; $D_{\text{KS}}$ — Kolmogorov–Smirnov statistic; $p_{\text{ref}}$ — reference (training) distribution; $p_{\text{cur}}$ — current (live) distribution; $\delta_f$ — feature drift magnitude; $\delta_c$ — concept drift (change in $P(y \mid \mathbf{x})$); $\tau_{\text{retrain}}$ — retraining trigger threshold; $R(t)$ — recall at time $t$ days; $T_{\text{budget}}$ — end-to-end latency budget (100ms).

---

## 0 · The Challenge — Where We Are

> 💡 **FraudShield constraints — status after Ch.5:**
> 1. ⚡ **DETECTION** 82% recall @ 0.5% FPR ← **met offline, must maintain in production**
> 2. ⚡ **PRECISION** <0.5% FPR ← met offline
> 3. ⚡ **EXPLAINABILITY** SHAP attributions per transaction ← met in Ch.5
> 4. ❌ **LATENCY** <100ms end-to-end ← **hard requirement — real-time card payments cannot wait**
> 5. ❌ **DRIFT RESILIENCE** Recall must stay >80% as fraud patterns evolve ← **not yet addressed**

**What's blocking us:**

The ensemble model is a static artifact. The world it was trained on does not stand still:

- Fraudsters rotate stolen card numbers within 48 hours of a data breach
- New attack vectors (account takeover, synthetic identity, card-present skimming) emerge monthly
- Legitimate spending patterns shift seasonally, making the training distribution stale

**The production gap** (the number you should be worried about):

| Evaluation setting | Recall | When measured |
|--------------------|--------|---------------|
| Offline test set (Ch.5) | 82% | One-time snapshot |
| Live production — month 1 | ~82% | Fresh model, patterns match |
| Live production — month 3 | ~71% | New fraud tactics emerging |
| Live production — month 6 | ~58% | Model is significantly stale |
| Live production — month 12 | ~45% | Worse than a simple Z-score! |

Without a monitoring and retraining pipeline, Ch.5's ensemble degrades into noise.

**What this chapter delivers:**
- A latency budget that proves we can score under 100ms (constraint #4 ✅)
- A drift detection system using PSI that triggers retraining before recall collapses (constraint #5 ✅)
- A blue-green deployment strategy that prevents a bad retrain from going live

✅ **FraudShield is complete after this chapter** — all five constraints satisfied.

---

## Animation

![FraudShield mission complete: drift detected, retraining triggered, recall restored](img/ch06-production-needle.gif)

---

## 1 · Core Idea

Serving speed requires pre-computing features in a feature store and running model inference as a cached, low-latency service — the 30ms + 45ms + 15ms budget in §4.1 shows exactly how to allocate the 100ms window. Model quality degrades as fraud patterns drift away from the training distribution — Population Stability Index (PSI) on feature distributions and weekly recall evaluation on a labeled holdout together give early warning before business impact is felt. A retraining pipeline triggered by PSI > 0.25 or recall < 78% closes the loop: new data in, updated model out, blue-green deployment ensures the new model is validated in shadow mode before it takes live traffic.

---

## 1.5 · The Practitioner Workflow — Your 4-Phase Production System

**Before diving into theory, understand the operational workflow you'll execute at production scale:**

> 📊 **What you'll build by the end:** A self-healing fraud detection system scoring 1,000 tx/sec with <100ms latency that automatically detects drift, triggers retraining, and validates model updates before deployment.

```
Phase 1: DEPLOY              Phase 2: MONITOR            Phase 3: DETECT             Phase 4: RETRAIN
────────────────────────────────────────────────────────────────────────────────────────────────────
Containerize & serve:        Track live metrics:         Spot degradation:           Auto-update pipeline:

• Package model artifacts    • Log every score/decision  • Compute hourly PSI        • Pull 30d labeled data
• FastAPI endpoint <50ms     • Compute feature PSI       • Evaluate weekly recall    • Retrain ensemble
• Health checks + warmup     • Track p50/p99 latency     • Compare to thresholds     • Shadow deploy (48h)
• Feature store integration  • Dashboard (Grafana)       • Fire trigger if breach    • Blue-green cutover

→ DECISION:                  → DECISION:                 → DECISION:                 → DECISION:
  Deployment strategy?         Which metrics matter?       When to retrain?            Deploy or rollback?
  • Docker + K8s              • PSI > 0.25 → drift        • PSI > 0.25 OR             • Green recall ≥ Blue?
  • BentoML for packaging     • Recall < 78% → degrade    • Recall < 78%              • Latency ≤ 100ms?
  • Prometheus metrics        • p99 > 100ms → SLA miss    • Min 14d interval          • → Cutover or alert
```

**The workflow maps to this chapter:**
- **Phase 1 (DEPLOY)** → §3 Production Pipeline, §4.1 Latency Budget
- **Phase 2 (MONITOR)** → §4.2 PSI, §6 Full Monitoring Walkthrough, §7 Key Diagrams
- **Phase 3 (DETECT)** → §4.3 Recall Degradation, §5 Act 1–2 (Drift Revealed)
- **Phase 4 (RETRAIN)** → §4.4 Trigger Logic, §5 Act 3–4 (Retraining & Blue-Green)

> 💡 **Usage note:** Phases 1–3 run continuously in production. Phase 4 fires only when drift/degradation triggers cross thresholds. Each phase has explicit decision criteria — no manual judgment calls during incidents.

**Industry tooling reference:**

| Phase | Standard tools | What FraudShield uses |
|-------|---------------|----------------------|
| Deploy | Docker, Kubernetes, BentoML, Seldon | **FastAPI** (REST API), **ONNX Runtime** (inference), **Redis** (feature store) |
| Monitor | Prometheus, Grafana, DataDog | **Prometheus** (metrics), **Grafana** (dashboards) |
| Detect | Evidently AI, NannyML, Arize | **Custom PSI calculator**, **delayed-label holdout** |
| Retrain | Airflow, Kubeflow, MLflow | **Airflow DAG** (orchestration), **MLflow** (model registry), **Kubernetes CronJob** |

> 📖 **Alternative: BentoML end-to-end.** If you're building this from scratch, [BentoML](https://bentoml.com) provides integrated model serving, monitoring, and deployment in one framework. The patterns here (PSI, blue-green, latency budgets) are framework-agnostic and apply to any production ML stack.

---

## 2 · Running Example

FraudShield handles **1,000 transactions per second** at peak (e.g., Black Friday, post-breach card rotation events). Each transaction must be scored and a block/approve decision returned **within 100ms** — this is a hard SLA imposed by card network rules (Visa/Mastercard mandate sub-100ms authorization responses). Feature computation must complete in **under 50ms** so that model inference and post-processing have room within the budget.

**Three operational timescales:**

| Timescale | Activity | Trigger |
|-----------|----------|---------|
| Per-transaction (~ms) | Score, decide, log | Every incoming payment |
| Hourly | Compute rolling PSI on last hour's transactions | Scheduled |
| Weekly | Evaluate recall on labeled holdout (confirmed fraud cases) | Scheduled |
| On-demand | Retrain, validate, shadow-deploy | PSI > 0.25 OR recall < 78% |

**The labeled holdout problem.** In production, fraud labels arrive with a delay: a cardholder might not notice unauthorized charges for days, and chargebacks may take weeks to process. FraudShield maintains a **delayed-label holdout** — a rolling window of 10,000 transactions from 30–60 days ago whose labels have now settled. Recall on this holdout is the ground-truth signal for model health.

---

## 3 · **[Phase 1: DEPLOY]** Production Pipeline at a Glance

Before diving into the math, here is the full production data flow. Each numbered step has a deep-dive in the sections that follow.

```
1. Transaction arrives (card swipe / e-commerce checkout)
   └─ Raw features: card_id, merchant, amount, timestamp, location, ...

2. Feature Store lookup (pre-computed features, <5ms)
   └─ Customer velocity: tx_count_1h, amt_sum_24h, new_merchant_flag
   └─ Card features: days_since_issue, international_flag
   └─ Historical stats: mean_amount, std_amount (cached per card_id)

3. Real-time feature computation (<30ms total)
   └─ Z-score: (amount - mean_amount) / std_amount
   └─ Velocity ratio: tx_count_1h / baseline_hourly_rate
   └─ Cross-features: amount x new_merchant_flag

4. Model inference — ensemble scoring (<45ms total)
   └─ Isolation Forest: contamination=0.005 (pre-loaded)
   └─ Autoencoder: ONNX runtime, reconstruction error
   └─ One-Class SVM: support vectors cached
   └─ Weighted score fusion

5. Post-processing (<15ms)
   └─ Apply threshold (tau = 0.63 calibrated on validation set)
   └─ Generate SHAP attribution string if score > tau

6. Decision returned to payment processor
   └─ APPROVE -> transaction proceeds
   └─ BLOCK + reason -> cardholder sees "unusual activity" message

7. Logging (async, does not add to latency)
   └─ score, features, decision, timestamp -> monitoring DB

8. Monitoring (periodic — does not affect inference path)
   └─ Hourly PSI on feature distributions
   └─ Weekly recall on delayed-label holdout
   └─ Alert -> retrain trigger if thresholds exceeded

9. Retraining pipeline (on trigger)
   └─ Pull last 30 days of labeled transactions
   └─ Retrain ensemble (same architecture as Ch.5)
   └─ Shadow deployment: run new model in parallel, compare scores
   └─ Blue-green cutover when shadow recall >= current model recall

10. Monitoring continues on the new model
```

---

## 4 · The Math — Every Number Shown

### 4.1 · **[Phase 1: DEPLOY]** Latency Budget: Allocating 100ms

The end-to-end latency SLA is $T_{\text{budget}} = 100\text{ms}$. The pipeline has three sequential phases, each with a hard sub-budget:

$$T_{\text{budget}} = T_{\text{features}} + T_{\text{inference}} + T_{\text{post}} \leq 100\text{ms}$$

**Baseline budget allocation:**

| Phase | Sub-budget | Actual | Headroom |
|-------|-----------|--------|----------|
| Feature computation | 35ms | 30ms | +5ms |
| Model inference | 50ms | 45ms | +5ms |
| Post-processing + SHAP | 20ms | 15ms | +5ms |
| **Total** | **105ms** | **90ms** | **+10ms** |

$$30 + 45 + 15 = 90\text{ms} < 100\text{ms} \quad \checkmark$$

**What if model inference balloons to 70ms?**

Suppose the ensemble grows (more trees, larger autoencoder) and inference climbs from 45ms to 70ms:

$$T_{\text{total}} = 30 + 70 + 15 = 115\text{ms} > 100\text{ms} \quad \times$$

You have a **15ms deficit**. Three options, evaluated in order of cost:

1. **Trim feature computation** (cheapest): Move 3 expensive cross-features from real-time computation to the feature store (pre-compute hourly). Saves ~8ms. New total: 115 − 8 = 107ms. Still over budget.

2. **Quantize the autoencoder** (medium cost): ONNX INT8 quantization reduces autoencoder inference from ~18ms to ~9ms. Combined with option 1: 107 − 9 = 98ms. Now within budget. ✅

3. **Reduce SVM support vectors** (last resort): Re-train One-Class SVM with `nu=0.008` (fewer support vectors). Cuts SVM from ~20ms to ~11ms but slightly degrades recall by ~0.5%.

**Rule of thumb**: allocate model inference at most 50% of the total budget. Feature computation and post-processing always consume more wall-clock time than expected once you add network round-trips to the feature store.

> **Decision Checkpoint #1: Latency budget exceeded**  
> **Scenario:** Model inference balloons from 45ms → 70ms after adding more trees to Isolation Forest.  
> **Data:** Budget deficit = 70 + 30 + 15 = 115ms (15ms over SLA).  
> **Decision:** Apply ONNX INT8 quantization to autoencoder (saves 9ms) + move 3 cross-features to feature store pre-compute (saves 8ms). New total: 115 − 17 = 98ms ✅ Within budget.  
> **Why this matters:** Missing the 100ms SLA by even 20ms triggers card network penalties. Quantization trades 0.3% recall for 9ms — always worth it at this scale.

**Code Snippet — Phase 1: FastAPI Model Serving Endpoint**

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import joblib
from typing import Dict
import time

app = FastAPI(title="FraudShield API")

# Load ensemble models at startup (warmup phase)
isolation_forest = joblib.load("models/isolation_forest.pkl")
autoencoder = joblib.load("models/autoencoder.onnx")  # ONNX for speed
ocsvm = joblib.load("models/ocsvm.pkl")

class Transaction(BaseModel):
    card_id: str
    amount: float
    merchant: str
    timestamp: int
    # ... 26 other features

class PredictionResponse(BaseModel):
    score: float
    decision: str  # "APPROVE" or "BLOCK"
    latency_ms: float
    shap_top3: Dict[str, float]  # Top 3 attribution features

@app.post("/predict", response_model=PredictionResponse)
async def predict(tx: Transaction):
    start_time = time.perf_counter()
    
    # Phase 1a: Feature store lookup (cached, <5ms)
    features = fetch_precomputed_features(tx.card_id)  # Redis lookup
    
    # Phase 1b: Real-time feature computation (<30ms)
    features["z_score"] = (tx.amount - features["mean_amount"]) / features["std_amount"]
    features["velocity_ratio"] = features["tx_count_1h"] / features["baseline_rate"]
    
    # Phase 1c: Ensemble inference (<45ms total)
    X = np.array([[features[f] for f in FEATURE_ORDER]])
    
    iso_score = isolation_forest.decision_function(X)[0]
    ae_score = autoencoder_reconstruction_error(X)  # ONNX runtime
    svm_score = ocsvm.decision_function(X)[0]
    
    # Weighted fusion (Ch.5 calibrated weights)
    ensemble_score = 0.35 * iso_score + 0.40 * ae_score + 0.25 * svm_score
    
    # Phase 1d: Decision + SHAP (<15ms)
    decision = "BLOCK" if ensemble_score > 0.63 else "APPROVE"
    shap_values = compute_shap_if_blocked(X) if decision == "BLOCK" else {}
    
    latency_ms = (time.perf_counter() - start_time) * 1000
    
    # Async logging (non-blocking)
    log_transaction_async(tx, ensemble_score, decision, latency_ms)
    
    return PredictionResponse(
        score=float(ensemble_score),
        decision=decision,
        latency_ms=latency_ms,
        shap_top3=shap_values
    )

@app.get("/health")
async def health_check():
    """Kubernetes liveness probe — verifies models loaded"""
    return {"status": "healthy", "models_loaded": 3}

# Run: uvicorn fraud_api:app --host 0.0.0.0 --port 8000 --workers 4
```

**Deployment checklist (Phase 1 complete):**
- ✅ Endpoint serves predictions in <100ms (p99 measured at 92ms)
- ✅ Health checks pass (Kubernetes readiness probe)
- ✅ Feature store integrated (Redis lookup <5ms)
- ✅ Async logging enabled (Prometheus metrics scraped every 15s)
- ✅ Model artifacts versioned (MLflow registry tracks v1.3.2)

> 💡 **Industry standard: BentoML** simplifies this entire workflow. The code above would be ~40 lines with BentoML's `@svc.api` decorator handling serialization, batching, and metrics automatically. For production systems, evaluate BentoML vs custom FastAPI based on team expertise and existing infrastructure.

---

### 4.2 · **[Phase 2: MONITOR]** PSI — Measuring Feature Drift

The **Population Stability Index** measures how much the distribution of a feature has shifted between training time (reference) and now (current):

$$\text{PSI} = \sum_{b=1}^{B} \left( A_b - E_b \right) \ln \frac{A_b}{E_b}$$

where $A_b$ is the fraction of current transactions falling in bin $b$, and $E_b$ is the fraction from the training set (the expected distribution).

**Interpretation thresholds:**

| PSI value | Interpretation | Action |
|-----------|---------------|--------|
| < 0.10 | No significant drift | Monitor, no action |
| 0.10 – 0.25 | Slight drift | Investigate; schedule review |
| > 0.25 | Significant drift | **Trigger retraining** |

**Worked example — Amount feature, 5 bins:**

We divide transaction amounts into 5 bins: <€50, €50–200, €200–500, €500–2000, >€2000.

| Bin | Range | Expected $E_b$ | Actual $A_b$ | $A_b - E_b$ | $\ln(A_b/E_b)$ | Contribution |
|-----|-------|---------------|-------------|-------------|----------------|-------------|
| 1 | <€50 | 0.40 | 0.25 | −0.15 | $\ln(0.625) = -0.470$ | $(-0.15)\times(-0.470) = +0.0705$ |
| 2 | €50–200 | 0.30 | 0.28 | −0.02 | $\ln(0.933) = -0.069$ | $(-0.02)\times(-0.069) = +0.0014$ |
| 3 | €200–500 | 0.15 | 0.22 | +0.07 | $\ln(1.467) = +0.383$ | $(+0.07)\times(+0.383) = +0.0268$ |
| 4 | €500–2000 | 0.10 | 0.17 | +0.07 | $\ln(1.700) = +0.531$ | $(+0.07)\times(+0.531) = +0.0372$ |
| 5 | >€2000 | 0.05 | 0.08 | +0.03 | $\ln(1.600) = +0.470$ | $(+0.03)\times(+0.470) = +0.0141$ |
| **Total** | | **1.00** | **1.00** | | | **PSI = 0.150** |

$$\text{PSI} = 0.0705 + 0.0014 + 0.0268 + 0.0372 + 0.0141 = \mathbf{0.150}$$

**Interpretation:** PSI = 0.150 falls in the 0.10–0.25 range — **slight drift**. The distribution has shifted toward higher-value transactions (bins 3–5 gained share while bins 1–2 lost share). This is consistent with a fraud ring targeting high-value purchases. Schedule a model review; if PSI climbs above 0.25 in the next measurement, trigger retraining.

> 💡 **Why the log ratio?** $\ln(A_b/E_b)$ measures the relative entropy direction — positive when $A > E$ (current has more in this bin than expected), negative when $A < E$. Multiplying by $(A_b - E_b)$ ensures both overshoots and undershoots contribute positively to PSI. The structure is: $\text{PSI} \approx \text{KL}(A \| E) + \text{KL}(E \| A)$ — a symmetric measure of distributional distance, the same idea as Jensen–Shannon divergence.

> **Decision Checkpoint #2: PSI threshold breach**  
> **Scenario:** Month 2 monitoring shows Amount feature PSI = 0.31 (> 0.25 threshold).  
> **Data:** Distribution shifted toward higher-value transactions — bins 3–5 (€200–€2000+) gained 17% share, bins 1–2 (< €200) lost share.  
> **Decision:** Fire retraining trigger immediately. Do not wait for weekly recall evaluation — feature drift of this magnitude will degrade recall within 7–10 days.  
> **Why this matters:** PSI is the **early warning**. Recall drops are the **damage report**. By the time recall hits 78%, you've already lost days of fraud detection quality. Act on PSI.

**Code Snippet — Phase 2: Prometheus Metrics Collection**

```python
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import numpy as np
from collections import defaultdict
from datetime import datetime, timedelta

# Prometheus metrics (scraped by Grafana every 15s)
predictions_total = Counter('fraud_predictions_total', 'Total predictions', ['decision'])
prediction_latency = Histogram('fraud_prediction_latency_ms', 'Prediction latency')
feature_psi = Gauge('fraud_feature_psi', 'PSI per feature', ['feature_name'])
recall_holdout = Gauge('fraud_recall_holdout', 'Weekly recall on delayed holdout')

class ProductionMonitor:
    def __init__(self, reference_distribution: dict):
        """
        reference_distribution: dict of {feature_name: np.array of bin edges and expected frequencies}
        Example: {"Amount": {"bins": [0, 50, 200, 500, 2000, np.inf], "expected": [0.40, 0.30, 0.15, 0.10, 0.05]}}
        """
        self.reference = reference_distribution
        self.hourly_buffer = defaultdict(list)  # Rolling 1h window per feature
        self.last_psi_compute = datetime.now()
    
    def log_transaction(self, features: dict, score: float, decision: str, latency_ms: float):
        """Called after every prediction — async, non-blocking"""
        # Update counters
        predictions_total.labels(decision=decision).inc()
        prediction_latency.observe(latency_ms)
        
        # Buffer features for PSI computation
        for feature_name, value in features.items():
            if feature_name in self.reference:
                self.hourly_buffer[feature_name].append(value)
    
    def compute_hourly_psi(self):
        """Scheduled every hour via cron or Airflow"""
        for feature_name, values in self.hourly_buffer.items():
            if len(values) < 100:  # Need min sample size
                continue
            
            ref = self.reference[feature_name]
            bins = ref["bins"]
            expected = np.array(ref["expected"])
            
            # Compute actual distribution
            counts, _ = np.histogram(values, bins=bins)
            actual = counts / counts.sum()
            
            # PSI formula
            psi = np.sum((actual - expected) * np.log((actual + 1e-10) / (expected + 1e-10)))
            
            # Push to Prometheus
            feature_psi.labels(feature_name=feature_name).set(psi)
            
            # Alert if breach
            if psi > 0.25:
                send_alert(f"PSI breach: {feature_name} = {psi:.3f}")
        
        # Clear buffer for next hour
        self.hourly_buffer.clear()
        self.last_psi_compute = datetime.now()
    
    def compute_weekly_recall(self, delayed_holdout_path: str):
        """Scheduled every Sunday via Airflow DAG"""
        # Load delayed-label holdout (30-60 days old, labels now settled)
        holdout = load_delayed_holdout(delayed_holdout_path)
        
        # Run inference on all holdout transactions
        y_true = holdout["is_fraud"]
        y_scores = [predict(tx) for tx in holdout["transactions"]]
        y_pred = (np.array(y_scores) > 0.63).astype(int)  # Threshold from Ch.5
        
        # Compute recall
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        # Push to Prometheus
        recall_holdout.set(recall)
        
        # Alert if breach
        if recall < 0.78:
            send_alert(f"Recall degraded: {recall:.3f} < 0.78 threshold")
        
        return recall

# Start Prometheus metrics server (scraped by Grafana)
if __name__ == "__main__":
    start_http_server(8001)  # Metrics available at http://localhost:8001/metrics
    monitor = ProductionMonitor(load_reference_distributions())
    
    # Hourly PSI check (run via cron: 0 * * * *)
    monitor.compute_hourly_psi()
    
    # Weekly recall check (run via cron: 0 0 * * SUN)
    monitor.compute_weekly_recall("data/holdout_delayed_labels.parquet")
```

**Grafana dashboard queries (Phase 2 monitoring):**

```promql
# PSI time series (last 7 days)
fraud_feature_psi{feature_name="Amount"}

# Recall trend (weekly measurements)
fraud_recall_holdout

# p99 latency (rolling 1h window)
histogram_quantile(0.99, rate(fraud_prediction_latency_ms_bucket[1h]))

# Fraud detection rate (% of transactions blocked)
rate(fraud_predictions_total{decision="BLOCK"}[1h]) / rate(fraud_predictions_total[1h])
```

**Phase 2 monitoring checklist:**
- ✅ PSI computed hourly for all 29 features
- ✅ Recall evaluated weekly on 10k delayed-label holdout
- ✅ Latency p50/p95/p99 tracked in real-time
- ✅ Alerts fire when PSI > 0.25 or Recall < 0.78
- ✅ Dashboards accessible to fraud ops team

> 💡 **Industry callout: Evidently AI** — If you're building monitoring from scratch, [Evidently AI](https://evidentlyai.com) provides pre-built PSI, KS-test, and drift dashboards with Prometheus integration. The code above replicates what Evidently does under the hood — use their library to save 200+ lines of monitoring code.

---

### 4.3 · **[Phase 3: DETECT]** Recall Degradation Model

When fraud patterns shift, recall decays exponentially. A simplified model: if fraud tactics change by 20% per month, the fraction of current fraud that our training-time model has "seen" decays geometrically:

$$R(t) = 82\% \times (1 - 0.20)^{t/30}$$

where $t$ is days since last training.

**Explicit arithmetic at key checkpoints:**

| Days since retrain | Calculation | Recall $R(t)$ |
|--------------------|-------------|---------------|
| $t = 0$ (just trained) | $82\% \times (0.80)^{0} = 82\% \times 1.000$ | **82.0%** |
| $t = 30$ (1 month) | $82\% \times (0.80)^{1} = 82\% \times 0.800$ | **65.6%** |
| $t = 60$ (2 months) | $82\% \times (0.80)^{2} = 82\% \times 0.640$ | **52.5%** |
| $t = 90$ (3 months) | $82\% \times (0.80)^{3} = 82\% \times 0.512$ | **42.0%** |

At $t = 90$ days without retraining, recall would have collapsed to 42% — worse than a naive rule that flags every transaction above €500.

**What does "20% drift per month" mean concretely?** In the September 2013 test dataset, 20% of all fraud used contactless card skimming at petrol stations. By month 3 of production, that vector has been patched by issuers, but a new vector (e-commerce account takeover) has appeared. Our model never saw account-takeover patterns, so its recall on that sub-population is near zero — dragging overall recall down sharply.

**Retraining every 30 days** resets recall to ~82% on each cycle. At the end of a 30-day cycle, recall has drifted to ~65.6% before retraining rescues it. More frequent retraining (every 14 days) keeps the floor higher:

$$R(14) = 82\% \times (0.80)^{14/30} = 82\% \times 0.905 = 74.2\%$$

> ⚠️ **This model is a planning tool, not a physical law.** Real drift rates depend on fraud ecosystem dynamics. The key insight is the shape: recall degrades fastest in the first weeks, then the rate slows as the remaining overlap between training and current fraud is more stable.

> **Decision Checkpoint #3: Recall degradation detected**  
> **Scenario:** Week 7 holdout evaluation shows recall = 76.8% (below 78% threshold). PSI for Amount = 0.24 (borderline). V14 (PCA component) PSI = 0.33 (significant).  
> **Data:** Two signals converging — both PSI and recall thresholds breached within 3 days.  
> **Decision:** Trigger retraining immediately. Investigation shows new fraud vector (account takeover) not present in training data. Model is blind to 15% of current fraud patterns.  
> **Why this matters:** The 78% recall threshold is the safety margin before SLA breach (80% target). Once you hit 78%, you have ~5–7 days before falling below contract minimum. Delay = business impact.

**Code Snippet — Phase 3: Drift Detection & Alert Logic**

```python
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Tuple
import pandas as pd

@dataclass
class DriftSignal:
    """Encapsulates one drift detection signal"""
    name: str
    value: float
    threshold: float
    breached: bool
    timestamp: datetime
    
    def alert_message(self) -> str:
        if self.breached:
            return f"[DRIFT ALERT] {self.name} = {self.value:.3f} > {self.threshold} threshold"
        return f"[OK] {self.name} = {self.value:.3f}"

class DriftDetector:
    def __init__(self, psi_threshold: float = 0.25, recall_threshold: float = 0.78):
        self.psi_threshold = psi_threshold
        self.recall_threshold = recall_threshold
        self.last_retrain = datetime.now()
        self.min_retrain_interval = timedelta(days=14)
    
    def check_psi_breach(self, feature_psi_scores: dict) -> Tuple[bool, list]:
        """
        Args:
            feature_psi_scores: dict of {feature_name: psi_value}
        Returns:
            (breach_detected, list of DriftSignal objects)
        """
        signals = []
        breach = False
        
        for feature, psi in feature_psi_scores.items():
            signal = DriftSignal(
                name=f"PSI_{feature}",
                value=psi,
                threshold=self.psi_threshold,
                breached=(psi > self.psi_threshold),
                timestamp=datetime.now()
            )
            signals.append(signal)
            if signal.breached:
                breach = True
        
        return breach, signals
    
    def check_recall_breach(self, holdout_recall: float) -> DriftSignal:
        """Check if recall on delayed-label holdout has degraded"""
        signal = DriftSignal(
            name="Holdout_Recall",
            value=holdout_recall,
            threshold=self.recall_threshold,
            breached=(holdout_recall < self.recall_threshold),
            timestamp=datetime.now()
        )
        return signal
    
    def should_trigger_retrain(self, psi_signals: list, recall_signal: DriftSignal) -> Tuple[bool, str]:
        """
        Retrain trigger logic: PSI breach OR recall breach, subject to min interval
        Returns: (should_trigger, reason)
        """
        # Check minimum interval constraint
        days_since_retrain = (datetime.now() - self.last_retrain).days
        if days_since_retrain < self.min_retrain_interval.days:
            return False, f"Too soon (last retrain {days_since_retrain}d ago, min {self.min_retrain_interval.days}d)"
        
        # Check PSI breach
        psi_breaches = [s for s in psi_signals if s.breached]
        if psi_breaches:
            features = ", ".join([s.name for s in psi_breaches])
            return True, f"PSI breach on features: {features}"
        
        # Check recall breach
        if recall_signal.breached:
            return True, f"Recall degraded to {recall_signal.value:.3f} < {self.recall_threshold}"
        
        return False, "No breach detected"

# Example usage (run hourly via Airflow)
if __name__ == "__main__":
    detector = DriftDetector(psi_threshold=0.25, recall_threshold=0.78)
    
    # Fetch latest metrics from Prometheus
    feature_psi = {
        "Amount": 0.31,
        "V14": 0.33,
        "Merchant": 0.19,
        # ... other 26 features
    }
    holdout_recall = 0.768  # From weekly evaluation
    
    # Check both signals
    psi_breach, psi_signals = detector.check_psi_breach(feature_psi)
    recall_signal = detector.check_recall_breach(holdout_recall)
    
    # Decision logic
    should_retrain, reason = detector.should_trigger_retrain(psi_signals, recall_signal)
    
    if should_retrain:
        print(f"✅ TRIGGER RETRAINING: {reason}")
        # Fire Airflow DAG: airflow dags trigger fraud_model_retrain
        trigger_retraining_pipeline()
    else:
        print(f"⏸️  No action: {reason}")
        # Log all signals to monitoring DB for trend analysis
        log_drift_signals(psi_signals + [recall_signal])
```

**Phase 3 detection thresholds (tunable dials):**

| Threshold | Default | Conservative | Aggressive | Tradeoff |
|-----------|---------|--------------|-----------|----------|
| PSI | 0.25 | 0.15 | 0.35 | Lower = more frequent retraining, less drift accumulation |
| Recall | 0.78 | 0.80 | 0.75 | Higher = less tolerance for degradation, earlier intervention |
| Min interval | 14 days | 21 days | 7 days | Longer = less operational churn, more drift between retrains |

> 💡 **Alert fatigue prevention:** If PSI fires every 3–4 days, the threshold is too tight. Raise to 0.30 or increase min interval to 21 days. The goal is **1 retrain per month** on average — more frequent = instability, less frequent = drift accumulation.

---

### 4.4 · **[Phase 4: RETRAIN]** Retraining Trigger Logic

Two independent signals can fire the retraining pipeline. Either alone is sufficient:

- **Signal 1 — Feature drift (PSI):** computed hourly on the last hour's transaction volume.
- **Signal 2 — Performance degradation:** computed weekly on the delayed-label holdout.

```python
def should_retrain(psi: float, holdout_recall: float) -> bool:
    # PSI > 0.25 => significant feature distribution shift.
    # Recall < 0.78 => slipped below safety margin
    # (target 80%; 78% gives 2-point buffer before SLA breach).
    drift_trigger = psi > 0.25
    performance_trigger = holdout_recall < 0.78

    if drift_trigger:
        print(f"  [TRIGGER] PSI={psi:.3f} > 0.25 — feature distribution shifted")
    if performance_trigger:
        print(f"  [TRIGGER] Recall={holdout_recall:.3f} < 0.78 — performance degraded")

    return drift_trigger or performance_trigger


# --- Example: month 2 monitoring cycle ---
psi_month2 = 0.31        # Amount feature distribution has shifted significantly
recall_month2 = 0.71     # Holdout recall has dropped below target

retrain = should_retrain(psi_month2, recall_month2)
# Output:
#   [TRIGGER] PSI=0.310 > 0.25 — feature distribution shifted
#   [TRIGGER] Recall=0.710 < 0.78 — performance degraded
# retrain = True
```

**Why two independent signals?**

PSI fires when feature distributions change, even before recall measurably degrades (labels are delayed). Recall fires even when PSI looks stable (concept drift — same feature distribution, but fraud behaviour within the distribution has changed). Together they cover both **covariate shift** and **concept drift**.

> **Decision Checkpoint #4: Blue-green deployment decision**  
> **Scenario:** Retraining complete. Shadow deployment runs for 48h. Green model: 81% recall @ 0.48% FPR, p99 latency = 91ms. Blue model: 71% recall @ 0.51% FPR, p99 latency = 87ms.  
> **Data:** Green wins on both detection metrics (recall +10pp, FPR −0.03pp). Latency delta = +4ms (within 10ms tolerance).  
> **Decision:** **Approve cutover.** Green becomes live at hour 50. Archive blue model to MLflow with tag "superseded_by_v1.4.0".  
> **Why this matters:** The 48h shadow window provides 172k transaction samples — statistically significant. If green had shown recall < blue or p99 > 100ms, rollback to blue with alert for manual review. Never deploy a model worse than what's running.

**Code Snippet — Phase 4: Blue-Green Deployment with Kubernetes**

```python
import subprocess
import time
from datetime import datetime, timedelta
import mlflow
from typing import Dict

class BlueGreenDeployment:
    def __init__(self, namespace: str = "fraud-detection"):
        self.namespace = namespace
        self.shadow_duration_hours = 48
        self.cutover_approved = False
    
    def trigger_retraining(self, training_window_days: int = 30):
        """
        Triggered by drift detector. Pulls labeled data, retrains ensemble, stages as green.
        """
        print(f"[RETRAIN] Starting retraining pipeline...")
        
        # Step 1: Pull labeled data (last N days with settled labels)
        end_date = datetime.now() - timedelta(days=30)  # 30-day label delay
        start_date = end_date - timedelta(days=training_window_days)
        
        training_data = fetch_labeled_transactions(start_date, end_date)
        print(f"[RETRAIN] Loaded {len(training_data)} transactions ({start_date.date()} to {end_date.date()})")
        
        # Step 2: Retrain ensemble (same architecture as Ch.5)
        print("[RETRAIN] Training Isolation Forest...")
        iso_model = train_isolation_forest(training_data)
        
        print("[RETRAIN] Training Autoencoder...")
        ae_model = train_autoencoder(training_data)
        
        print("[RETRAIN] Training One-Class SVM...")
        svm_model = train_ocsvm(training_data)
        
        # Step 3: Evaluate on held-out validation set (20% of training window)
        val_recall, val_fpr, val_latency = evaluate_ensemble(
            iso_model, ae_model, svm_model, training_data
        )
        print(f"[RETRAIN] Validation metrics: Recall={val_recall:.3f}, FPR={val_fpr:.4f}, p99={val_latency:.1f}ms")
        
        # Step 4: Save to MLflow with version tag
        new_version = mlflow.register_model(
            model_uri=f"models:/fraud_ensemble/staging",
            name="fraud_ensemble",
            tags={"retrain_date": datetime.now().isoformat(), "val_recall": val_recall}
        )
        print(f"[RETRAIN] Model registered as v{new_version} in MLflow")
        
        # Step 5: Deploy green to Kubernetes (shadow mode)
        self.deploy_green_shadow(new_version)
        
        return new_version
    
    def deploy_green_shadow(self, model_version: str):
        """Deploy green model in shadow mode — receives traffic copy but doesn't make decisions"""
        print(f"[DEPLOY] Deploying green model v{model_version} in shadow mode...")
        
        # Update Kubernetes deployment with new model version
        kubectl_cmd = f"""
        kubectl set image deployment/fraud-api-green \\
            fraud-api=fraud-api:v{model_version} \\
            -n {self.namespace}
        """
        subprocess.run(kubectl_cmd, shell=True, check=True)
        
        # Configure Istio to mirror 100% of traffic to green (shadow)
        istio_config = f"""
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: fraud-api-mirror
  namespace: {self.namespace}
spec:
  hosts:
  - fraud-api
  http:
  - route:
    - destination:
        host: fraud-api-blue    # Live traffic
        port:
          number: 8000
      weight: 100
    mirror:
      host: fraud-api-green      # Shadow traffic
      port:
        number: 8000
    mirrorPercentage:
      value: 100.0
"""
        with open("/tmp/fraud-api-mirror.yaml", "w") as f:
            f.write(istio_config)
        subprocess.run("kubectl apply -f /tmp/fraud-api-mirror.yaml", shell=True, check=True)
        
        print(f"[DEPLOY] Green shadow active. Will evaluate for {self.shadow_duration_hours}h.")
    
    def evaluate_shadow_metrics(self) -> Dict[str, float]:
        """
        After 48h shadow run, compare green vs blue on live traffic.
        Queries Prometheus for metrics collected during shadow period.
        """
        print(f"[EVAL] Evaluating shadow metrics (last {self.shadow_duration_hours}h)...")
        
        # Query Prometheus for green and blue metrics
        blue_metrics = query_prometheus(
            f'fraud_recall_live{{deployment="blue"}}[{self.shadow_duration_hours}h]'
        )
        green_metrics = query_prometheus(
            f'fraud_recall_shadow{{deployment="green"}}[{self.shadow_duration_hours}h]'
        )
        
        comparison = {
            "blue_recall": blue_metrics["recall"],
            "green_recall": green_metrics["recall"],
            "blue_fpr": blue_metrics["fpr"],
            "green_fpr": green_metrics["fpr"],
            "blue_p99_latency": blue_metrics["p99_latency_ms"],
            "green_p99_latency": green_metrics["p99_latency_ms"],
        }
        
        print(f"[EVAL] Blue: Recall={comparison['blue_recall']:.3f}, FPR={comparison['blue_fpr']:.4f}, p99={comparison['blue_p99_latency']:.1f}ms")
        print(f"[EVAL] Green: Recall={comparison['green_recall']:.3f}, FPR={comparison['green_fpr']:.4f}, p99={comparison['green_p99_latency']:.1f}ms")
        
        return comparison
    
    def decide_cutover(self, comparison: Dict[str, float]) -> bool:
        """
        Cutover decision logic:
        - Green recall >= Blue recall (no degradation)
        - Green p99 latency <= 100ms (SLA constraint)
        - Green FPR <= Blue FPR + 0.05% (precision tolerance)
        """
        recall_ok = comparison["green_recall"] >= comparison["blue_recall"]
        latency_ok = comparison["green_p99_latency"] <= 100.0
        fpr_ok = comparison["green_fpr"] <= (comparison["blue_fpr"] + 0.0005)
        
        cutover = recall_ok and latency_ok and fpr_ok
        
        if cutover:
            print("✅ [DECISION] APPROVE CUTOVER — All criteria met")
        else:
            reasons = []
            if not recall_ok:
                reasons.append(f"Recall degraded ({comparison['green_recall']:.3f} < {comparison['blue_recall']:.3f})")
            if not latency_ok:
                reasons.append(f"Latency breach ({comparison['green_p99_latency']:.1f}ms > 100ms)")
            if not fpr_ok:
                reasons.append(f"FPR increased ({comparison['green_fpr']:.4f} > {comparison['blue_fpr']:.4f} + 0.05%)")
            print(f"❌ [DECISION] REJECT CUTOVER — {'; '.join(reasons)}")
        
        return cutover
    
    def execute_cutover(self):
        """Flip Istio traffic from blue → green (green becomes new live model)"""
        print("[CUTOVER] Switching live traffic to green...")
        
        # Update Istio VirtualService to route 100% to green
        istio_config = f"""
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: fraud-api-live
  namespace: {self.namespace}
spec:
  hosts:
  - fraud-api
  http:
  - route:
    - destination:
        host: fraud-api-green    # Green is now live
        port:
          number: 8000
      weight: 100
"""
        with open("/tmp/fraud-api-live.yaml", "w") as f:
            f.write(istio_config)
        subprocess.run("kubectl apply -f /tmp/fraud-api-live.yaml", shell=True, check=True)
        
        print("✅ [CUTOVER] Complete. Green is now live. Blue archived.")
        
        # Scale down old blue deployment (keep for rollback if needed)
        subprocess.run(
            f"kubectl scale deployment/fraud-api-blue --replicas=1 -n {self.namespace}",
            shell=True
        )

# Full Phase 4 workflow (orchestrated by Airflow DAG)
if __name__ == "__main__":
    deployer = BlueGreenDeployment(namespace="fraud-detection")
    
    # 1. Retrain triggered by drift detector
    new_version = deployer.trigger_retraining(training_window_days=30)
    
    # 2. Shadow deployment (48h)
    time.sleep(48 * 3600)  # Wait for shadow evaluation period
    
    # 3. Evaluate shadow metrics
    comparison = deployer.evaluate_shadow_metrics()
    
    # 4. Cutover decision
    if deployer.decide_cutover(comparison):
        deployer.execute_cutover()
    else:
        print("[ROLLBACK] Green rejected. Blue remains live. Manual review required.")
        send_alert("Blue-green cutover rejected — investigate before next retrain")
```

**Phase 4 retraining pipeline metrics (last 6 months):**

| Retrain # | Trigger | Training window | Shadow recall | Cutover? | Time to deploy |
|-----------|---------|----------------|---------------|----------|----------------|
| 1 | PSI (Amount=0.31) | Oct 19–Nov 18 | 81% > 71% Blue | ✅ Yes | 52h |
| 2 | Recall (76%) | Dec 14–Jan 13 | 83% > 79% Blue | ✅ Yes | 50h |
| 3 | PSI (V14=0.34) | Feb 1–Mar 2 | 80% < 82% Blue | ❌ No | N/A (rollback) |
| 4 | Recall (77%) | Mar 20–Apr 19 | 84% > 80% Blue | ✅ Yes | 51h |

**Rollback reasons (Retrain #3):**  
Green model showed 80% recall on shadow traffic vs 82% blue — a 2pp degradation. Investigation revealed training data included a 3-day holiday period with atypical transaction patterns, biasing the model. Fix: exclude holiday periods from training windows going forward.

> 💡 **Industry callout: MLflow Model Registry + Kubernetes** — This is the production standard for versioned model deployment. MLflow tracks model lineage (which data trained it, validation metrics, hyperparameters). Kubernetes + Istio handle traffic routing and shadow evaluation. For teams without Kubernetes, **AWS SageMaker** or **Azure ML** provide managed equivalents with built-in blue-green deployment.

---

### 4.5 · KS Test — A Complementary Distribution Check

The **Kolmogorov–Smirnov (KS) test** offers an alternative to PSI for continuous features. Where PSI requires you to bin the data (and bin boundaries are arbitrary), KS works on raw empirical CDFs:

$$D_{\text{KS}} = \sup_x \left| F_{\text{ref}}(x) - F_{\text{cur}}(x) \right|$$

The statistic $D_{\text{KS}}$ is the maximum vertical distance between the two cumulative distribution functions.

**Worked example — Amount feature:**

Suppose we have 500 reference transactions and 500 current transactions. We compute the empirical CDFs and find:

| Threshold $x$ | $F_{\text{ref}}(x)$ | $F_{\text{cur}}(x)$ | $|F_{\text{ref}} - F_{\text{cur}}|$ |
|--------------|--------------------|--------------------|--------------------------------------|
| €50 | 0.40 | 0.25 | **0.15** |
| €200 | 0.70 | 0.53 | **0.17** |
| €500 | 0.85 | 0.75 | **0.10** |
| €2000 | 0.95 | 0.92 | 0.03 |

$D_{\text{KS}} = \max(0.15, 0.17, 0.10, 0.03) = \mathbf{0.17}$

With $n = 500$ samples per side, the critical value at $\alpha = 0.05$ is approximately $1.36 / \sqrt{500} \approx 0.061$. Since $0.17 > 0.061$, we **reject** the null hypothesis that the two distributions are the same — statistically significant drift detected.

**PSI vs KS — when to use which:**

| Criterion | PSI | KS test |
|-----------|-----|---------|
| Requires binning | Yes (arbitrary) | No |
| Interpretability | Direct: <0.1/0.1–0.25/>0.25 | Statistical p-value |
| Captures location shifts | Yes | Yes |
| Captures scale/shape shifts | Partial | Full |
| Common in industry | Very common (risk/credit) | Common (ML monitoring) |

FraudShield uses PSI as the primary trigger (industry standard, easy to explain to stakeholders) and KS as a secondary confirmatory test when PSI is borderline (0.20–0.25).

---

## 5 · Production Hardening Arc

> The arc that takes FraudShield from a static lab model to a self-maintaining production system.

### Act 1 — Online Evaluation Reveals Drift (Months 1–3)

FraudShield goes live on October 1. The first month is clean: recall on the delayed-label holdout is 82% — exactly matching offline evaluation. Confidence is high.

By month 2, the team notices the holdout recall has slipped to 77%. Below the 78% trigger, but only by 1 point — just under the wire. The PSI for the Amount feature is 0.19 (slight drift). No trigger fires, but the team flags it for review.

**Month 3: the alarm fires.** Holdout recall is 71%. PSI for Amount is 0.31. Both triggers fire simultaneously. Investigation reveals a fraud ring operating via account-takeover at e-commerce merchants — a pattern not present in the September 2013 training data. The static model never learned this signature.

---

### Act 2 — PSI Shows What Shifted

The team runs PSI on all 29 features. The distribution shift is concentrated:

| Feature | PSI (Month 3) | Interpretation |
|---------|--------------|----------------|
| Amount | 0.31 | **Significant** — higher-value purchases up |
| Merchant category (hash) | 0.28 | **Significant** — new merchant types in fraud |
| tx\_count\_1h | 0.08 | No drift |
| international\_flag | 0.11 | Slight drift |
| V14 (PCA component) | 0.33 | **Significant** — structural change in fraud signal |

V14 showing PSI = 0.33 is the key signal. V14 is a PCA component that the original training set showed was one of the strongest fraud indicators (high SHAP value in Ch.5). Its distribution shifting means the latent fraud signal itself has rotated — not just surface-level feature changes.

---

### Act 3 — Trigger Retraining

The retraining pipeline fires at end of month 2 (first crossing of thresholds). It:

1. Pulls 30 days of labeled transactions (Oct 1 – Oct 31, confirmed labels via chargebacks)
2. Retrains the ensemble with the same architecture (no hyperparameter changes)
3. Evaluates on a held-out slice: new model achieves **81% recall @ 0.5% FPR**
4. Stages the new model for shadow deployment

Total retraining pipeline time: **4.5 hours** (data pull 30min, training 3h, evaluation 1h).

---

### Act 4 — Blue-Green Deployment

Rather than swapping the model live, FraudShield uses **blue-green deployment**:

- **Blue** (current): old model, handling 100% of live traffic
- **Green** (new): new model, receiving a copy of all transactions but *not* making live decisions — shadow mode

For 48 hours, green model scores are logged alongside blue model scores. Comparison metrics:

| Metric | Blue (old model) | Green (new model) | Winner |
|--------|-----------------|-------------------|--------|
| Recall on shadow traffic | 71% | 81% | Green ✅ |
| FPR on shadow traffic | 0.51% | 0.48% | Green ✅ |
| p99 latency | 87ms | 91ms | Blue (minimal diff) |

Green wins on both detection metrics with acceptable latency. At hour 50, the load balancer is flipped: green becomes blue, old model is archived. Recall jumps back to 81% on live traffic — the loop is closed.

---

## 6 · Full Monitoring Walkthrough — 3-Month Timeline

The timeline below shows every PSI measurement, every recall evaluation, and the trigger/retrain/deploy events.

### Weekly Recall on Delayed-Label Holdout

| Date | Days since train | Model prediction $R(t)$ | Observed recall | Trigger? |
|------|-----------------|------------------------|-----------------|---------|
| Oct 7 (week 1) | 7 | 81.3% | 82.0% | No |
| Oct 14 (week 2) | 14 | 80.7% | 81.5% | No |
| Oct 21 (week 3) | 21 | 80.0% | 80.8% | No |
| Oct 28 (week 4) | 28 | 79.4% | 80.1% | No |
| Nov 4 (week 5) | 35 | 78.7% | 79.3% | No |
| Nov 11 (week 6) | 42 | 78.1% | 78.0% | No (borderline) |
| Nov 18 (week 7) | 49 | 77.5% | 76.8% | **Yes — recall < 78%** |

> 💡 The model prediction underestimates actual recall in weeks 1–5 because the 20%/month decay constant was calibrated conservatively. Real-world drift is slower initially, then accelerates when a new fraud vector breaks out (which happens around week 6–7 in this timeline).

### Monthly PSI Measurements (Amount Feature)

| Measurement | PSI | Band | Action |
|-------------|-----|------|--------|
| End of October | 0.09 | No drift | Continue monitoring |
| Mid-November | 0.19 | **Slight drift** | Flag for review |
| End of November | 0.31 | **Significant drift** | **Trigger retraining** |

### Full Timeline

```
Oct 1   --- Model deployed (blue). Recall = 82%.
Oct 7   --- Week 1 holdout eval: 82.0%. PSI = 0.09. Green.
Oct 14  --- Week 2 holdout eval: 81.5%. PSI = 0.09. Green.
Oct 21  --- Week 3 holdout eval: 80.8%. Green.
Oct 28  --- Week 4 holdout eval: 80.1%. PSI = 0.09. Green.

Nov 4   --- Week 5 holdout eval: 79.3%. Green (above 78%).
Nov 11  --- Week 6 holdout eval: 78.0%. BORDERLINE. Flag.
Nov 14  --- Mid-month PSI: 0.19 (slight). Monitor.
Nov 18  --- Week 7 holdout eval: 76.8%. TRIGGER (recall < 78%).
             PSI month-to-date: 0.24. Two signals converging.
Nov 18  --- Retraining pipeline started.
             Data window: Oct 19 -- Nov 18 (30 days, labels settled).
             Training time: 3h.
Nov 18  --- Evaluation: green model = 81% recall, 0.48% FPR. Approved.
Nov 18  --- Shadow deployment begins. Blue = live. Green = shadow.

Nov 20  --- Shadow complete (48h). Metrics:
             Green recall 81% > Blue recall 71% on shadow traffic.
             Cutover approved.
Nov 20  --- Blue-green cutover. Recall restores to 81%.
Nov 29  --- End-of-month PSI: 0.31 (significant). Second trigger fires
             but retraining already complete -- logged, no duplicate job.

Dec 1   --- New monitoring cycle begins. Recall = 81%.
Dec 7   --- Week 1 on new model: 81.8%. PSI = 0.07. Green.
Dec 14  --- Week 2: 81.3%. Green.
Dec 21  --- Week 3: 80.9%. PSI (mid-month) = 0.08. Green.
Dec 28  --- Week 4: 80.5%. Green. System stable.
```

### Key Numbers to Commit to Memory

- Recall drops **~1 percentage point per week** during a fraud-vector transition
- PSI crosses 0.10 (slight drift) at about **6 weeks** of accumulated drift in this scenario
- PSI crosses 0.25 (significant drift) at about **8–9 weeks**
- Retraining restores recall to within **1 point of original** (81% vs 82%)
- Shadow deployment takes **48 hours** — the main bottleneck is label settling for evaluation

### Interpreting the Arc

The three-month timeline illustrates a pattern you will see in virtually every deployed ML system:

1. **Honeymoon period** (weeks 1–4): the model is fresh, performance matches offline evaluation, confidence is high. This is the dangerous phase — teams often forget to instrument proper monitoring during this window.
2. **Silent degradation** (weeks 5–7): recall is sliding but still above threshold. PSI shows slight drift. Without monitoring, this would be invisible.
3. **Trigger and recovery** (week 7 onwards): both signals fire, retraining executes, blue-green cutover restores performance. The system has self-healed.

The key insight: **monitoring is not optional overhead** — it is the production system. The ensemble model from Ch.5 is only one component of FraudShield. The monitoring loop, retraining pipeline, and deployment strategy are the other half.

---

## 7 · Key Diagrams

### Production Pipeline

```mermaid
flowchart TD
    TX["Incoming Transaction\n(1,000 tx/sec peak)"]
    FS["Feature Store\nLookup\n< 5ms"]
    RT["Real-time Feature\nComputation\n< 30ms total"]
    ENS["Ensemble Inference\n< 45ms total"]
    POST["Post-processing\n+ SHAP\n< 15ms"]
    DEC{"Score > tau?"}
    APPROVE["APPROVE\nTotal: ~70ms"]
    BLOCK["BLOCK + Reason\nTotal: ~90ms"]
    LOG["Async Logging\n(non-blocking)"]
    MON["Monitoring DB\nPSI - Recall"]

    TX --> FS
    FS --> RT
    RT --> ENS
    ENS --> POST
    POST --> DEC
    DEC -->|"No"| APPROVE
    DEC -->|"Yes"| BLOCK
    POST --> LOG
    LOG --> MON

    style TX fill:#1e3a8a,stroke:#e2e8f0,stroke-width:2px,color:#ffffff
    style FS fill:#1d4ed8,stroke:#e2e8f0,stroke-width:2px,color:#ffffff
    style RT fill:#1d4ed8,stroke:#e2e8f0,stroke-width:2px,color:#ffffff
    style ENS fill:#1e3a8a,stroke:#e2e8f0,stroke-width:2px,color:#ffffff
    style POST fill:#1d4ed8,stroke:#e2e8f0,stroke-width:2px,color:#ffffff
    style DEC fill:#b45309,stroke:#e2e8f0,stroke-width:2px,color:#ffffff
    style APPROVE fill:#15803d,stroke:#e2e8f0,stroke-width:2px,color:#ffffff
    style BLOCK fill:#b91c1c,stroke:#e2e8f0,stroke-width:2px,color:#ffffff
    style LOG fill:#1d4ed8,stroke:#e2e8f0,stroke-width:1px,color:#ffffff
    style MON fill:#b45309,stroke:#e2e8f0,stroke-width:1px,color:#ffffff
```

### Monitoring & Retraining Loop

```mermaid
flowchart LR
    LIVE["Live Model\n(Blue)"]
    DATA["Transaction Stream\n+ Async Labels"]
    HOURLY["Hourly PSI\nComputation"]
    WEEKLY["Weekly Recall\non Holdout"]
    TRIGGER{"Retrain\nTrigger?"}
    PIPE["Retraining\nPipeline\n~4.5h"]
    SHADOW["Shadow Deployment\n(48h)"]
    EVAL{"Green better\nthan Blue?"}
    CUTOVER["Blue-Green\nCutover"]
    ALERT["Alert:\nManual Review"]

    LIVE --> DATA
    DATA --> HOURLY
    DATA --> WEEKLY
    HOURLY --> TRIGGER
    WEEKLY --> TRIGGER
    TRIGGER -->|"PSI > 0.25\nOR recall < 78%"| PIPE
    TRIGGER -->|"No"| LIVE
    PIPE --> SHADOW
    SHADOW --> EVAL
    EVAL -->|"Yes"| CUTOVER
    EVAL -->|"No"| ALERT
    CUTOVER --> LIVE

    style LIVE fill:#15803d,stroke:#e2e8f0,stroke-width:2px,color:#ffffff
    style DATA fill:#1d4ed8,stroke:#e2e8f0,stroke-width:2px,color:#ffffff
    style HOURLY fill:#1d4ed8,stroke:#e2e8f0,stroke-width:2px,color:#ffffff
    style WEEKLY fill:#1d4ed8,stroke:#e2e8f0,stroke-width:2px,color:#ffffff
    style TRIGGER fill:#b45309,stroke:#e2e8f0,stroke-width:2px,color:#ffffff
    style PIPE fill:#1e3a8a,stroke:#e2e8f0,stroke-width:2px,color:#ffffff
    style SHADOW fill:#1e3a8a,stroke:#e2e8f0,stroke-width:2px,color:#ffffff
    style EVAL fill:#b45309,stroke:#e2e8f0,stroke-width:2px,color:#ffffff
    style CUTOVER fill:#15803d,stroke:#e2e8f0,stroke-width:2px,color:#ffffff
    style ALERT fill:#b91c1c,stroke:#e2e8f0,stroke-width:2px,color:#ffffff
```

---

## 8 · Hyperparameter Dial

Three dials control the health of the monitoring and retraining system. Tuning them is a cost-benefit tradeoff: tighter thresholds catch drift faster but trigger more retraining cycles; looser thresholds reduce operational load but allow more recall degradation before correction.

| Dial | Default | Tighter (more sensitive) | Looser (less sensitive) | Primary risk of loosening |
|------|---------|--------------------------|-------------------------|--------------------------|
| **Retraining frequency cap** | Max 1 retrain per 14 days | 1 per 7 days | 1 per 30 days | Recall degrades for 4+ weeks before correction |
| **PSI threshold** ($\tau_{\text{PSI}}$) | 0.25 | 0.15 | 0.35 | Significant feature shifts go undetected longer |
| **Holdout size** | 10,000 transactions | 20,000 (more power) | 5,000 (noisier) | Recall estimates have ±3% CI instead of ±1.5% CI |
| **Shadow duration** | 48 hours | 96 hours (more data) | 12 hours (faster) | Insufficient traffic to detect edge-case regressions |

**Recommended starting point:**

- PSI threshold of 0.25 is the industry standard (Basel III credit risk guidance uses this same threshold)
- Holdout of 10,000 with 48h shadow is a reasonable cost/safety balance for a mid-size fintech
- Cap retraining at once per 14 days to avoid instability from overlapping retraining jobs

> 📖 **Connecting back to earlier chapters.** The PSI threshold controls false positives in drift detection the same way the anomaly threshold $\tau$ in Ch.3–Ch.5 controls false positives in fraud detection. The tradeoff is identical: lower threshold → more triggers → more operational overhead → less recall degradation. Both are precision/recall tradeoffs wearing different labels.

---

## 9 · What Can Go Wrong

### Feedback Loop Poisons Training Data

When the model blocks a transaction, **no label is generated** for that transaction — there is no ground truth for whether it was truly fraud or a false positive. If blocked transactions are excluded from the retraining dataset, the model trains only on what it allowed through. Over time, the training distribution drifts toward transactions that look legitimate to the current model — reinforcing its blind spots.

**Fix:** Use a small **holdout exploration rate** (0.1–1% of flagged transactions are randomly approved and labeled by the fraud operations team). This provides unbiased labels for training even in the blocked region.

---

### Alert Fatigue

If the model flags 5,000 transactions per day for human review and only 200 are confirmed fraud, the fraud operations team burns out reviewing 4,800 false positives. When humans stop reviewing alerts, confirmed fraud labels stop arriving, the delayed-label holdout empties, and the recall trigger never fires — even as the model degrades.

**Fix:** Monitor the **review queue depth** and **analyst throughput** as a leading indicator of label quality degradation. If confirmed fraud rate drops below 2% of reviewed alerts, the threshold $\tau$ needs recalibration.

---

### Retraining Too Frequent → Model Instability

If retraining fires every 3–4 days (e.g., because PSI threshold is too tight at 0.10), each new model is trained on only 3 days of labeled data. Small random fluctuations in fraud volume cause wild swings in model weights. The deployment cadence becomes impossible to manage and shadow evaluation is meaningless with too little data.

**Fix:** Enforce the **minimum retraining interval** (14 days) and **minimum labeled sample size** (≥5,000 confirmed fraud cases in the training window). Never train on a window shorter than 14 days.

---

### Shadow Mode Misses Edge Cases

Shadow deployment compares aggregate metrics (recall, FPR) averaged over 48 hours. A new fraud vector that is rare (0.01% of transactions) but catastrophic might be missed if those edge-case transactions simply did not appear in the shadow window. The new model passes shadow evaluation and goes live — but silently fails on that rare vector.

**Fix:** Maintain a **canary fraud set** — a curated collection of known fraud edge cases (synthetic but representative) that is evaluated on every candidate model before shadow deployment. Think of it as a regression test suite for model quality.

---

## 10 · Where This Reappears

The production patterns in this chapter are not unique to anomaly detection — they are the universal infrastructure of deployed ML.

| Concept | Where it reappears |
|---------|-------------------|
| **PSI / distribution monitoring** | FlixAI (Recommender Systems track Ch.6) — item popularity distribution shifts seasonally; same PSI logic applies to recommendation quality monitoring |
| **Blue-green deployment** | DevOps Fundamentals track (Ch.3) — the exact same pattern applied to web services; ML adds the shadow evaluation step |
| **Latency budgets** | Neural Networks track (Ch.16 — TensorBoard + production) — profiling inference latency for the California Housing model |
| **Feedback loop correction** | Multi-Agent AI track — agents that learn from their own decisions face an identical feedback loop problem; RLHF is a principled solution |
| **Concept drift** | AgentAI (Reinforcement Learning track) — the environment can drift (non-stationary MDPs); the same drift-detection and adaptation logic applies |

> ➡️ **The deeper unifying idea.** Every deployed ML system is a **control loop**: model predicts, world responds, model observes response, model updates. PSI and recall monitoring are the *sensor* in this loop. Retraining is the *actuator*. Blue-green deployment is the *safety interlock*. When you understand the loop, every new deployment challenge is a variation on the same theme.

---

## 11 · Progress Check — FraudShield Complete

**Verify your understanding:**

1. The FraudShield latency budget allocates 30ms + 45ms + 15ms = 90ms. If the feature store adds 8ms of network latency, does the system still meet the 100ms SLA?
   *(Answer: 90 + 8 = 98ms — yes, just within budget. But any further growth requires optimization.)*

2. Compute PSI for a two-bin Amount distribution: Expected = [70%, 30%], Actual = [50%, 50%].
   - Bin 1: $(0.50 - 0.70) \times \ln(0.50/0.70) = (-0.20) \times (-0.336) = +0.067$
   - Bin 2: $(0.50 - 0.30) \times \ln(0.50/0.30) = (+0.20) \times (+0.511) = +0.102$
   - PSI = $0.067 + 0.102 = 0.169$ — slight drift, schedule review.

3. If fraud patterns drift at 15%/month (less aggressive than our 20%), what is recall at $t = 90$ days?
   $$R(90) = 82\% \times (0.85)^{3} = 82\% \times 0.614 = 50.3\%$$
   Still severe! Even modest drift renders a static model unusable over 3 months.

4. You deploy the retraining pipeline and notice that after 4 consecutive retraining cycles, recall is oscillating between 78% and 82% rather than stabilizing. What are two likely causes?
   *(a) Training window too short — 30 days may not contain enough new fraud examples for stable learning; try a 45-day window. (b) Exploration rate too low — not enough labeled data from blocked transactions; increase from 0.1% to 0.5%.)*

---

> 💡 **FraudShield COMPLETE — All 5 constraints satisfied:**
>
> | Constraint | Status | Chapter where met |
> |------------|--------|------------------|
> | 1. DETECTION 82% recall @ 0.5% FPR | ✅ | Ch.5 (ensemble) |
> | 2. PRECISION <0.5% FPR | ✅ | Ch.5 (threshold calibration) |
> | 3. EXPLAINABILITY SHAP per transaction | ✅ | Ch.5 (SHAP integration) |
> | 4. LATENCY <100ms end-to-end | ✅ | **Ch.6** (30+45+15=90ms budget) |
> | 5. DRIFT RESILIENCE >80% recall maintained | ✅ | **Ch.6** (PSI + retrain pipeline) |

![FraudShield mission complete — all 5 constraints](img/ch06-production-progress-check.png)

---

## 12 · Bridge — Reinforcement Learning Track

FraudShield is a **supervised, batch-update** system: you collect labels, retrain periodically, and deploy. This works because chargebacks arrive with a delay but they do arrive — you have ground truth, eventually.

What happens when you don't have ground truth? What if the only signal is whether a transaction was disputed 45 days later, and you need to make decisions *now* to shape future fraud patterns? That is the **online decision problem** — and it is where the **Reinforcement Learning (RL)** track begins.

The bridge points:

- FraudShield's feedback loop (block → no label → training blind spot) is a special case of the **exploration-exploitation dilemma** in RL. The holdout exploration rate (1% random approvals) is structurally identical to **ε-greedy exploration** in Q-learning.
- The retraining trigger (PSI > 0.25 OR recall < 78%) is a hand-coded policy. RL trains a **policy automatically** from interaction with the environment — no hand-tuning of thresholds.
- Blue-green deployment maps to **policy evaluation and improvement** in RL: shadow mode is Monte Carlo policy evaluation; cutover is policy improvement.

> ➡️ **Next:** [Reinforcement Learning — AgentAI](../../06-rl) — learn to build agents that improve through trial and error, without waiting for batch labels. The first chapter introduces GridWorld: a fraud-probe environment where the agent must balance exploiting known safe transactions against exploring unknown ones.

---

*FraudShield | Anomaly Detection Track · Chapter 6 of 6*
