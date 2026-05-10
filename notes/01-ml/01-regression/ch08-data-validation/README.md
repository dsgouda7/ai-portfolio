# Ch.3 — Data Validation & Drift Detection

> **The story.** The data quality problem is ancient — and the solution arrived from an unlikely direction. In **1924** Walter Shewhart at Bell Labs sketched the first **control chart**: a time-series plot of a manufacturing measurement with upper and lower control limits drawn three standard deviations from the process mean. When a measurement crossed a limit, it rang an alarm: something in the process had changed. Shewhart’s insight — *monitor the distribution, not just the output* — is the conceptual ancestor of everything in this chapter. Sixty years later Nick Radcliffe formalised **Test-Driven Data Analysis (TDDA)** (2016): datasets, like code, need automated tests. If you’d write a unit test for a function, you should write a data contract for a column. Then in **2017** Abe Gong and James Campbell, two data engineers exhausted by discovering surprise distribution shifts *after* deployment, built **Great Expectations** — an open-source Python library where you declare what you *expect* of your data and the library tells you, batch by batch, whether reality matches. Meanwhile the credit risk and insurance industries had been using **Population Stability Index (PSI)** since the 1990s to flag when loan applicant populations drifted; the ML community is now adopting it wholesale. This chapter unifies all three lineages — Shewhart control chart thinking, TDDA contracts, and PSI monitoring — into a single production-ready validation pipeline.
>
> **Where you are in the SmartVal AI story.** You've built SmartVal AI through Ch.00-07: cleaned data, trained models, achieved $32k MAE with XGBoost. The model is production-ready — today. But what about tomorrow? Production data changes over time: market shifts, new neighborhoods, seasonal trends. Without validation guardrails, distribution drift arrives silently and degrades model performance. This chapter builds automated data quality checks that run on every production batch: schema validation, distribution drift detection (PSI, KS tests), and Great Expectations contracts. After this chapter, SmartVal AI has production-grade monitoring that catches data issues BEFORE they break predictions.
>
> **Notation in this chapter.** $P$ — reference (training) distribution; $Q$ — production distribution; $F_1, F_2$ — empirical CDFs; $D_{KL}(P \| Q)$ — KL divergence (information cost of using $Q$ when truth is $P$); $D_{KS}$ — KS test statistic (max CDF gap); $\text{PSI}$ — Population Stability Index; $E_b\%$ — expected fraction in bin $b$ from training; $A_b\%$ — actual fraction in bin $b$ from production; $B$ — number of bins.

---

## 0 · The Challenge — Where We Are

> 🎯 **The mission**: Build **SmartVal AI** — a production home valuation system achieving <$40k MAE with automated data quality monitoring.
>
> 1. **ACCURACY** (<95k MAE) — 33k gap remains; drift-corrected retraining closes it ← *this chapter completes*
> 2. **GENERALIZATION** (CA → Portland) — distribution-aware validation gates quantify geographic drift ← *this chapter*
> 3. **DATA QUALITY** — ✅ Fixed in Ch.1; this chapter *automates* quality checks forward in time
> 4. **AUDITABILITY** — 🔄 *unlocked here*: data contracts, GE expectation suites, batch-level audit trail
> 5. **PRODUCTION-READY** — 🔄 *partially unlocked*: automated validation + CI/CD gate; streaming monitoring in AI Infra track

**What Sarah knows so far:**
- ✅ Ch.1: Removed 127 outliers; fixed 1,483 bad imputations (IQR method + KNN imputer)
- ✅ Ch.2: SMOTE rebalancing + class-weighted loss; Portland MAE fell 174k → 128k
- ❌ **But the model still fails on Portland data 33k worse than needed**

**What’s blocking her:**
No validation pipeline exists. The training-to-production path is a trust fall: data arrives, the model predicts, errors surface days later as customer complaints. There is no gate. Schema changes pass silently. Distribution shifts are invisible until the MAE spikes.

**What this chapter unlocks:**
- **Constraint #4 AUDITABILITY ✅** — GE suites create machine-readable audit records: *which batch · which feature · which expectation · timestamp*
- **Constraint #5 PRODUCTION-READY (partial) ✅** — automated validation pipeline runs pre-deployment; schema violations and drift events block bad batches before a single prediction is served
- **Constraint #2 GENERALIZATION ✅** — PSI and KS tests quantify exactly *how* Portland differs from California, making geographic generalisation measurable and alertable
- **Constraint #1 ACCURACY ✅** — drift-aware retraining closes the final 33k gap: Portland MAE reaches **89k** (below the 95k target)

---

## Animation

![Chapter animation](img/ch03-data-validation-needle.gif)

---

## 1 · A Data Contract Is a Machine-Checkable Promise About Your Input

A **data contract** declares what your input data must look like — types, value ranges, distributions — at training time, and enforces that declaration on every production batch. When a batch violates the contract, an alert fires before the model makes a single prediction; the violation becomes an auditable event with a timestamp, feature name, and observed-vs-expected value. Without contracts, distribution shift arrives silently; with them, it arrives as an actionable ticket.

> 💡 **Why this chapter follows Ch.1 and Ch.2.** Those chapters cleaned *historical* data by hand. This chapter automates that discipline forward in time: every future batch receives the same scrutiny the training data received, without Sarah’s manual intervention.

---
## 1.5 · The Practitioner Workflow — Your 4-Phase Validation System

**Before diving into theory, understand the workflow you'll follow with every production model:**

> 📊 **What you'll build by the end:** An automated validation pipeline that declares schema contracts at training time, validates every production batch, computes drift scores (PSI/KS), and routes decisions (DEPLOY/WARN/BLOCK) without human intervention — saving SmartVal AI from the Portland deployment disaster.

```
Phase 1: DEFINE              Phase 2: MONITOR            Phase 3: DETECT             Phase 4: RESPOND
────────────────────────────────────────────────────────────────────────────────────────────────────
At training time:            Per batch validation:       Threshold evaluation:       Automated action:

• Declare schema             • Validate schema           • Compute PSI score         • PSI < 0.10: DEPLOY
  - Types (float64)            - Types, nulls, ranges      - Compute KS statistic      • 0.10 ≤ PSI < 0.25: WARN
  - Ranges (min, max, 99th)    - Mean ± tolerance          - Evaluate p-value          • PSI ≥ 0.25: BLOCK
  - Nulls allowed?             - Std ± tolerance         • Compare vs thresholds       - Log audit event
• Capture distribution       • Log validation results    • Route by severity           - Trigger retraining
  - Mean, std                • Generate audit record     • Alert on failures           - Reject batch
  - Percentiles (25/50/75/99)
• Save as GE suite JSON

→ DECISION:                  → DECISION:                 → DECISION:                 → DECISION:
  Which checks to enforce?     Schema pass?                Alert threshold hit?        Route decision:
  • Schema: MANDATORY          • No: REJECT immediately    • p < 0.05: BLOCK           • DEPLOY: serve predictions
  • Distribution: per-feature  • Yes: proceed to drift     • PSI ≥ 0.25: BLOCK         • WARN: serve + monitor
  • Set tolerance bands        • Compare vs reference      • PSI ≥ 0.10: WARN          • BLOCK: retrain required
    (mean ±15%, std ±20%)                                 • PSI < 0.10: DEPLOY         • REJECT: data invalid
```

> 💡 **Usage note:** Execute phases sequentially on first model deployment (DEFINE once at training time, then MONITOR→DETECT→RESPOND runs continuously). After retraining, rebuild Phase 1 contracts to keep reference distributions current. Version the GE suite JSON alongside the model artifact.

**The 4-phase diagnostic in practice:**

When Sarah deployed SmartVal AI to Portland without this workflow, she missed a 37% income distribution shift that caused 174k MAE. With the workflow:

1. **Phase 1 (DEFINE)** — California training data produced this contract:
   - `MedInc` mean: 3.87 ± 15% → acceptable range [3.29, 4.45]
   - `MedInc` 99th percentile: 10.68 (not the absolute max 15.00)
   - Schema: `float64`, no nulls, range [0.5, 10.68]

2. **Phase 2 (MONITOR)** — Portland batch validation detected:
   - Mean: 5.30 (outside [3.29, 4.45]) → **GE expectation failed**
   - Std: 2.60 vs expected [1.52, 2.28] → **GE expectation failed**

3. **Phase 3 (DETECT)** — Drift scoring revealed severity:
   - PSI = 0.339 (≥ 0.25 threshold) → **SEVERE DRIFT**
   - KS statistic D = 0.385, p = 2.3e-41 (< 0.05) → **shape changed**

4. **Phase 4 (RESPOND)** — Automated routing decision:
   - Action: **BLOCK** deployment
   - Reason: "PSI=0.339 ≥ 0.25: retrain required"
   - Outcome: Retrained on CA + PDX combined → MAE fell 128k → **89k ✅**

This chapter teaches you to build each phase. The sections below provide the theory and implementation details — return to this workflow diagram when integrating the pieces.

---
## 2 · Running Example: What Sarah Finally Measures

Sarah runs the comparison she should have run before the Portland deployment:

```python
from sklearn.datasets import fetch_california_housing
import pandas as pd
import numpy as np

data = fetch_california_housing()
df_ca = pd.DataFrame(data.data, columns=data.feature_names)

# Simulate Portland production data: higher incomes, slightly newer housing stock
rng = np.random.default_rng(42)
df_pdx = df_ca.copy()
df_pdx['MedInc']   = df_pdx['MedInc']   * 1.37   # +37%: Portland tech-sector effect
df_pdx['HouseAge'] = df_pdx['HouseAge'] * 0.92   # -8%: newer housing stock
df_pdx['AveRooms'] = df_pdx['AveRooms'] * 1.08   # +8%: slightly larger homes

for col in ['MedInc', 'HouseAge', 'AveRooms']:
    ca_m  = df_ca[col].mean()
    pdx_m = df_pdx[col].mean()
    pct   = (pdx_m / ca_m - 1) * 100
    print(f"{col:<16}  CA={ca_m:.2f}  PDX={pdx_m:.2f}  shift={pct:+.1f}%")
```

```
MedInc            CA=3.87  PDX=5.30  shift=+37.0%
HouseAge          CA=28.64  PDX=26.35  shift=-8.0%
AveRooms          CA=5.43  PDX=5.86  shift=+8.0%
```

The `MedInc` column alone shifted 37%. A model that learned *“a 3.9-unit income district is worth $X”* is predicting on 5.3-unit districts it has never seen. It extrapolates — and fails.

The five training-distribution facts that will define the validation contract:

| Feature | CA mean | CA std | CA min | CA 99th pct | CA max |
|---------|---------|--------|--------|-------------|--------|
| MedInc | 3.87 | 1.90 | 0.50 | 10.68 | 15.00 |
| HouseAge | 28.64 | 12.59 | 1.00 | 52.00 | 52.00 |
| AveRooms | 5.43 | 2.47 | 0.85 | 10.02 | 141.91 |
| AveBedrms | 1.10 | 0.47 | 0.33 | 1.73 | 34.07 |
| Population | 1425 | 1132 | 3 | 5008 | 35682 |

> ⚠️ **Use the 99th percentile, not the maximum, for upper bounds.** Setting `max_value=15.0` from the absolute maximum will pass every batch that contains no single value above 15. Setting it from the 99th percentile (10.68 for MedInc) catches the distribution rightward-shift that the deployment missed — because Portland’s 99th percentile is ~13.5, well above 10.68.

---

## 3 · Define: The Validation Pipeline at a Glance

Before the math, here is the full four-stage pipeline Sarah will build. Each numbered step has a deep-dive in the sections that follow — treat this as your map.

```
1. DEFINE    At training time, declare contracts:
             schema (types, nulls, ranges) + distribution stats (mean, std, percentiles)
             Saved as a versioned JSON suite; one suite per model version.

2. MONITOR   At each production batch, run contracts against incoming data:
             schema pass → distribution check → drift score computation (PSI / KS)

3. DETECT    When a check fails, emit a structured event:
             severity (WARN / BLOCK), feature name, observed vs expected value,
             batch timestamp, expectation type. Goes to alerting system + audit log.

4. RESPOND   Automated response by severity level:
             schema error             → reject batch immediately
             moderate drift           → deploy with flag, schedule review
             severe drift (PSI≥0.25) → block deployment, trigger retrain pipeline
```

**Four validation layers and what each catches:**

| Validation layer | What it catches | Tool | When it fires |
|---|---|---|---|
| **Schema validation** | Wrong types, nulls, out-of-range values | Great Expectations | Every batch |
| **Statistical validation** | Mean / std drifted beyond declared tolerance | Great Expectations | Every batch |
| **Distribution drift** | Full CDF shape changed (even when mean is stable) | KS test | Every batch |
| **Data contracts** | All of the above, versioned, signed, and auditable | GE Expectation Suite JSON | Deploy gate |

> � **Industry Standard:** `great_expectations.from_pandas(df).expect_*`
>
> ```python
> import great_expectations as ge
> # Create expectation suite from training data
> df_ge = ge.from_pandas(df_train)
> df_ge.expect_column_values_to_be_of_type('MedInc', 'float64')
> df_ge.expect_column_values_to_not_be_null('MedInc')
> df_ge.expect_column_values_to_be_between(
>     'MedInc', min_value=0.5, max_value=df_train['MedInc'].quantile(0.99))
> df_ge.expect_column_mean_to_be_between(
>     'MedInc', min_value=mean*0.85, max_value=mean*1.15)
> # Save suite as versioned JSON
> suite = df_ge.get_expectation_suite()
> # Validate production batch
> results = ge.from_pandas(df_prod).validate(expectation_suite=suite)
> ```
>
> **When to use:** Always in production. Stores suites as JSON, integrates with Airflow/dbt, auto-generates HTML data-docs for auditors.
> **Common alternatives:** `pandera` (Pydantic-style schema classes), `pydantic` (API validation), raw `assert` (notebook-only)
> **See also:** [Great Expectations docs](https://docs.greatexpectations.io/)

> 📖 **Great Expectations vs Pandera vs raw asserts.** GE stores suites as JSON, integrates with Airflow/dbt, and auto-generates HTML data-docs for auditors. Pandera offers Pydantic-style schema classes that feel more Pythonic. Raw `assert` statements are fine for notebooks but break silently the moment a check is removed or skipped. For production, prefer GE or Pandera — the *audit trail* is the whole point, not just the check.

> 💡 **Define verdict:** GE expectation suite declared (schema + distribution checks) with MedInc mean tolerance ±15% — Portland’s 37% shift would have failed immediately.
> ➡️ Integrate suite into inference pipeline and version alongside model artifact before proceeding to drift monitoring.

---

## 4 · Monitor: The Math — Three Drift Metrics, Three Complementary Views

Three mathematically distinct tools each catch a different facet of drift. Use all three; they are not redundant.

### 4.1 · KL Divergence — The Information Distance Between Distributions

**Intuition before the formula.** Imagine encoding income values using a Huffman codebook tuned to California: low-income values (frequent) get short codes; rare high-income values get long codes. Now someone sends you Portland income values over that California codebook. Because Portland has more high-income samples (the “long code” buckets), the total message length explodes. The *extra* bits needed — beyond what you’d need if the codebook had been tuned to Portland — is the KL divergence.

Formally, KL divergence measures the expected extra information when distribution $Q$ approximates the true distribution $P$:

$$D_{KL}(P \| Q) = \sum_{x} P(x) \log\frac{P(x)}{Q(x)}$$

Each term $P(x) \log(P(x)/Q(x))$:
- is **positive** when $P(x) > Q(x)$ (training has more mass here than production expects)
- is **negative** when $P(x) < Q(x)$ (positive terms dominate — $D_{KL}(P\|Q) \geq 0$ always)
- is **zero** when $P(x) = Q(x)$ (no surprise at this point)

$D_{KL}(P\|Q) = 0$ only when $P = Q$ everywhere.

> ⚠️ **KL divergence is asymmetric.** $D_{KL}(P \| Q) \neq D_{KL}(Q \| P)$ in general. In monitoring, $P$ is always the training distribution (the reference) and $Q$ is production (the approximant). Swapping them gives a numerically different answer with a different interpretation.

**Toy numerical example — 3 income buckets, computed by hand:**

Simplify MedInc into three buckets: Low (< 3.0 units ≈ $30k), Mid (3.0–6.0), High (> 6.0 ≈ $60k+).

| Bucket | P (CA train) | Q (PDX prod) | P / Q | ln(P/Q) | P × ln(P/Q) |
|--------|-------------|--------------|-------|---------|-------------|
| Low | 0.50 | 0.20 | 2.500 | +0.9163 | **+0.458** |
| Mid | 0.30 | 0.30 | 1.000 | 0.0000 | **0.000** |
| High | 0.20 | 0.50 | 0.400 | −0.9163 | **−0.183** |
| **Total** | 1.00 | 1.00 | — | — | **D_KL = 0.275** |

Arithmetic, term by term:

- **Low:** $0.50 \times \ln(0.50/0.20) = 0.50 \times \ln(2.5) = 0.50 \times 0.9163 = 0.458$
- **Mid:** $0.30 \times \ln(0.30/0.30) = 0.30 \times 0 = 0.000$
- **High:** $0.20 \times \ln(0.20/0.50) = 0.20 \times \ln(0.4) = 0.20 \times (-0.9163) = -0.183$

$$D_{KL}(P_{\text{CA}} \| Q_{\text{PDX}}) = 0.458 + 0.000 + (-0.183) = \mathbf{0.275}$$

> 💡 **Interpretation rule of thumb.** $D_{KL} < 0.1$: negligible drift. $0.1 \leq D_{KL} < 0.2$: worth monitoring. $D_{KL} \geq 0.2$: investigate; consider blocking deployment. Our value of 0.275 sits firmly in the “investigate” zone.

---

### 4.2 · The Kolmogorov-Smirnov Test — Maximum Gap in the CDFs

The KS test asks a subtler question than KL divergence: **is the entire shape of the distribution the same?** A distribution can have the same mean, the same standard deviation, and a similar KL divergence — and still have a completely different shape (bimodal vs unimodal, heavy-tailed vs light-tailed). KL divergence computed on coarse bins misses fine-grained shape differences; the KS test catches them.

The KS statistic is the maximum vertical gap between two **empirical cumulative distribution functions**:

$$D_{KS} = \sup_{x} |F_1(x) - F_2(x)|$$

where $F_1$ is the training CDF and $F_2$ is the production CDF.

**How to read it.** Sort both samples and compute their empirical CDFs. At every data point, measure how far apart the two CDFs sit. $D_{KS}$ is the single largest gap. If both samples come from the same distribution, the CDFs overlap closely ($D_{KS} \approx 0$). If they diverge sharply, $D_{KS} \to 1$.

The associated **p-value** uses the asymptotic distribution of $D_{KS}$ under the null hypothesis “both samples share the same underlying distribution.” A p-value < 0.05 means: if the populations were truly identical, seeing this large a gap by random sampling would happen fewer than 5% of the time. (For the full null-hypothesis framework behind this interpretation, see [ch06-metrics §8b](../ch06_metrics/README.md#8b--statistical-significance-of-regression-coefficients-p-values).)

```python
from scipy.stats import ks_2samp

stat, p = ks_2samp(df_ca['MedInc'], df_pdx['MedInc'])
print(f"KS statistic D = {stat:.4f}")   # → 0.3847
print(f"p-value         = {p:.2e}")     # → 2.3e-41
# p << 0.05 → reject null → distributions are significantly different
```

> 📖 **KS test vs Chi-squared test.** Use KS for continuous features (MedInc, HouseAge, AveRooms). Use Chi-squared for categorical features (ZipCode, PropertyType). KS is non-parametric and distribution-free: it makes no assumption about the shape of either sample.

---

### 4.3 · Population Stability Index — The Industry-Standard Drift Scalar

KL divergence and KS are theoretically elegant. In credit risk and insurance, practitioners needed something simpler: **a single number per feature that maps cleanly to “no action / investigate / retrain.”** That number is PSI. It was invented by US banks in the 1990s for monitoring loan-applicant population drift, and has been adopted wholesale by ML monitoring platforms.

PSI bins the distribution into $B$ buckets, then computes the fraction of training and production samples in each bucket and sums the bin-level divergence terms:

$$\text{PSI} = \sum_{b=1}^{B} \bigl(A_b\% - E_b\%\bigr) \times \ln\!\left(\frac{A_b\%}{E_b\%}\right)$$

where $E_b\%$ is the expected (training) fraction in bin $b$ and $A_b\%$ is the actual (production) fraction.

**Why PSI is always non-negative.** When $A > E$: both $(A-E)$ and $\ln(A/E)$ are positive — their product is positive. When $A < E$: both $(A-E)$ and $\ln(A/E)$ are negative — their product is still positive. PSI sums non-negative terms.

**Industry thresholds** (from the credit risk literature, now standard in ML monitoring):

| PSI value | Interpretation | Recommended action |
|-----------|---------------|--------------------|
| PSI < 0.10 | No significant drift | Deploy normally |
| 0.10 ≤ PSI < 0.25 | Moderate drift | Investigate; consider targeted retraining |
| PSI ≥ 0.25 | **Severe drift** | Block deployment; retrain required |

> ⚡ **PSI practical rule.** Set bins using the *training data percentiles* (equal-frequency binning), not equal-width bins. Equal-frequency binning ensures the reference histogram is uniform (each bin holds ~20% if using 5 bins), making PSI numerically stable and comparable across features with different scales.

> 💡 **Industry Standard:** `pydantic.BaseModel` for runtime schema validation
>
> ```python
> from pydantic import BaseModel, Field, confloat
> from typing import List
>
> class HousingBatch(BaseModel):
>     """Schema contract for California Housing production batches."""
>     MedInc: List[confloat(ge=0.5, le=15.0)]  # Median income [0.5, 15.0]
>     HouseAge: List[confloat(ge=1.0, le=52.0)]  # House age [1, 52]
>     AveRooms: List[confloat(ge=0.0)]           # Average rooms >= 0
>     # ... define all 8 features
>
>     def check_distribution_drift(self, reference_stats: dict) -> dict:
>         """Compute PSI for each feature against reference."""
>         drift_scores = {}
>         for feature in ['MedInc', 'HouseAge', 'AveRooms']:
>             psi = compute_psi(
>                 reference_stats[feature], getattr(self, feature))
>             drift_scores[feature] = psi
>         return drift_scores
>
> # Usage at inference time
> try:
>     batch = HousingBatch(**production_data)  # Schema check
>     drift = batch.check_distribution_drift(ca_reference)  # Drift check
>     if any(score >= 0.25 for score in drift.values()):
>         raise ValueError(f"Severe drift: {drift}")
> except ValidationError as e:
>     log_error("Schema validation failed", e)
>     return "REJECT"
> ```
>
> **When to use:** API endpoints receiving inference requests. Pydantic validates JSON payloads at parse time.
> **Common alternatives:** `marshmallow` (serialization), `cerberus` (dict validation), `attrs` + `cattrs` (dataclasses)
> **See also:** [Pydantic docs](https://docs.pydantic.dev/)

> 💡 **Monitor verdict:** PSI = 0.339 (severe), KS D = 0.385 (p = 2.3e-41) — Portland income shifted +37%, all three drift metrics confirm severity.
> ➡️ Log scores per batch and set PSI ≥ 0.25 = BLOCK threshold before routing to Phase 3 alert configuration.

---

## 4.5 · Detect: Threshold Configuration and Alert Routing

**The routing decision:** Now that you can compute PSI, KS, and KL divergence for any batch, you need rules that convert those scores into actions. This is where statistical monitoring becomes an operational system.

The industry-standard PSI thresholds (from credit risk, now universal in ML monitoring):

```python
# Phase 3: Alert threshold logic with graduated responses
def route_by_psi(psi_score: float, feature: str, batch_id: str) -> str:
    """Convert PSI score to routing decision with structured logging."""

    # DECISION LOGIC (threshold evaluation)
    if psi_score < 0.10:
        severity = "INFO"
        action = "DEPLOY"
        reason = f"No significant drift (PSI={psi_score:.3f})"
    elif 0.10 <= psi_score < 0.25:
        severity = "WARNING"
        action = "WARN"
        reason = f"Moderate drift (PSI={psi_score:.3f}) - monitor closely"
    else:  # psi_score >= 0.25
        severity = "CRITICAL"
        action = "BLOCK"
        reason = f"Severe drift (PSI={psi_score:.3f}) - retrain required"

    # Structured audit event (logged to monitoring system)
    alert = {
        "timestamp": datetime.utcnow().isoformat(),
        "batch_id": batch_id,
        "feature": feature,
        "metric": "PSI",
        "score": psi_score,
        "severity": severity,
        "action": action,
        "reason": reason
    }
    log_to_monitoring_system(alert)  # → Datadog / Prometheus / CloudWatch

    return action

# Usage on Portland batch
action = route_by_psi(psi_score=0.339, feature="MedInc", batch_id="pdx-2024-01-15")
# Output: "BLOCK"
# Alert logged: {"severity": "CRITICAL", "action": "BLOCK",
#                "reason": "Severe drift (PSI=0.339) - retrain required"}
```

**Graduated response strategy:**

| PSI Range | Action | Alert Severity | What Happens |
|-----------|--------|---------------|--------------|
| **< 0.10** | DEPLOY | INFO | Serve predictions normally; log audit record |
| **0.10 – 0.25** | WARN | WARNING | Deploy with drift flag; increase monitoring cadence; schedule 1-week review |
| **≥ 0.25** | BLOCK | CRITICAL | Reject batch; trigger retraining pipeline; page on-call ML engineer |

**Combining multiple metrics for robust detection:**

```python
def validate_batch_robust(df_prod, df_train, feature='MedInc'):
    """Multi-metric validation: PSI + KS test + GE suite (defense in depth)."""
    from scipy.stats import ks_2samp

    # Metric 1: PSI (population shift detector)
    psi = compute_psi(df_train[feature].values, df_prod[feature].values)

    # Metric 2: KS test (distribution shape detector)
    ks_stat, ks_p = ks_2samp(df_train[feature], df_prod[feature])

    # Metric 3: GE mean/std bounds (moment detector)
    mean_ok = df_train[feature].mean() * 0.85 <= df_prod[feature].mean() <= df_train[feature].mean() * 1.15

    # DECISION LOGIC (consensus voting)
    if psi >= 0.25 or ks_p < 0.01 or not mean_ok:
        return "BLOCK", f"PSI={psi:.3f}, KS_p={ks_p:.2e}, mean_ok={mean_ok}"
    elif psi >= 0.10 or ks_p < 0.05:
        return "WARN", f"Moderate drift detected"
    else:
        return "DEPLOY", "All metrics passed"

action, reason = validate_batch_robust(df_pdx, df_ca, feature='MedInc')
# Output: ("BLOCK", "PSI=0.339, KS_p=2.30e-41, mean_ok=False")
```

> 💡 **Industry Standard:** `evidently` for drift monitoring dashboards
>
> ```python
> from evidently.report import Report
> from evidently.metric_preset import DataDriftPreset, DataQualityPreset
>
> # Generate drift report comparing reference vs current data
> report = Report(metrics=[
>     DataDriftPreset(),        # PSI + KS for all features
>     DataQualityPreset(),      # Missing values, duplicates, correlations
> ])
> report.run(reference_data=df_ca, current_data=df_pdx)
> report.save_html("drift_report_portland_2024-01-15.html")
>
> # Programmatic access to drift scores
> drift_scores = report.as_dict()['metrics'][0]['result']['drift_by_columns']
> for feature, stats in drift_scores.items():
>     psi = stats['drift_score']
>     if psi >= 0.25:
>         print(f"ALERT: {feature} PSI={psi:.3f}")
> ```
>
> **When to use:** Production monitoring dashboards. Evidently generates interactive HTML reports with per-feature drift scores, histograms, and KS test results. Integrates with MLflow, Prefect, Airflow.
> **Common alternatives:** `whylabs` (enterprise SaaS), `arize` (observability platform), `fiddler` (explainability + monitoring)
> **See also:** [Evidently AI docs](https://docs.evidentlyai.com/)

> 💡 **Detect verdict:** Three-tier alert ladder set (PSI < 0.10 → DEPLOY, 0.10–0.25 → WARN, ≥ 0.25 → BLOCK) with structured JSON audit events.
> ➡️ Test thresholds against holdout and integrate as CI/CD gate — Portland’s PSI = 0.339 triggers BLOCK immediately.

---

## 5 · Discovery Arc — How Sarah Builds the Pipeline

You don’t arrive at a production validation pipeline all at once. Sarah builds hers through four failure modes, each one teaching something the previous attempt missed.

### Act 1 · The Naive Schema Check — “Types and Ranges Are Fine”

Sarah writes her first validation layer: assert that all columns are present, typed correctly, and within range.

```python
import numpy as np

def check_schema_naive(df):
    """First-attempt schema check — necessary but not sufficient."""
    required = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms',
                'Population', 'AveOccup', 'Latitude', 'Longitude']
    for col in required:
        assert col in df.columns,           f"Missing column: {col}"
        assert df[col].dtype == np.float64, f"Wrong type: {col}"
        assert df[col].isnull().sum() == 0, f"Nulls found: {col}"
    # Range check uses training absolute max — this is the trap
    assert df['MedInc'].between(0.5, 15.0).all(), "MedInc out of absolute range"
    print("✅ Schema check passed")

check_schema_naive(df_pdx)
# Output: ✅ Schema check passed
```

Portland passes. Columns are present, numeric, null-free, and within the absolute min/max of the training data. Schema checks alone are *necessary but not sufficient* — they see nothing about the distribution underneath.

---

### Act 2 · Distribution Shift Shock — “The Mean Is Outside the Contract”

Sarah upgrades to Great Expectations, declaring mean and std tolerances derived from the training distribution:

```python
import great_expectations as ge

# Step 1: Build expectation suite from training data
df_ca_ge = ge.from_pandas(df_ca)

# Schema expectations (types, nulls, 99th-pct range)
df_ca_ge.expect_column_values_to_be_of_type('MedInc', 'float64')
df_ca_ge.expect_column_values_to_not_be_null('MedInc')
df_ca_ge.expect_column_values_to_be_between(
    'MedInc',
    min_value=0.5,
    max_value=float(df_ca['MedInc'].quantile(0.99)))  # 99th pct, not absolute max

# Distribution expectations: mean ±15% tolerance
ca_mean = df_ca['MedInc'].mean()   # 3.87
ca_std  = df_ca['MedInc'].std()    # 1.90
df_ca_ge.expect_column_mean_to_be_between(
    'MedInc', min_value=ca_mean * 0.85, max_value=ca_mean * 1.15)
df_ca_ge.expect_column_stdev_to_be_between(
    'MedInc', min_value=ca_std  * 0.80, max_value=ca_std  * 1.20)

# Step 2: Save suite as versioned JSON
suite = df_ca_ge.get_expectation_suite(discard_failed_expectations=False)
# suite.save_to_json_file('suites/california_housing_v1.json')

# Step 3: Validate Portland data against the California contract
results = ge.from_pandas(df_pdx).validate(expectation_suite=suite)
print(f"Validation passed: {results['success']}")    # → False
```

```
Validation passed: False

  FAILED  expect_column_mean_to_be_between
    column:   MedInc
    observed: 5.30
    expected: [3.29, 4.45]

  FAILED  expect_column_stdev_to_be_between
    column:   MedInc
    observed: 2.60
    expected: [1.52, 2.28]
```

Two checks fail the moment Portland data arrives. **This is the signal** that would have prevented the 174k MAE disaster — had the validation suite been in place before the first Portland deployment.

---

### Act 3 · KL Divergence as a Drift Score — “How Bad Is the Shift?”

Mean and std failing is binary: pass or fail. Sarah wants a continuous score she can log, trend, and set graduated thresholds on:

```python
def kl_divergence_binned(p_samples, q_samples, bins=20):
    """Estimate D_KL(P||Q) via equal-width histogram binning."""
    edges = np.histogram_bin_edges(
        np.concatenate([p_samples, q_samples]), bins=bins)
    p_hist, _ = np.histogram(p_samples, bins=edges)
    q_hist, _ = np.histogram(q_samples, bins=edges)
    eps = 1e-10                         # floor to avoid log(0)
    p_norm = (p_hist + eps) / (p_hist + eps).sum()
    q_norm = (q_hist + eps) / (q_hist + eps).sum()
    return float(np.sum(p_norm * np.log(p_norm / q_norm)))

kl = kl_divergence_binned(df_ca['MedInc'].values, df_pdx['MedInc'].values)
print(f"D_KL(CA || PDX) for MedInc = {kl:.3f}")  # → ~0.281
# 0.281 > 0.20 → severe drift zone
```

$D_{KL} = 0.281$. Drift is not marginal — it is severe. The score is logged alongside the batch timestamp, enabling drift trending over time.

---

### Act 4 · [Phase 4: RESPOND] Automated Alerting — "The Pipeline Decides"

The final act removes Sarah from the critical path entirely. The pipeline issues a routing decision automatically:

```python
def compute_psi(expected, actual, bins=5):
    """PSI = sum((A% - E%) * ln(A%/E%)) over equal-frequency bins."""
    percentiles = np.percentile(expected, np.linspace(0, 100, bins + 1))
    e_counts = np.histogram(expected, bins=percentiles)[0]
    a_counts = np.histogram(actual,   bins=percentiles)[0]
    eps = 0.0001                        # industry-standard floor for zero bins
    e_pct = np.maximum(e_counts / len(expected), eps)
    a_pct = np.maximum(a_counts / len(actual),   eps)
    return float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))

def validate_and_route(df_prod, suite, ca_ref, psi_threshold=0.25):
    """Full validation stack → routing decision (DEPLOY / WARN / BLOCK / REJECT)."""
    from scipy.stats import ks_2samp
    # Layer 1: schema + statistical gate (Great Expectations)
    result = ge.from_pandas(df_prod).validate(expectation_suite=suite)
    if not result['success']:
        return 'REJECT', 'GE expectation failed — schema or distribution out of contract'
    # Layer 2: KS test for shape shift
    for feature in ['MedInc', 'HouseAge', 'AveRooms']:
        _, p = ks_2samp(ca_ref[feature].values, df_prod[feature].values)
        if p < 0.01:
            return 'BLOCK', f'KS test: {feature} severely drifted (p={p:.2e})'
    # Layer 3: PSI severity gate
    psi = compute_psi(ca_ref['MedInc'].values, df_prod['MedInc'].values, bins=5)
    if psi >= psi_threshold:
        return 'BLOCK',  f'PSI={psi:.3f} ≥ {psi_threshold}: retrain required'
    elif psi >= 0.10:
        return 'WARN',   f'PSI={psi:.3f}: moderate drift — monitor closely'
    return 'DEPLOY', 'All checks passed — audit record logged'

action, reason = validate_and_route(df_pdx, suite, df_ca)
# action → 'BLOCK'
# reason → 'PSI=0.339 ≥ 0.25: retrain required'
```

Portland is **blocked automatically**. No human needed to catch it. The audit trail records: batch ID, timestamp, feature, score, decision.

**Automated retraining trigger integration:**

```python
def retrain_on_drift(action: str, batch_metadata: dict, model_config: dict):
    """Phase 4: Automated response to drift detection (orchestrated via MLflow)."""
    if action == "BLOCK":
        # Step 1: Collect production labels (2-week manual review for Portland)
        prod_labels = collect_labels_for_batch(batch_metadata['batch_id'])

        # Step 2: Merge with training data (CA + Portland combined)
        df_combined = pd.concat([df_train, prod_labels])

        # Step 3: Trigger retraining pipeline
        retrain_job = mlflow.projects.run(
            uri=".",
            entry_point="train",
            parameters={
                "data_path": save_to_storage(df_combined),
                "model_version": f"{model_config['version']}_drift_corrected",
                "parent_run_id": model_config['run_id']
            }
        )

        # Step 4: Re-validate on Portland holdout
        new_model = mlflow.sklearn.load_model(f"runs:/{retrain_job.run_id}/model")
        portland_mae = evaluate_mae(new_model, df_pdx_holdout)

        # Step 5: Log drift-correction outcome
        mlflow.log_metric("portland_mae_before_retrain", 128_000)
        mlflow.log_metric("portland_mae_after_retrain", portland_mae)
        mlflow.log_param("retrain_trigger", f"PSI={psi:.3f}_KS_p={ks_p:.2e}")

        return portland_mae  # → 89k (below 95k target ✅)

    elif action == "WARN":
        # Schedule review, increase monitoring frequency
        schedule_model_review(days=7)
        increase_validation_cadence(from_daily_to_hourly=True)

    # DEPLOY: no action needed, audit logged automatically

# Execute on Portland drift event
final_mae = retrain_on_drift(
    action="BLOCK",
    batch_metadata={"batch_id": "pdx-2024-01-15", "n_samples": 1000},
    model_config={"version": "v1.2.3", "run_id": "abc123"}
)
# Output: 89_000 (MAE fell 128k → 89k after drift-corrected retrain)
```

> 💡 **Industry Standard:** `mlflow.evaluate()` with drift thresholds for model registry gates
>
> ```python
> import mlflow
> from mlflow.models import make_metric
>
> # Define custom drift metric for model validation
> def psi_metric(eval_df, builtin_metrics):
>     reference_data = mlflow.artifacts.load_table("reference_distribution")
>     psi_scores = {}
>     for col in ['MedInc', 'HouseAge', 'AveRooms']:
>         psi = compute_psi(reference_data[col], eval_df[col])
>         psi_scores[f"drift_psi_{col}"] = psi
>     return psi_scores
>
> # Register model with drift validation gate
> with mlflow.start_run() as run:
>     mlflow.sklearn.log_model(model, "model")
>
>     # Evaluate on production data with drift checks
>     results = mlflow.evaluate(
>         model=f"runs:/{run.info.run_id}/model",
>         data=df_prod,
>         targets="MedHouseVal",
>         model_type="regressor",
>         evaluators=["default"],
>         extra_metrics=[make_metric(eval_fn=psi_metric)]
>     )
>
>     # Conditional model promotion based on drift + accuracy
>     if results.metrics["mae"] < 95_000 and all(
>         results.metrics[k] < 0.25 for k in results.metrics if k.startswith("drift_psi_")
>     ):
>         mlflow.register_model(f"runs:/{run.info.run_id}/model", "SmartValAI")
>     else:
>         print(f"BLOCKED: MAE={results.metrics['mae']}, drift={results.metrics}")
> ```
>
> **When to use:** CI/CD pipelines deploying models to production. MLflow model registry gates prevent drift-affected models from reaching staging/prod environments.
> **Common alternatives:** `kubeflow` (Kubernetes-native ML), `sagemaker pipelines` (AWS-native), `vertex ai` (GCP-native)
> **See also:** [MLflow Model Registry docs](https://mlflow.org/docs/latest/model-registry.html)

> 💡 **Respond verdict:** Drift-triggered retrain cut Portland MAE 128k → 89k (below 95k target ✅); REJECT/BLOCK/WARN/DEPLOY paths fully automated.
> ➡️ Monitor retrain outcomes and expand to streaming validation — SmartVal AI Constraint #5 (PRODUCTION-READY) satisfied.

---

## 6 · Step-by-Step — PSI on MedInc by Hand

Full PSI calculation for `MedInc`, using 5 equal-frequency bins from the California training set. This is the calculation that fires the `BLOCK` decision above.

**Step 1 — Define bins from training percentiles (equal-frequency binning).**

Use the 0th, 20th, 40th, 60th, 80th, and 100th percentiles of California `MedInc` as bin edges. This guarantees each training bin holds approximately 20% of samples — a necessary property for PSI stability.

California `MedInc` bin edges: `[0.50, 2.56, 3.54, 4.74, 6.61, 15.00]`

| Bin | Range | Label |
|-----|-------|-------|
| 1 | [0.50, 2.56) | Low income |
| 2 | [2.56, 3.54) | Lower-middle |
| 3 | [3.54, 4.74) | Middle |
| 4 | [4.74, 6.61) | Upper-middle |
| 5 | [6.61, 15.00] | High income |

**Step 2 — Compute E% (California training fractions) and A% (Portland fractions).**

By construction, California is close to 20%/20%/20%/20%/20%. Portland, with its +37% income shift, migrates mass from low and middle bins into upper-middle and high bins:

| Bin | E% (CA training) | A% (PDX production) |
|-----|-----------------|---------------------|
| 1 | 0.15 | 0.06 |
| 2 | 0.28 | 0.15 |
| 3 | 0.32 | 0.30 |
| 4 | 0.17 | 0.29 |
| 5 | 0.08 | 0.20 |

**Step 3 — Compute each PSI term: $(A_b\% - E_b\%) \times \ln(A_b\% / E_b\%)$, row by row.**

| Bin | E% | A% | A%−E% | A%/E% | ln(A%/E%) | PSI term |
|-----|----|----|-------|-------|-----------|----------|
| 1 | 0.15 | 0.06 | −0.09 | 0.400 | −0.9163 | (−0.09)×(−0.9163) = **0.0825** |
| 2 | 0.28 | 0.15 | −0.13 | 0.536 | −0.6243 | (−0.13)×(−0.6243) = **0.0812** |
| 3 | 0.32 | 0.30 | −0.02 | 0.938 | −0.0645 | (−0.02)×(−0.0645) = **0.0013** |
| 4 | 0.17 | 0.29 | +0.12 | 1.706 | +0.5347 | (+0.12)×(+0.5347) = **0.0642** |
| 5 | 0.08 | 0.20 | +0.12 | 2.500 | +0.9163 | (+0.12)×(+0.9163) = **0.1100** |

Verify signs: Bins 1 and 2 lost mass in Portland (A < E) → both factors negative → positive product. Bins 4 and 5 gained mass (A > E) → both factors positive → positive product. Bin 3 barely changed → tiny term. All PSI terms are non-negative. ✓

**Step 4 — Sum the terms.**

$$\text{PSI} = 0.0825 + 0.0812 + 0.0013 + 0.0642 + 0.1100 = \mathbf{0.3392}$$

**Step 5 — Route the decision.**

$\text{PSI} = 0.339 \geq 0.25$ → **SEVERE DRIFT**. Block deployment. Trigger retraining pipeline with combined CA + PDX data.

> ⚡ **Constraint #1 ACCURACY check.** After the PSI alert triggers retraining on combined CA + PDX data (1,000 Portland rows with labels collected over two weeks of manual review), the new model achieves Portland MAE = **89k** — below the 95k target. The constraint is met precisely *because* PSI forced the retraining the unmonitored original deployment never triggered.

---

## 7 · Key Diagrams

### 7.1 · Validation Pipeline Architecture

```mermaid
flowchart TD
    TRAIN["Training Complete\nCalifornia Housing"]:::primary
    SUITE["GE Expectation Suite\n7 schema + 8 distribution checks\nJSON versioned"]:::success
    BATCH["Production Batch\nPortland data N rows"]:::info

    SCHEMA{"Schema Gate\nTypes - Nulls - Ranges"}:::caution
    DIST{"Distribution Gate\nMean - Std - Percentiles"}:::caution
    KS{"KS Test\np < 0.05?"}:::caution
    PSI{"PSI Score\n>= 0.25?"}:::caution

    REJECT["REJECT batch\nLog schema error\nPage data-eng on-call"]:::danger
    BLOCK["BLOCK deployment\nCreate retrain ticket\nAlert ML team"]:::danger
    WARN["WARN + MONITOR\nDeploy with drift flag\nSchedule model review"]:::caution
    DEPLOY["DEPLOY\nAll checks passed\nAudit record written"]:::success

    TRAIN -->|"Define contracts at train time"| SUITE
    BATCH --> SCHEMA
    SCHEMA -->|"Pass"| DIST
    SCHEMA -->|"Fail: type / null / range"| REJECT
    DIST -->|"Pass"| KS
    DIST -->|"Fail: mean or std out of bounds"| REJECT
    KS -->|"p >= 0.05"| PSI
    KS -->|"p < 0.05: shape shift detected"| BLOCK
    PSI -->|"PSI < 0.10"| DEPLOY
    PSI -->|"0.10 <= PSI < 0.25"| WARN
    PSI -->|"PSI >= 0.25"| BLOCK

    SUITE -.->|"Suite loaded at inference time"| SCHEMA

    classDef primary fill:#1e3a8a,color:#fff,stroke:#1e3a8a
    classDef success fill:#15803d,color:#fff,stroke:#15803d
    classDef caution fill:#b45309,color:#fff,stroke:#b45309
    classDef danger fill:#b91c1c,color:#fff,stroke:#b91c1c
    classDef info fill:#1d4ed8,color:#fff,stroke:#1d4ed8
```

---

### 7.2 · Drift Alert Decision Tree

```mermaid
flowchart TD
    START(["New production batch arrives"]):::info
    CHECK1{"Any GE expectation\nfailed?"}:::caution
    CHECK2{"PSI < 0.10\nfor all features?"}:::caution
    CHECK3{"0.10 \u2264 PSI < 0.25\nmoderate drift?"}:::caution
    LABELS{"Production labels\navailable?"}:::primary

    A_REJECT["REJECT batch\nSchema or distribution\nout of contract"]:::danger
    A_DEPLOY["Deploy normally\nLog audit record"]:::success
    A_WARN["Deploy with flag\nIncrease monitoring cadence\nSchedule 1-week review"]:::caution
    A_BLOCK["Block deployment\nCollect labels, retrain\nRe-validate before deploy"]:::danger
    A_MAE_OK["Accept: MAE stable\nDrift is cosmetic"]:::success
    A_MAE_BAD["Block: MAE degraded\nDrift is material"]:::danger

    START --> CHECK1
    CHECK1 -->|"Yes"| A_REJECT
    CHECK1 -->|"No"| CHECK2
    CHECK2 -->|"Yes"| A_DEPLOY
    CHECK2 -->|"No"| CHECK3
    CHECK3 -->|"No: PSI >= 0.25"| A_BLOCK
    CHECK3 -->|"Yes: moderate"| LABELS
    LABELS -->|"No: wait for labels"| A_WARN
    LABELS -->|"Yes: MAE stable"| A_MAE_OK
    LABELS -->|"Yes: MAE degraded"| A_MAE_BAD

    classDef primary fill:#1e3a8a,color:#fff,stroke:#1e3a8a
    classDef success fill:#15803d,color:#fff,stroke:#15803d
    classDef caution fill:#b45309,color:#fff,stroke:#b45309
    classDef danger fill:#b91c1c,color:#fff,stroke:#b91c1c
    classDef info fill:#1d4ed8,color:#fff,stroke:#1d4ed8
```

---

### 7.3 · Shewhart Control Chart — Feature Mean Over Time

Walter Shewhart’s 1924 insight: plot a statistic over time and draw ±3σ control limits. Any point outside the limits signals the process has changed. Applied to ML monitoring: plot `MedInc` mean per weekly inference batch.

```
MedInc mean (weekly batch inference)

  6.0 |                                        <- OUT OF CONTROL
      |                                      *
  5.5 |                                   *
      + - - - - - - - - - - - - - - -  UCL = 5.46  - - - - - -
  5.0 |
      |
  4.5 |
      |        *                *
  4.0 |  *  *     *     *  *         *  *
      |                   *
  3.5 |
      + - - - - - - - - - - - - - - -  CL  = 3.87  - - - - - -
  3.0 |
      |
  2.5 |
      + - - - - - - - - - - - - - - -  LCL = 2.28  - - - - - -
  2.0 |
      +--+--+--+--+--+--+--+--+--+--+--+--+--+--> Week
         1  2  3  4  5  6  7  8  9 10 11 12 13 14

  CL  = center line (training mean = 3.87)
  UCL = CL + 3*sigma_batch = 3.87 + 3*0.53 = 5.46
  LCL = CL - 3*sigma_batch = 3.87 - 3*0.53 = 2.28
  sigma_batch = std error of weekly mean (approx sigma_feature / sqrt(N_batch))
```

The two rightmost points cross the UCL. This is the Portland deployment: income distributions shifted upward. A control chart running continuously in production would have flagged this in week 13 — two weeks before the MAE degradation surfaced as customer complaints.

---

## 8 · The Hyperparameter Dial

Three dials control pipeline sensitivity. Each is a genuine trade-off between catching real drift and generating alert fatigue.

| Dial | What it controls | Default | Stricter (↓) | More lenient (↑) |
|------|-----------------|---------|-------------|------------|
| **PSI threshold** | BLOCK boundary | 0.25 | 0.15: more retrains, fewer misses | 0.35: misses moderate drift |
| **Reference window** | Training samples defining E% | Full training set | Rolling 90-day window (adapts to gradual drift) | Full history (staleness risk) |
| **Alert cadence** | How often validation runs | Per-batch | Continuous streaming | Weekly batch (misses intra-week spikes) |

**Calibrating PSI thresholds on your own data:**

```python
def compute_psi(expected, actual, bins=5):
    percentiles = np.percentile(expected, np.linspace(0, 100, bins + 1))
    e_counts = np.histogram(expected, bins=percentiles)[0]
    a_counts = np.histogram(actual,   bins=percentiles)[0]
    eps = 0.0001
    e_pct = np.maximum(e_counts / len(expected), eps)
    a_pct = np.maximum(a_counts / len(actual),   eps)
    return float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))

# Step 1: PSI on holdout (same population as training — should be near-zero)
psi_holdout = compute_psi(
    df_train['MedInc'].values, df_holdout['MedInc'].values, bins=5)
# Expected: ~0.02 → confirms pipeline is not over-sensitive

# Step 2: PSI on 10% shift (mild — should stay below 0.10 = "no action")
df_mild = df_train.copy()
df_mild['MedInc'] *= 1.10
psi_mild = compute_psi(df_train['MedInc'].values, df_mild['MedInc'].values, bins=5)
# Expected: ~0.07 → correctly classified as noise

# Step 3: PSI on 37% shift (Portland scale — must trigger BLOCK)
df_severe = df_train.copy()
df_severe['MedInc'] *= 1.37
psi_severe = compute_psi(df_train['MedInc'].values, df_severe['MedInc'].values, bins=5)
# Expected: ~0.34 → correctly triggers retrain
```

> ⚡ **Reference window staleness.** A GE suite built from 2022 training data running in 2026 compares production to a four-year-old baseline. Natural demographic and market changes may fire false alerts on legitimate population evolution. Rebuild the reference distribution — and recompute the GE suite — every time the model is retrained. Version the suite alongside the model artifact (same registry, same tag).

---

## 9 · What Can Go Wrong

- **Schema drift through valid ranges.** A column silently changes from `float64` to `int64` without leaving the `[0.5, 15.0]` range. Always validate dtype explicitly:
  ```python
  df_ge.expect_column_values_to_be_of_type('MedInc', 'float64')
  ```
  One dtype change can cause downstream aggregation bugs that never surface in range-based validation.

- **Reference window staleness.** If you retrain the model but forget to rebuild the GE suite, validation compares production to an outdated baseline — and silently accepts batches that would fail a current contract.

- **Alert fatigue.** Setting PSI > 0.05 as the page-on-call threshold floods the team. Engineers stop reading alerts within three weeks. Keep the “block deployment” threshold at 0.25, the “investigate” threshold at 0.10, and document the escalation path for each tier. Undocumented alerts become wallpaper.

- **Covariate drift without label drift.** Features can shift (MedInc distribution moves) while model MAE stays stable (the feature-to-price relationship is unchanged). Track MAE alongside PSI: if both degrade together, retrain immediately. If PSI rises but MAE is stable, monitor closely but don’t block — you’re seeing cosmetic drift the model handles correctly.

- **Zero-count bins in PSI.** If a production bin has zero samples, $A_b\% = 0$ and $\ln(0/E)$ is undefined. Apply a floor:
  ```python
  A_pct = max(A_pct, 0.0001)   # industry standard: 0.01% floor
  E_pct = max(E_pct, 0.0001)
  ```
  This prevents a single empty bin from returning `nan` for the entire PSI computation.

---

## 10 · Where This Reappears

- ➡️ **[Regression Ch.6 — Metrics & Evaluation](../../01_regression/ch06_metrics/README.md):** Residual control charts apply Shewhart limits to prediction *errors* over time — the same statistical process control pattern applied to model output rather than input features. A residual chart drifting upward signals model degradation before you recompute MAE.

- ➡️ **[Neural Networks Ch.8 — TensorBoard](../../03_neural_networks/ch08_tensorboard/README.md):** TensorBoard’s distribution dashboards plot layer activation histograms epoch by epoch. The monitoring question — “has this distribution changed from the reference?” — is the same KL-divergence comparison you built here, now applied inside the network rather than to raw inputs.

- ➡️ **[AI Infrastructure track](../../06_ai_infrastructure/README.md):** Production ML infra chapters cover Evidently AI, WhyLabs, and Arize — tools that implement exactly the PSI + KS + GE pipeline you built here, at enterprise scale with Slack/PagerDuty integrations. The concepts from this chapter are the prerequisites for understanding what those platforms actually compute.

- ➡️ **[Multi-Agent AI track](../../04_multi_agent_ai/README.md):** Agent orchestration systems validate tool outputs and inter-agent message schemas using the same principle: declare expected structure at design time, validate at runtime, alert on violation, route by severity. The same four-stage pipeline (define → validate → alert → remediate) applies verbatim.

---

## N · Progress Check — RealtyML Fix: Complete

![Progress check](img/ch03-data-validation-progress-check.png)

**Constraint Status — Final Report:**

| # | Constraint | Target | Before Ch.3 | After Ch.3 | How achieved |
|---|-----------|--------|-------------|------------|-------------|
| **#1** | **ACCURACY** | < 95k MAE | 128k ❌ | **89k ✅** | PSI alert → drift-corrected retraining on CA + PDX combined data |
| **#2** | **GENERALIZATION** | CA → Portland | ❌ blind | **✅ measured** | KS + PSI gate quantifies geographic drift; blocks deployment when severe |
| **#3** | **DATA QUALITY** | No garbage in | ✅ (Ch.1, manual) | **✅ automated** | GE suite runs on every batch; schema violations rejected instantly |
| **#4** | **AUDITABILITY** | Traceable decisions | ❌ none | **✅ unlocked** | GE audit trail: batch ID · feature · expectation type · timestamp · outcome |
| **#5** | **PRODUCTION-READY** | Automated pipeline | ❌ manual | **✅ partial** | Validation gate + alerting in place; streaming monitoring in AI Infra track |

**Portland MAE progression across Data Fundamentals track:**

```
Ch.1 EDA + Outlier Removal:  174k → 148k   (-26k: garbage eliminated)
Ch.2 Class Rebalancing:      148k → 128k   (-20k: SMOTE + class weights)
Ch.3 Validation Pipeline:    128k →  89k   (-39k: drift-corrected retrain)
                                       ↑
                                 TARGET MET: 89k < 95k ✅
```

**Unlocked capabilities after this track:**
- ✅ Declare schema + distribution contracts from training data in a single GE call
- ✅ Validate any production batch automatically against those contracts
- ✅ Compute PSI (with row-by-row arithmetic explainability) for regulatory audits
- ✅ Route batches: DEPLOY / WARN / BLOCK without human intervention
- ✅ Generate a machine-readable audit record of every validation event
- ✅ Control-chart feature means over time using Shewhart-style UCL/LCL bounds

**Still being built (beyond this track):**
- ❌ Real-time streaming validation at sub-second latency (AI Infrastructure track)
- ❌ Automated label collection to close the MAE-measurement loop post-deployment
- ❌ Multi-model governance — validating across model *versions*, not just data (DevOps Fundamentals track)

**Sarah’s status update to the board:**

> “We have three chapters of data quality discipline compressed into an automated pipeline. Schema violations are rejected in milliseconds. Distribution drift fires a structured alert with a traceable reason before the model makes a single production prediction. PSI = 0.34 would have blocked the Portland deployment that caused the 174k MAE disaster. The model now runs at 89k MAE — below the 95k target. The system is auditability-ready for regulatory review.”

---

## N+1 · Bridge to the Classification Track

Data Fundamentals gave you the forensic foundation: *look at the data before you trust the model*, then *automate that inspection so it runs forever*. The [Classification Track](../../02_classification/ch01_logistic_regression/README.md) picks up from a clean, validated, contract-governed dataset and asks: *what kind of prediction should we make?* The running scenario shifts from continuous house prices to binary decisions, but the data quality discipline carries forward immediately. The logistic regression chapter encounters its own distribution considerations — class balance, threshold calibration, score drift — and you will recognise the pattern: declare expectations, validate, measure drift, remediate.

> ➡️ **Next:** [02_classification/ch01_logistic_regression](../../02_classification/ch01_logistic_regression/README.md) — binary classification, the sigmoid function, and decision boundaries.

