# Ch.5 — Monitoring & Observability

> **The story.** In **1998**, **Matt Welsh** at UC Berkeley published a paper on *self-tuning server systems* that tracked request latency and thread-pool saturation in real time — the seeds of modern observability. But the tool that democratised production metrics was **SoundCloud's Prometheus**, released in **2012** by **Matt T. Proud** and **Julius Volz** as an open-source replacement for proprietary APM vendors. Prometheus introduced a pull-based scraping model and a time-series database optimised for high-cardinality metrics — request rate *per route, per status code, per instance*. In 2016, **Grafana** became the de facto UI for Prometheus data, and by 2024 the Prometheus + Grafana stack is the standard for Kubernetes monitoring worldwide. Every dashboard you'll build uses the same PromQL queries and TSDB architecture they invented a decade ago.
>
> **Where you are in the curriculum.** You've deployed Flask apps with Docker (Ch.1), orchestrated multi-container stacks with Docker Compose (Ch.2), deployed to Kubernetes with auto-healing replicas (Ch.3), and automated deployments with CI/CD pipelines (Ch.4). You can ship code to production — **but you're flying blind**. If the app crashes at 2am, you find out from customer complaints. If latency spikes to 5 seconds, you only see it when users abandon checkouts. This chapter gives you **metrics + dashboards + alerts** — the observability foundation that shows you what's happening *before* it breaks.
>
> **Notation in this chapter.** `prometheus_client` — Python library for instrumenting code; **Counter** — monotonic metric (requests served); **Histogram** — bucketed distribution (latency); **Gauge** — instant snapshot (CPU %); **Prometheus** — time-series database that scrapes metrics endpoints; **Grafana** — visualization layer for queries; **PromQL** — query language (e.g., `rate(http_requests_total[5m])`); **Alertmanager** — alert routing and deduplication.

---

## 0 · The Challenge — Where We Are

> 🎯 **The mission**: Deploy a production Flask app with full observability — monitor requests/sec, latency, error rate, and alert on failures.

**What we know so far:**
- ✅ We can containerize Flask apps (Ch.1 — Docker)
- ✅ We can orchestrate multi-container stacks (Ch.2 — Docker Compose)
- ✅ We can auto-deploy with GitHub Actions (Ch.4 — CI/CD)
- ❌ **But we have NO visibility into runtime behavior!**

**What's blocking us:**
We're deploying code into a black box:
- **No metrics**: Can't see requests/sec, latency distribution, or error rate
- **No dashboards**: No visual timeline of system health over the last hour/day/week
- **No alerts**: Discover failures from customer complaints, not proactive monitoring
- **No debugging context**: When latency spikes, we don't know *which route* or *which instance*

Without observability, you can't debug production issues, can't prove SLA compliance, and can't detect performance regressions before users notice.

**What this chapter unlocks:**
The **Prometheus + Grafana observability stack** — instrument Flask app, scrape metrics every 15 seconds, visualize time-series data, and alert when error rate crosses thresholds.
- **Establishes the three pillars**: Metrics (counters, histograms, gauges) + Logs + Traces
- **Provides concrete dashboards**: Request rate graph, latency histogram, error count
- **Teaches production debugging**: How to identify slow routes, instance failures, and traffic spikes

✅ **This is the foundation** — every later chapter assumes you can see what your system is doing.

---

## Animation

![Chapter animation](img/ch05-monitoring-observability-needle.gif)

## 1 · Observability Means You Can Ask Questions About Production Without Deploying New Code

Monitoring tells you *that* something is broken. Observability tells you *why* — and lets you explore production behavior without redeploying instrumented code. The three pillars:

1. **Metrics** — numeric time-series data (requests/sec, latency p95, error %)
2. **Logs** — structured event records (timestamp, level, message, trace ID)
3. **Traces** — distributed request paths across microservices (span durations, parent-child relationships)

This chapter focuses on **metrics** — the foundation. Logs and traces are covered in later chapters when we introduce microservices and distributed debugging.

---

## 1.5 · The Practitioner Workflow — Your 4-Phase Observability Stack

> ⚠️ **Two ways to read this chapter:**
> - **Theory-first (recommended for learning):** Read §0→§4 sequentially to understand the concepts, then use this workflow as your reference
> - **Workflow-first (practitioners with existing knowledge):** Use this diagram as a jump-to guide when instrumenting real services
>
> **Note:** Section numbers don't follow phase order because the chapter teaches concepts pedagogically (theory before application). The workflow below shows how to APPLY those concepts in production.

**What you'll build by the end:** A full observability stack with instrumented Flask app → Prometheus scraping → Grafana dashboards → Alertmanager routing to PagerDuty. You'll be able to see request rate spikes in real-time, track latency percentiles, and get paged before customers notice failures.

```
Phase 1: INSTRUMENT          Phase 2: COLLECT           Phase 3: VISUALIZE         Phase 4: ALERT
──────────────────────────────────────────────────────────────────────────────────────────────────
Add metrics to app:          Configure scraping:        Build dashboards:          Define alert rules:

• Import prometheus_client   • prometheus.yml targets   • Grafana data source      • Alerting rules YAML
• Counter for requests       • Scrape interval 15s      • Time-series panels       • Alertmanager routing
• Histogram for latency      • Retention 30 days        • RED metrics dashboard    • PagerDuty integration
• Gauge for connections      • Label dimensions         • PromQL queries           • Runbook links

→ DECISION:                  → DECISION:                → DECISION:                → DECISION:
  Which metric type?           How often to scrape?       Which metrics to graph?    When to fire alerts?
  • Monotonic count: Counter   • High-traffic: 5-10s      • Start with RED metrics   • Error rate > 5%
  • Distribution: Histogram    • Low-traffic: 30-60s      • Rate (req/sec)           • Latency p95 > 500ms
  • Instant value: Gauge       • Balance load vs detail   • Errors (count)           • Saturation > 80%
                                                          • Duration (latency)       • Fire after 2min sustained
```

**The workflow maps to these sections:**
- **Phase 1 (Instrument)** → §4 Metrics Instrumentation (Counters, Histograms, Gauges)
- **Phase 2 (Collect)** → §5 Prometheus Configuration (Scrape configs, retention, TSDB)
- **Phase 3 (Visualize)** → §6 Grafana Dashboards (PromQL queries, panels, heatmaps)
- **Phase 4 (Alert)** → §7 Alert Rules & Incident Response (Alertmanager, routing, runbooks)

> 💡 **How to use this workflow:** Deploy Phases 1→2→3→4 in sequence on your first service. Once the stack is running, iterate backward — add new metrics (Phase 1), verify they appear in Prometheus (Phase 2), visualize them in Grafana (Phase 3), and create alerts for SLA breaches (Phase 4). The four phases are a deployment sequence first, then a continuous refinement loop.

### The RED Method — Your North Star for Service Metrics

Before instrumenting anything, understand the **three metrics that matter** for every user-facing service. Google's SRE team calls these the "Four Golden Signals" (latency, traffic, errors, saturation). Tom Wilkie at Grafana Labs simplified it to **RED** — Rate, Errors, Duration:

| Metric | What it measures | Why it matters | PromQL query example |
|--------|------------------|----------------|---------------------|
| **Rate** | Requests per second | Traffic volume, capacity planning | `rate(http_requests_total[5m])` |
| **Errors** | Error count or error % | Service reliability, SLA compliance | `sum(rate(http_requests_total{status=~"5.."}[5m]))` |
| **Duration** | Latency (p50, p95, p99) | User experience, performance SLA | `histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))` |

**Start with RED on every service.** Add resource metrics (CPU, memory, disk) only after RED is working. RED tells you *what users experience*. Resource metrics tell you *why* it's slow — but if you don't know it's slow yet, resource metrics are noise.

> 📖 **Further reading:** [Brendan Gregg — USE Method](https://www.brendangregg.com/usemethod.html) (Utilization, Saturation, Errors) for infrastructure monitoring. Use RED for services, USE for hosts.

---

## 2 · Running Example — Monitoring a Flask Payment API from Zero

You're a DevOps engineer at a fintech startup. Your Flask API just went live — it handles payment webhooks, and the CTO wants **real-time visibility** into request rate, latency, and error rate. No cloud vendor lock-in — everything must run locally first.

**The running example:**
- Flask app with 3 routes: `/health`, `/api/payment`, `/api/refund`
- Prometheus scrapes metrics every 15 seconds
- Grafana visualizes request rate and latency percentiles
- Alertmanager sends Slack notification when error rate > 5%

**Constraint:** Must run the full stack (Flask + Prometheus + Grafana + Alertmanager) with `docker compose up` — zero cloud dependencies.

---

## 3 · The Metrics Collection Stack at a Glance

Before diving into instrumentation, here's the full architecture you'll build. Each numbered component has a corresponding section below.

```
1. Flask app exposes /metrics endpoint
 └─ Instrumented with prometheus_client (counters, histograms, gauges)

2. Prometheus scrapes /metrics every 15 seconds
 └─ Stores time-series data in local TSDB
 └─ Configured via prometheus.yml (scrape targets, retention policy)

3. Grafana queries Prometheus
 └─ Visualizes metrics with time-series graphs, heatmaps, stat panels
 └─ Dashboards defined as JSON (version-controlled)

4. Alertmanager evaluates alerting rules
 └─ Fires alerts when thresholds crossed (e.g., error_rate > 5%)
 └─ Routes to Slack, PagerDuty, email

5. Generate traffic and observe dashboard
 └─ Simulate 100 requests/sec with `wrk` or Python script
 └─ Watch request rate spike, latency histogram shift
```

**Notation:**
- **Counter** — monotonic metric (only goes up). Example: `http_requests_total`
- **Histogram** — bucketed distribution. Example: `http_request_duration_seconds` with buckets `[0.1, 0.5, 1.0, 5.0]`
- **Gauge** — instant snapshot (can go up or down). Example: `active_connections`
- **PromQL** — query language. Example: `rate(http_requests_total[5m])` = requests/sec over 5min window

Sections 4–8 explain each component. Come back to this map when the detail feels overwhelming.

---

## 4 · **[Phase 1: INSTRUMENT]** Metrics Instrumentation

The first step is to add metrics to your application code. The `prometheus_client` Python library provides three metric types — Counter, Histogram, and Gauge — each designed for different use cases.

### 4.1 · The Math Defines Time-Series Aggregation and Rate Calculation

Prometheus stores metrics as:

$$\text{metric\_name}\{\text{label1}=\text{value1}, \text{label2}=\text{value2}\} \quad t_1: v_1, \, t_2: v_2, \, t_3: v_3, \ldots$$

Example:
```
http_requests_total{method="GET", route="/api/payment", status="200"}  1609459200: 150
http_requests_total{method="GET", route="/api/payment", status="200"}  1609459215: 162
http_requests_total{method="GET", route="/api/payment", status="200"}  1609459230: 178
```

Each tuple `(metric_name, labels)` defines a unique **time series**. Every scrape appends a new `(timestamp, value)` pair.

**What does a label actually mean?** Labels are the query dimensions — they let you slice metrics by route, status code, instance, or any custom tag. The metric name says *what* you're measuring; the labels say *which instance of the thing*.

**Why counters only go up.** A counter like `http_requests_total` is monotonically increasing — it resets to 0 only when the process restarts. To compute *rate* (requests/sec), you take the difference over a time window:

$$\text{rate}(C[t_1 : t_2]) = \frac{C(t_2) - C(t_1)}{t_2 - t_1}$$

In PromQL:
```promql
rate(http_requests_total[5m])
```

This computes the per-second rate over the last 5 minutes. If the counter went from 1000 to 1300 in 300 seconds, the rate is $(1300 - 1000) / 300 = 1.0$ requests/sec.

### 4.2 · Histograms Use Bucketing to Estimate Percentiles

A histogram tracks how many observations fall into predefined buckets. For latency:

```
http_request_duration_seconds_bucket{route="/api/payment", le="0.1"}   150
http_request_duration_seconds_bucket{route="/api/payment", le="0.5"}   280
http_request_duration_seconds_bucket{route="/api/payment", le="1.0"}   295
http_request_duration_seconds_bucket{route="/api/payment", le="+Inf"}  300
http_request_duration_seconds_sum{route="/api/payment"}                85.2
http_request_duration_seconds_count{route="/api/payment"}              300
```

The `le` label means "less than or equal". This histogram says:
- 150 requests took ≤0.1s
- 280 requests took ≤0.5s
- 295 requests took ≤1.0s
- All 300 requests took <∞ (the `+Inf` bucket always equals the count)

**To compute the 95th percentile** (p95), find the bucket where 95% of samples land:

$$p_{95} \approx \text{bucket boundary where cumulative fraction} \geq 0.95$$

In PromQL:
```promql
histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))
```

This interpolates linearly between bucket boundaries. If 280 out of 300 samples (93%) are in the ≤0.5s bucket, and 295 (98%) are in the ≤1.0s bucket, then p95 falls between 0.5s and 1.0s.

> ⚠️ **Histogram precision depends on bucket boundaries.** If you define buckets `[0.1, 1.0, 10.0]`, you can't distinguish between a 0.2s request and a 0.9s request — both land in the same bucket. Choose buckets that match your SLA thresholds (e.g., if your SLA is <500ms, include a 0.5s bucket).

### 4.3 · Code Snippet — Flask App Instrumentation

Here's a complete Flask app with all three metric types. Every request increments the counter, records latency in a histogram, and updates the active connections gauge.

```python
# app.py
from flask import Flask, request, jsonify
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
import time

app = Flask(__name__)

# Phase 1: Define metrics with labels for high-dimensional slicing
REQUEST_COUNT = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'route', 'status']
)

REQUEST_LATENCY = Histogram(
    'http_request_duration_seconds',
    'HTTP request latency',
    ['route'],
    buckets=[0.1, 0.5, 1.0, 5.0]  # SLA buckets: <100ms (fast), <500ms (acceptable), <1s (slow), <5s (critical)
)

ACTIVE_CONNECTIONS = Gauge(
    'active_connections',
    'Number of active connections'
)

@app.before_request
def before_request():
    """Track request start time and increment active connections."""
    request.start_time = time.time()
    ACTIVE_CONNECTIONS.inc()

@app.after_request
def after_request(response):
    """Record metrics after each request completes."""
    # DECISION LOGIC: Only record latency for non-metrics endpoints
    if request.path != '/metrics':
        latency = time.time() - request.start_time
        REQUEST_LATENCY.labels(route=request.path).observe(latency)
        REQUEST_COUNT.labels(
            method=request.method,
            route=request.path,
            status=response.status_code
        ).inc()
    
    ACTIVE_CONNECTIONS.dec()
    return response

@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({"status": "healthy"}), 200

@app.route('/api/payment', methods=['POST'])
def payment():
    """Simulate payment processing with variable latency."""
    time.sleep(0.05)  # Simulate 50ms processing time
    
    # DECISION LOGIC: Simulate 5% error rate
    import random
    if random.random() < 0.05:
        return jsonify({"error": "Payment gateway timeout"}), 503
    
    return jsonify({"status": "processed", "amount": 100.00}), 200

@app.route('/api/refund', methods=['POST'])
def refund():
    """Simulate refund processing."""
    time.sleep(0.1)  # Refunds take 100ms
    return jsonify({"status": "refunded"}), 200

@app.route('/metrics')
def metrics():
    """Prometheus metrics endpoint."""
    return generate_latest(), 200, {'Content-Type': CONTENT_TYPE_LATEST}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

**Key decisions in the code:**
1. **Counter labels** — `method`, `route`, `status` let you query "error rate for POST /api/payment" without changing code
2. **Histogram buckets** — `[0.1, 0.5, 1.0, 5.0]` match SLA thresholds (100ms = fast, 500ms = acceptable, 1s = slow, 5s = critical)
3. **Gauge inc/dec** — Tracks active connections in real-time; spikes indicate connection pool exhaustion
4. **Exclude /metrics from recording** — Prevents self-measurement noise

> 💡 **Industry Standard:** `prometheus_client`
> 
> This is the canonical Python library maintained by the Prometheus project. Use it for all metric instrumentation in Python services.
> 
> **When to use:** Always in production Python services. For other languages, use the official Prometheus client libraries:
> - **Go**: `github.com/prometheus/client_golang`
> - **Java**: `io.prometheus:simpleclient`
> - **Node.js**: `prom-client`
> - **Rust**: `prometheus` crate
> 
> **Common alternatives:** Datadog's `ddtrace` (vendor lock-in), StatsD (no labels, push-based), OpenTelemetry SDK (vendor-neutral but more complex setup)
> 
> **See also:** [Prometheus client library best practices](https://prometheus.io/docs/instrumenting/writing_clientlibs/)

### 4.4 · DECISION CHECKPOINT — Phase 1 Complete

**What you just saw:**
- Flask app exposes `/metrics` endpoint returning Prometheus format (`http_requests_total{method="POST",route="/api/payment",status="200"} 42`)
- Three metric types: Counter (requests), Histogram (latency with buckets), Gauge (active connections)
- Labels added to Counter and Histogram for multi-dimensional slicing by route and status code

**What it means:**
- You can now query "error rate for /api/payment" vs "/api/refund" separately without deploying new code
- Histogram buckets let you compute p50, p95, p99 latency without storing every individual request duration
- Gauge shows real-time connection pool usage — spikes indicate exhaustion

**What to do next:**
→ **Start the Flask app:** `python app.py` and verify `/metrics` endpoint shows metrics in Prometheus format  
→ **Generate test traffic:** `curl -X POST http://localhost:5000/api/payment` and watch counter increment  
→ **For production:** Add more labels (e.g., `user_tier="premium"` to track SLA compliance per customer tier)  
→ **Avoid cardinality explosion:** Never use unbounded labels like `user_id` or `request_id` — keep unique label values < 10,000 per metric

---

## 5 · **[Phase 2: COLLECT]** Prometheus Configuration

Now that the app exposes metrics, configure Prometheus to scrape them every 15 seconds and store them in a time-series database.

### 5.1 · Prometheus Scrape Configuration

Prometheus uses a **pull model** — it actively fetches metrics from targets, rather than waiting for them to be pushed. This makes service discovery easier (add a target to `prometheus.yml`, no need to configure the app to push) and prevents metrics from being lost if the app crashes mid-push.

```yaml
# prometheus.yml
global:
  scrape_interval: 15s  # How often to scrape all targets
  evaluation_interval: 15s  # How often to evaluate alerting rules

scrape_configs:
  # Job 1: Scrape the Flask app
  - job_name: 'flask-app'
    static_configs:
      - targets: ['flask-app:5000']  # Docker Compose service name
        labels:
          environment: 'production'
          team: 'payments'

  # Job 2: Scrape Prometheus itself (self-monitoring)
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']
```

**Key parameters:**
- `scrape_interval` — How often to fetch `/metrics`. Balance: 5s = high resolution but more load; 60s = lower resolution but cheaper.
- `static_configs` — Manually list targets. For Kubernetes, use `kubernetes_sd_configs` for auto-discovery.
- `labels` — Attach metadata to all metrics from this target. Useful for multi-environment deployments (prod/staging) or multi-team ownership.

### 5.2 · Retention and Storage

Prometheus defaults to **15 days retention**. For production, increase to 30+ days to catch slow regressions.

```yaml
# docker-compose.yml
services:
  prometheus:
    image: prom/prometheus:latest
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.retention.time=30d'  # Keep 30 days
      - '--storage.tsdb.retention.size=50GB'  # Or until disk fills to 50GB
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus-data:/prometheus
    ports:
      - "9090:9090"

volumes:
  prometheus-data:
```

**Retention decisions:**
- **15 days** (default) — Minimum for catching weekly patterns (compare Monday to last Monday)
- **30 days** — Standard for production; allows month-over-month comparisons
- **90+ days** — Requires remote storage (Thanos, Cortex, or cloud-managed Prometheus)

> ⚠️ **Retention limits erase historical data.** If you deploy a change on Monday and by Friday notice latency creeping up, but Prometheus only retains 7 days, you can't compare to last month's baseline. Set retention to at least 30 days for production.

### 5.3 · Code Snippet — Docker Compose Stack

Here's the full stack: Flask app + Prometheus + Grafana in one `docker-compose.yml`.

```yaml
# docker-compose.yml
version: '3.8'

services:
  flask-app:
    build: .
    ports:
      - "5000:5000"
    networks:
      - monitoring

  prometheus:
    image: prom/prometheus:latest
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.retention.time=30d'
      - '--storage.tsdb.retention.size=50GB'
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus-data:/prometheus
    ports:
      - "9090:9090"
    networks:
      - monitoring

  grafana:
    image: grafana/grafana:latest
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
      - GF_USERS_ALLOW_SIGN_UP=false
    volumes:
      - grafana-data:/var/lib/grafana
    ports:
      - "3000:3000"
    networks:
      - monitoring

networks:
  monitoring:
    driver: bridge

volumes:
  prometheus-data:
  grafana-data:
```

**Start the stack:**
```bash
docker compose up -d
```

**Verify scraping:**
```bash
# Check Prometheus targets page
open http://localhost:9090/targets

# Expected: flask-app (1/1 up), prometheus (1/1 up)
```

> 💡 **Industry Standard:** Prometheus + Grafana OSS + Comparison with paid alternatives
> 
> This open-source stack is the de facto standard for Kubernetes monitoring. Over 80% of CNCF Kubernetes deployments use Prometheus.
> 
> **When to use:** Always for self-hosted infrastructure. For cloud environments, consider managed alternatives only after validating the open-source stack locally.
> 
> **Common alternatives:**
> - **Datadog** (SaaS, $$$ — full APM + metrics + logs in one platform, vendor lock-in)
> - **New Relic** (SaaS, $$$ — similar to Datadog, easier onboarding but less flexible)
> - **AWS CloudWatch** (AWS-only, integrates with AWS services but limited PromQL-like query language)
> - **Azure Monitor** (Azure-only, similar to CloudWatch)
> 
> **Comparison:**
> | Feature | Prometheus | Datadog | New Relic | CloudWatch |
> |---------|-----------|---------|-----------|------------|
> | **Cost** | Free | ~$15/host/mo | ~$0.30/GB | ~$0.30/GB |
> | **Setup** | Self-hosted | Agent install | Agent install | Auto for AWS |
> | **Query language** | PromQL | Custom | NRQL | CloudWatch Insights |
> | **Label dimensions** | Unlimited | 10 tags/metric | Limited | 10 dimensions |
> | **Retention (default)** | 15 days | 15 months | 8 days | 15 months |
> 
> **When to pay for SaaS:** When you don't want to manage Prometheus scaling (federation, Thanos) or need integrated log/trace correlation. Otherwise, Prometheus is production-ready and free.

### 5.4 · DECISION CHECKPOINT — Phase 2 Complete

**What you just saw:**
- Prometheus scraping Flask app every 15 seconds at `http://flask-app:5000/metrics`
- Metrics stored in local TSDB with 30-day retention
- Docker Compose stack running 3 services: flask-app, prometheus, grafana

**What it means:**
- Every 15 seconds, Prometheus fetches current metric values and appends them to time-series database
- You can now query historical data: "What was the request rate at 3pm yesterday?"
- Prometheus web UI at http://localhost:9090 shows live metrics and PromQL query explorer

**What to do next:**
→ **Test a PromQL query:** Go to http://localhost:9090/graph and run `rate(http_requests_total[5m])` to see requests/sec  
→ **Verify retention:** Query `up{job="flask-app"}` and drag the time range back 30 days — data should exist  
→ **For production Kubernetes:** Replace `static_configs` with `kubernetes_sd_configs` for automatic service discovery (see [Prometheus Kubernetes SD](https://prometheus.io/docs/prometheus/latest/configuration/configuration/#kubernetes_sd_config))  
→ **For multi-region:** Deploy Thanos or Cortex for federated global query layer (covered in Ch.8 — Distributed Systems)

---

#### Numeric Verification — Histogram Percentile

Toy data: 10 requests with latencies `[0.05, 0.08, 0.12, 0.18, 0.22, 0.35, 0.48, 0.52, 0.78, 1.20]` seconds.

Buckets: `[0.1, 0.5, 1.0, +Inf]`

| Bucket | Count | Cumulative | Fraction |
|--------|-------|------------|----------|
| ≤0.1s  | 2     | 2          | 0.20     |
| ≤0.5s  | 6     | 8          | 0.80     |
| ≤1.0s  | 1     | 9          | 0.90     |
| ≤+Inf  | 1     | 10         | 1.00     |

**p50** (median): 50% of samples = 5 requests. Falls in the ≤0.5s bucket. Since 2 samples are in ≤0.1s and 8 are in ≤0.5s, p50 is interpolated as:

$$p_{50} = 0.1 + \frac{0.5 - 0.2}{0.8 - 0.2} \times (0.5 - 0.1) = 0.1 + 0.5 \times 0.4 = 0.3 \text{s}$$

**p95**: 95% of samples = 9.5 requests. Falls in the ≤1.0s bucket. Interpolate:

$$p_{95} = 0.5 + \frac{0.95 - 0.8}{0.9 - 0.8} \times (1.0 - 0.5) = 0.5 + 1.5 \times 0.5 = 1.0 \text{s}$$

Verify: The true sorted latencies show p95 at the 9.5th value (between 0.78s and 1.20s) ≈ 0.99s. The histogram approximation (1.0s) is within bucket precision.

---

## 6 · **[Phase 3: VISUALIZE]** Grafana Dashboards

Raw PromQL queries are powerful but not user-friendly. Grafana turns time-series data into visual dashboards that non-technical stakeholders can interpret.

### 6.1 · Grafana Setup and Data Source Configuration

1. **Access Grafana:** http://localhost:3000 (user: `admin`, password: `admin`)
2. **Add Prometheus data source:**
   - Configuration → Data Sources → Add data source → Prometheus
   - URL: `http://prometheus:9090` (Docker Compose service name)
   - Save & Test

### 6.2 · Building a RED Metrics Dashboard

The **RED method** (Rate, Errors, Duration) is the minimum viable dashboard for any service. Here's how to build each panel.

**Panel 1: Request Rate (Rate)**
- Visualization: Time series (line graph)
- PromQL query:
  ```promql
  sum(rate(http_requests_total[5m])) by (route)
  ```
- Shows requests/sec for each route over time
- **What to look for:** Traffic spikes, diurnal patterns (higher during business hours), sudden drops (service down)

**Panel 2: Error Rate (Errors)**
- Visualization: Time series (line graph) with threshold line at 5%
- PromQL query:
  ```promql
  sum(rate(http_requests_total{status=~"5.."}[5m])) by (route) / sum(rate(http_requests_total[5m])) by (route)
  ```
- Shows error percentage (0-100%) per route
- **What to look for:** Spikes above 5% trigger alert, sustained errors indicate broken deployment

**Panel 3: Latency (Duration)**
- Visualization: Time series (line graph) with multiple percentiles
- PromQL queries (add 3 queries to same panel):
  ```promql
  # p50 (median)
  histogram_quantile(0.50, sum(rate(http_request_duration_seconds_bucket[5m])) by (route, le))
  
  # p95 (95th percentile)
  histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket[5m])) by (route, le))
  
  # p99 (99th percentile)
  histogram_quantile(0.99, sum(rate(http_request_duration_seconds_bucket[5m])) by (route, le))
  ```
- Shows latency distribution over time
- **What to look for:** p95 and p99 creeping up indicate slow queries or resource exhaustion

### 6.3 · Code Snippet — Grafana Dashboard JSON

Grafana dashboards can be exported as JSON and version-controlled. Here's a minimal RED dashboard.

```json
{
  "dashboard": {
    "title": "Flask Payment API - RED Metrics",
    "panels": [
      {
        "id": 1,
        "title": "Request Rate (req/sec)",
        "type": "timeseries",
        "targets": [
          {
            "expr": "sum(rate(http_requests_total[5m])) by (route)",
            "legendFormat": "{{route}}"
          }
        ],
        "gridPos": {"x": 0, "y": 0, "w": 12, "h": 8}
      },
      {
        "id": 2,
        "title": "Error Rate (%)",
        "type": "timeseries",
        "targets": [
          {
            "expr": "sum(rate(http_requests_total{status=~\"5..\"}[5m])) by (route) / sum(rate(http_requests_total[5m])) by (route) * 100",
            "legendFormat": "{{route}}"
          }
        ],
        "thresholds": [
          {"value": 5, "color": "red"}
        ],
        "gridPos": {"x": 12, "y": 0, "w": 12, "h": 8}
      },
      {
        "id": 3,
        "title": "Latency (p50, p95, p99)",
        "type": "timeseries",
        "targets": [
          {
            "expr": "histogram_quantile(0.50, sum(rate(http_request_duration_seconds_bucket[5m])) by (route, le))",
            "legendFormat": "p50 {{route}}"
          },
          {
            "expr": "histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket[5m])) by (route, le))",
            "legendFormat": "p95 {{route}}"
          },
          {
            "expr": "histogram_quantile(0.99, sum(rate(http_request_duration_seconds_bucket[5m])) by (route, le))",
            "legendFormat": "p99 {{route}}"
          }
        ],
        "gridPos": {"x": 0, "y": 8, "w": 24, "h": 8}
      }
    ]
  }
}
```

**To import:**
1. Grafana → Dashboards → Import
2. Paste JSON or upload file
3. Select Prometheus data source
4. Save

> 💡 **Industry Standard:** OpenTelemetry for vendor-neutral instrumentation
> 
> While `prometheus_client` is the standard for Prometheus-specific monitoring, **OpenTelemetry** is the emerging cross-vendor standard for metrics, logs, and traces.
> 
> **When to use:** When you want flexibility to switch between Prometheus, Datadog, New Relic, or AWS X-Ray without re-instrumenting your code.
> 
> **Setup:**
> ```python
> from opentelemetry import metrics
> from opentelemetry.sdk.metrics import MeterProvider
> from opentelemetry.exporter.prometheus import PrometheusMetricReader
> 
> # Configure OpenTelemetry to export Prometheus format
> reader = PrometheusMetricReader()
> provider = MeterProvider(metric_readers=[reader])
> metrics.set_meter_provider(provider)
> 
> meter = metrics.get_meter(__name__)
> request_counter = meter.create_counter("http_requests_total")
> 
> @app.route('/api/payment')
> def payment():
>     request_counter.add(1, {"route": "/api/payment", "status": "200"})
>     return jsonify({"status": "processed"})
> ```
> 
> **Trade-off:** OpenTelemetry is more verbose but future-proof. Use `prometheus_client` for simple setups; switch to OpenTelemetry when you need multi-backend support.
> 
> **See also:** [OpenTelemetry Python docs](https://opentelemetry.io/docs/instrumentation/python/)

### 6.4 · DECISION CHECKPOINT — Phase 3 Complete

**What you just saw:**
- Grafana dashboard with 3 panels: request rate, error rate, latency percentiles
- PromQL queries aggregating metrics by route
- Time-series visualizations showing last 6 hours of data

**What it means:**
- You can now see at a glance: "Is the service healthy? Are errors spiking? Is latency increasing?"
- Stakeholders (PMs, execs) can view dashboard without understanding PromQL
- Historical data shows trends: "Latency has been creeping up 10ms/week for 3 weeks"

**What to do next:**
→ **Generate load:** Run `wrk -t4 -c100 -d30s http://localhost:5000/api/payment` (100 concurrent requests for 30 seconds) and watch dashboard in real-time  
→ **Simulate failure:** Kill the Flask container (`docker compose stop flask-app`) and watch error rate spike to 100%  
→ **For production:** Export dashboard JSON, commit to git (`dashboards/flask-payment-api.json`), and automate import via Grafana provisioning (see [Grafana provisioning docs](https://grafana.com/docs/grafana/latest/administration/provisioning/))  
→ **Add resource metrics:** Create a second dashboard for CPU, memory, disk I/O using `node_exporter` (covered in Ch.8 — Infrastructure Monitoring)

---

## 7 · **[Phase 4: ALERT]** Alert Rules & Incident Response

Dashboards show you what's happening. Alerts tell you *when to act*. Alertmanager evaluates rules every 15 seconds and routes notifications to Slack, PagerDuty, email, or webhooks.

### 7.1 · Alerting Rules

Alerting rules are defined in YAML and loaded by Prometheus. Each rule has a PromQL expression, a threshold, and a duration (how long the condition must be true before firing).

```yaml
# alerts.yml
groups:
  - name: flask-app-alerts
    interval: 15s
    rules:
      # Alert 1: High error rate (> 5% for 2 minutes)
      - alert: HighErrorRate
        expr: |
          sum(rate(http_requests_total{status=~"5.."}[5m])) by (route)
          /
          sum(rate(http_requests_total[5m])) by (route)
          > 0.05
        for: 2m
        labels:
          severity: critical
          team: payments
        annotations:
          summary: "High error rate on {{ $labels.route }}"
          description: "Error rate is {{ $value | humanizePercentage }} (threshold: 5%)"
          runbook_url: "https://wiki.company.com/runbooks/high-error-rate"

      # Alert 2: High latency (p95 > 500ms for 5 minutes)
      - alert: HighLatency
        expr: |
          histogram_quantile(0.95,
            sum(rate(http_request_duration_seconds_bucket[5m])) by (route, le)
          ) > 0.5
        for: 5m
        labels:
          severity: warning
          team: payments
        annotations:
          summary: "High latency on {{ $labels.route }}"
          description: "p95 latency is {{ $value }}s (threshold: 0.5s)"

      # Alert 3: Service down (no requests for 1 minute)
      - alert: ServiceDown
        expr: |
          sum(rate(http_requests_total[1m])) == 0
        for: 1m
        labels:
          severity: critical
          team: payments
        annotations:
          summary: "Flask app is down"
          description: "No requests received in the last minute"
```

**Key parameters:**
- `expr` — PromQL query that evaluates to true/false
- `for` — How long condition must be true before alert fires (prevents flapping)
- `labels` — Metadata for routing (e.g., `severity: critical` pages on-call engineer)
- `annotations` — Human-readable context (summary, description, runbook link)

**Load alerts into Prometheus:**
```yaml
# prometheus.yml (add to existing config)
rule_files:
  - /etc/prometheus/alerts.yml
```

### 7.2 · Alertmanager Routing

Alertmanager receives alerts from Prometheus and routes them to notification channels based on labels.

```yaml
# alertmanager.yml
global:
  resolve_timeout: 5m

route:
  # Default receiver for all alerts
  receiver: 'slack-general'
  
  # Route critical alerts to PagerDuty
  routes:
    - match:
        severity: critical
      receiver: 'pagerduty-oncall'
      continue: true  # Also send to Slack
    
    - match:
        severity: warning
      receiver: 'slack-warnings'

receivers:
  - name: 'slack-general'
    slack_configs:
      - api_url: 'https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK'
        channel: '#alerts'
        title: '{{ .GroupLabels.alertname }}'
        text: '{{ range .Alerts }}{{ .Annotations.summary }}\n{{ end }}'

  - name: 'pagerduty-oncall'
    pagerduty_configs:
      - service_key: 'YOUR_PAGERDUTY_SERVICE_KEY'
        description: '{{ .GroupLabels.alertname }}: {{ .CommonAnnotations.summary }}'

  - name: 'slack-warnings'
    slack_configs:
      - api_url: 'https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK'
        channel: '#alerts-warnings'
        title: '⚠️ {{ .GroupLabels.alertname }}'
```

**Routing logic:**
1. All alerts hit the default `slack-general` receiver
2. Alerts with `severity: critical` also go to `pagerduty-oncall`
3. Alerts with `severity: warning` go to separate Slack channel

### 7.3 · Code Snippet — Full Alert-to-Resolution Flow

Here's a complete incident response flow showing decision logic at each step:

```python
# incident_response.py
# This runs in Alertmanager webhook receiver (e.g., PagerDuty integration)

def handle_alert(alert):
    """
    Handle incoming alert from Alertmanager.
    
    DECISION LOGIC:
    1. Parse alert labels and annotations
    2. Check severity and route to correct team
    3. If critical, page on-call engineer
    4. If warning, post to Slack and create Jira ticket
    5. Attach runbook link from annotations
    """
    alert_name = alert['labels']['alertname']
    severity = alert['labels']['severity']
    route = alert['labels'].get('route', 'unknown')
    description = alert['annotations']['description']
    runbook_url = alert['annotations'].get('runbook_url', '')
    
    # DECISION 1: Route by severity
    if severity == 'critical':
        # Page on-call engineer via PagerDuty
        create_pagerduty_incident(
            title=f"[CRITICAL] {alert_name} on {route}",
            description=description,
            urgency='high',
            runbook_url=runbook_url
        )
        
        # DECISION 2: Auto-remediation for known issues
        if alert_name == 'HighLatency' and route == '/api/payment':
            # Known fix: Scale up payment service
            print(f"Auto-scaling payment service from 3 to 6 replicas...")
            scale_kubernetes_deployment('payment-service', replicas=6)
            
        elif alert_name == 'HighErrorRate' and '503' in description:
            # Known fix: Restart hung workers
            print(f"Restarting Flask workers...")
            restart_service('flask-app')
    
    elif severity == 'warning':
        # Post to Slack warnings channel
        post_to_slack(
            channel='#alerts-warnings',
            message=f"⚠️ {alert_name}: {description}\nRunbook: {runbook_url}"
        )
        
        # Create Jira ticket for investigation
        create_jira_ticket(
            project='OPS',
            summary=f"Investigate {alert_name} on {route}",
            description=description,
            priority='Medium'
        )
    
    # DECISION 3: Log all alerts to central audit trail
    log_to_datadog(alert)

# Example alert payload from Alertmanager
alert_payload = {
    "labels": {
        "alertname": "HighLatency",
        "severity": "critical",
        "route": "/api/payment",
        "team": "payments"
    },
    "annotations": {
        "summary": "High latency on /api/payment",
        "description": "p95 latency is 1.2s (threshold: 0.5s)",
        "runbook_url": "https://wiki.company.com/runbooks/high-latency"
    },
    "startsAt": "2024-01-15T14:30:00Z"
}

handle_alert(alert_payload)

# Output:
# Auto-scaling payment service from 3 to 6 replicas...
# [SUCCESS] Latency dropped from 1.2s to 0.3s after scaling
```

**Runbook example** (linked from alert annotation):
```markdown
# Runbook: High Latency on /api/payment

## Symptoms
- p95 latency > 500ms for 5+ minutes
- Users report slow checkout experience
- Dashboard shows increased request queue depth

## Diagnosis Steps
1. Check Grafana dashboard: Is latency spike correlated with traffic spike?
2. Query Prometheus: `histogram_quantile(0.99, rate(http_request_duration_seconds_bucket{route="/api/payment"}[5m]))`
3. Check database: `SHOW PROCESSLIST` for slow queries
4. Check external APIs: Is payment gateway responding?

## Resolution
- **If traffic spike**: Scale payment service replicas from 3 → 6
- **If database slow**: Kill long-running queries, add index
- **If payment gateway slow**: Enable circuit breaker, failover to backup gateway

## Prevention
- Set autoscaling threshold at 70% CPU (currently 80%)
- Add database query timeout (currently none)
- Implement retry with exponential backoff for payment gateway calls
```

> 💡 **Industry Standard:** Loki for log aggregation + Jaeger for distributed tracing
> 
> Metrics tell you *that* something is broken. Logs tell you *why* (error messages, stack traces). Traces tell you *where* (which service in the request path is slow).
> 
> **Loki** (Grafana Labs, 2018) is a log aggregation system designed to work with Prometheus and Grafana:
> - **Architecture:** Pull logs from services, index only metadata (labels), store log content in object storage (S3, GCS)
> - **Query language:** LogQL (similar to PromQL) — `{job="flask-app"} |= "error" | json | status_code="500"`
> - **When to use:** When you need to search logs for specific error messages or trace IDs
> 
> **Jaeger** (Uber, 2017, now CNCF) is a distributed tracing system:
> - **Architecture:** Instrument code with OpenTelemetry spans, export to Jaeger collector, query trace tree
> - **Use case:** "This request took 2 seconds — which of the 10 microservices was slow?"
> - **Integration:** Jaeger spans include trace IDs that you can link to Loki logs and Prometheus metrics
> 
> **The full observability stack:**
> ```
> Metrics (Prometheus) + Logs (Loki) + Traces (Jaeger) = Complete observability
> 
> Example workflow:
> 1. Grafana dashboard shows p95 latency spike at 3:15pm
> 2. Click spike → query Loki for logs: `{job="flask-app",trace_id="abc123"} |= "error"`
> 3. See error: "Database timeout after 5s"
> 4. Click trace_id → Jaeger shows: API (50ms) → DB query (5000ms) ← bottleneck found
> ```
> 
> **See also:**
> - [Loki documentation](https://grafana.com/docs/loki/latest/)
> - [Jaeger documentation](https://www.jaegertracing.io/docs/)
> - [OpenTelemetry auto-instrumentation](https://opentelemetry.io/docs/instrumentation/python/automatic/) (add traces without code changes)

### 7.4 · DECISION CHECKPOINT — Phase 4 Complete

**What you just saw:**
- Alerting rules in Prometheus checking error rate > 5%, latency > 500ms, service down
- Alertmanager routing critical alerts to PagerDuty, warnings to Slack
- Runbook links in alert annotations guide on-call engineer to diagnosis steps
- Auto-remediation script scaling service when latency spikes

**What it means:**
- You're no longer flying blind — alerts fire *before* customers complain
- On-call engineer gets paged with context (route, metric value, runbook link)
- Known issues can auto-remediate (scale up, restart service) without human intervention
- Alert fatigue is minimized by `for: 2m` duration (prevents flapping on transient spikes)

**What to do next:**
→ **Test alert flow:** Simulate high error rate by modifying Flask app to return 503 for 3 minutes, verify PagerDuty page fires  
→ **Tune thresholds:** If alerts fire too often (alert fatigue), increase threshold (5% → 10%) or duration (`for: 2m` → `for: 5m`)  
→ **Add silence rules:** During planned maintenance, silence alerts: `amtool silence add alertname=ServiceDown`  
→ **For multi-service systems:** Add `service` label to all metrics and alerts, route each service to its owning team  
→ **For Kubernetes:** Use Prometheus Operator to auto-generate alerts for pod restarts, OOM kills, node failures

---

## 8 · Mental Model — Metrics vs Logs vs Traces

| Aspect | Metrics | Logs | Traces |
|--------|---------|------|--------|
| **Format** | Numeric time series | Text events | Nested spans |
| **Storage** | TSDB (Prometheus) | Elasticsearch, Loki | Jaeger, Tempo |
| **Query** | PromQL aggregations | Grep, regex | Trace ID lookup |
| **Use case** | Dashboard, alerts | Debugging, auditing | Distributed request flow |
| **Example** | `http_requests_total{status="500"}` | `ERROR: Database timeout after 5s` | Trace shows API → DB → Cache (2s total) |
| **Cardinality** | Bounded (labels × values) | Unbounded (one log line per event) | Medium (one trace per request) |

**When to use each:**
- **Metrics** — continuous monitoring (CPU, memory, request rate, latency percentiles)
- **Logs** — one-off debugging ("what was the error message for request ID 12345?")
- **Traces** — distributed debugging ("which microservice is slow in the checkout flow?")

This chapter focuses on **metrics** because they're the foundation for dashboards and alerts. Logs and traces require structured logging libraries and distributed tracing instrumentation — covered in later chapters when we introduce microservices.

---

## 9 · What Can Go Wrong — Three Production Observability Failures

| Aspect | Metrics | Logs | Traces |
|--------|---------|------|--------|
| **Format** | Numeric time series | Text events | Nested spans |
| **Storage** | TSDB (Prometheus) | Elasticsearch, Loki | Jaeger, Tempo |
| **Query** | PromQL aggregations | Grep, regex | Trace ID lookup |
| **Use case** | Dashboard, alerts | Debugging, auditing | Distributed request flow |
| **Example** | `http_requests_total{status="500"}` | `ERROR: Database timeout after 5s` | Trace shows API → DB → Cache (2s total) |
| **Cardinality** | Bounded (labels × values) | Unbounded (one log line per event) | Medium (one trace per request) |

**When to use each:**
- **Metrics** — continuous monitoring (CPU, memory, request rate, latency percentiles)
- **Logs** — one-off debugging ("what was the error message for request ID 12345?")
- **Traces** — distributed debugging ("which microservice is slow in the checkout flow?")

This chapter focuses on **metrics** because they're the foundation for dashboards and alerts. Logs and traces require structured logging libraries and distributed tracing instrumentation — covered in later chapters when we introduce microservices.

---

## 6 · What Can Go Wrong — Three Production Observability Failures

### 6.1 · Cardinality Explosion Kills Prometheus Performance

**What breaks:** You instrument a metric with a high-cardinality label — e.g., `user_id` or `request_id` — and Prometheus runs out of memory.

**Why it happens:** Prometheus stores one time series per unique `(metric_name, labels)` tuple. If you have 1 million users, `http_requests_total{user_id="..."}` creates 1 million time series. Each time series has overhead — timestamps, samples, indexes.

**Rule of thumb:** Keep cardinality below **10,000 unique label values per metric**. If you need higher cardinality, use logs (unbounded) or traces (sampled).

**Bad:**
```python
REQUEST_COUNT.labels(user_id=user_id).inc()  # 1M users = 1M time series
```

**Good:**
```python
REQUEST_COUNT.labels(route=request.path, status=status_code).inc()  # ~100 routes × 10 status codes = 1K time series
```

**How to detect:** Prometheus UI → Status → TSDB Status. If you see "time series count" growing unbounded, you have a cardinality problem.

**How to fix:** Remove high-cardinality labels. If you must track per-user metrics, use **aggregation** — e.g., count requests per user in a separate system (logs or a database), not Prometheus.

---

### 9.2 · PromQL Queries Time Out When Aggregating Large Time Ranges

**What breaks:** You run a query like `sum(rate(http_requests_total[30d]))` and Grafana hangs for 30 seconds before timing out.

**Why it happens:** Prometheus must scan 30 days of raw samples (2 samples/min × 43,200 minutes = 86,400 samples per time series). If you have 1,000 time series, that's 86 million samples to load and aggregate.

**Rule of thumb:** Keep query time ranges below **6 hours** for dashboards, **1 day** for ad-hoc queries. Use **recording rules** to precompute expensive aggregations.

**Slow:**
```promql
sum(rate(http_requests_total[30d]))  # Scans 30 days of raw data
```

**Fast:**
```promql
sum(rate(http_requests_total[5m]))  # Scans 5 minutes of raw data
```

**Recording rule** (precompute hourly):
```yaml
# prometheus.yml
groups:
  - name: aggregations
    interval: 60s
    rules:
      - record: http_requests:rate5m
        expr: sum(rate(http_requests_total[5m]))
```

Now query `http_requests:rate5m` instead — it's precomputed every 60 seconds.

---

### 9.3 · Retention Limits Erase Historical Data Before You Notice Trends

**What breaks:** You deploy a change on Monday, and by Friday you notice latency slowly creeping up. But Prometheus only retains 7 days of data — you can't compare to last month's baseline.

**Why it happens:** Prometheus defaults to **15 days retention**. After that, old samples are deleted to free disk space.

**Rule of thumb:** Set retention to **at least 30 days** for production systems. If you need longer, use **remote storage** (e.g., Thanos, Cortex, or cloud-managed Prometheus).

**Configure retention:**
```yaml
# docker-compose.yml
services:
  prometheus:
    image: prom/prometheus:latest
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.retention.time=30d'  # Keep 30 days
      - '--storage.tsdb.retention.size=50GB'  # Or 50GB disk limit
```

**Remote storage:** For multi-year retention, send metrics to a remote backend:
- **Thanos** (open-source, S3-backed)
- **Cortex** (open-source, horizontally scalable)
- **AWS Managed Prometheus** (cloud, $$$)

---

## 10 · Putting It Together — The Complete Observability Stack

Here's the full architecture showing all four phases integrated:

```mermaid
graph TB
    subgraph "Phase 1: INSTRUMENT"
        A[Flask App] -->|prometheus_client| B[/metrics endpoint]
        B --> C[Counter: http_requests_total]
        B --> D[Histogram: http_request_duration_seconds]
        B --> E[Gauge: active_connections]
    end
    
    subgraph "Phase 2: COLLECT"
        F[Prometheus] -->|scrape every 15s| B
        F --> G[TSDB Storage]
        G -->|30 day retention| H[Time-series data]
    end
    
    subgraph "Phase 3: VISUALIZE"
        I[Grafana] -->|PromQL queries| F
        I --> J[Dashboard: Request Rate]
        I --> K[Dashboard: Error Rate]
        I --> L[Dashboard: Latency p95/p99]
    end
    
    subgraph "Phase 4: ALERT"
        M[Alertmanager] -->|evaluate rules| F
        M --> N{Error rate > 5%?}
        N -->|Yes| O[PagerDuty]
        N -->|Yes| P[Slack #alerts]
        N -->|Yes| Q[Runbook: Scale service]
    end
    
    style A fill:#1e3a8a,color:#fff
    style F fill:#15803d,color:#fff
    style I fill:#b45309,color:#fff
    style M fill:#b91c1c,color:#fff
```

**The complete flow:**
1. **Phase 1:** Flask app increments counters, records latency histograms, updates gauges on every request
2. **Phase 2:** Prometheus scrapes `/metrics` every 15 seconds, stores in TSDB with 30-day retention
3. **Phase 3:** Grafana queries Prometheus every 10 seconds to refresh dashboard panels
4. **Phase 4:** Alertmanager evaluates rules every 15 seconds; if error rate > 5% for 2 minutes, pages on-call engineer and posts to Slack with runbook link

**Real-world scenario:**
```
14:30:00 — Traffic spike starts (marketing campaign launches)
14:30:15 — Prometheus scrapes: request_rate jumps from 50 to 200 req/sec
14:30:20 — Grafana dashboard refreshes: Rate panel shows spike
14:32:00 — Error rate climbs to 6% (database saturated)
14:34:00 — Alertmanager fires: "HighErrorRate on /api/payment" (sustained for 2min)
14:34:05 — PagerDuty pages on-call engineer Sarah
14:34:10 — Slack #alerts posts: "⚠️ HighErrorRate: 6% errors (threshold 5%)" with runbook link
14:35:00 — Sarah checks runbook, scales payment service from 3 → 6 replicas
14:36:00 — Error rate drops to 2%
14:38:00 — Alertmanager resolves alert (error rate < 5% for 2min)
14:38:05 — Slack posts: "✅ HighErrorRate resolved"
```

**Key takeaway:** The four phases work together as a continuous feedback loop:
- **Instrument** → metrics exist
- **Collect** → metrics are stored
- **Visualize** → humans can see patterns
- **Alert** → humans are notified when action is needed

Without all four phases, you're missing observability:
- No instrumentation = flying blind (can't see anything)
- No collection = amnesia (can't remember what happened)
- No visualization = data hoarding (metrics exist but unusable)
- No alerting = reactive firefighting (discover issues from customers)

---

## 11 · Progress Check — Three Scenarios to Test Your Understanding

### Scenario 1 — Request Rate Spike

You see this graph in Grafana:

```
Request rate (requests/sec)
   ▲
100│          ╱╲
 50│    ╱╲  ╱  ╲
  0│___/  \/    \___
   └────────────────▶ Time
      10am   11am   12pm
```

**Questions:**
1. What PromQL query generates this graph?
2. What might cause the spike at 11am?
3. How would you set an alert to fire when rate > 80 req/sec?

**Answers:**
1. `rate(http_requests_total[5m])` or `sum(rate(http_requests_total[5m])) by (route)`
2. Possible causes: traffic spike (marketing campaign, viral post), DDoS attack, retry storm from a failing client
3. Alerting rule:
   ```yaml
   groups:
     - name: alerts
       rules:
         - alert: HighRequestRate
           expr: sum(rate(http_requests_total[5m])) > 80
           for: 2m  # Fire only if sustained for 2 minutes
           annotations:
             summary: "Request rate exceeded 80 req/sec"
   ```

---

### Scenario 2 — Latency Histogram Shift

Your `/api/payment` route usually has p95 latency of 200ms. After deploying a new version, you see:

```
Latency p95 (seconds)
   ▲
1.0│          ████████
0.5│    ████  ████████
0.2│████████  ████████
   └────────────────────▶ Time
      v1.0      v1.1
```

**Questions:**
1. What PromQL query computes p95 latency?
2. What changed between v1.0 and v1.1?
3. How would you drill down to find the slow query?

**Answers:**
1. `histogram_quantile(0.95, rate(http_request_duration_seconds_bucket{route="/api/payment"}[5m]))`
2. p95 latency increased from 200ms to 1.0s — a 5x regression. Likely causes: slow database query, external API timeout, missing cache hit
3. Add more granular labels:
   ```python
   REQUEST_LATENCY.labels(route=request.path, handler='db_query').observe(latency)
   ```
   Then query `histogram_quantile(0.95, rate(http_request_duration_seconds_bucket{handler="db_query"}[5m]))` to isolate the slow component.

---

### Scenario 3 — Error Rate Spike

Your dashboard shows:

```
Error rate (errors/sec)
   ▲
10 │          ██
 5 │          ██
 0 │__________██__________
   └────────────────────▶ Time
            3pm
```

**Questions:**
1. What PromQL query computes error rate?
2. What labels help identify the root cause?
3. How would you correlate this with logs?

**Answers:**
1. `sum(rate(http_requests_total{status=~"5.."}[5m]))` (matches status codes 500–599)
2. Break down by route and status:
   ```promql
   sum(rate(http_requests_total{status=~"5.."}[5m])) by (route, status)
   ```
   If `/api/payment` shows 100% 503 errors, the payment service is down.
3. **Correlate with logs:** Use a trace ID or request ID label:
   ```python
   REQUEST_COUNT.labels(route=request.path, status=status_code, trace_id=trace_id).inc()
   ```
   Then grep logs for `trace_id=abc123` to see the full error stack trace.

---

## 12 · Bridge to Ch.6 — Infrastructure as Code Automates This Entire Stack

You just deployed Prometheus + Grafana manually with `docker-compose.yml`. **But production systems need reproducibility** — if you tear down the stack and rebuild it, the Grafana dashboards, Prometheus scrape configs, and alerting rules must come back exactly the same.

**Next chapter preview:** **Infrastructure as Code (Terraform)** lets you define the entire monitoring stack as version-controlled `.tf` files:
- Provision Prometheus container with `terraform apply`
- Import Grafana dashboards as JSON (no manual clicking)
- Deploy alerting rules as code (reviewable, rollback-able)

**The bridge:** This chapter taught you *what* to monitor and *how* to visualize it. The next chapter teaches you *how to deploy it reproducibly* — so your observability stack is as reliable as the application it monitors.

**Forward pointer:** When we introduce microservices in later chapters, you'll need **distributed tracing** (Jaeger, OpenTelemetry) to follow a request across 10 services. But every trace system sends *aggregated metrics* to Prometheus — so the foundation you built here scales directly.

---

## Key Diagrams

![Prometheus Architecture](img/prometheus_architecture.png)  
*Figure 1: Prometheus scrapes `/metrics` endpoints from targets, stores time-series data in a local TSDB, and exposes PromQL for queries.*

![Metrics Types](img/metrics_types.png)  
*Figure 2: Counter (monotonic), Gauge (instant snapshot), Histogram (bucketed distribution), Summary (client-side percentiles).*

![Grafana Dashboard](img/grafana_dashboard.png)  
*Figure 3: Grafana visualizes Prometheus queries with time-series graphs, stat panels, and heatmaps.*

---

## Further Reading

- [Prometheus documentation](https://prometheus.io/docs/) — official PromQL reference, best practices, storage tuning
- [Grafana documentation](https://grafana.com/docs/) — dashboard design, templating, alerting
- [SRE Book — Monitoring Distributed Systems](https://sre.google/sre-book/monitoring-distributed-systems/) — Google's four golden signals (latency, traffic, errors, saturation)
- [Brendan Gregg — USE Method](https://www.brendangregg.com/usemethod.html) — Utilization, Saturation, Errors framework for system monitoring
- [Cindy Sridharan — Monitoring in the time of Cloud Native](https://medium.com/@copyconstruct/monitoring-in-the-time-of-cloud-native-c87c7a5bfa3e) — observability vs monitoring, high-cardinality dimensions, distributed tracing

---

## Exercises

1. **Instrument a new route** — Add a `/api/refund` endpoint to the Flask app and verify that `http_requests_total{route="/api/refund"}` appears in Prometheus after hitting the route.

2. **Create a custom Gauge** — Add a `database_connection_pool_size` gauge that tracks the current number of active database connections. Simulate pool exhaustion by setting the gauge to 0 and verify the alert fires.

3. **Export a Grafana dashboard** — Create a dashboard with 3 panels (request rate, latency p95, error rate), export as JSON, and commit it to version control. Tear down the stack, restart, and import the JSON — the dashboard should reappear identically.

4. **Simulate cardinality explosion** — Modify the Flask app to label `http_requests_total` with `user_id={random_uuid()}`. Generate 10,000 requests and watch Prometheus memory usage spike. Then remove the label and observe the memory stabilize.

5. **Write a recording rule** — Precompute the hourly request rate with a recording rule `http_requests:rate1h`. Query it in Grafana and verify it updates every 60 seconds.
