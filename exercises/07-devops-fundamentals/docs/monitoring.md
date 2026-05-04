# Monitoring and Observability

## Overview
The ML DevOps Demo includes comprehensive monitoring and observability using Prometheus, Grafana, and structured logging.

## Prometheus Metrics

### Application Metrics

#### Prediction Metrics
- `ml_predictions_total` - Total number of predictions made
- `ml_prediction_duration_seconds` - Histogram of prediction latency
- `ml_errors_total` - Total errors by error type

#### System Metrics
- `process_cpu_seconds_total` - CPU time used
- `process_resident_memory_bytes` - Memory usage
- `process_open_fds` - Open file descriptors

### Accessing Metrics

```bash
# View raw metrics
curl http://localhost:5000/metrics

# In Kubernetes
kubectl port-forward svc/ml-api 9090:9090 -n ml-dev
curl http://localhost:9090/metrics
```

## Grafana Dashboards

### Setup Grafana

```bash
# Deploy Grafana
kubectl apply -f kubernetes/grafana.yaml

# Access Grafana
kubectl port-forward svc/grafana 3000:3000
# Visit http://localhost:3000 (default: admin/admin)
```

### Key Dashboards

#### 1. API Performance Dashboard
- Request rate
- Latency (p50, p95, p99)
- Error rate
- Success rate

#### 2. Resource Usage Dashboard
- CPU utilization
- Memory usage
- Pod count
- Network I/O

#### 3. ML Model Dashboard
- Predictions per second
- Prediction latency
- Model load time
- Error types distribution

### Example PromQL Queries

```promql
# Request rate (requests per second)
rate(ml_predictions_total[5m])

# 95th percentile latency
histogram_quantile(0.95, rate(ml_prediction_duration_seconds_bucket[5m]))

# Error rate
rate(ml_errors_total[5m])

# CPU usage
rate(process_cpu_seconds_total[5m])
```

## Logging

### Structured Logging

All logs are in JSON format for easy parsing:

```json
{
  "time": "2024-01-15T10:30:00",
  "level": "INFO",
  "message": "Prediction successful",
  "prediction_time": 0.045,
  "environment": "prod"
}
```

### Log Levels by Environment

- **Development**: DEBUG
- **Staging**: INFO
- **Production**: WARNING

### Viewing Logs

```bash
# View application logs
kubectl logs -f deployment/ml-api -n ml-dev

# View last 100 lines
kubectl logs --tail=100 deployment/ml-api -n ml-dev

# View logs from all pods
kubectl logs -l app=ml-api -n ml-dev --all-containers=true

# Follow logs from specific pod
kubectl logs -f <pod-name> -n ml-dev
```

### Log Aggregation

For production, integrate with:
- **ELK Stack** (Elasticsearch, Logstash, Kibana)
- **Loki** (Grafana Loki)
- **Cloud Logging** (GCP Stackdriver, AWS CloudWatch)

Example Loki configuration:
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: loki-config
data:
  loki.yaml: |
    auth_enabled: false
    server:
      http_listen_port: 3100
```

## Health Checks

### Endpoints

#### 1. Liveness Probe (`/health`)
Checks if the application is alive
```bash
curl http://localhost:5000/health
```

Response:
```json
{
  "status": "healthy",
  "cpu_percent": 15.2,
  "memory_percent": 45.8
}
```

#### 2. Readiness Probe (`/ready`)
Checks if the application is ready to serve traffic
```bash
curl http://localhost:5000/ready
```

Response:
```json
{
  "status": "ready",
  "model_loaded": true,
  "memory_percent": 45.8
}
```

### Kubernetes Probe Configuration

```yaml
livenessProbe:
  httpGet:
    path: /health
    port: 5000
  initialDelaySeconds: 30
  periodSeconds: 10
  timeoutSeconds: 3
  failureThreshold: 3

readinessProbe:
  httpGet:
    path: /ready
    port: 5000
  initialDelaySeconds: 10
  periodSeconds: 5
  timeoutSeconds: 3
  failureThreshold: 3
```

## Alerting

### Prometheus Alerts

Create alert rules in `prometheus-rules.yaml`:

```yaml
groups:
- name: ml-api-alerts
  rules:
  - alert: HighErrorRate
    expr: rate(ml_errors_total[5m]) > 0.05
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High error rate detected"
      description: "Error rate is {{ $value }} errors/sec"
  
  - alert: HighLatency
    expr: histogram_quantile(0.95, rate(ml_prediction_duration_seconds_bucket[5m])) > 0.5
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High prediction latency"
      description: "P95 latency is {{ $value }} seconds"
  
  - alert: PodDown
    expr: up{job="ml-api"} == 0
    for: 1m
    labels:
      severity: critical
    annotations:
      summary: "ML API pod is down"
```

### Alert Channels

Configure alert notifications:
- **Slack** - Team notifications
- **PagerDuty** - On-call alerts
- **Email** - Summary reports

## Performance Monitoring

### Key Metrics to Track

1. **Latency**
   - p50, p95, p99 prediction time
   - API response time

2. **Throughput**
   - Requests per second
   - Predictions per second

3. **Error Rate**
   - 4xx errors (client errors)
   - 5xx errors (server errors)

4. **Resource Usage**
   - CPU utilization
   - Memory usage
   - Disk I/O

5. **Model Metrics**
   - Model load time
   - Prediction accuracy drift (if tracking)

### SLIs and SLOs

Define Service Level Indicators and Objectives:

```yaml
SLIs:
  - availability: 99.9%
  - latency_p95: < 100ms
  - error_rate: < 0.1%

SLOs:
  - availability: 99.5%
  - latency_p95: < 200ms
  - error_rate: < 1%
```

## Dashboard URLs

After deployment:

- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000
- **ML API**: http://localhost:5000
- **Metrics Endpoint**: http://localhost:5000/metrics
