# Diagram Generation Scripts

This directory contains Python scripts to generate visual diagrams for Ch.5: Monitoring & Observability.

## Scripts

1. **gen_ch05_prometheus_architecture.py** — Prometheus architecture diagram
   - Shows: Scrape targets → TSDB → Query flow
   - Output: `../img/prometheus_architecture.png`

2. **gen_ch05_metrics_types.py** — Metrics types comparison
   - Shows: Counter, Gauge, Histogram, Summary with examples
   - Output: `../img/metrics_types.png`

3. **gen_ch05_grafana_dashboard.py** — Sample Grafana dashboard layout
   - Shows: Time-series graphs, stat panels, heatmaps
   - Output: `../img/grafana_dashboard.png`

## Requirements

```bash
pip install matplotlib numpy
```

## Generate All Diagrams

```bash
python gen_ch05_prometheus_architecture.py
python gen_ch05_metrics_types.py
python gen_ch05_grafana_dashboard.py
```

Or run all at once:
```bash
python gen_ch05_prometheus_architecture.py && \
python gen_ch05_metrics_types.py && \
python gen_ch05_grafana_dashboard.py
```

## Output

All diagrams are saved to `../img/` with dark background theme matching the repository style.
