# Monitoring & Observability Diagrams

This directory contains generated diagrams for Ch.5: Monitoring & Observability.

## To Generate Diagrams

Run the generation scripts from the `gen_scripts/` directory:

```bash
cd gen_scripts
pip install matplotlib numpy pillow
python gen_ch05_prometheus_architecture.py
python gen_ch05_metrics_types.py
python gen_ch05_grafana_dashboard.py
python gen_ch05_needle_animation.py
```

## Expected Files

- `prometheus_architecture.png` — Prometheus scraping flow
- `metrics_types.png` — Counter, Gauge, Histogram, Summary comparison
- `grafana_dashboard.png` — Sample Grafana dashboard layout
- `ch05-monitoring-observability-needle.gif` — Animated error rate needle
- `ch05-monitoring-observability-needle.png` — Static frame of needle animation

## Note

Diagrams are not committed to git to keep the repository size small. Generate them locally as needed.
