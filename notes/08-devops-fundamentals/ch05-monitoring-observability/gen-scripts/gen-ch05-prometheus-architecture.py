#!/usr/bin/env python3
"""
Generate Prometheus Architecture Diagram

Shows the flow: Scrape targets → TSDB → Query (Grafana)
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib
matplotlib.use('Agg')  # Headless backend

# Set dark background style
plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(14, 8), facecolor='#1a1a2e')
ax.set_facecolor('#1a1a2e')
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

# Title
ax.text(5, 9.5, 'Prometheus Architecture', 
        fontsize=24, fontweight='bold', ha='center', va='top',
        color='#ffffff')

# Color palette (from authoring guidelines)
color_primary = '#1e3a8a'      # Primary blue
color_success = '#15803d'      # Success green
color_caution = '#b45309'      # Caution orange
color_info = '#1d4ed8'         # Info blue
color_text = '#ffffff'         # White text
color_box_bg = '#2d3748'       # Box background

# 1. Scrape Targets (left side)
targets = [
    ('Flask App\n:5000/metrics', 1.5, 7),
    ('Node Exporter\n:9100/metrics', 1.5, 5.5),
    ('Postgres\n:9187/metrics', 1.5, 4)
]

for label, x, y in targets:
    box = FancyBboxPatch((x-0.6, y-0.4), 1.2, 0.8,
                         boxstyle="round,pad=0.1", 
                         edgecolor=color_success, facecolor=color_box_bg,
                         linewidth=2)
    ax.add_patch(box)
    ax.text(x, y, label, ha='center', va='center', 
            fontsize=10, color=color_text, fontweight='bold')

# Group label for targets
ax.text(1.5, 8.2, 'Scrape Targets', ha='center', va='bottom',
        fontsize=12, fontweight='bold', color=color_success)

# 2. Prometheus Server (center)
prom_x, prom_y = 5, 5.5
prom_width, prom_height = 2.5, 3

# Main Prometheus box
prom_box = FancyBboxPatch((prom_x - prom_width/2, prom_y - prom_height/2), 
                          prom_width, prom_height,
                          boxstyle="round,pad=0.15", 
                          edgecolor=color_primary, facecolor=color_box_bg,
                          linewidth=3)
ax.add_patch(prom_box)

# Prometheus components
ax.text(prom_x, prom_y + 1.2, 'Prometheus Server', ha='center', va='center',
        fontsize=14, fontweight='bold', color=color_text)

# Retrieval component
retrieval_box = FancyBboxPatch((prom_x - 1, prom_y + 0.3), 2, 0.5,
                               boxstyle="round,pad=0.05",
                               edgecolor=color_info, facecolor='#1a2332',
                               linewidth=1.5)
ax.add_patch(retrieval_box)
ax.text(prom_x, prom_y + 0.55, 'Retrieval (Scraper)', ha='center', va='center',
        fontsize=9, color=color_text)

# TSDB component
tsdb_box = FancyBboxPatch((prom_x - 1, prom_y - 0.4), 2, 0.5,
                          boxstyle="round,pad=0.05",
                          edgecolor=color_caution, facecolor='#2d2416',
                          linewidth=1.5)
ax.add_patch(tsdb_box)
ax.text(prom_x, prom_y - 0.15, 'TSDB (Time-Series DB)', ha='center', va='center',
        fontsize=9, color=color_text)

# PromQL component
promql_box = FancyBboxPatch((prom_x - 1, prom_y - 1.1), 2, 0.5,
                            boxstyle="round,pad=0.05",
                            edgecolor=color_success, facecolor='#1a2d1a',
                            linewidth=1.5)
ax.add_patch(promql_box)
ax.text(prom_x, prom_y - 0.85, 'PromQL Engine', ha='center', va='center',
        fontsize=9, color=color_text)

# 3. Grafana (right side)
grafana_x, grafana_y = 8.5, 6.5
grafana_box = FancyBboxPatch((grafana_x - 0.8, grafana_y - 0.6), 1.6, 1.2,
                             boxstyle="round,pad=0.1",
                             edgecolor=color_caution, facecolor=color_box_bg,
                             linewidth=2)
ax.add_patch(grafana_box)
ax.text(grafana_x, grafana_y + 0.2, 'Grafana', ha='center', va='center',
        fontsize=12, fontweight='bold', color=color_text)
ax.text(grafana_x, grafana_y - 0.15, 'Dashboards', ha='center', va='center',
        fontsize=9, color=color_text, style='italic')

# 4. Alertmanager (right side, below Grafana)
alert_x, alert_y = 8.5, 4.5
alert_box = FancyBboxPatch((alert_x - 0.8, alert_y - 0.6), 1.6, 1.2,
                           boxstyle="round,pad=0.1",
                           edgecolor='#b91c1c', facecolor=color_box_bg,
                           linewidth=2)
ax.add_patch(alert_box)
ax.text(alert_x, alert_y + 0.2, 'Alertmanager', ha='center', va='center',
        fontsize=12, fontweight='bold', color=color_text)
ax.text(alert_x, alert_y - 0.15, 'Alerts & Routing', ha='center', va='center',
        fontsize=9, color=color_text, style='italic')

# 5. Notification channels (far right)
notifications = [
    ('📧 Email', 8.5, 2.8),
    ('💬 Slack', 8.5, 2.2),
    ('📟 PagerDuty', 8.5, 1.6)
]

for label, x, y in notifications:
    ax.text(x, y, label, ha='center', va='center',
            fontsize=9, color='#fbbf24', bbox=dict(boxstyle='round,pad=0.3',
                                                     facecolor='#2d2416',
                                                     edgecolor='#fbbf24',
                                                     linewidth=1))

# Arrows
# Targets → Prometheus (scrape)
for label, x, y in targets:
    arrow = FancyArrowPatch((x + 0.6, y), (prom_x - prom_width/2 - 0.1, prom_y + 0.55),
                           arrowstyle='->', mutation_scale=20, linewidth=2,
                           color=color_success, alpha=0.8)
    ax.add_patch(arrow)
    # Label "HTTP GET every 15s"
    if y == 7:
        ax.text(2.8, y - 0.3, 'HTTP GET\nevery 15s', ha='center', va='center',
                fontsize=8, color=color_success, style='italic')

# Internal flow: Retrieval → TSDB
arrow_internal1 = FancyArrowPatch((prom_x, prom_y + 0.05), (prom_x, prom_y - 0.15),
                                 arrowstyle='->', mutation_scale=15, linewidth=2,
                                 color=color_info, alpha=0.8)
ax.add_patch(arrow_internal1)

# Internal flow: TSDB → PromQL
arrow_internal2 = FancyArrowPatch((prom_x, prom_y - 0.65), (prom_x, prom_y - 0.85),
                                 arrowstyle='->', mutation_scale=15, linewidth=2,
                                 color=color_caution, alpha=0.8)
ax.add_patch(arrow_internal2)

# Prometheus → Grafana (query)
arrow_grafana = FancyArrowPatch((prom_x + prom_width/2 + 0.1, prom_y - 0.85),
                               (grafana_x - 0.8, grafana_y - 0.2),
                               arrowstyle='<->', mutation_scale=20, linewidth=2,
                               color=color_caution, alpha=0.8)
ax.add_patch(arrow_grafana)
ax.text(6.8, prom_y - 0.2, 'PromQL\nqueries', ha='center', va='center',
        fontsize=8, color=color_caution, style='italic')

# Prometheus → Alertmanager
arrow_alert = FancyArrowPatch((prom_x + prom_width/2 + 0.1, prom_y - 1.2),
                             (alert_x - 0.8, alert_y),
                             arrowstyle='->', mutation_scale=20, linewidth=2,
                             color='#b91c1c', alpha=0.8)
ax.add_patch(arrow_alert)
ax.text(6.8, 4.2, 'Alert\nrules', ha='center', va='center',
        fontsize=8, color='#b91c1c', style='italic')

# Alertmanager → Notifications
for label, x, y in notifications:
    arrow_notif = FancyArrowPatch((alert_x + 0.8, alert_y - 0.4), (x - 0.5, y),
                                 arrowstyle='->', mutation_scale=15, linewidth=1.5,
                                 color='#fbbf24', alpha=0.8)
    ax.add_patch(arrow_notif)

# Key annotations
annotations = [
    ('Pull-based scraping', 1.5, 2.8, color_success),
    ('Local TSDB storage\n15d–30d retention', prom_x, 2.2, color_caution),
    ('HTTP API on :9090', prom_x + 1.8, 8.5, color_info)
]

for text, x, y, color in annotations:
    ax.text(x, y, text, ha='center', va='center',
            fontsize=8, color=color, style='italic',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#1a1a2e',
                     edgecolor=color, linewidth=1, alpha=0.8))

# Footer
ax.text(5, 0.5, 'Prometheus scrapes metrics from /metrics endpoints → stores in TSDB → exposes PromQL for querying',
        ha='center', va='center', fontsize=10, color='#9ca3af', style='italic')

plt.tight_layout()
plt.savefig('../img/prometheus_architecture.png', dpi=150, facecolor='#1a1a2e', bbox_inches='tight')
print("✅ Generated: ../img/prometheus_architecture.png")
