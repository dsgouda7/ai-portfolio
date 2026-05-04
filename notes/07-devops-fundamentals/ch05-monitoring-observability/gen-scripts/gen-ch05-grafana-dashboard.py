#!/usr/bin/env python3
"""
Generate Grafana Dashboard Visualization

Shows a sample dashboard layout with time-series graphs, stat panels, and heatmaps
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np
import matplotlib
matplotlib.use('Agg')

# Set dark background style
plt.style.use('dark_background')
fig = plt.figure(figsize=(16, 10), facecolor='#0b0c0e')

# Color palette
color_success = '#15803d'
color_caution = '#b45309'
color_danger = '#b91c1c'
color_info = '#1d4ed8'
color_text = '#ffffff'
color_panel_bg = '#1a1d23'
color_border = '#2d3748'

# Main title
fig.text(0.5, 0.97, 'Flask App Monitoring Dashboard', 
         fontsize=22, fontweight='bold', ha='center', va='top', color=color_text)

# Timestamp
fig.text(0.5, 0.94, 'Last refresh: 2026-04-26 14:32:15 | Auto-refresh: 10s', 
         fontsize=10, ha='center', va='top', color='#9ca3af', style='italic')

# Create grid layout: 3 rows, 4 columns
# Row 1: 4 stat panels (small)
# Row 2: 2 time-series graphs (large)
# Row 3: 1 heatmap (full width)

# Helper function to create panel
def create_panel(ax, title, panel_type='graph'):
    ax.set_facecolor(color_panel_bg)
    ax.spines['top'].set_color(color_border)
    ax.spines['bottom'].set_color(color_border)
    ax.spines['left'].set_color(color_border)
    ax.spines['right'].set_color(color_border)
    ax.spines['top'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['right'].set_linewidth(1.5)
    
    # Panel title
    ax.text(0.05, 0.95, title, transform=ax.transAxes, 
            fontsize=11, fontweight='bold', va='top', color=color_text)

# Row 1: Stat Panels
stat_panels = [
    (0.05, 0.68, 0.20, 0.20, 'Total Requests (1h)', '12,847', color_info),
    (0.28, 0.68, 0.20, 0.20, 'Active Connections', '23', color_success),
    (0.51, 0.68, 0.20, 0.20, 'Error Rate', '3.2%', color_caution),
    (0.74, 0.68, 0.20, 0.20, 'p95 Latency', '285ms', color_success),
]

for x, y, w, h, title, value, color in stat_panels:
    ax = fig.add_axes([x, y, w, h])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # Panel background
    panel = Rectangle((0, 0), 1, 1, facecolor=color_panel_bg, 
                     edgecolor=color_border, linewidth=2)
    ax.add_patch(panel)
    
    # Title
    ax.text(0.5, 0.85, title, ha='center', va='top', 
            fontsize=10, color='#9ca3af')
    
    # Value (large)
    ax.text(0.5, 0.45, value, ha='center', va='center',
            fontsize=28, fontweight='bold', color=color)
    
    # Sparkline (mini trend)
    t = np.linspace(0, 1, 20)
    sparkline = 0.15 + 0.05 * np.sin(t * 8) + np.random.normal(0, 0.01, 20)
    ax.plot(np.linspace(0.1, 0.9, 20), sparkline, 
            color=color, linewidth=1.5, alpha=0.6)

# Row 2: Time-Series Graphs
# Left: Request Rate
ax_rate = fig.add_axes([0.05, 0.38, 0.43, 0.25])
create_panel(ax_rate, '📈 Request Rate (req/sec) — by route')

t = np.linspace(0, 60, 100)
rate_health = 3 + 0.5 * np.sin(t / 10) + np.random.normal(0, 0.2, 100)
rate_payment = 1.5 + 0.3 * np.sin(t / 8) + np.random.normal(0, 0.15, 100)
rate_refund = 0.5 + 0.1 * np.sin(t / 12) + np.random.normal(0, 0.05, 100)

ax_rate.plot(t, rate_health, label='/health', color=color_success, linewidth=2)
ax_rate.plot(t, rate_payment, label='/api/payment', color=color_info, linewidth=2)
ax_rate.plot(t, rate_refund, label='/api/refund', color=color_caution, linewidth=2)
ax_rate.fill_between(t, rate_health, alpha=0.1, color=color_success)
ax_rate.fill_between(t, rate_payment, alpha=0.1, color=color_info)
ax_rate.fill_between(t, rate_refund, alpha=0.1, color=color_caution)
ax_rate.set_xlabel('Time (seconds)', fontsize=9, color='#9ca3af')
ax_rate.set_ylabel('Requests/sec', fontsize=9, color='#9ca3af')
ax_rate.legend(loc='upper left', framealpha=0.8, facecolor=color_panel_bg, 
               edgecolor=color_border, fontsize=9)
ax_rate.grid(True, alpha=0.15, linestyle='--')
ax_rate.tick_params(colors='#9ca3af', labelsize=8)

# Right: Latency Percentiles
ax_latency = fig.add_axes([0.52, 0.38, 0.43, 0.25])
create_panel(ax_latency, '⏱️ Latency Percentiles (seconds)')

t = np.linspace(0, 60, 100)
latency_p50 = 0.15 + 0.02 * np.sin(t / 10) + np.random.normal(0, 0.01, 100)
latency_p95 = 0.28 + 0.05 * np.sin(t / 8) + np.random.normal(0, 0.02, 100)
latency_p99 = 0.45 + 0.1 * np.sin(t / 6) + np.random.normal(0, 0.03, 100)

ax_latency.plot(t, latency_p50, label='p50', color=color_success, linewidth=2)
ax_latency.plot(t, latency_p95, label='p95', color=color_caution, linewidth=2)
ax_latency.plot(t, latency_p99, label='p99', color=color_danger, linewidth=2)
ax_latency.fill_between(t, latency_p50, alpha=0.1, color=color_success)
ax_latency.fill_between(t, latency_p95, alpha=0.1, color=color_caution)
ax_latency.fill_between(t, latency_p99, alpha=0.1, color=color_danger)

# SLA threshold line
ax_latency.axhline(y=0.5, color='#ef4444', linestyle='--', linewidth=1.5, 
                   alpha=0.7, label='SLA threshold (500ms)')

ax_latency.set_xlabel('Time (seconds)', fontsize=9, color='#9ca3af')
ax_latency.set_ylabel('Latency (s)', fontsize=9, color='#9ca3af')
ax_latency.legend(loc='upper left', framealpha=0.8, facecolor=color_panel_bg,
                  edgecolor=color_border, fontsize=9)
ax_latency.grid(True, alpha=0.15, linestyle='--')
ax_latency.tick_params(colors='#9ca3af', labelsize=8)

# Row 3: Heatmap
ax_heatmap = fig.add_axes([0.05, 0.08, 0.90, 0.25])
create_panel(ax_heatmap, '🔥 Latency Heatmap — Request distribution by latency bucket')

# Generate heatmap data (time vs latency buckets)
time_buckets = 60
latency_buckets = ['0-100ms', '100-250ms', '250-500ms', '500ms-1s', '1s-2s', '2s+']
heatmap_data = np.random.poisson(20, (len(latency_buckets), time_buckets))

# Make higher latencies less frequent
for i in range(len(latency_buckets)):
    heatmap_data[i] = heatmap_data[i] * (0.9 ** i)

# Spike at time 30 (simulated load spike)
heatmap_data[:, 28:33] *= 2.5

im = ax_heatmap.imshow(heatmap_data, aspect='auto', cmap='YlOrRd', 
                       interpolation='nearest', origin='lower')

ax_heatmap.set_yticks(range(len(latency_buckets)))
ax_heatmap.set_yticklabels(latency_buckets, fontsize=9)
ax_heatmap.set_xlabel('Time (seconds)', fontsize=9, color='#9ca3af')
ax_heatmap.set_ylabel('Latency Bucket', fontsize=9, color='#9ca3af')
ax_heatmap.tick_params(colors='#9ca3af', labelsize=8)

# Colorbar
cbar = plt.colorbar(im, ax=ax_heatmap, orientation='vertical', pad=0.01)
cbar.set_label('Request Count', fontsize=9, color='#9ca3af')
cbar.ax.tick_params(colors='#9ca3af', labelsize=8)

# Annotations
ax_heatmap.annotate('Load spike\n(2x requests)', 
                   xy=(30, 3), xytext=(40, 4.5),
                   arrowprops=dict(arrowstyle='->', color=color_danger, lw=2),
                   fontsize=9, color=color_danger, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='#2d1a1a', 
                            edgecolor=color_danger, linewidth=1.5))

# Footer with PromQL queries
query_text = '''
PromQL Queries Used:
• Request Rate:  sum(rate(http_requests_total[5m])) by (route)
• Latency p95:   histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket[5m])) by (le, route))
• Error Rate:    sum(rate(http_requests_total{status=~"5.."}[5m])) / sum(rate(http_requests_total[5m])) * 100
'''

fig.text(0.05, 0.02, query_text, ha='left', va='bottom',
         fontsize=8, family='monospace', color='#9ca3af',
         bbox=dict(boxstyle='round', facecolor='#1a1d23', alpha=0.9, 
                  edgecolor=color_border, linewidth=1))

# Refresh indicator
fig.text(0.95, 0.02, '🔄 Auto-refresh: ON', ha='right', va='bottom',
         fontsize=9, color=color_success, fontweight='bold',
         bbox=dict(boxstyle='round', facecolor='#1a2d1a', alpha=0.9,
                  edgecolor=color_success, linewidth=1))

plt.savefig('../img/grafana_dashboard.png', dpi=150, facecolor='#0b0c0e', bbox_inches='tight')
print("✅ Generated: ../img/grafana_dashboard.png")
