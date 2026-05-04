#!/usr/bin/env python3
"""
Generate Metrics Types Diagram

Shows Counter, Gauge, Histogram, and Summary with visual examples
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np
import matplotlib
matplotlib.use('Agg')

# Set dark background style
plt.style.use('dark_background')
fig, axes = plt.subplots(2, 2, figsize=(14, 10), facecolor='#1a1a2e')
fig.suptitle('Prometheus Metrics Types', fontsize=20, fontweight='bold', color='#ffffff', y=0.98)

# Color palette
color_success = '#15803d'
color_caution = '#b45309'
color_info = '#1d4ed8'
color_danger = '#b91c1c'
color_text = '#ffffff'

# 1. COUNTER (top-left)
ax1 = axes[0, 0]
ax1.set_facecolor('#1a1a2e')
ax1.set_title('Counter — Monotonic (only goes up)', fontsize=14, fontweight='bold', 
              color=color_success, pad=15)

# Generate counter data
t = np.linspace(0, 60, 100)
counter_values = np.cumsum(np.random.poisson(2, 100))  # Monotonic increase

ax1.plot(t, counter_values, color=color_success, linewidth=2.5)
ax1.fill_between(t, counter_values, alpha=0.2, color=color_success)
ax1.set_xlabel('Time (seconds)', fontsize=10, color=color_text)
ax1.set_ylabel('Request Count', fontsize=10, color=color_text)
ax1.grid(True, alpha=0.2, linestyle='--')
ax1.tick_params(colors=color_text)

# Annotations
ax1.annotate('Process restart\n(counter resets to 0)', 
            xy=(30, 20), xytext=(40, 80),
            arrowprops=dict(arrowstyle='->', color=color_danger, lw=2),
            fontsize=9, color=color_danger, ha='left')
ax1.axvline(x=30, color=color_danger, linestyle='--', linewidth=1.5, alpha=0.7)

ax1.text(0.05, 0.95, '📊 Example: http_requests_total\n\n'
                      '✓ Always increases\n'
                      '✓ Use rate() to get req/sec\n'
                      '✓ Resets on process restart',
         transform=ax1.transAxes, fontsize=9, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='#2d3748', alpha=0.8, edgecolor=color_success),
         color=color_text)

# 2. GAUGE (top-right)
ax2 = axes[0, 1]
ax2.set_facecolor('#1a1a2e')
ax2.set_title('Gauge — Instant Snapshot (up or down)', fontsize=14, fontweight='bold',
              color=color_info, pad=15)

# Generate gauge data (CPU usage)
t = np.linspace(0, 60, 100)
gauge_values = 50 + 30 * np.sin(t / 10) + np.random.normal(0, 5, 100)
gauge_values = np.clip(gauge_values, 0, 100)

ax2.plot(t, gauge_values, color=color_info, linewidth=2.5)
ax2.fill_between(t, gauge_values, alpha=0.2, color=color_info)
ax2.axhline(y=80, color=color_danger, linestyle='--', linewidth=1.5, alpha=0.7, label='Alert threshold')
ax2.set_xlabel('Time (seconds)', fontsize=10, color=color_text)
ax2.set_ylabel('CPU Usage (%)', fontsize=10, color=color_text)
ax2.set_ylim(0, 100)
ax2.grid(True, alpha=0.2, linestyle='--')
ax2.tick_params(colors=color_text)
ax2.legend(loc='upper right', framealpha=0.8, facecolor='#2d3748', edgecolor=color_danger)

ax2.text(0.05, 0.95, '📊 Example: cpu_usage_percent\n\n'
                      '✓ Can increase or decrease\n'
                      '✓ Current value matters\n'
                      '✓ No rate() needed',
         transform=ax2.transAxes, fontsize=9, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='#2d3748', alpha=0.8, edgecolor=color_info),
         color=color_text)

# 3. HISTOGRAM (bottom-left)
ax3 = axes[1, 0]
ax3.set_facecolor('#1a1a2e')
ax3.set_title('Histogram — Bucketed Distribution', fontsize=14, fontweight='bold',
              color=color_caution, pad=15)

# Generate histogram data (latency distribution)
latencies = np.random.lognormal(mean=np.log(0.2), sigma=0.6, size=1000)
latencies = np.clip(latencies, 0, 5)

buckets = [0.1, 0.5, 1.0, 2.0, 5.0]
bucket_labels = ['≤0.1s', '≤0.5s', '≤1.0s', '≤2.0s', '≤5.0s']

hist_counts, _ = np.histogram(latencies, bins=[0] + buckets)
cumulative_counts = np.cumsum(hist_counts)

bars = ax3.bar(range(len(bucket_labels)), cumulative_counts, 
               color=color_caution, alpha=0.7, edgecolor=color_caution, linewidth=2)

# Highlight p95
p95_idx = np.searchsorted(cumulative_counts, 0.95 * len(latencies))
if p95_idx < len(bars):
    bars[p95_idx].set_color(color_danger)
    bars[p95_idx].set_edgecolor(color_danger)
    ax3.annotate(f'p95 ≈ {buckets[p95_idx]}s', 
                xy=(p95_idx, cumulative_counts[p95_idx]), 
                xytext=(p95_idx + 0.5, cumulative_counts[p95_idx] + 100),
                arrowprops=dict(arrowstyle='->', color=color_danger, lw=2),
                fontsize=10, color=color_danger, fontweight='bold')

ax3.set_xticks(range(len(bucket_labels)))
ax3.set_xticklabels(bucket_labels, fontsize=9)
ax3.set_xlabel('Latency Bucket', fontsize=10, color=color_text)
ax3.set_ylabel('Cumulative Count', fontsize=10, color=color_text)
ax3.grid(True, alpha=0.2, linestyle='--', axis='y')
ax3.tick_params(colors=color_text)

ax3.text(0.05, 0.95, '📊 Example: http_request_duration_seconds\n\n'
                      '✓ Pre-defined buckets\n'
                      '✓ Compute percentiles with histogram_quantile()\n'
                      '✓ Trade precision for efficiency',
         transform=ax3.transAxes, fontsize=9, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='#2d3748', alpha=0.8, edgecolor=color_caution),
         color=color_text)

# 4. SUMMARY (bottom-right)
ax4 = axes[1, 1]
ax4.set_facecolor('#1a1a2e')
ax4.set_title('Summary — Pre-computed Quantiles', fontsize=14, fontweight='bold',
              color='#8b5cf6', pad=15)

# Summary pre-computes percentiles client-side
quantiles = ['p50', 'p90', 'p95', 'p99']
values = [0.15, 0.28, 0.42, 0.85]

bars = ax4.barh(quantiles, values, color='#8b5cf6', alpha=0.7, edgecolor='#8b5cf6', linewidth=2)

# Add value labels
for i, (q, v) in enumerate(zip(quantiles, values)):
    ax4.text(v + 0.03, i, f'{v:.2f}s', va='center', fontsize=10, color=color_text, fontweight='bold')

ax4.set_xlabel('Latency (seconds)', fontsize=10, color=color_text)
ax4.set_ylabel('Quantile', fontsize=10, color=color_text)
ax4.set_xlim(0, max(values) * 1.2)
ax4.grid(True, alpha=0.2, linestyle='--', axis='x')
ax4.tick_params(colors=color_text)

ax4.text(0.05, 0.95, '📊 Example: rpc_duration_seconds_summary\n\n'
                      '✓ Client computes percentiles\n'
                      '✓ No bucketing overhead\n'
                      '✓ Cannot aggregate across instances\n'
                      '⚠️ Less common (use Histogram instead)',
         transform=ax4.transAxes, fontsize=9, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='#2d3748', alpha=0.8, edgecolor='#8b5cf6'),
         color=color_text)

# Comparison table at bottom
comparison_text = '''
┌─────────────┬──────────────────────────────┬─────────────────────────────────┬────────────────────────────┐
│ Type        │ Use Case                     │ Aggregation                     │ PromQL Example             │
├─────────────┼──────────────────────────────┼─────────────────────────────────┼────────────────────────────┤
│ Counter     │ Requests, errors, bytes sent │ rate(), increase()              │ rate(requests_total[5m])   │
│ Gauge       │ CPU, memory, queue size      │ avg(), min(), max()             │ avg(cpu_percent)           │
│ Histogram   │ Latency, response size       │ histogram_quantile()            │ histogram_quantile(0.95,…) │
│ Summary     │ Latency (rare, pre-computed) │ Cannot aggregate across labels  │ rpc_duration{quantile=...} │
└─────────────┴──────────────────────────────┴─────────────────────────────────┴────────────────────────────┘
'''

fig.text(0.5, 0.02, comparison_text, ha='center', va='bottom', 
         fontsize=8, family='monospace', color='#9ca3af',
         bbox=dict(boxstyle='round', facecolor='#1a1a2e', alpha=0.9, edgecolor='#4b5563', linewidth=1))

plt.tight_layout(rect=[0, 0.08, 1, 0.96])
plt.savefig('../img/metrics_types.png', dpi=150, facecolor='#1a1a2e', bbox_inches='tight')
print("✅ Generated: ../img/metrics_types.png")
