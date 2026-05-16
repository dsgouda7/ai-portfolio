"""
Generate latency comparison diagram: Direct DB vs Feature Store.

Shows the dramatic latency improvement from using a feature store with Redis.
"""

import matplotlib.pyplot as plt
import numpy as np

plt.style.use('dark_background')
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6), facecolor='#1a1a2e')

colors = {
    'db': '#b91c1c',
    'feast': '#15803d',
    'bg': '#1a1a2e'
}

# === LEFT: Bar chart comparison ===
methods = ['Direct DB\n(PostgreSQL)', 'Feature Store\n(Redis)']
p50_latencies = [320, 6]
p95_latencies = [380, 8]
p99_latencies = [450, 12]

x = np.arange(len(methods))
width = 0.25

bars1 = ax1.bar(x - width, p50_latencies, width, label='p50', color='#1d4ed8', alpha=0.9)
bars2 = ax1.bar(x, p95_latencies, width, label='p95', color='#b45309', alpha=0.9)
bars3 = ax1.bar(x + width, p99_latencies, width, label='p99', color='#b91c1c', alpha=0.9)

ax1.set_ylabel('Latency (ms)', fontsize=12, color='white', fontweight='bold')
ax1.set_title('Feature Lookup Latency', fontsize=14, color='white', fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(methods, fontsize=11, color='white')
ax1.legend(loc='upper left', facecolor=colors['bg'], edgecolor='white', labelcolor='white')
ax1.tick_params(colors='white')
ax1.set_facecolor(colors['bg'])
ax1.grid(axis='y', alpha=0.3, color='white')
ax1.axhline(y=10, color='yellow', linestyle='--', linewidth=2, alpha=0.7, label='10ms target')

# Add value labels on bars
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}ms',
                ha='center', va='bottom', fontsize=9, color='white', fontweight='bold')

# Add improvement annotation
ax1.annotate('', xy=(1, 8), xytext=(0, 380),
            arrowprops=dict(arrowstyle='->', color='#15803d', lw=3, alpha=0.7))
ax1.text(0.5, 200, '97% faster!', ha='center', fontsize=12, color='#15803d',
        bbox=dict(boxstyle='round,pad=0.5', facecolor=colors['bg'], edgecolor='#15803d'),
        fontweight='bold')

# === MIDDLE: Latency breakdown ===
categories = ['Feature\nQuery', 'Model\nInference', 'Response\nSerialization', 'Total']

# Before (Direct DB)
before_values = [380, 950, 50, 1380]
# After (Feature Store)
after_values = [8, 950, 50, 1008]

x_pos = np.arange(len(categories))
width = 0.35

bars_before = ax2.bar(x_pos - width/2, before_values, width, label='Before (Direct DB)',
                      color=colors['db'], alpha=0.9)
bars_after = ax2.bar(x_pos + width/2, after_values, width, label='After (Feature Store)',
                     color=colors['feast'], alpha=0.9)

ax2.set_ylabel('Latency (ms)', fontsize=12, color='white', fontweight='bold')
ax2.set_title('End-to-End Inference Latency', fontsize=14, color='white', fontweight='bold')
ax2.set_xticks(x_pos)
ax2.set_xticklabels(categories, fontsize=10, color='white')
ax2.legend(loc='upper left', facecolor=colors['bg'], edgecolor='white', labelcolor='white')
ax2.tick_params(colors='white')
ax2.set_facecolor(colors['bg'])
ax2.grid(axis='y', alpha=0.3, color='white')
ax2.axhline(y=2000, color='red', linestyle='--', linewidth=2, alpha=0.7)
ax2.text(3.5, 2050, '2s SLA', fontsize=9, color='red', ha='right')

# Add value labels
for bars in [bars_before, bars_after]:
    for bar in bars:
        height = bar.get_height()
        if height > 50:  # Only show non-trivial values
            ax2.text(bar.get_x() + bar.get_width()/2., height/2,
                    f'{int(height)}ms',
                    ha='center', va='center', fontsize=9, color='white', fontweight='bold')

# === RIGHT: Distribution plot ===
np.random.seed(42)

# Generate latency distributions
db_latencies = np.random.normal(380, 40, 1000)
feast_latencies = np.random.normal(8, 2, 1000)

# Violin plot
parts = ax3.violinplot([db_latencies, feast_latencies],
                       positions=[1, 2],
                       widths=0.6,
                       showmeans=True,
                       showextrema=True)

# Color the violin plots
for i, pc in enumerate(parts['bodies']):
    pc.set_facecolor(colors['db'] if i == 0 else colors['feast'])
    pc.set_alpha(0.7)
    pc.set_edgecolor('white')
    pc.set_linewidth(2)

# Customize mean, min, max lines
for partname in ('cbars', 'cmins', 'cmaxes', 'cmeans'):
    vp = parts[partname]
    vp.set_edgecolor('white')
    vp.set_linewidth(2)

ax3.set_ylabel('Latency (ms)', fontsize=12, color='white', fontweight='bold')
ax3.set_title('Latency Distribution', fontsize=14, color='white', fontweight='bold')
ax3.set_xticks([1, 2])
ax3.set_xticklabels(['Direct DB', 'Feature Store'], fontsize=11, color='white')
ax3.tick_params(colors='white')
ax3.set_facecolor(colors['bg'])
ax3.grid(axis='y', alpha=0.3, color='white')
ax3.axhline(y=10, color='yellow', linestyle='--', linewidth=2, alpha=0.7)

# Add percentile annotations
db_p95 = np.percentile(db_latencies, 95)
feast_p95 = np.percentile(feast_latencies, 95)

ax3.text(1.5, db_p95, f'p95: {db_p95:.0f}ms', fontsize=9, color='white',
        bbox=dict(boxstyle='round,pad=0.3', facecolor=colors['db'], edgecolor='white'))
ax3.text(2.5, feast_p95, f'p95: {feast_p95:.0f}ms', fontsize=9, color='white',
        bbox=dict(boxstyle='round,pad=0.3', facecolor=colors['feast'], edgecolor='white'))

# Overall title and summary
fig.suptitle('Feature Store Latency Impact', fontsize=16, color='white', fontweight='bold', y=0.98)

# Add summary box at bottom
summary_text = (
    '💡 Before: 4 PostgreSQL replicas, 380ms p95 → 2.8s total (violated 2s SLA)\n'
    '✅ After: 1 Redis instance, 8ms p95 → 1.4s total (30% better than target)\n'
    '💰 Cost: $800/month → $120/month (85% reduction)'
)
fig.text(0.5, 0.02, summary_text, ha='center', fontsize=11, color='white',
        bbox=dict(boxstyle='round,pad=0.8', facecolor='#15803d', edgecolor='white', linewidth=2),
        fontweight='bold')

plt.tight_layout(rect=[0, 0.08, 1, 0.96])
plt.savefig('../img/ch08-latency-comparison.png', dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
print("✓ Generated: ch08-latency-comparison.png")
plt.close()
