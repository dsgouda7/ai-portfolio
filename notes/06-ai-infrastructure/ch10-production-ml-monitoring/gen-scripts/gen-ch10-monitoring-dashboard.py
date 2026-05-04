"""
gen_ch10_monitoring_dashboard.py
Generates: ../img/ch10_monitoring_dashboard.png

Mockup of an Evidently AI monitoring dashboard showing:
- Data drift metrics over time
- Model performance trends
- Alert indicators
- Feature importance changes

Visual representation of a production monitoring UI.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
import matplotlib.patches as mpatches

OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "img", "ch10_monitoring_dashboard.png")

# Ensure output directory exists
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

# Create figure with dark background (dashboard theme)
fig = plt.figure(figsize=(16, 9), facecolor="#0f1117")

# Create main grid
gs = fig.add_gridspec(4, 3, hspace=0.35, wspace=0.25, 
                     left=0.05, right=0.95, top=0.90, bottom=0.08)

# ============ HEADER ============
header_ax = fig.add_axes([0.05, 0.92, 0.9, 0.06])
header_ax.set_facecolor("#1a1d29")
header_ax.axis("off")

header_ax.text(0.02, 0.5, "Evidently ML Monitoring Dashboard", 
               ha="left", va="center", fontsize=22, fontweight="bold", 
               color="white", transform=header_ax.transAxes)

header_ax.text(0.98, 0.5, "Model: fraud-detector-v2.3  |  Last updated: 2 min ago", 
               ha="right", va="center", fontsize=11, color="#94a3b8",
               transform=header_ax.transAxes, family="monospace")

# Status indicator
status_circle = plt.Circle((0.92, 0.5), 0.015, color="#10b981", 
                          transform=header_ax.transAxes)
header_ax.add_patch(status_circle)

# ============ PANEL 1: DATA DRIFT SCORE ============
ax1 = fig.add_subplot(gs[0, 0])
ax1.set_facecolor("#1a1d29")
ax1.set_title("Data Drift Score (PSI)", fontsize=12, fontweight="bold", 
              color="white", pad=10)

# Time series with drift threshold
days = np.arange(30)
drift_score = 0.05 + 0.02 * np.sin(days / 5) + np.random.normal(0, 0.01, 30)
drift_score[25:] += 0.08  # Simulate drift spike

ax1.plot(days, drift_score, color="#3b82f6", linewidth=2)
ax1.axhline(y=0.10, color="#f59e0b", linestyle="--", linewidth=1.5, label="Warning")
ax1.axhline(y=0.15, color="#ef4444", linestyle="--", linewidth=1.5, label="Critical")
ax1.fill_between(days[25:], 0, drift_score[25:], alpha=0.3, color="#ef4444")

ax1.set_xlabel("Days", fontsize=10, color="#94a3b8")
ax1.set_ylabel("PSI", fontsize=10, color="#94a3b8")
ax1.tick_params(colors="#94a3b8", labelsize=8)
ax1.legend(fontsize=8, loc="upper left", facecolor="#1a1d29", edgecolor="#94a3b8")
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
ax1.spines["bottom"].set_color("#94a3b8")
ax1.spines["left"].set_color("#94a3b8")

# Alert annotation
ax1.text(27, 0.14, "⚠ ALERT", ha="center", va="bottom", fontsize=9,
        color="#ef4444", fontweight="bold", 
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#ef4444", alpha=0.2))

# ============ PANEL 2: MODEL ACCURACY ============
ax2 = fig.add_subplot(gs[0, 1])
ax2.set_facecolor("#1a1d29")
ax2.set_title("Model Accuracy (Rolling 7-day)", fontsize=12, fontweight="bold", 
              color="white", pad=10)

accuracy = 0.92 + 0.02 * np.sin(days / 4) + np.random.normal(0, 0.005, 30)
accuracy[25:] -= 0.05  # Performance degradation

ax2.plot(days, accuracy, color="#10b981", linewidth=2)
ax2.axhline(y=0.90, color="#f59e0b", linestyle="--", linewidth=1.5, alpha=0.6)

ax2.set_xlabel("Days", fontsize=10, color="#94a3b8")
ax2.set_ylabel("Accuracy", fontsize=10, color="#94a3b8")
ax2.set_ylim(0.84, 0.96)
ax2.tick_params(colors="#94a3b8", labelsize=8)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
ax2.spines["bottom"].set_color("#94a3b8")
ax2.spines["left"].set_color("#94a3b8")

# Highlight degradation zone
ax2.fill_between(days[25:], 0.84, accuracy[25:], alpha=0.2, color="#f59e0b")

# ============ PANEL 3: PREDICTION VOLUME ============
ax3 = fig.add_subplot(gs[0, 2])
ax3.set_facecolor("#1a1d29")
ax3.set_title("Predictions per Hour", fontsize=12, fontweight="bold", 
              color="white", pad=10)

hours = np.arange(24)
volume = 5000 + 2000 * np.sin(hours / 3) + np.random.normal(0, 300, 24)

bars = ax3.bar(hours, volume, color="#8b5cf6", alpha=0.8)
# Highlight current hour
bars[-1].set_color("#a78bfa")
bars[-1].set_alpha(1.0)

ax3.set_xlabel("Hour (last 24h)", fontsize=10, color="#94a3b8")
ax3.set_ylabel("Count", fontsize=10, color="#94a3b8")
ax3.tick_params(colors="#94a3b8", labelsize=8)
ax3.spines["top"].set_visible(False)
ax3.spines["right"].set_visible(False)
ax3.spines["bottom"].set_color("#94a3b8")
ax3.spines["left"].set_color("#94a3b8")

# ============ PANEL 4: FEATURE DRIFT HEATMAP ============
ax4 = fig.add_subplot(gs[1:3, 0])
ax4.set_facecolor("#1a1d29")
ax4.set_title("Feature Drift Heatmap (PSI)", fontsize=12, fontweight="bold", 
              color="white", pad=10)

features = ["transaction_amt", "user_age", "device_type", 
           "time_of_day", "location", "merchant_cat"]
drift_matrix = np.random.rand(6, 10) * 0.2
# Add some drift spikes
drift_matrix[1, 7:] = 0.25
drift_matrix[4, 5:9] = 0.22

im = ax4.imshow(drift_matrix, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=0.3)
ax4.set_yticks(np.arange(len(features)))
ax4.set_yticklabels(features, fontsize=9, color="#94a3b8")
ax4.set_xlabel("Time Window (days)", fontsize=10, color="#94a3b8")
ax4.tick_params(colors="#94a3b8", labelsize=8)

# Colorbar
cbar = plt.colorbar(im, ax=ax4, pad=0.02)
cbar.set_label("PSI", fontsize=9, color="#94a3b8")
cbar.ax.tick_params(labelsize=8, colors="#94a3b8")

# ============ PANEL 5: PREDICTION DISTRIBUTION ============
ax5 = fig.add_subplot(gs[1:3, 1])
ax5.set_facecolor("#1a1d29")
ax5.set_title("Prediction Distribution: Training vs Production", fontsize=12, 
              fontweight="bold", color="white", pad=10)

train_pred = np.random.beta(2, 10, 1000)
prod_pred = np.random.beta(2, 8, 1000)  # Slight shift

ax5.hist(train_pred, bins=30, alpha=0.6, color="#3b82f6", label="Training", density=True)
ax5.hist(prod_pred, bins=30, alpha=0.6, color="#f59e0b", label="Production (last 7d)", density=True)

ax5.set_xlabel("Fraud Probability", fontsize=10, color="#94a3b8")
ax5.set_ylabel("Density", fontsize=10, color="#94a3b8")
ax5.legend(fontsize=9, facecolor="#1a1d29", edgecolor="#94a3b8")
ax5.tick_params(colors="#94a3b8", labelsize=8)
ax5.spines["top"].set_visible(False)
ax5.spines["right"].set_visible(False)
ax5.spines["bottom"].set_color("#94a3b8")
ax5.spines["left"].set_color("#94a3b8")

# ============ PANEL 6: ALERT LOG ============
ax6 = fig.add_subplot(gs[1:3, 2])
ax6.set_facecolor("#1a1d29")
ax6.set_title("Recent Alerts", fontsize=12, fontweight="bold", 
              color="white", pad=10)
ax6.axis("off")

alerts = [
    ("2h ago", "🔴 Data drift detected: user_age", "#ef4444"),
    ("5h ago", "🟡 Latency p99 > 200ms", "#f59e0b"),
    ("12h ago", "🔴 Accuracy drop: 0.92 → 0.87", "#ef4444"),
    ("1d ago", "🟢 Model retrain completed", "#10b981"),
    ("2d ago", "🟡 Prediction volume spike", "#f59e0b"),
]

for i, (time, message, color) in enumerate(alerts):
    y_pos = 0.85 - i * 0.16
    
    # Alert box
    ax6.add_patch(Rectangle((0.05, y_pos - 0.06), 0.9, 0.12,
                            facecolor="#262b3d", edgecolor=color, 
                            linewidth=1.5, transform=ax6.transAxes))
    
    ax6.text(0.08, y_pos + 0.02, time, ha="left", va="center",
            fontsize=8, color="#94a3b8", transform=ax6.transAxes,
            family="monospace")
    
    ax6.text(0.08, y_pos - 0.03, message, ha="left", va="center",
            fontsize=9, color="white", transform=ax6.transAxes)

# ============ PANEL 7: SUMMARY STATS ============
ax7 = fig.add_subplot(gs[3, :])
ax7.set_facecolor("#1a1d29")
ax7.axis("off")

stats = [
    ("Total Predictions", "3.2M", "#3b82f6"),
    ("Avg Latency", "87ms", "#10b981"),
    ("Error Rate", "0.03%", "#10b981"),
    ("Features Drifting", "2/12", "#f59e0b"),
    ("Model Version", "v2.3", "#8b5cf6"),
    ("Uptime", "99.97%", "#10b981"),
]

stat_width = 1.0 / len(stats)
for i, (label, value, color) in enumerate(stats):
    x_pos = i * stat_width + stat_width / 2
    
    # Stat box
    ax7.add_patch(Rectangle((i * stat_width + 0.01, 0.2), stat_width - 0.02, 0.6,
                            facecolor="#262b3d", edgecolor=color, 
                            linewidth=2, transform=ax7.transAxes))
    
    ax7.text(x_pos, 0.7, value, ha="center", va="center",
            fontsize=16, fontweight="bold", color=color, 
            transform=ax7.transAxes)
    
    ax7.text(x_pos, 0.35, label, ha="center", va="center",
            fontsize=9, color="#94a3b8", transform=ax7.transAxes)

# Save figure
plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"✓ Saved: {OUT_PATH}")
