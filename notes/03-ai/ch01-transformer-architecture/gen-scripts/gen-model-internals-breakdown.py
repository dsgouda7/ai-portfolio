"""Generate model-internals-breakdown.png — parameter distribution and VRAM usage.

Shows:
  1. Pie chart: Where the 7B parameters live (embeddings, attention, FFN, output)
  2. Bar chart: VRAM usage breakdown (weights, activations, KV cache)
  3. Precision comparison table (fp16, int8, int4)
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

plt.xkcd(scale=0.25, length=80, randomness=0.8)
plt.rcParams["path.effects"] = []

fig = plt.figure(figsize=(18, 10), facecolor="white")
fig.suptitle("LLaMA 7B — Parameter Distribution & VRAM Usage",
             fontsize=20, fontweight="bold", y=0.97)

# Color palette
BLUE = "#2E86C1"
ORANGE = "#E67E22"
GREEN = "#27AE60"
RED = "#E74C3C"
PURPLE = "#8E44AD"
YELLOW = "#F39C12"
DARK = "#2C3E50"
GREY = "#95A5A6"
LIGHTGREY = "#ECF0F1"

# Create grid
gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.30,
                      left=0.08, right=0.95, top=0.88, bottom=0.08)
ax_pie = fig.add_subplot(gs[0, 0])
ax_vram = fig.add_subplot(gs[0, 1])
ax_precision = fig.add_subplot(gs[1, :])

# ============= PIE CHART: PARAMETER BREAKDOWN =============
ax_pie.set_title("Where the 6.7B Parameters Live", fontsize=14,
                 fontweight="bold", color=DARK, pad=15)

# Data (LLaMA 7B breakdown)
# Token embeddings: 131M (2%)
# 32 layers × 158M each = 5,050M (76%)
# Output LM head: 131M (2%)
# Within each layer:
#   - Attention (QKV + O): 67M (1%)
#   - FFN: 90M (1.4%)

components = ["Token\nEmbeddings\n(131M)",
              "Attention\n(32 layers)\n(2,144M)",
              "Feed-Forward\nNetworks\n(32 layers)\n(2,883M)",
              "Output\nLM Head\n(131M)",
              "Layer Norms\n& Other\n(20M)"]

sizes = [131, 2144, 2883, 131, 20]  # Millions of parameters
colors = [BLUE, GREEN, ORANGE, RED, PURPLE]
explode = (0.05, 0.05, 0.1, 0.05, 0)  # Explode FFN slice

wedges, texts, autotexts = ax_pie.pie(sizes, labels=components, colors=colors,
                                       autopct=lambda pct: f"{pct:.1f}%\n({pct/100*6.7:.1f}B)",
                                       explode=explode, startangle=90,
                                       textprops=dict(fontsize=10, color="white",
                                                     fontweight="bold"))

for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontsize(9)
    autotext.set_fontweight('bold')

# Add center annotation
ax_pie.text(0, 0, "Total:\n6.7B\nparams", ha="center", va="center",
           fontsize=12, fontweight="bold", color=DARK,
           bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                    edgecolor=DARK, lw=2))

# ============= BAR CHART: VRAM USAGE =============
ax_vram.set_title("VRAM Usage During Inference\n(fp16, seq_len=2048, batch=1)",
                 fontsize=14, fontweight="bold", color=DARK, pad=15)

categories = ["Model\nWeights", "Activations\n(temp)", "KV Cache\n(per request)"]
vram_gb = [13.4, 1.0, 1.0]
bar_colors = [BLUE, ORANGE, PURPLE]

bars = ax_vram.bar(categories, vram_gb, color=bar_colors, edgecolor="white",
                   lw=3, alpha=0.8, width=0.6)

# Add values on bars
for bar, val in zip(bars, vram_gb):
    height = bar.get_height()
    ax_vram.text(bar.get_x() + bar.get_width()/2., height + 0.3,
                f'{val:.1f} GB', ha="center", va="bottom",
                fontsize=12, fontweight="bold", color=DARK)

ax_vram.set_ylabel("VRAM (GB)", fontsize=12, fontweight="bold", color=DARK)
ax_vram.set_ylim(0, 16)
ax_vram.grid(axis='y', alpha=0.3, linestyle='--')
ax_vram.set_axisbelow(True)

# Add total annotation
total_vram = sum(vram_gb)
ax_vram.text(1, 14.5, f"Total: {total_vram:.1f} GB\n(fits RTX 4090: 24 GB)",
            ha="center", fontsize=11, color=DARK, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.5", facecolor=YELLOW,
                     edgecolor=DARK, lw=2, alpha=0.7))

# ============= TABLE: PRECISION COMPARISON =============
ax_precision.set_title("Precision Tradeoffs — Same Model, Different VRAM Footprints",
                      fontsize=14, fontweight="bold", color=DARK, pad=15)
ax_precision.axis("off")

# Table data
table_data = [
    ["Precision", "Bytes/Param", "7B Model\nVRAM", "70B Model\nVRAM",
     "Quality Impact", "Inference Speed"],
    ["fp32\n(full)", "4", "26.8 GB", "280 GB", "Baseline\n(overkill)", "1.0×"],
    ["fp16 / bf16\n(half)", "2", "13.4 GB", "140 GB", "<0.1% loss\n(standard)", "1.0×"],
    ["int8\n(quantized)", "1", "6.7 GB", "70 GB", "<1% loss\n(production)", "1.5-2×"],
    ["int4\n(aggressive)", "0.5", "3.4 GB", "35 GB", "2-5% loss\n(reasoning tasks)", "2-4×"],
]

# Color rows
row_colors = [DARK, LIGHTGREY, "#D5F4E6", "#FFF9E6", "#FFE5E5"]
text_colors = ["white", DARK, DARK, DARK, DARK]

# Draw table
table = ax_precision.table(cellText=table_data, cellLoc='center', loc='center',
                           colWidths=[0.15, 0.12, 0.12, 0.12, 0.20, 0.15])

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2.5)

# Style cells
for i, row in enumerate(table_data):
    for j, cell_text in enumerate(row):
        cell = table[(i, j)]
        cell.set_facecolor(row_colors[i])
        cell.set_text_props(color=text_colors[i], fontweight='bold' if i == 0 else 'normal')
        cell.set_edgecolor('white')
        cell.set_linewidth(2)

# Add key insights box
ax_precision.text(0.5, -0.35,
                 "KEY INSIGHTS\n\n"
                 "• FFN dominates parameters (43% of total)\n"
                 "• KV cache scales linearly with batch size (1 GB per request)\n"
                 "• int8 quantization: 2× memory savings, <1% quality loss → production sweet spot\n"
                 "• int4 quantization: 4× memory savings, enables 70B on consumer GPUs (4× RTX 4090)",
                 ha="center", fontsize=10, color=DARK,
                 bbox=dict(boxstyle="round,pad=0.8", facecolor=LIGHTGREY,
                          edgecolor=DARK, lw=2.5),
                 transform=ax_precision.transAxes)

plt.tight_layout()
output_path = Path(__file__).parent.parent / "img" / "model-internals-breakdown.png"
output_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output_path, dpi=150, facecolor="white", bbox_inches="tight")
print(f"✓ Generated: {output_path}")
plt.close()
