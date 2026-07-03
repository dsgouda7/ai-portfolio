"""Generate kv-cache-inference.png — autoregressive generation with KV caching.

Shows:
  1. Prefill phase: process entire prompt, compute full attention matrix
  2. Decode step 1: generate token 1, compute attention only for new token vs cached
  3. Decode step 2: generate token 2, reuse growing cache
  4. Memory/compute savings visualization
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch, Circle
import numpy as np

plt.xkcd(scale=0.25, length=80, randomness=0.8)
plt.rcParams["path.effects"] = []

fig = plt.figure(figsize=(18, 12), facecolor="white")
fig.suptitle("Autoregressive Generation with KV Cache",
             fontsize=20, fontweight="bold", y=0.98)

# Color palette
BLUE = "#2E86C1"
ORANGE = "#E67E22"
GREEN = "#27AE60"
RED = "#E74C3C"
PURPLE = "#8E44AD"
DARK = "#2C3E50"
GREY = "#95A5A6"
LIGHTGREY = "#ECF0F1"
YELLOW = "#F39C12"

def draw_matrix(ax, x, y, rows, cols, scale=0.4, color=BLUE, label="", alpha=0.6):
    """Draw a matrix representation"""
    for i in range(rows):
        for j in range(cols):
            rect = Rectangle((x + j*scale, y - i*scale), scale*0.9, scale*0.9,
                           facecolor=color, edgecolor="white", lw=1.5, alpha=alpha)
            ax.add_patch(rect)
    if label:
        ax.text(x + (cols*scale)/2, y + 0.5, label,
               ha="center", fontsize=9, fontweight="bold", color=DARK)

def draw_token_box(ax, x, y, text, color, w=0.8, h=0.5):
    """Draw a token box"""
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                         facecolor=color, edgecolor="white", lw=2)
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center",
           color="white", fontsize=9, fontweight="bold")

# Create main axis
ax = fig.add_subplot(111)
ax.set_xlim(0, 18)
ax.set_ylim(0, 16)
ax.axis("off")

# ============= PHASE 0: PROMPT =============
y_prompt = 15
ax.text(9, y_prompt + 0.5, "Input Prompt", ha="center", fontsize=13,
        fontweight="bold", color=DARK,
        bbox=dict(boxstyle="round,pad=0.4", facecolor=LIGHTGREY,
                 edgecolor=DARK, lw=2))

# Draw prompt tokens
prompt_tokens = ["The", "river", "bank", "was"]
x_start = 5
for i, tok in enumerate(prompt_tokens):
    draw_token_box(ax, x_start + i*1.2, y_prompt - 0.8, tok, BLUE)

ax.text(9, y_prompt - 1.6, f"n = {len(prompt_tokens)} tokens",
       ha="center", fontsize=9, color=GREY, style="italic")

# ============= PHASE 1: PREFILL =============
y_prefill = 12
ax.text(1, y_prefill + 1.5, "PREFILL PHASE", ha="left", fontsize=14,
        fontweight="bold", color=DARK,
        bbox=dict(boxstyle="round,pad=0.5", facecolor=BLUE,
                 edgecolor="white", lw=3, alpha=0.2))

arrow = FancyArrowPatch((9, y_prompt - 1.2), (9, y_prefill + 2),
                       arrowstyle="-|>", color=DARK, lw=3)
ax.add_patch(arrow)

ax.text(1, y_prefill + 0.8,
        "Process ALL prompt tokens in parallel",
        ha="left", fontsize=10, color=DARK, fontweight="bold")

# Full attention matrix
matrix_x = 6
matrix_y = y_prefill
draw_matrix(ax, matrix_x, matrix_y, rows=4, cols=4, scale=0.6, color=GREEN, alpha=0.5)

# Causal mask overlay
for i in range(4):
    for j in range(4):
        if j > i:  # Upper triangle (masked)
            rect = Rectangle((matrix_x + j*0.6, matrix_y - i*0.6), 0.54, 0.54,
                           facecolor=RED, edgecolor="white", lw=1, alpha=0.4)
            ax.add_patch(rect)

ax.text(matrix_x + 1.2, matrix_y + 1, "Attention Matrix $(4 \\times 4)$",
       ha="center", fontsize=10, fontweight="bold", color=DARK)

# Token labels
token_labels = ["T1", "T2", "T3", "T4"]
for i, label in enumerate(token_labels):
    ax.text(matrix_x + i*0.6 + 0.3, matrix_y + 0.4, label,
           ha="center", fontsize=8, color=DARK)
    ax.text(matrix_x - 0.3, matrix_y - i*0.6 - 0.3, label,
           ha="right", fontsize=8, color=DARK)

# KV cache initialization
cache_x = 12
ax.text(cache_x, y_prefill + 0.5, "KV Cache", ha="left", fontsize=11,
       fontweight="bold", color=PURPLE)
ax.text(cache_x, y_prefill,
        "Store K, V for all 4 tokens\n"
        "across all L layers",
        ha="left", fontsize=9, color=DARK,
        bbox=dict(boxstyle="round,pad=0.4", facecolor=PURPLE,
                 edgecolor="white", lw=2, alpha=0.2))

# Cache boxes
for i in range(4):
    draw_token_box(ax, cache_x + i*0.6, y_prefill - 1, f"K{i+1}", PURPLE, w=0.5, h=0.4)
    draw_token_box(ax, cache_x + i*0.6, y_prefill - 1.6, f"V{i+1}", PURPLE, w=0.5, h=0.4)

ax.text(1, y_prefill - 2.5,
        "Cost: $O(n^2 \\cdot d_{\\text{model}})$ for attention\n"
        "Memory: Store $(n \\times d_{\\text{model}})$ per layer",
        ha="left", fontsize=9, color=DARK, style="italic",
        bbox=dict(boxstyle="round,pad=0.4", facecolor=YELLOW,
                 edgecolor=DARK, lw=1.5, alpha=0.7))

# ============= PHASE 2: DECODE STEP 1 =============
y_decode1 = 7.5
ax.text(1, y_decode1 + 1.5, "DECODE STEP 1", ha="left", fontsize=14,
        fontweight="bold", color=DARK,
        bbox=dict(boxstyle="round,pad=0.5", facecolor=ORANGE,
                 edgecolor="white", lw=3, alpha=0.2))

arrow = FancyArrowPatch((9, y_prefill - 3), (9, y_decode1 + 2),
                       arrowstyle="-|>", color=DARK, lw=3)
ax.add_patch(arrow)

ax.text(1, y_decode1 + 0.8,
        "Generate token 5: compute Q₅ · [K₁, K₂, K₃, K₄]",
        ha="left", fontsize=10, color=DARK, fontweight="bold")

# New token
new_tok_x = 5
draw_token_box(ax, new_tok_x, y_decode1, "flooded", ORANGE, w=1.2, h=0.5)
ax.text(new_tok_x + 0.6, y_decode1 - 0.6, "Token 5 (new)",
       ha="center", fontsize=8, color=GREY, style="italic")

# Attention with cache
matrix_x2 = 7
ax.text(matrix_x2, y_decode1 + 0.5, "Attn: Q₅ vs cached K",
       ha="left", fontsize=9, fontweight="bold", color=DARK)

# Single row (new query against cached keys)
for j in range(4):
    rect = Rectangle((matrix_x2 + j*0.6, y_decode1 - 0.5), 0.54, 0.5,
                   facecolor=GREEN, edgecolor="white", lw=1.5, alpha=0.7)
    ax.add_patch(rect)

ax.text(matrix_x2 + 1.2, y_decode1 - 1.3, "Only 1 new row computed",
       ha="center", fontsize=8, color=DARK, style="italic")

# Update cache
cache_x2 = 12
ax.text(cache_x2, y_decode1 + 0.5, "Update Cache", ha="left", fontsize=11,
       fontweight="bold", color=PURPLE)

for i in range(4):
    draw_token_box(ax, cache_x2 + i*0.6, y_decode1 - 0.5, f"K{i+1}", GREY, w=0.5, h=0.4)
    draw_token_box(ax, cache_x2 + i*0.6, y_decode1 - 1.1, f"V{i+1}", GREY, w=0.5, h=0.4)

# New K, V
draw_token_box(ax, cache_x2 + 2.4, y_decode1 - 0.5, "K₅", ORANGE, w=0.5, h=0.4)
draw_token_box(ax, cache_x2 + 2.4, y_decode1 - 1.1, "V₅", ORANGE, w=0.5, h=0.4)

ax.text(cache_x2 + 1.45, y_decode1 - 1.8, "Append K₅, V₅",
       ha="center", fontsize=8, color=DARK, style="italic")

ax.text(1, y_decode1 - 2.5,
        "Cost: $O(n \\cdot d_{\\text{model}})$ for attention (linear!)\n"
        "No recomputation of prior tokens",
        ha="left", fontsize=9, color=DARK, style="italic",
        bbox=dict(boxstyle="round,pad=0.4", facecolor=YELLOW,
                 edgecolor=DARK, lw=1.5, alpha=0.7))

# ============= PHASE 3: DECODE STEP 2 =============
y_decode2 = 3.5
ax.text(1, y_decode2 + 1.5, "DECODE STEP 2", ha="left", fontsize=14,
        fontweight="bold", color=DARK,
        bbox=dict(boxstyle="round,pad=0.5", facecolor=ORANGE,
                 edgecolor="white", lw=3, alpha=0.2))

arrow = FancyArrowPatch((9, y_decode1 - 3), (9, y_decode2 + 2),
                       arrowstyle="-|>", color=DARK, lw=3)
ax.add_patch(arrow)

ax.text(1, y_decode2 + 0.8,
        "Generate token 6: Q₆ · [K₁...K₅]",
        ha="left", fontsize=10, color=DARK, fontweight="bold")

# New token
draw_token_box(ax, new_tok_x, y_decode2, "by", ORANGE, w=0.8, h=0.5)
ax.text(new_tok_x + 0.4, y_decode2 - 0.6, "Token 6",
       ha="center", fontsize=8, color=GREY, style="italic")

# Attention (now 5 cached keys)
ax.text(matrix_x2, y_decode2 + 0.5, "Attn: Q₆ vs cached K",
       ha="left", fontsize=9, fontweight="bold", color=DARK)

for j in range(5):
    rect = Rectangle((matrix_x2 + j*0.6, y_decode2 - 0.5), 0.54, 0.5,
                   facecolor=GREEN, edgecolor="white", lw=1.5, alpha=0.7)
    ax.add_patch(rect)

ax.text(matrix_x2 + 1.5, y_decode2 - 1.3, "Cache grows: now 5 keys",
       ha="center", fontsize=8, color=DARK, style="italic")

# Cache continues growing
cache_x3 = 12
ax.text(cache_x3 + 1.5, y_decode2 + 0.5, "Cache: 6 tokens",
       ha="center", fontsize=11, fontweight="bold", color=PURPLE)

for i in range(6):
    if i < 5:
        col = GREY
    else:
        col = ORANGE
    draw_token_box(ax, cache_x3 + i*0.5, y_decode2 - 0.5, "", col, w=0.45, h=0.35)
    draw_token_box(ax, cache_x3 + i*0.5, y_decode2 - 1, "", col, w=0.45, h=0.35)

ax.text(cache_x3 + 1.5, y_decode2 - 1.6, "Continues until EOS",
       ha="center", fontsize=8, color=DARK, style="italic")

# ============= PERFORMANCE BOX =============
ax.text(1, 1,
        "SPEEDUP: 10-20× vs recomputing full sequence\n"
        "MEMORY: $(2 \\times L \\times n \\times d_{\\text{model}} \\times \\text{precision})$\n"
        "LLaMA 7B @ 2k tokens: ~1 GB cache per request",
        ha="left", fontsize=10, color=DARK,
        bbox=dict(boxstyle="round,pad=0.6", facecolor=GREEN,
                 edgecolor=DARK, lw=3, alpha=0.3))

plt.tight_layout()
output_path = Path(__file__).parent.parent / "img" / "kv-cache-inference.png"
output_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output_path, dpi=150, facecolor="white", bbox_inches="tight")
print(f"✓ Generated: {output_path}")
plt.close()
