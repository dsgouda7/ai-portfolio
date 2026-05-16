"""Generate multihead-attention-flow.png — detailed visualization of multi-head attention mechanism.

Shows:
  1. Input tokens (with embeddings + positional encoding)
  2. Q/K/V projection for multiple heads
  3. Attention score computation (QK^T, scaling, masking, softmax)
  4. Weighted sum with V
  5. Concatenation and output projection
  6. Residual connection + layer norm
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
from matplotlib.patches import ConnectionPatch
import numpy as np

# Gentle xkcd for sketchy aesthetic while keeping text readable
plt.xkcd(scale=0.25, length=80, randomness=0.8)
plt.rcParams["path.effects"] = []

fig = plt.figure(figsize=(20, 14), facecolor="white")
fig.suptitle("Multi-Head Self-Attention — Complete Data Flow",
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

# Create main axis
ax = fig.add_subplot(111)
ax.set_xlim(0, 20)
ax.set_ylim(0, 18)
ax.axis("off")

# ============= HELPER FUNCTIONS =============
def draw_box(ax, x, y, w, h, text, color, text_color="white", fontsize=10):
    """Draw a rounded box with text"""
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                         facecolor=color, edgecolor="white", lw=2.5)
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center",
            color=text_color, fontweight="bold", fontsize=fontsize,
            bbox=dict(boxstyle="round,pad=0.2", facecolor=color,
                     edgecolor="none", alpha=0.8))

def draw_arrow(ax, x1, y1, x2, y2, color=DARK, lw=2, style="-|>"):
    """Draw an arrow between two points"""
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                           arrowstyle=style, color=color, lw=lw,
                           connectionstyle="arc3,rad=0", zorder=2)
    ax.add_patch(arrow)

def draw_matrix(ax, x, y, rows, cols, scale=0.3, color=BLUE, label="", label_offset=0.3):
    """Draw a matrix representation"""
    cell_w = scale
    cell_h = scale
    # Draw grid
    for i in range(rows):
        for j in range(cols):
            intensity = 0.3 + 0.6 * np.random.random()  # Random shading
            rect = Rectangle((x + j*cell_w, y - i*cell_h), cell_w, cell_h,
                           facecolor=color, edgecolor="white", lw=0.8,
                           alpha=intensity)
            ax.add_patch(rect)
    # Label
    if label:
        ax.text(x + (cols*cell_w)/2, y + label_offset, label,
               ha="center", fontsize=9, fontweight="bold", color=DARK)

# ============= STEP 1: INPUT TOKENS =============
y_input = 16
ax.text(10, 17.2, "Step 0: Input Sequence", ha="center", fontsize=12,
        fontweight="bold", color=DARK,
        bbox=dict(boxstyle="round,pad=0.4", facecolor=LIGHTGREY, edgecolor=DARK, lw=2))

# Draw 4 example tokens
tokens = ["The", "river", "bank", "..."]
token_colors = [BLUE, GREEN, ORANGE, GREY]
x_start = 6
for i, (tok, col) in enumerate(zip(tokens, token_colors)):
    x = x_start + i * 2
    draw_box(ax, x, y_input, 1.5, 0.6, tok, col, fontsize=10)
    # Add dimension indicator
    ax.text(x + 0.75, y_input - 0.5, f"d={4096}", ha="center",
           fontsize=7, color=GREY, style="italic")

ax.text(10, y_input - 1.2, "Each token: embedding + positional encoding",
        ha="center", fontsize=9, color=DARK, style="italic")

# ============= STEP 2: Q/K/V PROJECTION =============
y_proj = 13
draw_arrow(ax, 10, y_input - 0.2, 10, y_proj + 2, color=DARK, lw=2.5)

ax.text(10, 14.5, "Step 1: Project to Q, K, V (per head)", ha="center",
        fontsize=12, fontweight="bold", color=DARK,
        bbox=dict(boxstyle="round,pad=0.4", facecolor=LIGHTGREY, edgecolor=DARK, lw=2))

# Show 3 heads (simplified from 32)
head_labels = ["Head 1", "Head 2", "Head h"]
head_x_positions = [3, 10, 17]
head_colors = [BLUE, ORANGE, PURPLE]

for i, (x_head, h_label, h_color) in enumerate(zip(head_x_positions, head_labels, head_colors)):
    # Head label
    ax.text(x_head, y_proj + 1.5, h_label, ha="center", fontsize=10,
           fontweight="bold", color=h_color)

    # Q, K, V boxes
    qkv_y = y_proj
    for j, (label, offset) in enumerate([("Q", -1), ("K", 0), ("V", 1)]):
        draw_box(ax, x_head + offset - 0.4, qkv_y, 0.8, 0.5, label, h_color, fontsize=9)
        # Show dimensions
        ax.text(x_head + offset, qkv_y - 0.4, f"(n×{128})", ha="center",
               fontsize=7, color=GREY, style="italic")

ax.text(10, y_proj - 1.5,
        "Linear projections: X W_Q, X W_K, X W_V\n"
        "d_model=4096 → d_k=128 per head (4096/32 heads)",
        ha="center", fontsize=8, color=DARK, style="italic",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#F8F9F9",
                 edgecolor=GREY, lw=1))

# ============= STEP 3: ATTENTION SCORES =============
y_scores = 9
# Arrow from Q/K to scores
draw_arrow(ax, 10, y_proj - 0.8, 10, y_scores + 3, color=DARK, lw=2.5)

ax.text(10, 11.5, "Step 2: Compute Attention Scores", ha="center",
        fontsize=12, fontweight="bold", color=DARK,
        bbox=dict(boxstyle="round,pad=0.4", facecolor=LIGHTGREY, edgecolor=DARK, lw=2))

# Show attention score matrix for one head (head 1)
x_matrix = 5
draw_matrix(ax, x_matrix, y_scores + 1.5, rows=4, cols=4, scale=0.5,
           color=BLUE, label="", label_offset=0)

# Add labels to matrix
ax.text(x_matrix - 0.8, y_scores + 1, "The\nriver\nbank\n...",
       fontsize=8, color=DARK, va="top")
ax.text(x_matrix + 1, y_scores + 2.2, "The  river  bank  ...",
       fontsize=8, color=DARK, ha="center")

# Formulas and explanations
ax.text(x_matrix + 4, y_scores + 1.5,
        "scores = QKᵀ / √d_k\n\n"
        "• Dot product every query\n"
        "  with every key\n"
        "• Scale by √128 = 11.3\n"
        "• Result: (n × n) matrix",
        ha="left", fontsize=9, color=DARK,
        bbox=dict(boxstyle="round,pad=0.4", facecolor=YELLOW,
                 edgecolor=DARK, lw=1.5, alpha=0.7))

# Causal mask visualization
x_mask = 13
ax.text(x_mask + 1, y_scores + 1.8, "Causal Mask\n(decoder only)",
       ha="center", fontsize=9, fontweight="bold", color=RED)
# Draw triangular mask
for i in range(4):
    for j in range(4):
        if j > i:  # Upper triangle = masked
            rect = Rectangle((x_mask + j*0.5, y_scores + 1.5 - i*0.5),
                           0.5, 0.5, facecolor=RED, edgecolor="white",
                           lw=0.8, alpha=0.5)
            ax.add_patch(rect)
        else:  # Lower triangle = visible
            rect = Rectangle((x_mask + j*0.5, y_scores + 1.5 - i*0.5),
                           0.5, 0.5, facecolor=GREEN, edgecolor="white",
                           lw=0.8, alpha=0.3)
            ax.add_patch(rect)

ax.text(x_mask + 1, y_scores - 0.5, "Set future tokens\nto -∞",
       ha="center", fontsize=8, color=RED, style="italic")

# ============= STEP 4: SOFTMAX =============
y_softmax = 6.5
draw_arrow(ax, 10, y_scores - 0.5, 10, y_softmax + 1.5, color=DARK, lw=2.5)

ax.text(10, 7.8, "Step 3: Softmax → Attention Weights", ha="center",
        fontsize=12, fontweight="bold", color=DARK,
        bbox=dict(boxstyle="round,pad=0.4", facecolor=LIGHTGREY, edgecolor=DARK, lw=2))

# Show weights matrix (normalized version of scores)
x_weights = 5
draw_matrix(ax, x_weights, y_softmax + 0.8, rows=4, cols=4, scale=0.5,
           color=GREEN, label="", label_offset=0)

# Show example weights for "bank" token
ax.text(x_weights + 4, y_softmax + 0.8,
        'Token "bank" weights:\n'
        "The:     0.08\n"
        "river:   0.62  ← high!\n"
        "bank:    0.25\n"
        "...:      0.00  (masked)",
        ha="left", fontsize=9, color=DARK, family="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor=GREEN,
                 edgecolor=DARK, lw=1.5, alpha=0.3))

ax.text(x_weights + 4, y_softmax - 0.5,
        "Each row sums to 1.0",
        ha="left", fontsize=8, color=DARK, style="italic")

# ============= STEP 5: WEIGHTED SUM WITH V =============
y_output_head = 4
draw_arrow(ax, 10, y_softmax - 0.3, 10, y_output_head + 1.8, color=DARK, lw=2.5)

ax.text(10, 5.5, "Step 4: Weighted Sum (attention_weights × V)", ha="center",
        fontsize=12, fontweight="bold", color=DARK,
        bbox=dict(boxstyle="round,pad=0.4", facecolor=LIGHTGREY, edgecolor=DARK, lw=2))

# Show output per head
for i, (x_head, h_label, h_color) in enumerate(zip(head_x_positions, head_labels, head_colors)):
    draw_box(ax, x_head - 0.6, y_output_head, 1.2, 0.5, f"{h_label}\nout", h_color, fontsize=8)
    ax.text(x_head, y_output_head - 0.4, f"(n×{128})", ha="center",
           fontsize=7, color=GREY, style="italic")

# ============= STEP 6: CONCATENATE HEADS =============
y_concat = 2.5
for x_head in head_x_positions:
    draw_arrow(ax, x_head, y_output_head - 0.2, 10, y_concat + 0.8, color=DARK, lw=1.5)

ax.text(10, 3.2, "Step 5: Concatenate All Heads", ha="center",
        fontsize=12, fontweight="bold", color=DARK,
        bbox=dict(boxstyle="round,pad=0.4", facecolor=LIGHTGREY, edgecolor=DARK, lw=2))

draw_box(ax, 7, y_concat, 6, 0.6, "Concat(head₁, head₂, ..., headₕ)", PURPLE, fontsize=10)
ax.text(10, y_concat - 0.5, f"Shape: (n × {32*128} = n × 4096)", ha="center",
       fontsize=8, color=GREY, style="italic")

# ============= STEP 7: OUTPUT PROJECTION =============
y_final = 1
draw_arrow(ax, 10, y_concat - 0.2, 10, y_final + 0.8, color=DARK, lw=2.5)

ax.text(10, 1.6, "Step 6: Output Projection (W_O)", ha="center",
        fontsize=12, fontweight="bold", color=DARK,
        bbox=dict(boxstyle="round,pad=0.4", facecolor=LIGHTGREY, edgecolor=DARK, lw=2))

draw_box(ax, 7, y_final, 6, 0.6, "MultiHead(X)", DARK, fontsize=10)
ax.text(10, y_final - 0.5, "Shape: (n × 4096) — same as input", ha="center",
       fontsize=8, color=GREY, style="italic")

# ============= KEY INSIGHTS BOX =============
ax.text(1, 8,
        "KEY INSIGHTS\n\n"
        "• Q = 'what am I looking for?'\n"
        "• K = 'what do I offer?'\n"
        "• V = 'what info do I carry?'\n\n"
        "• Each head specializes:\n"
        "  - Syntactic patterns\n"
        "  - Positional proximity\n"
        "  - Semantic relationships\n\n"
        "• Attention is a lookup:\n"
        "  Query matches Keys,\n"
        "  retrieves Values",
        fontsize=9, color=DARK, va="top",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#FFF9E6",
                 edgecolor=ORANGE, lw=2.5))

# ============= COMPUTE BOX =============
ax.text(18.5, 8,
        "COMPUTE\n"
        "(LLaMA 7B)\n\n"
        f"• d_model = 4096\n"
        f"• h = 32 heads\n"
        f"• d_k = 128\n\n"
        "Per head:\n"
        "• QKᵀ: n² × d_k\n"
        "• Softmax: n²\n"
        "• ×V: n² × d_k\n\n"
        "For n=512 tokens:\n"
        "~8M ops/head\n"
        "×32 heads\n"
        "×32 layers\n"
        "= ~8B FLOPs",
        fontsize=8, color=DARK, va="top",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#E8F6F3",
                 edgecolor=BLUE, lw=2.5))

plt.tight_layout()
output_path = Path(__file__).parent.parent / "img" / "multihead-attention-flow.png"
output_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output_path, dpi=150, facecolor="white", bbox_inches="tight")
print(f"✓ Generated: {output_path}")
plt.close()
