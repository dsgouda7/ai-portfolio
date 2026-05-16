from pathlib import Path
"""Generate GPU Architecture.png — 4-panel xkcd-style concept sketch.

Panels:
  (1) GPU hardware stack — SM / tensor cores / register file / shared mem
  (2) Memory hierarchy pyramid with bandwidth & capacity
  (3) Roofline model — compute-bound vs memory-bound
  (4) Matrix multiply tiling — how a matmul maps to SMs
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon
import numpy as np

# Gentle xkcd: keep sketchy look but keep text legible (no stroke outlines).
plt.xkcd(scale=0.3, length=100, randomness=1)
plt.rcParams["path.effects"] = []

fig = plt.figure(figsize=(18, 11), facecolor="white")
fig.suptitle("GPU Architecture Fundamentals", fontsize=22,
             fontweight="bold", y=0.98)

gs = fig.add_gridspec(2, 2, hspace=0.50, wspace=0.28,
                      left=0.05, right=0.97, top=0.91, bottom=0.06)
ax_stack = fig.add_subplot(gs[0, 0])
ax_mem = fig.add_subplot(gs[0, 1])
ax_roof = fig.add_subplot(gs[1, 0])
ax_tile = fig.add_subplot(gs[1, 1])

BLUE, ORANGE, GREEN, RED, PURPLE, DARK, GREY = (
    "#2E86C1", "#E67E22", "#27AE60", "#E74C3C", "#8E44AD", "#2C3E50", "#BDC3C7")

# ═══════════════════════ PANEL 1 — GPU hardware stack ═══════════════
ax_stack.set_title("1 · GPU Hardware Stack (one SM)",
                   fontsize=13, fontweight="bold", color=DARK)
ax_stack.set_xlim(0, 10); ax_stack.set_ylim(0, 10); ax_stack.axis("off")

# SM outer box
ax_stack.add_patch(FancyBboxPatch((0.6, 1.0), 8.8, 7.5,
                                  boxstyle="round,pad=0.1",
                                  facecolor="#EBF5FB", edgecolor=BLUE, lw=2))
ax_stack.text(5, 8.1, "Streaming Multiprocessor (SM)",
              ha="center", fontweight="bold", fontsize=11, color=BLUE)

# Tensor cores row
ax_stack.add_patch(FancyBboxPatch((1.0, 6.2), 8.0, 1.4,
                                  boxstyle="round,pad=0.05",
                                  facecolor=PURPLE, edgecolor="white", lw=1.5))
ax_stack.text(5, 6.9, "Tensor Cores  (matmul engines, fp16/bf16/fp8)",
              ha="center", va="center", color="white",
              fontweight="bold", fontsize=10)

# CUDA cores row
ax_stack.add_patch(FancyBboxPatch((1.0, 4.6), 8.0, 1.3,
                                  boxstyle="round,pad=0.05",
                                  facecolor=ORANGE, edgecolor="white", lw=1.5))
ax_stack.text(5, 5.25, "CUDA Cores  (fp32/int32, general SIMT)",
              ha="center", va="center", color="white",
              fontweight="bold", fontsize=10)

# Warp scheduler
ax_stack.add_patch(FancyBboxPatch((1.0, 3.3), 3.8, 1.0,
                                  boxstyle="round,pad=0.05",
                                  facecolor=GREEN, edgecolor="white", lw=1.5))
ax_stack.text(2.9, 3.8, "Warp Scheduler\n(32 threads / warp)",
              ha="center", va="center", color="white",
              fontweight="bold", fontsize=9)

# Register file
ax_stack.add_patch(FancyBboxPatch((5.2, 3.3), 3.8, 1.0,
                                  boxstyle="round,pad=0.05",
                                  facecolor=BLUE, edgecolor="white", lw=1.5))
ax_stack.text(7.1, 3.8, "Register File\n(fast, per-thread)",
              ha="center", va="center", color="white",
              fontweight="bold", fontsize=9)

# Shared memory / L1
ax_stack.add_patch(FancyBboxPatch((1.0, 1.5), 8.0, 1.5,
                                  boxstyle="round,pad=0.05",
                                  facecolor=DARK, edgecolor="white", lw=1.5))
ax_stack.text(5, 2.25, "Shared Memory / L1 Cache  (on-chip, programmable)",
              ha="center", va="center", color="white",
              fontweight="bold", fontsize=10)

ax_stack.text(5, 0.3,
              "GPU = many SMs in parallel + HBM. 100+ SMs on H100/H200.",
              ha="center", fontsize=9, color="#555", style="italic")

# ═══════════════════════ PANEL 2 — Memory hierarchy ═════════════════
ax_mem.set_title("2 · Memory Hierarchy — capacity vs bandwidth",
                 fontsize=13, fontweight="bold", color=DARK)
ax_mem.set_xlim(0, 10); ax_mem.set_ylim(0, 10); ax_mem.axis("off")

# Pyramid of tiers
levels = [
    (8.4, 2.6, "Registers",        "KB  /  ~20 TB/s",     BLUE),
    (7.2, 3.6, "Shared mem / L1",  "~200 KB / SM  /  ~15 TB/s", GREEN),
    (6.0, 4.8, "L2 cache",         "~60 MB  /  ~5 TB/s",  ORANGE),
    (4.2, 6.5, "HBM (VRAM)",       "80-192 GB  /  ~3 TB/s", PURPLE),
    (2.0, 8.5, "Host DRAM (PCIe)", "TB  /  ~64 GB/s",     RED),
]
for y_top, width_half, name, spec, c in levels:
    y = y_top
    ax_mem.add_patch(FancyBboxPatch((5 - width_half/2, y - 0.45), width_half, 0.9,
                                    boxstyle="round,pad=0.05",
                                    facecolor=c, edgecolor="white", lw=1.5))
    ax_mem.text(5, y, name, ha="center", va="center",
                color="white", fontweight="bold", fontsize=10)
    ax_mem.text(5, y - 0.7, spec, ha="center", fontsize=8.5, color=DARK)

ax_mem.annotate("", xy=(0.5, 1.5), xytext=(0.5, 9),
                arrowprops=dict(arrowstyle="-|>", color=DARK, lw=2))
ax_mem.text(0.2, 5, "farther =\nbigger,\nslower",
            rotation=90, fontsize=9, color=DARK,
            fontweight="bold", ha="center", va="center")
ax_mem.text(5, 0.4,
            "Every off-chip hop costs 10-1000x latency.",
            ha="center", fontsize=9, color="#555", style="italic")

# ═══════════════════════ PANEL 3 — Roofline ═════════════════════════
ax_roof.set_title("3 · Roofline Model", fontsize=13,
                  fontweight="bold", color=DARK)
# x = arithmetic intensity (FLOP/byte), y = achieved TFLOP/s
ai = np.logspace(-2, 3, 200)
peak_compute = 1000.0         # TFLOP/s
peak_bw = 3.0                 # TB/s  ~ 3 * 1e12 B/s
# bw-bound line: perf = AI * BW
bw_line = ai * peak_bw * 1000  # convert TB/s * FLOP/byte -> GFLOP/s then TFLOP/s
roof = np.minimum(bw_line, peak_compute)
ax_roof.plot(ai, roof, "-", color=BLUE, lw=3)
ax_roof.fill_between(ai, roof, 0.1, alpha=0.12, color=BLUE)
# knee
knee_ai = peak_compute / (peak_bw * 1000)
ax_roof.axvline(knee_ai, color=GREY, ls=":", lw=1.5)

# Label regions
ax_roof.text(0.05, 500, "memory-\nbound", color=RED, fontsize=12,
             fontweight="bold", ha="left")
ax_roof.text(200, 1050, "compute-bound", color=GREEN, fontsize=12,
             fontweight="bold", ha="center")

# Kernel dots
kernels = [
    ("element-wise", 0.05, 0.05 * peak_bw * 1000, BLUE),
    ("attention QK", 2.0, 2.0 * peak_bw * 1000, ORANGE),
    ("GEMM (tiled)", 100.0, peak_compute, GREEN),
    ("conv 3x3",     20.0, min(20 * peak_bw * 1000, peak_compute), PURPLE),
]
for name, x, y, c in kernels:
    ax_roof.plot(x, y, "o", color=c, markersize=14,
                 markeredgecolor="white", markeredgewidth=2, zorder=5)
    ax_roof.annotate(name, (x, y), textcoords="offset points",
                     xytext=(8, 8), fontsize=9, color=c, fontweight="bold")

ax_roof.set_xscale("log"); ax_roof.set_yscale("log")
ax_roof.set_xlabel("arithmetic intensity (FLOP / byte)",
                   fontsize=10, color=DARK)
ax_roof.set_ylabel("achievable  TFLOP/s", fontsize=10, color=DARK)
ax_roof.set_ylim(1, 2000); ax_roof.set_xlim(0.01, 1000)
ax_roof.text(0.5, -0.28,
             "Below the knee: bandwidth-limited. Above: FLOPs-limited.\n"
             "Optimise the tight side of the roof.",
             transform=ax_roof.transAxes, ha="center",
             fontsize=9, color="#555", style="italic")

# ═══════════════════════ PANEL 4 — Matmul tiling ════════════════════
ax_tile.set_title("4 · Tiled MatMul — C = A * B",
                  fontsize=13, fontweight="bold", color=DARK)
ax_tile.set_xlim(0, 12); ax_tile.set_ylim(0, 10); ax_tile.axis("off")

# A matrix (left)
for i in range(4):
    for j in range(3):
        ax_tile.add_patch(plt.Rectangle((0.5 + j*0.7, 7.0 - i*0.7),
                                        0.65, 0.65,
                                        facecolor=BLUE, edgecolor="white", lw=1))
# highlight one row-tile
for j in range(3):
    ax_tile.add_patch(plt.Rectangle((0.5 + j*0.7, 7.0 - 1*0.7),
                                    0.65, 0.65,
                                    facecolor=PURPLE, edgecolor="white", lw=1.5))
ax_tile.text(1.55, 8.5, "A (M x K)", ha="center", fontsize=10,
             fontweight="bold", color=BLUE)

# B matrix
for i in range(3):
    for j in range(4):
        ax_tile.add_patch(plt.Rectangle((5.0 + j*0.7, 7.0 - i*0.7),
                                        0.65, 0.65,
                                        facecolor=ORANGE, edgecolor="white", lw=1))
for i in range(3):
    ax_tile.add_patch(plt.Rectangle((5.0 + 1*0.7, 7.0 - i*0.7),
                                    0.65, 0.65,
                                    facecolor=PURPLE, edgecolor="white", lw=1.5))
ax_tile.text(6.4, 8.5, "B (K x N)", ha="center", fontsize=10,
             fontweight="bold", color=ORANGE)

# = sign
ax_tile.text(3.9, 6.2, "*", ha="center", fontsize=20,
             fontweight="bold", color=DARK)
ax_tile.text(8.2, 6.2, "=", ha="center", fontsize=20,
             fontweight="bold", color=DARK)

# C matrix
for i in range(4):
    for j in range(4):
        c = GREEN if (i == 1 and j == 1) else "#D5F5E3"
        ax_tile.add_patch(plt.Rectangle((9.0 + j*0.7, 7.0 - i*0.7),
                                        0.65, 0.65,
                                        facecolor=c, edgecolor="white", lw=1))
ax_tile.text(10.4, 8.5, "C (M x N)", ha="center", fontsize=10,
             fontweight="bold", color=GREEN)

# Explanation
ax_tile.text(6, 3.5,
             "Each output tile C[i,j] = row-tile of A * col-tile of B.\n"
             "Tile dims chosen to fit in shared memory.\n"
             "All tiles computed in parallel across SMs.",
             ha="center", fontsize=10, color=DARK)
ax_tile.text(6, 1.2,
             "High arithmetic intensity (reuse tiles many times) ->\n"
             "matmul is the compute-bound ideal for GPUs.",
             ha="center", fontsize=9, color="#555", style="italic")

out = str(Path(__file__).resolve().parent.parent / "img" / "GPU Architecture.png")
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
print(f"Saved -> {out}")
