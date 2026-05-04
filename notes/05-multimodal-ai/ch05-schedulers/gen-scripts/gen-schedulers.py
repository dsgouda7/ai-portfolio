from pathlib import Path
"""Generate Schedulers.png — 4-panel xkcd-style concept sketch.

Panels:
  (1) DDPM 1000 steps vs DDIM 20 steps — step spacing
  (2) Sampling trajectory in latent 2D
  (3) Quality (FID) vs step count per scheduler
  (4) Scheduler family table
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

plt.xkcd(scale=0.8, length=100, randomness=2)

fig = plt.figure(figsize=(18, 11), facecolor="white")
fig.suptitle("Schedulers — From 1000 Steps to 20 Without Retraining",
             fontsize=20, fontweight="bold", y=0.98)

gs = fig.add_gridspec(2, 2, hspace=0.55, wspace=0.28,
                      left=0.05, right=0.97, top=0.91, bottom=0.06)
ax_steps = fig.add_subplot(gs[0, 0])
ax_traj = fig.add_subplot(gs[0, 1])
ax_qs = fig.add_subplot(gs[1, 0])
ax_tab = fig.add_subplot(gs[1, 1])

BLUE, ORANGE, GREEN, RED, PURPLE, DARK, GREY = (
    "#2E86C1", "#E67E22", "#27AE60", "#E74C3C", "#8E44AD", "#2C3E50", "#BDC3C7")

# ═══════════════════════ PANEL 1 — Step spacing ════════════════════
ax_steps.set_title("1 · DDPM (1000) vs DDIM (20) — Step Spacing",
                   fontsize=13, fontweight="bold", color=DARK)
ax_steps.set_xlim(0, 10); ax_steps.set_ylim(0, 10); ax_steps.axis("off")

ax_steps.text(5, 9.3, "noise  ->  clean  (t axis)",
              ha="center", fontsize=10, color=DARK)

# DDPM - many short ticks
ax_steps.text(0.5, 7.3, "DDPM", fontweight="bold", color=BLUE,
              fontsize=11)
ax_steps.plot([1.5, 9.5], [7.3, 7.3], "-", color=BLUE, lw=2)
for x in np.linspace(1.5, 9.5, 60):
    ax_steps.plot([x, x], [7.1, 7.5], "-", color=BLUE, lw=0.8)
ax_steps.text(5.5, 6.5, "1000 small stochastic steps",
              ha="center", fontsize=9, color=DARK)

# DDIM - fewer ticks
ax_steps.text(0.5, 4.8, "DDIM", fontweight="bold", color=GREEN,
              fontsize=11)
ax_steps.plot([1.5, 9.5], [4.8, 4.8], "-", color=GREEN, lw=2)
for x in np.linspace(1.5, 9.5, 20):
    ax_steps.plot([x, x], [4.6, 5.0], "-", color=GREEN, lw=1.2)
ax_steps.text(5.5, 4.0, "20 deterministic steps (jumps)",
              ha="center", fontsize=9, color=DARK)

# DPM-Solver++
ax_steps.text(0.5, 2.3, "DPM++", fontweight="bold", color=PURPLE,
              fontsize=11)
ax_steps.plot([1.5, 9.5], [2.3, 2.3], "-", color=PURPLE, lw=2)
for x in np.linspace(1.5, 9.5, 12):
    ax_steps.plot([x, x], [2.1, 2.5], "-", color=PURPLE, lw=1.4)
ax_steps.text(5.5, 1.5, "8–12 steps — higher-order ODE solver",
              ha="center", fontsize=9, color=DARK)

ax_steps.text(5, 0.4,
              "Same trained UNet. Scheduler decides HOW to hop.",
              ha="center", fontsize=9, color="#555", style="italic")

# ═══════════════════════ PANEL 2 — Trajectory ═══════════════════════
ax_traj.set_title("2 · Sampling Trajectory in Latent Space",
                  fontsize=13, fontweight="bold", color=DARK)
ax_traj.set_xlim(-3, 3); ax_traj.set_ylim(-3, 3)
ax_traj.set_xticks([]); ax_traj.set_yticks([])

# "Data manifold"
theta = np.linspace(0, 2*np.pi, 200)
ax_traj.fill(np.cos(theta)*1.0, np.sin(theta)*1.0,
             color=GREEN, alpha=0.15)
ax_traj.text(0, 0, "data\nmanifold", ha="center", va="center",
             color=GREEN, fontweight="bold", fontsize=10)

# DDPM - noisy wandering path
rng = np.random.default_rng(7)
ddpm_x = [2.5]; ddpm_y = [2.5]
for _ in range(30):
    ddpm_x.append(ddpm_x[-1] + rng.normal(-0.1, 0.15))
    ddpm_y.append(ddpm_y[-1] + rng.normal(-0.1, 0.15))
ax_traj.plot(ddpm_x, ddpm_y, "-o", color=BLUE, lw=1.2,
             markersize=3, label="DDPM 1000")

# DDIM - direct path
ddim_x = np.linspace(-2.5, -0.4, 8)
ddim_y = np.linspace(2.3, 0.6, 8) + rng.normal(0, 0.05, 8)
ax_traj.plot(ddim_x, ddim_y, "-s", color=GREEN, lw=2,
             markersize=6, label="DDIM 20")

# DPM++ - even fewer
dpm_x = np.linspace(2.3, 0.5, 5)
dpm_y = np.linspace(-2.3, -0.4, 5) + rng.normal(0, 0.08, 5)
ax_traj.plot(dpm_x, dpm_y, "-^", color=PURPLE, lw=2,
             markersize=8, label="DPM++ 10")

ax_traj.legend(loc="upper right", fontsize=9, framealpha=0.9)
ax_traj.text(0, -3.4,
             "Fewer but bigger jumps. Accurate only because the model is well-trained.",
             transform=ax_traj.transData, ha="center",
             fontsize=9, color="#555", style="italic")

# ═══════════════════════ PANEL 3 — Quality vs steps ═════════════════
ax_qs.set_title("3 · Quality (FID, lower better) vs Steps",
                fontsize=13, fontweight="bold", color=DARK)

steps = np.array([5, 10, 20, 50, 100, 500, 1000])
ddpm = [70, 50, 32, 18, 12, 7, 6.5]
ddim = [22, 12, 8.5, 7.5, 7.2, 7.0, 7.0]
dpm  = [10, 7.5, 7.0, 6.9, 6.9, 6.9, 6.9]

ax_qs.plot(steps, ddpm, "-o", color=BLUE,   lw=2.5, markersize=6,
           label="DDPM")
ax_qs.plot(steps, ddim, "-s", color=GREEN,  lw=2.5, markersize=6,
           label="DDIM")
ax_qs.plot(steps, dpm,  "-^", color=PURPLE, lw=2.5, markersize=7,
           label="DPM++")
ax_qs.set_xscale("log")
ax_qs.set_xlabel("sampling steps  (log)", fontsize=10, color=DARK)
ax_qs.set_ylabel("FID", fontsize=10, color=DARK)
ax_qs.legend(fontsize=10, loc="upper right", framealpha=0.9)

ax_qs.text(0.5, -0.25,
           "Higher-order solvers reach good quality in 8-20 steps.\n"
           "DDPM needs 100+ to catch up.",
           transform=ax_qs.transAxes, ha="center",
           fontsize=9, color="#555", style="italic")

# ═══════════════════════ PANEL 4 — Table ═══════════════════════════
ax_tab.set_title("4 · Scheduler Family — When to Pick What",
                 fontsize=13, fontweight="bold", color=DARK)
ax_tab.set_xlim(0, 10); ax_tab.set_ylim(0, 10); ax_tab.axis("off")

headers = ["scheduler", "type", "steps", "use for"]
rows = [
    ("DDPM",       "stochastic",   "500-1000", "reference / training sanity"),
    ("DDIM",       "deterministic","20-50",    "fast sampling, reproducible"),
    ("PNDM",       "pseudo-ODE",   "50",       "old SD default"),
    ("DPM-Solver", "ODE",          "15-25",    "quality + speed"),
    ("DPM++",      "higher-order", "8-15",     "modern default"),
    ("Euler a",    "ancestral",    "20-30",    "creative variation"),
]
col_x = [0.5, 3.2, 5.4, 6.8]
ax_tab.text(col_x[0], 8.8, headers[0], fontweight="bold",
            fontsize=10, color=DARK)
for i, h in enumerate(headers[1:], 1):
    ax_tab.text(col_x[i], 8.8, h, fontweight="bold", fontsize=10, color=DARK)
ax_tab.axhline(8.5, xmin=0.03, xmax=0.97, color=DARK, lw=1)

for j, r in enumerate(rows):
    y = 7.8 - j * 1.05
    for i, cell in enumerate(r):
        ax_tab.text(col_x[i], y, cell, fontsize=9.3, color=DARK)
    ax_tab.axhline(y - 0.45, xmin=0.03, xmax=0.97, color=GREY,
                   lw=0.3, alpha=0.5)

ax_tab.text(5, 0.5,
            "You only pay TRAINING cost once. Inference cost is your scheduler choice.",
            ha="center", fontsize=9.5, color="#555", style="italic",
            fontweight="bold")

out = str(Path(__file__).resolve().parent.parent / "img" / "Schedulers.png")
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
print(f"Saved -> {out}")
