from pathlib import Path
"""Generate Local Diffusion Lab.png — 4-panel xkcd-style concept sketch.

Panels:
  (1) End-to-end pipeline: prompt -> CLIP -> UNet x T -> VAE -> image
  (2) VRAM breakdown on a 12 GB consumer GPU
  (3) Latency breakdown per step
  (4) Quality vs speed knobs (steps / resolution / scheduler / half-prec)
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

plt.xkcd(scale=0.8, length=100, randomness=2)

fig = plt.figure(figsize=(18, 11), facecolor="white")
fig.suptitle("Local Diffusion Lab — Assembling the Full Pipeline",
             fontsize=20, fontweight="bold", y=0.98)

gs = fig.add_gridspec(2, 2, hspace=0.55, wspace=0.28,
                      left=0.05, right=0.97, top=0.91, bottom=0.06)
ax_pipe = fig.add_subplot(gs[0, 0])
ax_vram = fig.add_subplot(gs[0, 1])
ax_lat = fig.add_subplot(gs[1, 0])
ax_knob = fig.add_subplot(gs[1, 1])

BLUE, ORANGE, GREEN, RED, PURPLE, DARK, GREY = (
    "#2E86C1", "#E67E22", "#27AE60", "#E74C3C", "#8E44AD", "#2C3E50", "#BDC3C7")

# ═══════════════════════ PANEL 1 — Pipeline ════════════════════════
ax_pipe.set_title("1 · End-to-End Inference Pipeline",
                  fontsize=13, fontweight="bold", color=DARK)
ax_pipe.set_xlim(0, 10); ax_pipe.set_ylim(0, 10); ax_pipe.axis("off")

stages = [
    (1.2, "prompt",    GREEN,  '"a red fox"'),
    (3.0, "CLIP text", BLUE,   "77x768 emb"),
    (4.8, "noise z_T", GREY,   "(4,64,64)"),
    (6.8, "UNet x N",  PURPLE, "denoise loop"),
    (9.0, "VAE dec",   ORANGE, "512x512"),
]
for x, name, c, body in stages:
    ax_pipe.add_patch(FancyBboxPatch((x - 0.7, 5.8), 1.4, 2.2,
                                     boxstyle="round,pad=0.05",
                                     facecolor=c, edgecolor="white", lw=2))
    ax_pipe.text(x, 7.2, name, ha="center", va="center",
                 color="white", fontweight="bold", fontsize=9.5)
    ax_pipe.text(x, 6.2, body, ha="center", fontsize=8, color="white",
                 family="monospace")
for i in range(4):
    x1, x2 = stages[i][0] + 0.75, stages[i + 1][0] - 0.75
    ax_pipe.annotate("", xy=(x2, 7.0), xytext=(x1, 7.0),
                     arrowprops=dict(arrowstyle="-|>", color=DARK, lw=1.4))

# Cross-attention arrow from CLIP to UNet
ax_pipe.annotate("", xy=(6.1, 6.4), xytext=(3.0, 5.7),
                 arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=1.5,
                                 connectionstyle="arc3,rad=-0.3"))
ax_pipe.text(4.5, 5.2, "cross-attn", fontsize=9, color=ORANGE,
             fontweight="bold", ha="center")

# Safety gate
ax_pipe.add_patch(FancyBboxPatch((3.0, 2.5), 4.0, 1.4,
                                 boxstyle="round,pad=0.1",
                                 facecolor=RED, edgecolor="white", lw=2))
ax_pipe.text(5, 3.2, "NSFW / safety filter",
             ha="center", va="center", color="white",
             fontweight="bold", fontsize=10)
ax_pipe.annotate("", xy=(5, 2.5), xytext=(5, 5.7),
                 arrowprops=dict(arrowstyle="-|>", color=DARK, lw=1.4,
                                 linestyle=":"))

ax_pipe.text(5, 1.2,
             "Three components share a GPU. Offloading CPU<->GPU saves VRAM at latency cost.",
             ha="center", fontsize=9, color="#555", style="italic")

# ═══════════════════════ PANEL 2 — VRAM ═════════════════════════════
ax_vram.set_title("2 · VRAM Budget (SD1.5, fp16, 512^2, 12GB GPU)",
                  fontsize=13, fontweight="bold", color=DARK)

parts = ["UNet", "VAE", "CLIP text", "latents / buffers", "CUDA + Python"]
vals  = [3.2, 0.7, 0.35, 1.8, 1.6]
colors = [PURPLE, ORANGE, GREEN, BLUE, GREY]
total = sum(vals)

bottom = 0
for p, v, c in zip(parts, vals, colors):
    ax_vram.bar([0], [v], bottom=bottom, color=c, edgecolor="white",
                width=0.5, label=f"{p}  {v:.1f} GB")
    ax_vram.text(0, bottom + v/2, p, ha="center", va="center",
                 color="white", fontweight="bold", fontsize=9)
    bottom += v

# budget line
ax_vram.axhline(12, color=RED, ls="--", lw=2)
ax_vram.text(0.45, 12.2, "12 GB budget", color=RED,
             fontweight="bold", fontsize=10)

ax_vram.set_ylim(0, 14)
ax_vram.set_xticks([])
ax_vram.set_ylabel("VRAM (GB)", fontsize=10, color=DARK)
ax_vram.legend(fontsize=8, loc="upper right", framealpha=0.9)

ax_vram.text(0.5, -0.2,
             f"total ~{total:.1f} GB -> fits with room to spare.\n"
             "SDXL doubles this. Enable xFormers + attention slicing if OOM.",
             transform=ax_vram.transAxes, ha="center",
             fontsize=9, color="#555", style="italic")

# ═══════════════════════ PANEL 3 — Latency ═════════════════════════
ax_lat.set_title("3 · Per-Request Latency (25 steps, RTX 3060)",
                 fontsize=13, fontweight="bold", color=DARK)

segments = [
    ("CLIP encode",  0.05),
    ("UNet 25 steps", 2.50),
    ("VAE decode",   0.25),
    ("safety check", 0.10),
]
x = 0.0
colors = [GREEN, PURPLE, ORANGE, RED]
for (name, d), c in zip(segments, colors):
    ax_lat.barh([0], [d], left=x, color=c, edgecolor="white",
                height=0.5, label=f"{name} {d:.2f}s")
    if d > 0.2:
        ax_lat.text(x + d/2, 0, name, ha="center", va="center",
                    color="white", fontweight="bold", fontsize=9)
    x += d

ax_lat.set_xlim(0, 3.2)
ax_lat.set_yticks([])
ax_lat.set_xlabel("seconds", fontsize=10, color=DARK)
ax_lat.legend(fontsize=8, loc="lower right", framealpha=0.9)

ax_lat.text(0.5, -0.45,
            "UNet dominates. Reducing steps / using DPM++ is the biggest win.\n"
            "VAE decode is a fixed cost per output image.",
            transform=ax_lat.transAxes, ha="center",
            fontsize=9, color="#555", style="italic")

# ═══════════════════════ PANEL 4 — Knobs ═══════════════════════════
ax_knob.set_title("4 · Quality / Speed Knobs",
                  fontsize=13, fontweight="bold", color=DARK)
ax_knob.set_xlim(0, 10); ax_knob.set_ylim(0, 10); ax_knob.axis("off")

knobs = [
    (8.5, "scheduler",      BLUE,   "DPM++ / LCM  -> 8-15 steps"),
    (6.8, "steps",          ORANGE, "25 -> 15 saves ~40% latency"),
    (5.1, "resolution",     PURPLE, "512 vs 1024 -> 4x cost"),
    (3.4, "precision",      GREEN,  "fp16 / bf16 / int8 quantisation"),
    (1.7, "offload / slice",RED,    "CPU offload, attention slicing"),
]
for y, name, c, body in knobs:
    ax_knob.add_patch(FancyBboxPatch((0.5, y - 0.65), 9.0, 1.3,
                                     boxstyle="round,pad=0.05",
                                     facecolor=c, edgecolor="white", lw=2))
    ax_knob.text(2.2, y, name, ha="center", va="center",
                 color="white", fontweight="bold", fontsize=10.5)
    ax_knob.text(6.5, y, body, ha="center", va="center",
                 color="white", fontsize=9.5)

ax_knob.text(5, 0.5,
             "Distilled models (LCM / Turbo) collapse T to 1-4 steps with small quality loss.",
             ha="center", fontsize=9, color="#555", style="italic")

out = str(Path(__file__).resolve().parent.parent / "img" / "Local Diffusion Lab.png")
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
print(f"Saved -> {out}")
