"""
gen_ch07.py — generate visual assets for Ch.7 MLE & Loss Functions
───────────────────────────────────────────────────────────────────
Produces:
  img/ch07-mle-loss-functions-needle.gif
      Conceptual animation: the likelihood needle climbs toward its maximum
      as theta moves from an arbitrary starting value to the MLE estimate.
      At each stage the loss it implies is labelled.

  img/ch07-mle-loss-functions-progress-check.png
      UnifiedAI constraint dashboard after Ch.7.

Usage (from chapter root):
  python gen_scripts/gen_ch07.py
"""

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation, PillowWriter

# ── paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
IMG_DIR = SCRIPT_DIR.parent / "img"
IMG_DIR.mkdir(exist_ok=True)

# ── palette (authoring-guide standard) ────────────────────────────────────────
BG       = "#1a1a2e"
FG       = "#e2e8f0"
PRIMARY  = "#1e3a8a"
SUCCESS  = "#15803d"
CAUTION  = "#b45309"
DANGER   = "#b91c1c"
INFO     = "#1d4ed8"
ACCENT   = "#38bdf8"   # needle / highlight colour

RNG = np.random.default_rng(42)


# ══════════════════════════════════════════════════════════════════════════════
# 1.  NEEDLE GIF — likelihood climbing to the MLE
# ══════════════════════════════════════════════════════════════════════════════

# Toy Gaussian MLE scenario ── 3 data points, sweep theta (slope w)
# p(y|x;w) = N(w*x, sigma^2),  sigma=0.5
# True MLE: w* ~ 0.500  (see ch07 §4.6 toy example)

X_TOY = np.array([3.0, 5.0, 7.0])
Y_TOY = np.array([1.5, 2.3, 3.8])
SIGMA = 0.5

def gaussian_log_likelihood(w: float) -> float:
    y_hat = w * X_TOY
    ll = -np.sum((Y_TOY - y_hat) ** 2) / (2 * SIGMA ** 2) - \
         len(X_TOY) * np.log(np.sqrt(2 * np.pi) * SIGMA)
    return float(ll)


# theta sweep from 0.1 → 0.60 (MLE ~0.50)
W_RANGE = np.linspace(0.05, 0.70, 300)
LL_CURVE = np.array([gaussian_log_likelihood(w) for w in W_RANGE])
W_MLE = W_RANGE[np.argmax(LL_CURVE)]

# --- animation stages: (label, theta_current, caption) -----------------------
NEEDLE_STAGES = [
    {
        "label": "theta = 0.10  (far from MLE)",
        "w": 0.10,
        "caption": "Initial guess.  Log-likelihood is low — the model assigns\n"
                   "low probability to the observed house prices.",
    },
    {
        "label": "theta = 0.25  (improving)",
        "w": 0.25,
        "caption": "Gradient ascent on the log-likelihood.  Predictions improve;\n"
                   "probability of the data rises.",
    },
    {
        "label": "theta = 0.40  (near MLE)",
        "w": 0.40,
        "caption": "Almost at the maximum.  MSE is also nearly minimised —\n"
                   "the two objectives are equivalent.",
    },
    {
        "label": f"theta = {W_MLE:.3f}  (MLE = MSE minimiser  [DONE])",
        "w": W_MLE,
        "caption": "Maximum likelihood estimate.  This is identical to the\n"
                   "MSE minimiser: Gaussian noise → MSE loss.",
    },
]

GAUGE_MIN   = float(np.min(LL_CURVE))
GAUGE_MAX   = float(np.max(LL_CURVE))
N_TRANS     = 20   # frames for smooth interpolation between stages
N_HOLD      = 10   # frames to hold at each stage


def _ll_to_angle(ll: float) -> float:
    """Map log-likelihood value to needle angle (180°=low, 0°=high)."""
    frac = (ll - GAUGE_MIN) / (GAUGE_MAX - GAUGE_MIN + 1e-12)
    frac = np.clip(frac, 0.0, 1.0)
    return 180.0 - frac * 180.0


def _draw_ll_gauge(ax: plt.Axes, w: float, label: str, caption: str) -> None:
    """Draw the semicircular log-likelihood gauge for a given w."""
    ax.set_facecolor(BG)
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-0.6, 1.5)
    ax.set_aspect("equal")
    ax.axis("off")

    theta_arc = np.linspace(0, np.pi, 300)

    # coloured zones: danger=low LL, caution=medium, success=high
    zones = [
        (np.linspace(np.pi, np.pi * 2 / 3, 100), DANGER,  0.30),
        (np.linspace(np.pi * 2 / 3, np.pi / 3, 100), CAUTION, 0.30),
        (np.linspace(np.pi / 3, 0, 100),          SUCCESS, 0.30),
    ]
    for th, col, alpha in zones:
        ax.fill_between(np.cos(th), np.zeros_like(th),
                        np.sin(th) * 0.9, color=col, alpha=alpha)

    # outer arc
    ax.plot(np.cos(theta_arc), np.sin(theta_arc), color=FG, lw=2, alpha=0.6)

    # tick marks at 5 positions along the LL scale
    for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
        ll_tick = GAUGE_MIN + frac * (GAUGE_MAX - GAUGE_MIN)
        angle_deg = _ll_to_angle(ll_tick)
        angle_rad = np.deg2rad(angle_deg)
        ax.plot(
            [0.80 * np.cos(angle_rad), 0.95 * np.cos(angle_rad)],
            [0.80 * np.sin(angle_rad), 0.95 * np.sin(angle_rad)],
            color=FG, lw=1.2, alpha=0.6,
        )
        ax.text(
            1.12 * np.cos(angle_rad), 1.12 * np.sin(angle_rad),
            f"{ll_tick:.1f}", ha="center", va="center",
            fontsize=7, color=FG, alpha=0.7,
        )

    # likelihood needle
    ll_current = gaussian_log_likelihood(w)
    angle_rad  = np.deg2rad(_ll_to_angle(ll_current))
    ax.annotate(
        "",
        xy=(0.72 * np.cos(angle_rad), 0.72 * np.sin(angle_rad)),
        xytext=(0, 0),
        arrowprops=dict(arrowstyle="-|>", color=ACCENT, lw=3.5,
                        mutation_scale=16),
    )

    # hub
    ax.add_patch(plt.Circle((0, 0), 0.07, color=FG, zorder=5))

    # LL readout
    ll_color = SUCCESS if ll_current > (GAUGE_MIN + 0.7 * (GAUGE_MAX - GAUGE_MIN)) \
               else (CAUTION if ll_current > GAUGE_MIN + 0.3 * (GAUGE_MAX - GAUGE_MIN) \
               else DANGER)
    ax.text(0, -0.22, f"L(theta) = {ll_current:.3f}",
            ha="center", va="center", fontsize=12, fontweight="bold",
            color=ll_color)

    # MSE readout (equivalent metric)
    y_hat  = w * X_TOY
    mse    = float(np.mean((Y_TOY - y_hat) ** 2))
    ax.text(0, -0.38, f"MSE(θ) = {mse:.4f}",
            ha="center", va="center", fontsize=10, color=FG, alpha=0.8)

    # stage label
    ax.text(0, 1.45, label, ha="center", va="top",
            fontsize=10, fontweight="bold", color=FG)

    # caption
    ax.text(0, -0.52, caption, ha="center", va="top",
            fontsize=8.5, color=FG, alpha=0.8, linespacing=1.4)

    # x-axis label
    ax.text(-1.15, 0.02, "Low\nℓ(θ)", ha="center", va="center",
            fontsize=8, color=DANGER, alpha=0.8)
    ax.text(1.15, 0.02, "High\nℓ(θ)", ha="center", va="center",
            fontsize=8, color=SUCCESS, alpha=0.8)

    # MLE arrow
    mle_angle_rad = np.deg2rad(_ll_to_angle(gaussian_log_likelihood(W_MLE)))
    ax.annotate(
        "MLE",
        xy=(1.0 * np.cos(mle_angle_rad), 1.0 * np.sin(mle_angle_rad)),
        xytext=(1.2 * np.cos(mle_angle_rad), 1.2 * np.sin(mle_angle_rad)),
        arrowprops=dict(arrowstyle="->", color=SUCCESS, lw=1.2),
        ha="center", va="center", fontsize=8, color=SUCCESS,
    )


def build_needle_gif() -> None:
    print("  Building needle GIF …", flush=True)
    fig, ax = plt.subplots(figsize=(5.8, 4.4))
    fig.patch.set_facecolor(BG)

    # build frame list with smooth interpolation between stages
    frames: list[tuple[float, int]] = []
    for si, stage in enumerate(NEEDLE_STAGES):
        if si == 0:
            for _ in range(N_HOLD):
                frames.append((stage["w"], si))
        else:
            prev_w = NEEDLE_STAGES[si - 1]["w"]
            for f in range(N_TRANS):
                t = f / N_TRANS
                t_s = t * t * (3 - 2 * t)   # smoothstep easing
                w_interp = prev_w + (stage["w"] - prev_w) * t_s
                frames.append((w_interp, si))
            for _ in range(N_HOLD):
                frames.append((stage["w"], si))

    def update(frame_idx: int) -> None:
        ax.clear()
        w_cur, si = frames[frame_idx]
        s = NEEDLE_STAGES[si]
        _draw_ll_gauge(ax, w_cur, s["label"], s["caption"])

    ani = FuncAnimation(fig, update, frames=len(frames), interval=80, blit=False)
    out_path = IMG_DIR / "ch07-mle-loss-functions-needle.gif"
    ani.save(str(out_path), writer=PillowWriter(fps=12))
    plt.close(fig)
    print(f"    → saved {out_path.name}  ({out_path.stat().st_size // 1024} KB)")


# ══════════════════════════════════════════════════════════════════════════════
# 2.  PROGRESS CHECK PNG — UnifiedAI constraint dashboard
# ══════════════════════════════════════════════════════════════════════════════

CONSTRAINTS = [
    {
        "name":   "#1  ACCURACY",
        "target": "≤$28k MAE + ≥95% acc.",
        "status": "FOUNDATION",
        "detail": "MLE gives the principled loss for\nboth tasks. Loss selection is now\ntheoretically justified.",
        "color":  CAUTION,
        "pct":    0.65,
    },
    {
        "name":   "#2  GENERALIZATION",
        "target": "Unseen districts / nationwide",
        "status": "SUPPORTED",
        "detail": "Correct loss = correct gradient =\nbetter generalisation. Spatial\ncross-validation recommended.",
        "color":  CAUTION,
        "pct":    0.60,
    },
    {
        "name":   "#3  MULTI-TASK",
        "target": "Value + Classification",
        "status": "UNIFIED",
        "detail": "Gaussian → MSE (regression)\nBernoulli → BCE (classification)\nSame MLE framework, different noise.",
        "color":  SUCCESS,
        "pct":    0.80,
    },
    {
        "name":   "#4  INTERPRETABILITY",
        "target": "Explainable to stakeholders",
        "status": "ENHANCED",
        "detail": "Loss choice now justifiable:\n'MSE because residuals are Gaussian'\nis a testable, auditable claim.",
        "color":  SUCCESS,
        "pct":    0.75,
    },
    {
        "name":   "#5  PRODUCTION",
        "target": "<100ms + TensorBoard + monitor",
        "status": "PENDING",
        "detail": "Loss derived [Y]  Monitoring [N]\nNo TensorBoard yet — Ch.8 adds\nreal-time loss/metric logging.",
        "color":  DANGER,
        "pct":    0.40,
    },
]


def build_progress_check_png() -> None:
    print("  Building progress-check PNG …", flush=True)
    fig, axes = plt.subplots(
        1, len(CONSTRAINTS), figsize=(14, 4.5),
        gridspec_kw={"wspace": 0.35},
    )
    fig.patch.set_facecolor(BG)

    for ax, c in zip(axes, CONSTRAINTS):
        ax.set_facecolor(BG)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

        # background card
        card = mpatches.FancyBboxPatch(
            (0.04, 0.04), 0.92, 0.92,
            boxstyle="round,pad=0.02",
            facecolor=PRIMARY, edgecolor=c["color"], linewidth=2,
            transform=ax.transAxes, clip_on=False,
        )
        ax.add_patch(card)

        # progress bar
        bar_y, bar_h = 0.14, 0.08
        ax.add_patch(mpatches.FancyBboxPatch(
            (0.08, bar_y), 0.84, bar_h,
            boxstyle="round,pad=0.01",
            facecolor="#1e293b", edgecolor="none",
            transform=ax.transAxes,
        ))
        ax.add_patch(mpatches.FancyBboxPatch(
            (0.08, bar_y), 0.84 * c["pct"], bar_h,
            boxstyle="round,pad=0.01",
            facecolor=c["color"], edgecolor="none",
            transform=ax.transAxes,
        ))
        ax.text(0.50, bar_y + bar_h / 2, f"{int(c['pct']*100)}%",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=8.5, fontweight="bold", color=FG)

        # status badge
        ax.text(0.50, 0.92, c["status"],
                transform=ax.transAxes, ha="center", va="top",
                fontsize=9.5, fontweight="bold", color=c["color"])

        # constraint name
        ax.text(0.50, 0.82, c["name"],
                transform=ax.transAxes, ha="center", va="top",
                fontsize=9, fontweight="bold", color=FG)

        # target
        ax.text(0.50, 0.72, c["target"],
                transform=ax.transAxes, ha="center", va="top",
                fontsize=7.5, color=FG, alpha=0.75, style="italic")

        # detail text
        ax.text(0.50, 0.55, c["detail"],
                transform=ax.transAxes, ha="center", va="top",
                fontsize=7.5, color=FG, alpha=0.80, linespacing=1.45)

    # title
    fig.text(0.50, 0.98,
             "Ch.7 MLE & Loss Functions — UnifiedAI Progress Check",
             ha="center", va="top", fontsize=13, fontweight="bold",
             color=FG)
    fig.text(0.50, 0.93,
             "Gaussian → MSE  ·  Bernoulli → BCE  ·  Laplacian → MAE  "
             "·  Every loss is a noise assumption",
             ha="center", va="top", fontsize=9, color=FG, alpha=0.75)

    out_path = IMG_DIR / "ch07-mle-loss-functions-progress-check.png"
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight",
                facecolor=BG)
    plt.close(fig)
    print(f"    → saved {out_path.name}  ({out_path.stat().st_size // 1024} KB)")


# ══════════════════════════════════════════════════════════════════════════════
# main
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Generating Ch.7 visual assets …")
    build_needle_gif()
    build_progress_check_png()
    print("Done.")
