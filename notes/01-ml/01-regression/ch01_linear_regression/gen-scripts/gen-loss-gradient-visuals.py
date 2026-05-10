"""Generate three visuals for the Gradient Descent Lens section of Ch.01.

Outputs (saved next to this script):
    loss_curves_mae_vs_mse.png    -- static: MAE and MSE loss curves side-by-side
                                     with the kink at e=0 annotated
    gradient_at_zero.gif          -- animation: zoom in on |e| kink showing
                                     left/right derivative disagreement, then
                                     pan to e^2 which smoothly hits zero
    huber_gradient_comparison.png -- static: gradient magnitude of MAE, MSE,
                                     Huber (δ=30k), RMSE across error range

Run:  python gen_loss_gradient_visuals.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

# ── palette (matches repo conventions) ────────────────────────────────────────
BLUE   = "#2E86C1"
GREEN  = "#27AE60"
ORANGE = "#E67E22"
RED    = "#C0392B"
PURPLE = "#8E44AD"
DARK   = "#2C3E50"
GREY   = "#95A5A6"
BG     = "white"

HERE = Path(__file__).parent


# ══════════════════════════════════════════════════════════════════════════════
# Image 1 — loss_curves_mae_vs_mse.png
# Three-panel: |e|, e^2, and both overlaid; kink annotated on MAE panel
# ══════════════════════════════════════════════════════════════════════════════
def make_loss_curves() -> Path:
    e = np.linspace(-3, 3, 800)  # error in $10k units (so 1.0 = $10k)
    mae   = np.abs(e)
    mse   = e ** 2
    rmse  = np.sqrt(mse)
    delta = 0.8  # δ ~ $8k for illustration clarity
    huber = np.where(np.abs(e) <= delta,
                     0.5 * e**2,
                     delta * np.abs(e) - 0.5 * delta**2)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor=BG)
    fig.suptitle("Loss functions: shape, kink, and gradient behaviour",
                 fontsize=13, fontweight="bold", color=DARK, y=1.01)

    # ── Panel A: MAE = |e| ────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(e, mae, color=RED, lw=2.8, label=r"$L = |e|$  (MAE)")
    ax.axvline(0, color=DARK, lw=0.8, ls="--", alpha=0.5)
    ax.axhline(0, color=DARK, lw=0.8, ls="--", alpha=0.5)

    # annotate the kink
    ax.plot(0, 0, "o", ms=9, color=RED, markerfacecolor="white",
            markeredgewidth=2.2, zorder=5, label="kink at e=0 (not differentiable)")

    # draw the two one-sided tangent arrows diverging from the kink
    kink_x, kink_y = 0.0, 0.0
    arrow_kw = dict(arrowstyle="->", lw=1.6, mutation_scale=14)
    ax.annotate("", xy=(-1.0, 1.0), xytext=(kink_x, kink_y),
                arrowprops=dict(**arrow_kw, color=ORANGE))
    ax.annotate("", xy=(+1.0, 1.0), xytext=(kink_x, kink_y),
                arrowprops=dict(**arrow_kw, color=ORANGE))
    ax.text(-1.35, 0.72, r"slope $= -1$", color=ORANGE, fontsize=9.5, ha="right")
    ax.text(+0.12, 0.72, r"slope $= +1$", color=ORANGE, fontsize=9.5, ha="left")
    ax.text(0.07, -0.25,
            r"left limit $\neq$ right limit" "\n" r"$\Rightarrow$ no derivative here!",
            color=RED, fontsize=8.8, ha="left", style="italic",
            bbox=dict(boxstyle="round,pad=0.25", fc="#FFF3F3", ec=RED, lw=0.8))

    ax.set_title("MAE  ($L = |e|$)\nhas a kink at $e = 0$",
                 fontsize=11, fontweight="bold", color=DARK)
    ax.set_xlabel(r"error  $e$  (\$10k units)", fontsize=10)
    ax.set_ylabel("loss $L$", fontsize=10)
    ax.set_xlim(-3.1, 3.1); ax.set_ylim(-0.4, 3.2)
    ax.legend(fontsize=8.5, loc="upper center")
    ax.grid(alpha=0.2)

    # ── Panel B: MSE = e² ─────────────────────────────────────────────────
    ax = axes[1]
    ax.plot(e, mse, color=GREEN, lw=2.8, label=r"$L = e^2$  (MSE)")
    ax.axvline(0, color=DARK, lw=0.8, ls="--", alpha=0.5)
    ax.axhline(0, color=DARK, lw=0.8, ls="--", alpha=0.5)

    # show the tangent at zero — perfectly horizontal
    ax.plot(0, 0, "o", ms=9, color=GREEN, zorder=5,
            label="smooth at e=0: gradient = 0")
    tangent_x = np.linspace(-0.6, 0.6, 50)
    ax.plot(tangent_x, np.zeros_like(tangent_x), color=ORANGE, lw=2.4,
            ls="-", zorder=4, alpha=0.9)
    ax.annotate("tangent is flat\n(gradient $= 0$ at minimum)",
                xy=(0.0, 0.0), xytext=(1.0, 1.2),
                arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.4),
                color=ORANGE, fontsize=9, ha="left",
                bbox=dict(boxstyle="round,pad=0.2", fc="#F0FFF0", ec=GREEN, lw=0.8))

    # show gradient at a non-zero point e = 1.5  (slope = 2×1.5 = 3)
    e0 = 1.5; tang_run = 0.45
    tang_xs = np.array([e0-tang_run, e0+tang_run])
    tang_ys = e0**2 + 2*e0*(tang_xs - e0)
    ax.plot(tang_xs, tang_ys, color=BLUE, lw=2.0, ls="--", alpha=0.85)
    ax.text(e0+tang_run+0.05, tang_ys[-1],
            r"slope $= 2e = 3$" "\n(proportional to error)",
            color=BLUE, fontsize=8.5, va="center")

    ax.set_title("MSE  ($L = e^2$)\nsmooth — gradient $= 2e$ everywhere",
                 fontsize=11, fontweight="bold", color=DARK)
    ax.set_xlabel(r"error  $e$  (\$10k units)", fontsize=10)
    ax.set_ylabel("loss $L$", fontsize=10)
    ax.set_xlim(-3.1, 3.1); ax.set_ylim(-0.4, 9.5)
    ax.legend(fontsize=8.5, loc="upper center")
    ax.grid(alpha=0.2)

    # ── Panel C: Huber overlays MAE and MSE ───────────────────────────────
    ax = axes[2]
    ax.plot(e, mae,   color=RED,   lw=1.8, ls="--", alpha=0.55, label=r"MAE $|e|$")
    ax.plot(e, np.clip(mse, 0, 6), color=GREEN, lw=1.8, ls="--", alpha=0.55,
            label=r"MSE $e^2$ (clipped)")
    ax.plot(e, huber, color=PURPLE, lw=3.0, label=f"Huber (delta={delta})")
    ax.axvline( delta, color=GREY, lw=1.0, ls=":", alpha=0.7)
    ax.axvline(-delta, color=GREY, lw=1.0, ls=":", alpha=0.7)
    ax.axvline(0, color=DARK, lw=0.8, ls="--", alpha=0.5)

    # mark the transition thresholds
    ax.plot([ delta, -delta], [huber[np.searchsorted(e,  delta)],
                                huber[np.searchsorted(e, -delta)]],
            "D", ms=7, color=PURPLE, markerfacecolor="white", markeredgewidth=1.8,
            zorder=5, label=f"transition at |e|=delta={delta}")
    ax.text( delta+0.08, 0.25, rf"$\delta$", color=GREY, fontsize=11)
    ax.text(-delta-0.18, 0.25, rf"$-\delta$", color=GREY, fontsize=11)

    # smoothness annotation at zero
    ax.plot(0, 0, "o", ms=8, color=PURPLE, zorder=6,
            markerfacecolor="white", markeredgewidth=2.0)
    ax.text(0.08, -0.22, "gradient $= 0$ here too ✓",
            color=PURPLE, fontsize=8.5, style="italic",
            bbox=dict(boxstyle="round,pad=0.2", fc="#F8F0FF", ec=PURPLE, lw=0.7))

    ax.set_title(f"Huber ($\\delta = {delta}$): MSE inside, MAE outside\n"
                 r"$C^1$ smooth — no kink, capped gradient",
                 fontsize=11, fontweight="bold", color=DARK)
    ax.set_xlabel(r"error  $e$  (\$10k units)", fontsize=10)
    ax.set_ylabel("loss $L$", fontsize=10)
    ax.set_xlim(-3.1, 3.1); ax.set_ylim(-0.4, 3.2)
    ax.legend(fontsize=8.5, loc="upper center")
    ax.grid(alpha=0.2)

    plt.tight_layout()
    out = HERE / "loss_curves_mae_vs_mse.png"
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"saved {out}")
    return out


# ══════════════════════════════════════════════════════════════════════════════
# Animation — gradient_at_zero.gif
# Two-act animation:
#   Act 1 — Camera starts wide on |e|, then zooms into the kink at e=0.
#            At maximum zoom, the two mismatched tangent arrows are drawn
#            and labelled. A "CANNOT DIFFERENTIATE" stamp appear.
#   Act 2 — Wipe transition to e^2. Same zoom-in, but this time the
#            tangent is flat. "GRADIENT = 0 ✓" stamp appears.
#   Act 3 — Zoom back out to show both curves together, then the Huber
#            curve fades in on top as the "best of both" conclusion.
# ══════════════════════════════════════════════════════════════════════════════
def make_gradient_animation() -> Path:
    FPS = 18
    # act durations in frames
    F_WIDE   = 12   # hold wide view
    F_ZOOM   = 22   # zoom into kink
    F_HOLD   = 22   # hold at kink
    F_REVEAL = 18   # reveal arrows / stamp
    F_HOLD2  = 18   # hold stamp
    F_WIPE   = 14   # crossfade to MSE panel
    F_ZOOM2  = 20   # zoom into e² minimum
    F_HOLD3  = 22   # hold at minimum
    F_REVEAL2= 16   # reveal "gradient = 0" tangent
    F_HOLD4  = 20   # hold
    F_ZOUT   = 20   # zoom back out
    F_HUBER  = 24   # Huber fade-in + final hold
    TOTAL = (F_WIDE + F_ZOOM + F_HOLD + F_REVEAL + F_HOLD2 +
             F_WIPE + F_ZOOM2 + F_HOLD3 + F_REVEAL2 + F_HOLD4 +
             F_ZOUT + F_HUBER)

    e_full = np.linspace(-3.2, 3.2, 800)
    mae_y  = np.abs(e_full)
    mse_y  = e_full ** 2
    delta  = 0.8
    huber_y = np.where(np.abs(e_full) <= delta,
                       0.5 * e_full**2,
                       delta * np.abs(e_full) - 0.5 * delta**2)

    def smoothstep(t):
        t = np.clip(t, 0.0, 1.0)
        return t * t * (3.0 - 2.0 * t)

    def lerp(a, b, t):
        return a + (b - a) * t

    fig, ax = plt.subplots(figsize=(9, 6), facecolor=BG)
    ax.grid(alpha=0.15)
    ax.set_xlabel(r"error  $e$  (\$10k units)", fontsize=11)
    ax.set_ylabel("loss $L(e)$", fontsize=11)
    ax.axvline(0, color=DARK, lw=0.7, ls="--", alpha=0.4)
    ax.axhline(0, color=DARK, lw=0.7, ls="--", alpha=0.4)

    (curve_mae,)   = ax.plot([], [], color=RED,    lw=2.8, zorder=2, label=r"MAE $= |e|$")
    (curve_mse,)   = ax.plot([], [], color=GREEN,  lw=2.8, zorder=2, label=r"MSE $= e^2$")
    (curve_huber,) = ax.plot([], [], color=PURPLE, lw=2.8, zorder=3,
                             alpha=0.0, label=fr"Huber ($\delta={delta}$)")
    (tangent_left,)  = ax.plot([], [], color=ORANGE, lw=2.4, ls="-", zorder=4)
    (tangent_right,) = ax.plot([], [], color=ORANGE, lw=2.4, ls="-", zorder=4)
    (tangent_flat,)  = ax.plot([], [], color=BLUE,   lw=2.6, ls="-", zorder=4)
    kink_dot,    = ax.plot([], [], "o", ms=10, color=RED,   markerfacecolor="white",
                           markeredgewidth=2.4, zorder=6)
    min_dot,     = ax.plot([], [], "o", ms=10, color=GREEN, markerfacecolor="white",
                           markeredgewidth=2.4, zorder=6)

    stamp_bad  = ax.text(0.50, 0.88, "", transform=ax.transAxes, ha="center",
                         fontsize=12.5, fontweight="bold", color=RED, alpha=0.0,
                         bbox=dict(boxstyle="round,pad=0.35", fc="#FFF0F0",
                                   ec=RED, lw=1.5))
    stamp_good = ax.text(0.50, 0.88, "", transform=ax.transAxes, ha="center",
                         fontsize=12.5, fontweight="bold", color=GREEN, alpha=0.0,
                         bbox=dict(boxstyle="round,pad=0.35", fc="#F0FFF0",
                                   ec=GREEN, lw=1.5))
    caption    = ax.text(0.50, 0.04, "", transform=ax.transAxes, ha="center",
                         fontsize=10.5, color=DARK, style="italic",
                         bbox=dict(boxstyle="round,pad=0.3", fc="white",
                                   ec=GREY, lw=0.8, alpha=0.85))
    title_txt  = ax.set_title("", fontsize=12, fontweight="bold", color=DARK)
    legend     = ax.legend(loc="upper right", fontsize=9, framealpha=0.92)

    # Wide view limits
    WIDE_XL, WIDE_XR = -3.2, 3.2
    WIDE_YB, WIDE_YT = -0.3, 4.0

    # Kink zoom limits (very tight)
    KINK_HALF = 0.22
    KINK_XL, KINK_XR = -KINK_HALF, KINK_HALF
    KINK_YB, KINK_YT = -0.04, KINK_HALF*1.1

    # MSE min zoom limits
    MIN_HALF = 0.24
    MIN_XL, MIN_XR = -MIN_HALF, MIN_HALF
    MIN_YB, MIN_YT = -0.008, MIN_HALF**2 * 1.4

    # Back-out limits (both curves)
    BOTH_YT = 3.8

    # ── cumulative frame counter ──
    cuts = np.cumsum([0, F_WIDE, F_ZOOM, F_HOLD, F_REVEAL, F_HOLD2,
                      F_WIPE, F_ZOOM2, F_HOLD3, F_REVEAL2, F_HOLD4,
                      F_ZOUT, F_HUBER])

    def update(frame):
        # helper: which act are we in?
        act = np.searchsorted(cuts, frame, side="right") - 1
        t   = (frame - cuts[act]) / max(1, cuts[act+1] - cuts[act])  # 0→1 within act
        s   = smoothstep(t)

        # defaults: hide everything, reset
        for artist in [tangent_left, tangent_right, tangent_flat, kink_dot, min_dot]:
            artist.set_data([], [])
        for txt in [stamp_bad, stamp_good]:
            txt.set_alpha(0.0)
        curve_huber.set_alpha(0.0)

        # ── Act 0: wide view, show MAE ──────────────────────────────────
        if act == 0:
            ax.set_xlim(WIDE_XL, WIDE_XR)
            ax.set_ylim(WIDE_YB, WIDE_YT)
            curve_mae.set_data(e_full, mae_y)
            curve_mse.set_data([], [])
            title_txt.set_text("MAE loss  $L = |e|$")
            caption.set_text("The absolute value looks simple — but it hides a problem at $e = 0$")

        # ── Act 1: zoom into kink ───────────────────────────────────────
        elif act == 1:
            xl = lerp(WIDE_XL, KINK_XL, s)
            xr = lerp(WIDE_XR, KINK_XR, s)
            yb = lerp(WIDE_YB, KINK_YB, s)
            yt = lerp(WIDE_YT, KINK_YT, s)
            ax.set_xlim(xl, xr); ax.set_ylim(yb, yt)
            curve_mae.set_data(e_full, mae_y)
            curve_mse.set_data([], [])
            title_txt.set_text("Zooming in on $e = 0$ …")
            caption.set_text("Approaching the minimum — what does the slope look like?")

        # ── Act 2: hold at kink, show dot ──────────────────────────────
        elif act == 2:
            ax.set_xlim(KINK_XL, KINK_XR); ax.set_ylim(KINK_YB, KINK_YT)
            curve_mae.set_data(e_full, mae_y)
            curve_mse.set_data([], [])
            kink_dot.set_data([0.0], [0.0])
            title_txt.set_text("At $e = 0$: a sharp corner (kink)")
            caption.set_text("Even zoomed all the way in, $|e|$ never becomes a straight line at this point")

        # ── Act 3: reveal left/right tangent arrows ─────────────────────
        elif act == 3:
            ax.set_xlim(KINK_XL, KINK_XR); ax.set_ylim(KINK_YB, KINK_YT)
            curve_mae.set_data(e_full, mae_y)
            kink_dot.set_data([0.0], [0.0])
            run = KINK_HALF * 0.8 * s
            tangent_left.set_data(  [-run, 0.0], [run, 0.0])
            tangent_right.set_data( [0.0,  run], [0.0, run])
            title_txt.set_text(r"Left slope $= -1$,  right slope $= +1$")
            caption.set_text(r"The two one-sided slopes DISAGREE  →  ordinary derivative undefined!")

        # ── Act 4: stamp "NOT DIFFERENTIABLE" ──────────────────────────
        elif act == 4:
            ax.set_xlim(KINK_XL, KINK_XR); ax.set_ylim(KINK_YB, KINK_YT)
            curve_mae.set_data(e_full, mae_y)
            kink_dot.set_data([0.0], [0.0])
            run = KINK_HALF * 0.8
            tangent_left.set_data(  [-run, 0.0], [run, 0.0])
            tangent_right.set_data( [0.0,  run], [0.0, run])
            stamp_bad.set_text("⚠  NOT differentiable at $e = 0$")
            stamp_bad.set_alpha(min(1.0, t * 3))
            title_txt.set_text(r"Left slope $= -1$,  right slope $= +1$")
            caption.set_text("Standard backprop cannot compute this gradient → training fails near perfect predictions")

        # ── Act 5: wipe to MSE panel ────────────────────────────────────
        elif act == 5:
            xl = lerp(KINK_XL, WIDE_XL, s)
            xr = lerp(KINK_XR, WIDE_XR, s)
            yb = lerp(KINK_YB, WIDE_YB, s)
            yt = lerp(KINK_YT, WIDE_YT, s)
            ax.set_xlim(xl, xr); ax.set_ylim(yb, yt)
            mae_alpha = max(0.0, 1.0 - s * 2)
            mse_alpha = min(1.0, s * 2)
            curve_mae.set_data(e_full, mae_y)
            curve_mae.set_alpha(mae_alpha)
            curve_mse.set_data(e_full, mse_y)
            curve_mse.set_alpha(mse_alpha)
            title_txt.set_text("Now look at MSE  $L = e^2$")
            caption.set_text("A parabola — same minimum, but totally smooth")

        # ── Act 6: zoom into MSE minimum ────────────────────────────────
        elif act == 6:
            curve_mae.set_alpha(0.0)
            curve_mse.set_alpha(1.0)
            xl = lerp(WIDE_XL, MIN_XL, s)
            xr = lerp(WIDE_XR, MIN_XR, s)
            yb = lerp(WIDE_YB, MIN_YB, s)
            yt = lerp(WIDE_YT, MIN_YT, s)
            ax.set_xlim(xl, xr); ax.set_ylim(yb, yt)
            curve_mse.set_data(e_full, mse_y)
            title_txt.set_text("Zooming into the MSE minimum at $e = 0$ …")
            caption.set_text("Even at maximum zoom, the curve looks like a smooth parabola")

        # ── Act 7: hold at minimum ───────────────────────────────────────
        elif act == 7:
            curve_mae.set_alpha(0.0)
            curve_mse.set_alpha(1.0)
            ax.set_xlim(MIN_XL, MIN_XR); ax.set_ylim(MIN_YB, MIN_YT)
            curve_mse.set_data(e_full, mse_y)
            min_dot.set_data([0.0], [0.0])
            title_txt.set_text("At $e = 0$: the bottom of the bowl")
            caption.set_text("The parabola has a well-defined tangent here — completely flat")

        # ── Act 8: reveal flat tangent ──────────────────────────────────
        elif act == 8:
            curve_mae.set_alpha(0.0)
            curve_mse.set_alpha(1.0)
            ax.set_xlim(MIN_XL, MIN_XR); ax.set_ylim(MIN_YB, MIN_YT)
            curve_mse.set_data(e_full, mse_y)
            min_dot.set_data([0.0], [0.0])
            run = MIN_HALF * 0.85 * s
            tangent_flat.set_data([-run, run], [0.0, 0.0])
            title_txt.set_text(r"$(d/de)(e^2)\,|_{e=0} = 2 \times 0 = 0$")
            caption.set_text("Gradient is exactly zero → optimizer correctly stops here")

        # ── Act 9: stamp "GRADIENT = 0 ✓" ──────────────────────────────
        elif act == 9:
            curve_mae.set_alpha(0.0)
            curve_mse.set_alpha(1.0)
            ax.set_xlim(MIN_XL, MIN_XR); ax.set_ylim(MIN_YB, MIN_YT)
            curve_mse.set_data(e_full, mse_y)
            min_dot.set_data([0.0], [0.0])
            run = MIN_HALF * 0.85
            tangent_flat.set_data([-run, run], [0.0, 0.0])
            stamp_good.set_text(r"✓  Gradient $= 0$ — optimizer converges cleanly")
            stamp_good.set_alpha(min(1.0, t * 3))
            title_txt.set_text(r"$(d/de)(e^2)\,|_{e=0} = 0$  ← correct")
            caption.set_text("This is why MSE is gradient descent's best friend")

        # ── Act 10: zoom back out — show both curves ─────────────────────
        elif act == 10:
            curve_mae.set_alpha(1.0)
            curve_mse.set_alpha(1.0)
            xl = lerp(MIN_XL, WIDE_XL, s)
            xr = lerp(MIN_XR, WIDE_XR, s)
            yb = lerp(MIN_YB, WIDE_YB, s)
            yt = lerp(MIN_YT, BOTH_YT, s)
            ax.set_xlim(xl, xr); ax.set_ylim(yb, yt)
            curve_mae.set_data(e_full, mae_y)
            curve_mae.set_alpha(s)
            curve_mse.set_data(e_full, mse_y)
            run = MIN_HALF * 0.85 * (1 - s)
            if run > 1e-6:
                tangent_flat.set_data([-run, run], [0.0, 0.0])
            title_txt.set_text("MAE vs MSE — side by side")
            caption.set_text("Same task, different shape: one has a kink, one doesn't")

        # ── Act 11: Huber fades in ───────────────────────────────────────
        elif act == 11:
            ax.set_xlim(WIDE_XL, WIDE_XR); ax.set_ylim(WIDE_YB, BOTH_YT)
            curve_mae.set_data(e_full, mae_y);   curve_mae.set_alpha(0.45)
            curve_mse.set_data(e_full, mse_y);   curve_mse.set_alpha(0.45)
            curve_huber.set_data(e_full, huber_y)
            curve_huber.set_alpha(min(1.0, s * 2))
            title_txt.set_text(fr"Huber ($\delta={delta}$): MSE inside, MAE outside — $C^1$ smooth")
            caption.set_text("Huber inherits MSE's gradient=0 at zero AND MAE's capped gradient for outliers")

        return (curve_mae, curve_mse, curve_huber,
                tangent_left, tangent_right, tangent_flat,
                kink_dot, min_dot, stamp_bad, stamp_good, caption, title_txt)

    anim = FuncAnimation(fig, update, frames=TOTAL, interval=1000//FPS, blit=False)
    out  = HERE / "gradient_at_zero.gif"
    anim.save(str(out), writer=PillowWriter(fps=FPS))
    plt.close(fig)
    print(f"saved {out}")
    return out


# ══════════════════════════════════════════════════════════════════════════════
# Image 2 — huber_gradient_comparison.png
# Four-panel: the *gradient* (dL/de) of MAE, MSE, RMSE, Huber
# Shows: MAE's jump at 0, MSE's linear ramp, RMSE's jump, Huber's capped linear
# ══════════════════════════════════════════════════════════════════════════════
def make_gradient_comparison() -> Path:
    # error axis in $10k units  (3 = $30k)
    e     = np.linspace(-3.2, 3.2, 2000)
    delta = 0.8   # δ ~ $8k for visual clarity

    # gradients (dL/de):
    grad_mse   = 2 * e                                      # always defined
    grad_mae   = np.where(e > 0, +1.0,
                 np.where(e < 0, -1.0, np.nan))             # undefined at 0
    grad_rmse  = np.where(e > 0, +1.0,
                 np.where(e < 0, -1.0, np.nan))             # same kink
    grad_huber = np.where(np.abs(e) <= delta,
                          e,
                          delta * np.sign(e))               # capped at ±δ

    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), facecolor=BG)
    fig.suptitle(
        r"Gradients $\frac{dL}{de}$: which losses can gradient descent actually use?",
        fontsize=13, fontweight="bold", color=DARK, y=1.01
    )
    panels = [
        (axes[0,0], grad_mse,   GREEN,  "MSE  ($e^2$)",
         r"$\frac{d}{de}(e^2) = 2e$",
         "Defined everywhere. Zero at $e$=0.\nScales with error size (large errors get large steps).",
         True,  None),
        (axes[0,1], grad_mae,   RED,    "MAE  ($|e|$)",
         r"$\frac{d}{de}|e| = \mathrm{sign}(e)$",
         "Undefined at $e$=0 (kink).\nConstant ±1 everywhere else — no scaling.\n"
         "Cannot use with standard autodiff.",
         False, None),
        (axes[1,0], grad_rmse,  ORANGE, "RMSE  ($\\sqrt{e^2}$)",
         r"$\frac{d}{de}\sqrt{e^2} = \mathrm{sign}(e)$",
         "Same kink at $e$=0 as MAE.\nAlso loses MSE's proportional scaling.\n"
         "Never use as a training loss.",
         False, None),
        (axes[1,1], grad_huber, PURPLE, f"Huber  ($\\delta={delta}$)",
         "dL/de = e  when |e| ≤ δ\n" + r"dL/de = δ·sign($e$)  when |e|>δ",
         f"Smooth at $e$=0 (inner formula gives $e$=0).\n"
         f"Proportional for $|e|\\leq\\delta$, capped at $\\pm\\delta$ for outliers.",
         True,  delta),
    ]

    for ax, grad, color, title, formula, note, ok, delt in panels:
        # shade the undefined region for MAE/RMSE
        if not ok:
            ax.axvspan(-0.06, 0.06, color="#FFE8E8", alpha=0.9, zorder=0)
            ax.text(0.0, 0.0, "⊗", ha="center", va="center",
                    fontsize=14, color=RED, zorder=4,
                    bbox=dict(boxstyle="circle,pad=0.1", fc="white", ec=RED, lw=1.2))

        ax.plot(e, grad, color=color, lw=2.6, zorder=2)
        ax.axhline(0, color=DARK, lw=0.8, ls="--", alpha=0.4)
        ax.axvline(0, color=DARK, lw=0.8, ls="--", alpha=0.4)

        if delt is not None:
            ax.axhline(+delt, color=GREY, lw=1.0, ls=":", alpha=0.7)
            ax.axhline(-delt, color=GREY, lw=1.0, ls=":", alpha=0.7)
            ax.axvline(+delt, color=GREY, lw=1.0, ls=":", alpha=0.7)
            ax.axvline(-delt, color=GREY, lw=1.0, ls=":", alpha=0.7)
            ax.text(delt+0.08,  delt*0.82, rf"+$\delta$", color=GREY, fontsize=10)
            ax.text(delt+0.08, -delt*0.82, rf"$-\delta$", color=GREY, fontsize=10)

        ax.set_title(title, fontsize=11.5, fontweight="bold", color=DARK, pad=6)
        ax.set_xlabel(r"error  $e$  (\$10k units)", fontsize=9.5)
        ax.set_ylabel(r"gradient  $dL/de$", fontsize=9.5)
        ax.grid(alpha=0.18)
        ax.set_xlim(-3.2, 3.2)

        # formula box top-right
        ax.text(0.97, 0.97, formula, transform=ax.transAxes,
                ha="right", va="top", fontsize=9.0,
                color=color,
                bbox=dict(boxstyle="round,pad=0.3", fc="white",
                          ec=color, lw=0.9, alpha=0.9))

        # coloured verdict badge
        badge_color = GREEN if ok else RED
        badge_text  = "✓  use as training loss" if ok else "✗  do NOT use with autodiff"
        ax.text(0.03, 0.97, badge_text, transform=ax.transAxes,
                ha="left", va="top", fontsize=8.8, fontweight="bold",
                color=badge_color,
                bbox=dict(boxstyle="round,pad=0.3",
                          fc="#F0FFF0" if ok else "#FFF0F0",
                          ec=badge_color, lw=0.9))

        # note at bottom
        ax.text(0.5, -0.22, note, transform=ax.transAxes,
                ha="center", va="top", fontsize=8.5, color=DARK,
                style="italic", wrap=True)

    plt.subplots_adjust(hspace=0.52, wspace=0.28)
    out = HERE / "huber_gradient_comparison.png"
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"saved {out}")
    return out


if __name__ == "__main__":
    make_loss_curves()
    make_gradient_comparison()
    make_gradient_animation()
    print("All done.")
