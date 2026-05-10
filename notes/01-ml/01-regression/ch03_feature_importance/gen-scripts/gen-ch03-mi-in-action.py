"""
Generates ch03-mi-in-action.gif

Side-by-side animation: same x values, two y relationships animated in
parallel. Left panel starts as a diagonal line (linear), right panel starts
as a parabola (U-shaped). Both panels display the live Pearson ρ and MI score.

Key story:
  Left  (linear):    ρ stays high, MI stays high → ruler and detective agree
  Right (U-shaped):  ρ ≈ 0 throughout, MI is HIGH → ruler fails, detective wins

The animation morphs each scatter from sparse → full, then holds on the
final frame for 1 second before looping.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.stats import pearsonr
from sklearn.feature_selection import mutual_info_regression

DARK_BG  = "#1a1a2e"
CLR_LIN  = "#e94560"   # red for linear panel
CLR_QUAD = "#4ecdc4"   # teal for quadratic panel
CLR_RHO  = "#f7b731"   # amber for ρ label
CLR_MI   = "#a29bfe"   # purple for MI label
TEXT_CLR = "#e2e8f0"

N        = 250
DRAW_F   = 50   # frames to draw points
HOLD_F   = 30   # hold at end


def make_data():
    np.random.seed(42)
    x = np.linspace(-3, 3, N)
    y_lin  = 0.85 * x + np.random.randn(N) * 0.55   # linear
    y_quad = x ** 2  + np.random.randn(N) * 0.55    # U-shaped
    return x, y_lin, y_quad


def score(x, y):
    rho = pearsonr(x, y)[0]
    mi  = mutual_info_regression(x.reshape(-1, 1), y, random_state=42)[0]
    return rho, mi


def make_animation():
    x, y_lin, y_quad = make_data()

    rho_l, mi_l = score(x, y_lin)
    rho_q, mi_q = score(x, y_quad)

    fig, (ax_l, ax_q) = plt.subplots(1, 2, figsize=(10, 4.2),
                                      facecolor=DARK_BG)
    fig.suptitle("Mutual Information Catches What Pearson Misses",
                 color=TEXT_CLR, fontsize=12, fontweight="bold", y=1.02)

    for ax, title, clr, ylabel in [
        (ax_l, "Linear relationship", CLR_LIN, "y"),
        (ax_q, "U-shaped relationship (y = x²)", CLR_QUAD, ""),
    ]:
        ax.set_facecolor(DARK_BG)
        for spine in ["bottom", "left"]:
            ax.spines[spine].set_color(TEXT_CLR)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(colors=TEXT_CLR, labelsize=8)
        ax.set_xlabel("x", color=TEXT_CLR, fontsize=10)
        ax.set_ylabel(ylabel, color=TEXT_CLR, fontsize=10)
        ax.set_title(title, color=clr, fontsize=10, pad=6)

    ax_l.set_xlim(-3.3, 3.3); ax_l.set_ylim(-4.5, 4.5)
    ax_q.set_xlim(-3.3, 3.3); ax_q.set_ylim(-1.5, 11.5)

    scat_l = ax_l.scatter([], [], c=CLR_LIN,  alpha=0.55, s=14, zorder=3)
    scat_q = ax_q.scatter([], [], c=CLR_QUAD, alpha=0.55, s=14, zorder=3)

    def make_labels(ax, rho, mi, rho_clr=CLR_RHO, mi_clr=CLR_MI, alpha=0):
        t_rho = ax.text(0.04, 0.95, f"Pearson ρ = {rho:+.2f}",
                        transform=ax.transAxes, ha="left", va="top",
                        color=rho_clr, fontsize=11, fontweight="bold",
                        alpha=alpha)
        t_mi  = ax.text(0.04, 0.82, f"MI = {mi:.2f}",
                        transform=ax.transAxes, ha="left", va="top",
                        color=mi_clr, fontsize=11, fontweight="bold",
                        alpha=alpha)
        return t_rho, t_mi

    t_rho_l, t_mi_l = make_labels(ax_l, rho_l, mi_l)
    t_rho_q, t_mi_q = make_labels(ax_q, rho_q, mi_q)

    # Final annotation on the quadratic panel
    ann = ax_q.text(
        0.5, 0.45,
        "ρ ≈ 0 → ruler says: no signal\nMI high → detective: strong link!",
        transform=ax_q.transAxes, ha="center", va="center",
        color=TEXT_CLR, fontsize=8.5, alpha=0,
        bbox=dict(boxstyle="round,pad=0.4", fc="#16213e", ec=CLR_QUAD, lw=1.2)
    )

    total_frames = DRAW_F + HOLD_F

    def animate(frame):
        n_show = min(N, max(1, int((frame + 1) / DRAW_F * N)))
        progress = min(1.0, frame / (DRAW_F - 1))
        label_alpha = max(0.0, (frame - DRAW_F * 0.6) / (DRAW_F * 0.4))
        label_alpha = min(1.0, label_alpha)

        scat_l.set_offsets(np.column_stack([x[:n_show], y_lin[:n_show]]))
        scat_q.set_offsets(np.column_stack([x[:n_show], y_quad[:n_show]]))

        for t in [t_rho_l, t_mi_l, t_rho_q, t_mi_q]:
            t.set_alpha(label_alpha)

        ann_alpha = max(0.0, (frame - DRAW_F * 0.85) / (DRAW_F * 0.15))
        ann.set_alpha(min(1.0, ann_alpha))

        return scat_l, scat_q, t_rho_l, t_mi_l, t_rho_q, t_mi_q, ann

    ani = animation.FuncAnimation(
        fig, animate,
        frames=total_frames,
        interval=70,
        blit=True
    )

    out_path = "../img/ch03-mi-in-action.gif"
    ani.save(out_path, writer="pillow", fps=14, dpi=110)
    plt.close()
    print(f"Saved {out_path}")


if __name__ == "__main__":
    make_animation()
