"""
Generates ch03-pearson-in-action.gif

Shows three scatter plots animating in sequence:
  1. MedInc vs MedHouseVal  — strong positive (ρ = 0.69)
  2. HouseAge vs MedHouseVal — near-zero (ρ ≈ 0)
  3. AveOccup vs MedHouseVal — weak negative (ρ ≈ -0.02)
Each panel draws the best-fit line after the points settle,
then labels the ρ score. Teaches: high ρ → tight line band;
low ρ → cloud; negative ρ → downward slope.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression

DARK_BG   = "#1a1a2e"
ACCENT1   = "#e94560"   # red – strong positive
ACCENT2   = "#4ecdc4"   # teal – near-zero
ACCENT3   = "#f7b731"   # amber – weak negative
LINE_CLR  = "#e2e8f0"
TEXT_CLR  = "#e2e8f0"
FRAMES_PER_PANEL = 40   # frames to draw points
LINE_FRAMES      = 15   # frames to fade in the line
HOLD_FRAMES      = 20   # frames to hold after label appears
TOTAL_PER_PANEL  = FRAMES_PER_PANEL + LINE_FRAMES + HOLD_FRAMES

np.random.seed(42)


def load_data():
    housing = fetch_california_housing(as_frame=True)
    df = housing.frame.sample(400, random_state=42)
    pairs = [
        ("MedInc",    "MedHouseVal", ACCENT1, "Income (×$10k)"),
        ("HouseAge",  "MedHouseVal", ACCENT2, "House Age (years)"),
        ("AveOccup",  "MedHouseVal", ACCENT3, "Avg Occupancy"),
    ]
    return df, pairs


def fit_line(x, y):
    reg = LinearRegression().fit(x.values.reshape(-1, 1), y.values)
    x_range = np.linspace(x.min(), x.max(), 100)
    y_range = reg.predict(x_range.reshape(-1, 1))
    return x_range, y_range


def make_animation():
    df, pairs = load_data()
    n_pts = len(df)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), facecolor=DARK_BG)
    fig.suptitle("Pearson Correlation in Action", color=TEXT_CLR,
                 fontsize=13, fontweight="bold", y=1.02)
    plt.subplots_adjust(wspace=0.35)

    panel_artists = []
    rho_vals = []

    for ax, (feat, target, color, xlabel) in zip(axes, pairs):
        ax.set_facecolor(DARK_BG)
        for spine in ["bottom", "left"]:
            ax.spines[spine].set_color(TEXT_CLR)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(colors=TEXT_CLR, labelsize=8)
        ax.set_xlabel(xlabel, color=TEXT_CLR, fontsize=9)
        ax.set_ylabel("House Value (×$100k)", color=TEXT_CLR, fontsize=9)

        x_data = df[feat]
        y_data = df[target]
        rho = np.corrcoef(x_data, y_data)[0, 1]
        rho_vals.append(rho)

        x_range, y_range = fit_line(x_data, y_data)

        # Static full scatter (invisible at start)
        scat = ax.scatter(x_data, y_data, c=color, alpha=0, s=12, zorder=3)
        # Animated scatter (we'll update offsets frame by frame)
        scat_anim = ax.scatter([], [], c=color, alpha=0.55, s=12, zorder=3)
        # Regression line
        (line,) = ax.plot(x_range, y_range, color=LINE_CLR,
                          lw=1.8, alpha=0, zorder=4)
        # ρ label
        label = ax.text(
            0.97, 0.05,
            f"ρ = {rho:+.2f}",
            transform=ax.transAxes,
            ha="right", va="bottom",
            color=color, fontsize=12, fontweight="bold", alpha=0
        )

        x_rng = x_data.max() - x_data.min()
        ax.set_xlim(x_data.min() - 0.05 * x_rng,
                    x_data.max() + 0.05 * x_rng)
        ax.set_ylim(y_data.min() - 0.1, y_data.max() + 0.1)

        panel_artists.append((scat_anim, line, label, x_data, y_data))

    n_panels = len(pairs)
    total_frames = TOTAL_PER_PANEL * n_panels + HOLD_FRAMES

    def animate(frame):
        for panel_idx, (scat_anim, line, label, x_data, y_data) in enumerate(panel_artists):
            panel_start = panel_idx * TOTAL_PER_PANEL
            local = frame - panel_start

            if local < 0:
                # Panel hasn't started yet
                scat_anim.set_offsets(np.empty((0, 2)))
                line.set_alpha(0)
                label.set_alpha(0)
            elif local < FRAMES_PER_PANEL:
                # Drawing points
                n_show = max(1, int((local + 1) / FRAMES_PER_PANEL * n_pts))
                offsets = np.column_stack([x_data.values[:n_show],
                                           y_data.values[:n_show]])
                scat_anim.set_offsets(offsets)
                line.set_alpha(0)
                label.set_alpha(0)
            elif local < FRAMES_PER_PANEL + LINE_FRAMES:
                # All points visible; fade in line
                scat_anim.set_offsets(np.column_stack([x_data.values,
                                                        y_data.values]))
                alpha = (local - FRAMES_PER_PANEL) / LINE_FRAMES
                line.set_alpha(alpha)
                label.set_alpha(0)
            else:
                # Line fully drawn; show label
                scat_anim.set_offsets(np.column_stack([x_data.values,
                                                        y_data.values]))
                line.set_alpha(1.0)
                label.set_alpha(1.0)

        return [a[0] for a in panel_artists] + \
               [a[1] for a in panel_artists] + \
               [a[2] for a in panel_artists]

    ani = animation.FuncAnimation(
        fig, animate,
        frames=total_frames,
        interval=60,
        blit=True
    )

    out_path = "../img/ch03-pearson-in-action.gif"
    ani.save(out_path, writer="pillow", fps=16, dpi=110)
    plt.close()
    print(f"Saved {out_path}")


if __name__ == "__main__":
    make_animation()
