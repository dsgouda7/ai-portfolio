import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing

DARK_BG = "#1a1a2e"
ACCENT1 = "#e94560"
ACCENT2 = "#0f3460"
TEXT_COLOR = "#e2e8f0"


def gen():
    data = fetch_california_housing()
    pop = data.data[:, 4]  # Population column
    log_pop = np.log1p(pop)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), facecolor=DARK_BG)

    for ax in axes:
        ax.set_facecolor(DARK_BG)
        ax.tick_params(colors=TEXT_COLOR)
        ax.spines['bottom'].set_color(TEXT_COLOR)
        ax.spines['left'].set_color(TEXT_COLOR)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    axes[0].hist(pop, bins=60, color=ACCENT1, alpha=0.8)
    axes[0].set_title("Population (raw)\nRight-skewed", color=TEXT_COLOR)
    axes[0].set_xlabel("Value", color=TEXT_COLOR)
    axes[0].set_ylabel("Count", color=TEXT_COLOR)

    axes[1].hist(log_pop, bins=60, color=ACCENT2, alpha=0.8, edgecolor=TEXT_COLOR, linewidth=0.3)
    axes[1].set_title("log1p(Population)\nMore symmetric", color=TEXT_COLOR)
    axes[1].set_xlabel("Value", color=TEXT_COLOR)

    plt.suptitle("Log Transform Compresses Right-Tailed Features", color=TEXT_COLOR, fontsize=12)
    plt.tight_layout()
    plt.savefig("../img/ch03-log-transform.png", dpi=120, facecolor=DARK_BG)
    plt.close()
    print("Saved img/ch03-log-transform.png")


if __name__ == "__main__":
    gen()
