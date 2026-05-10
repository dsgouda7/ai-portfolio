"""Generate Ch.18 multi-head attention animation GIF.

Output (saved next to this script):
    multihead_attention.gif — walks through scaled dot-product attention one
                              query at a time, then replays the same sequence
                              under three different heads to show that each
                              head learns a different relationship pattern.

The 8 "tokens" are the California Housing features used throughout Ch.18,
so the visual continues the chapter's running example.

Run:  python gen_ch18_animations.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import FancyArrowPatch

DARK = "#2C3E50"
GREY = "#BDC3C7"
DIM = "#ECF0F1"
BLUE = "#2E86C1"     # query token highlight
ORANGE = "#E67E22"   # attention beams
GREEN = "#27AE60"
RED = "#C0392B"

HERE = Path(__file__).parent

TOKENS = ["MedInc", "HouseAge", "AveRooms", "AveBedrms",
          "Pop",    "AveOccup",  "Lat",     "Long"]
T = len(TOKENS)


# ──────────────────────────────────────────────────────────────────────────────
# Three hand-crafted attention matrices — one per head — that illustrate
# different learned relationship patterns. Each row is a softmax distribution.
# ──────────────────────────────────────────────────────────────────────────────
def _row_softmax(M):
    M = np.asarray(M, dtype=float)
    M = M - M.max(axis=1, keepdims=True)
    E = np.exp(M)
    return E / E.sum(axis=1, keepdims=True)


def build_heads():
    # Head 1 — "feature-family" head: rooms↔bedrooms, pop↔occupancy, lat↔long
    logits1 = np.eye(T) * 2.0
    pairs = [(2, 3), (3, 2), (4, 5), (5, 4), (6, 7), (7, 6), (0, 1), (1, 0)]
    for i, j in pairs:
        logits1[i, j] += 1.6
    logits1 += 0.05 * np.ones((T, T))
    # Head 2 — "geography" head: every token attends mostly to Lat/Long (6, 7)
    logits2 = np.full((T, T), 0.0)
    logits2[:, 6] += 2.0
    logits2[:, 7] += 2.0
    logits2 += np.eye(T) * 0.6
    # Head 3 — "income" head: every token attends strongly to MedInc (0)
    logits3 = np.full((T, T), 0.0)
    logits3[:, 0] += 2.4
    logits3 += np.eye(T) * 0.5
    return [
        ("Head 1 — feature family\n(rooms↔bedrooms, pop↔occup, lat↔long)",
         _row_softmax(logits1)),
        ("Head 2 — geography\n(everyone attends to Lat & Long)",
         _row_softmax(logits2)),
        ("Head 3 — income anchor\n(everyone attends to MedInc)",
         _row_softmax(logits3)),
    ]


# ──────────────────────────────────────────────────────────────────────────────
# Layout helpers
# ──────────────────────────────────────────────────────────────────────────────
def token_positions(y=0.0, width=7.0):
    xs = np.linspace(-width / 2, width / 2, T)
    return np.column_stack([xs, np.full(T, y)])


def animate_attention() -> Path:
    heads = build_heads()

    fig = plt.figure(figsize=(11.5, 6.2), facecolor="white")
    gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1.0], wspace=0.22)
    ax_tok = fig.add_subplot(gs[0, 0])
    ax_mat = fig.add_subplot(gs[0, 1])

    fig.suptitle(
        "Scaled dot-product attention, one query at a time — replayed across 3 heads",
        fontsize=12, fontweight="bold", color=DARK,
    )

    # ── left panel: tokens at top (queries) and bottom (keys) + beams ───────
    pos_q = token_positions(y=1.0, width=8.5)
    pos_k = token_positions(y=-1.0, width=8.5)

    ax_tok.set_xlim(-5.2, 5.2)
    ax_tok.set_ylim(-2.5, 2.2)
    ax_tok.set_axis_off()
    ax_tok.text(0, 2.0, "Queries  (who is asking?)",
                ha="center", fontsize=10, color=DARK, fontweight="bold")
    ax_tok.text(0, -1.9, "Keys / Values  (who answers?)",
                ha="center", fontsize=10, color=DARK, fontweight="bold")

    # draw token boxes
    q_boxes, q_labels = [], []
    k_boxes, k_labels = [], []
    for i in range(T):
        qb, = ax_tok.plot([pos_q[i, 0]], [pos_q[i, 1]], "s",
                          ms=28, markerfacecolor="white",
                          markeredgecolor=DARK, markeredgewidth=1.2, zorder=3)
        kb, = ax_tok.plot([pos_k[i, 0]], [pos_k[i, 1]], "s",
                          ms=28, markerfacecolor="white",
                          markeredgecolor=DARK, markeredgewidth=1.2, zorder=3)
        q_boxes.append(qb); k_boxes.append(kb)
        ql = ax_tok.text(pos_q[i, 0], pos_q[i, 1] + 0.42, TOKENS[i],
                         ha="center", fontsize=8.2, color=DARK,
                         family="monospace")
        kl = ax_tok.text(pos_k[i, 0], pos_k[i, 1] - 0.42, TOKENS[i],
                         ha="center", fontsize=8.2, color=DARK,
                         family="monospace")
        q_labels.append(ql); k_labels.append(kl)

    # pre-build one beam (FancyArrowPatch) per (q, k) pair; we'll update
    # thickness/alpha each frame.
    beams = {}
    for qi in range(T):
        for kj in range(T):
            arr = FancyArrowPatch(
                (pos_q[qi, 0], pos_q[qi, 1] - 0.18),
                (pos_k[kj, 0], pos_k[kj, 1] + 0.18),
                arrowstyle="-", color=ORANGE, alpha=0.0, lw=0.5,
                connectionstyle="arc3,rad=0.08", zorder=1,
            )
            ax_tok.add_patch(arr)
            beams[(qi, kj)] = arr

    # weight bars on the keys (how much total attention mass hit this key)
    bar_artists = []

    banner = ax_tok.text(0, -2.35, "", ha="center", fontsize=10.5,
                         color=DARK, fontweight="bold")

    # ── right panel: full attention matrix, filled row by row ──────────────
    im = ax_mat.imshow(np.zeros((T, T)), vmin=0.0, vmax=0.7,
                       cmap="Oranges", aspect="equal")
    ax_mat.set_xticks(range(T)); ax_mat.set_yticks(range(T))
    ax_mat.set_xticklabels(TOKENS, rotation=45, ha="right", fontsize=7.5)
    ax_mat.set_yticklabels(TOKENS, fontsize=7.5)
    ax_mat.set_xlabel("Key position", fontsize=9)
    ax_mat.set_ylabel("Query position", fontsize=9)
    row_highlight = ax_mat.add_patch(plt.Rectangle(
        (-0.5, -0.5), T, 1.0, fill=False, edgecolor=BLUE,
        linewidth=2.2, zorder=5, visible=False))
    head_title = ax_mat.set_title("", fontsize=10, color=DARK)

    # ── frame schedule ──────────────────────────────────────────────────────
    # For each head:
    #   - one frame per query (T frames), building the matrix row by row
    #   - 4 "hold" frames at the end with full matrix + all-keys view
    frames = []
    for h_idx, (title, A) in enumerate(heads):
        for qi in range(T):
            frames.append(("build", h_idx, qi))
        for _ in range(4):
            frames.append(("hold", h_idx, None))

    def reset_tokens():
        for b in q_boxes + k_boxes:
            b.set_markerfacecolor("white")
            b.set_markeredgecolor(DARK)
            b.set_markeredgewidth(1.2)
        for arr in beams.values():
            arr.set_alpha(0.0)
            arr.set_linewidth(0.5)

    def render_head_matrix(h_idx, up_to_row):
        _, A = heads[h_idx]
        shown = np.zeros_like(A)
        if up_to_row is None:
            shown = A
        elif up_to_row >= 0:
            shown[: up_to_row + 1] = A[: up_to_row + 1]
        im.set_data(shown)
        head_title.set_text(heads[h_idx][0])

    def update(k: int):
        kind, h_idx, qi = frames[k]
        _, A = heads[h_idx]
        reset_tokens()

        if kind == "build":
            # highlight the current query token (blue)
            q_boxes[qi].set_markerfacecolor(BLUE)
            q_boxes[qi].set_markeredgecolor(BLUE)
            q_boxes[qi].set_markeredgewidth(2.0)
            # draw beams to every key with width ∝ attention weight
            row = A[qi]
            for kj in range(T):
                w = float(row[kj])
                arr = beams[(qi, kj)]
                arr.set_linewidth(0.3 + 6.0 * w)
                arr.set_alpha(min(0.15 + w * 1.6, 0.95))
                # glow the most-attended keys
                if w > 0.18:
                    k_boxes[kj].set_markerfacecolor(ORANGE)
                    k_boxes[kj].set_markeredgecolor(ORANGE)
                    k_boxes[kj].set_markeredgewidth(2.0)
            banner.set_text(
                f"query = '{TOKENS[qi]}'   "
                f"softmax(q·K/√d) → distribution over keys (row sums to 1)"
            )
            render_head_matrix(h_idx, qi)
            row_highlight.set_visible(True)
            row_highlight.set_xy((-0.5, qi - 0.5))

        else:   # hold: full matrix visible, all tokens calm
            for b in q_boxes + k_boxes:
                b.set_markeredgecolor(DARK)
            banner.set_text(
                "full attention matrix for this head — every row is its own softmax"
            )
            render_head_matrix(h_idx, None)
            row_highlight.set_visible(False)

        return []

    anim = FuncAnimation(
        fig, update, frames=len(frames), interval=520,
        blit=False, repeat=True,
    )
    out = HERE / "multihead_attention.gif"
    anim.save(out, writer=PillowWriter(fps=2))
    plt.close(fig)
    return out


if __name__ == "__main__":
    p = animate_attention()
    print(f"wrote {p}")
