"""Generate ch17 (bridge chapter) animations.

Outputs (saved next to this script):
    hard_vs_soft_lookup.gif      — the single most important animation:
                                   Python dict (exact match → one value) vs
                                   attention (softmax weights over all keys →
                                   weighted blend of all values).
    softmax_temperature.gif      — same score vector, tau swept from 0.1
                                   (one-hot / argmax) up to 10 (uniform).
    rnn_vs_attention.gif         — sequential token-by-token RNN vs all
                                   positions attending to all positions in
                                   a single parallel step.
    attention_matrix_build.gif   — 8×8 self-attention matrix filling in row
                                   by row on the California Housing features.
    permutation_equivariance.png — shuffle the input, the output shuffles
                                   identically — motivates positional encoding.

Run:  python gen_ch17_animations.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

# ── house palette (matches ch01 / ch18) ─────────────────────────────────────
DARK   = "#2C3E50"
GREY   = "#95A5A6"
DIM    = "#ECF0F1"
BLUE   = "#2E86C1"
ORANGE = "#E67E22"
GREEN  = "#27AE60"
RED    = "#C0392B"
PURPLE = "#8E44AD"

HERE = Path(__file__).parent


def softmax(x, tau=1.0, axis=-1):
    x = np.asarray(x, dtype=float) / tau
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


# ═════════════════════════════════════════════════════════════════════════════
# 1.  Hard vs Soft Lookup
# ═════════════════════════════════════════════════════════════════════════════
def animate_hard_vs_soft() -> Path:
    fruits   = ["apple", "banana", "cherry"]
    prices   = np.array([0.80, 0.30, 2.50])
    # 3-D "embeddings" for each fruit (simple one-hot-ish anchors)
    emb      = np.array([
        [0.9, 0.1, 0.0],   # apple
        [0.1, 0.9, 0.0],   # banana
        [0.0, 0.0, 1.0],   # cherry
    ])
    # Query slides from "exact apple" → "something between apple & banana".
    # The hard dict can only match exactly one of them; the soft lookup blends.
    query_end = np.array([0.7, 0.3, 0.0])  # apple-ish but not exactly apple

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12.0, 5.6), facecolor="white")
    fig.suptitle("Hard lookup (Python dict)   vs   Soft lookup (attention)",
                 fontsize=13, fontweight="bold", color=DARK)

    # ── LEFT PANEL : hard lookup ───────────────────────────────────────────
    ax_l.set_xlim(0, 1); ax_l.set_ylim(0, 1); ax_l.axis("off")
    ax_l.set_title("prices['apple'] → exactly one value",
                   fontsize=11, color=DARK)

    # draw the dict as three rows
    row_y = [0.72, 0.52, 0.32]
    key_patches, val_patches = [], []
    for name, y, p in zip(fruits, row_y, prices):
        kp = FancyBboxPatch((0.08, y - 0.06), 0.28, 0.12,
                            boxstyle="round,pad=0.02",
                            facecolor=DIM, edgecolor=GREY, linewidth=1.2)
        vp = FancyBboxPatch((0.56, y - 0.06), 0.28, 0.12,
                            boxstyle="round,pad=0.02",
                            facecolor=DIM, edgecolor=GREY, linewidth=1.2)
        ax_l.add_patch(kp); ax_l.add_patch(vp)
        ax_l.text(0.22, y, f'"{name}"', ha="center", va="center",
                  fontsize=11, color=DARK, fontweight="bold")
        ax_l.text(0.70, y, f"${p:.2f}", ha="center", va="center",
                  fontsize=11, color=DARK)
        ax_l.annotate("", xy=(0.55, y), xytext=(0.37, y),
                      arrowprops=dict(arrowstyle="->", color=GREY, lw=1.2))
        key_patches.append(kp); val_patches.append(vp)

    # query label at top
    ax_l.text(0.5, 0.92, 'query: "apple"', ha="center", va="center",
              fontsize=12, color=BLUE, fontweight="bold")
    # output box at bottom
    out_l_bg = FancyBboxPatch((0.30, 0.08), 0.40, 0.14,
                              boxstyle="round,pad=0.02",
                              facecolor="white", edgecolor=DARK, linewidth=1.6)
    ax_l.add_patch(out_l_bg)
    out_l_txt = ax_l.text(0.5, 0.15, "output = ?", ha="center", va="center",
                          fontsize=12, color=DARK, fontweight="bold")

    # ── RIGHT PANEL : soft lookup ──────────────────────────────────────────
    ax_r.set_xlim(0, 1); ax_r.set_ylim(0, 1); ax_r.axis("off")
    ax_r.set_title("softmax( q · kᵢ ) · vᵢ  →  weighted blend of every value",
                   fontsize=11, color=DARK)

    for name, y, p in zip(fruits, row_y, prices):
        ax_r.add_patch(FancyBboxPatch((0.08, y - 0.06), 0.28, 0.12,
                                      boxstyle="round,pad=0.02",
                                      facecolor=DIM, edgecolor=GREY, linewidth=1.2))
        ax_r.add_patch(FancyBboxPatch((0.64, y - 0.06), 0.22, 0.12,
                                      boxstyle="round,pad=0.02",
                                      facecolor=DIM, edgecolor=GREY, linewidth=1.2))
        ax_r.text(0.22, y, f"key: {name}", ha="center", va="center",
                  fontsize=10.5, color=DARK, fontweight="bold")
        ax_r.text(0.75, y, f"${p:.2f}", ha="center", va="center",
                  fontsize=10.5, color=DARK)

    # query label
    q_txt = ax_r.text(0.5, 0.92, "query: q", ha="center", va="center",
                      fontsize=12, color=BLUE, fontweight="bold")

    # weight-arrow placeholders (width encodes softmax weight)
    arrow_artists = []
    weight_texts = []
    for y in row_y:
        arr = ax_r.annotate("", xy=(0.62, y), xytext=(0.38, y),
                            arrowprops=dict(arrowstyle="-|>",
                                            color=ORANGE, lw=1.0,
                                            shrinkA=0, shrinkB=0))
        arrow_artists.append(arr)
        wt = ax_r.text(0.50, y + 0.05, "", ha="center", va="center",
                       fontsize=9.5, color=ORANGE, fontweight="bold")
        weight_texts.append(wt)

    out_r_bg = FancyBboxPatch((0.30, 0.08), 0.40, 0.14,
                              boxstyle="round,pad=0.02",
                              facecolor="white", edgecolor=DARK, linewidth=1.6)
    ax_r.add_patch(out_r_bg)
    out_r_txt = ax_r.text(0.5, 0.15, "output = ?", ha="center", va="center",
                          fontsize=12, color=DARK, fontweight="bold")

    # footer comparison caption
    fig.text(0.5, 0.02,
             "Same data. Hard lookup returns one value; attention returns a blend weighted by similarity.",
             ha="center", fontsize=10, color=GREY, style="italic")

    # ── frames ──────────────────────────────────────────────────────────────
    HOLD_START = 8
    FADE       = 18      # slide query from apple → apple-ish
    HOLD_END   = 18
    TOTAL      = HOLD_START + FADE + HOLD_END

    # Hard-lookup animation: just highlight the apple row + arrow + output.
    # Left side "animation" is really just a reveal on frame 1.
    def update(frame):
        # interpolation factor 0 → 1 during the FADE segment
        if frame < HOLD_START:
            t = 0.0
        elif frame < HOLD_START + FADE:
            t = (frame - HOLD_START) / FADE
        else:
            t = 1.0

        # LEFT: highlight apple on frame 1 and stick.
        for i, (kp, vp) in enumerate(zip(key_patches, val_patches)):
            if i == 0:
                kp.set_facecolor("#FDEBD0"); kp.set_edgecolor(ORANGE)
                vp.set_facecolor("#FDEBD0"); vp.set_edgecolor(ORANGE)
            else:
                kp.set_facecolor(DIM); kp.set_edgecolor(GREY)
                vp.set_facecolor(DIM); vp.set_edgecolor(GREY)
        out_l_txt.set_text("output = $0.80")

        # RIGHT: query slides from one-hot-apple to query_end
        q_start = np.array([1.0, 0.0, 0.0])
        q       = (1 - t) * q_start + t * query_end
        # scores = q · keys (rows of emb)
        scores  = emb @ q
        weights = softmax(scores, tau=0.35)  # low tau → concentrated when t small
        out_price = float(weights @ prices)

        q_txt.set_text(f"query: q = [{q[0]:.2f}, {q[1]:.2f}, {q[2]:.2f}]")
        for i, (arr, wt_txt, w) in enumerate(zip(arrow_artists, weight_texts, weights)):
            # arrow line width encodes attention weight; alpha too
            lw    = 0.5 + 6.0 * w
            alpha = 0.15 + 0.85 * w
            arr.arrow_patch.set_linewidth(lw)
            arr.arrow_patch.set_alpha(alpha)
            wt_txt.set_text(f"w={w:.2f}")
            wt_txt.set_alpha(0.3 + 0.7 * w)

        out_r_txt.set_text(f"output = ${out_price:.2f}   (weighted blend)")
        return []

    anim = FuncAnimation(fig, update, frames=TOTAL, interval=110, blit=False)
    out = HERE / "hard_vs_soft_lookup.gif"
    anim.save(out, writer=PillowWriter(fps=9))
    plt.close(fig)
    return out


# ═════════════════════════════════════════════════════════════════════════════
# 2.  Softmax temperature sweep
# ═════════════════════════════════════════════════════════════════════════════
def animate_softmax_temperature() -> Path:
    scores = np.array([2.0, 1.0, 0.1, -0.5, -1.2])
    taus   = np.concatenate([
        np.linspace(0.10, 1.00, 20),
        np.linspace(1.00, 10.0, 20),
    ])
    n = len(scores)

    fig, ax = plt.subplots(figsize=(8.5, 5.0), facecolor="white")
    fig.suptitle("Softmax temperature τ — same scores, different distributions",
                 fontsize=12, fontweight="bold", color=DARK)
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(range(n))
    ax.set_xticklabels([f"s{i+1}={s:+.1f}" for i, s in enumerate(scores)],
                       fontsize=10, color=DARK)
    ax.set_ylabel("softmax weight", fontsize=10.5, color=DARK)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=GREY, alpha=0.3, linewidth=0.6)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    bars = ax.bar(range(n), softmax(scores, tau=1.0),
                  color=BLUE, edgecolor=DARK, linewidth=0.8)
    tau_text = ax.text(0.98, 0.92, "", transform=ax.transAxes,
                       ha="right", va="top", fontsize=13, color=ORANGE,
                       fontweight="bold")
    caption = ax.text(0.02, 0.92, "", transform=ax.transAxes,
                      ha="left", va="top", fontsize=10, color=DARK,
                      style="italic")

    def update(i):
        tau = taus[i]
        p   = softmax(scores, tau=tau)
        for b, h in zip(bars, p):
            b.set_height(h)
        tau_text.set_text(f"τ = {tau:.2f}")
        if tau < 0.25:
            caption.set_text("τ → 0 : collapses to one-hot (hard argmax; no gradient)")
        elif tau < 1.5:
            caption.set_text("τ ≈ 1 : standard softmax — the useful regime")
        else:
            caption.set_text("τ large : distribution flattens — no preference, no information")
        return list(bars) + [tau_text, caption]

    anim = FuncAnimation(fig, update, frames=len(taus), interval=120, blit=False)
    out = HERE / "softmax_temperature.gif"
    anim.save(out, writer=PillowWriter(fps=9))
    plt.close(fig)
    return out


# ═════════════════════════════════════════════════════════════════════════════
# 3.  RNN-sequential vs attention-parallel
# ═════════════════════════════════════════════════════════════════════════════
def animate_rnn_vs_attention() -> Path:
    words = ["the", "cat", "sat", "on", "the", "mat"]
    T = len(words)

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(10.5, 7.2),
                                         facecolor="white",
                                         gridspec_kw={"height_ratios": [1, 1.1],
                                                      "hspace": 0.55})
    fig.suptitle("Ch.8 RNN — sequential (T steps)     vs     Ch.18 Attention — parallel (1 step)",
                 fontsize=12, fontweight="bold", color=DARK)

    # --- RNN panel -----------------------------------------------------------
    ax_top.set_xlim(0, T + 1); ax_top.set_ylim(-1.8, 1.8); ax_top.axis("off")

    token_x = np.arange(1, T + 1)
    # token boxes (top row)
    token_patches = []
    for x, w in zip(token_x, words):
        p = FancyBboxPatch((x - 0.35, 0.55), 0.70, 0.55,
                           boxstyle="round,pad=0.02",
                           facecolor=DIM, edgecolor=GREY, linewidth=1.0)
        ax_top.add_patch(p)
        ax_top.text(x, 0.82, w, ha="center", va="center",
                    fontsize=10, color=DARK, fontweight="bold")
        token_patches.append(p)
    # hidden-state boxes (bottom row)
    h_patches, h_texts = [], []
    for x in token_x:
        p = FancyBboxPatch((x - 0.35, -1.10), 0.70, 0.55,
                           boxstyle="round,pad=0.02",
                           facecolor="white", edgecolor=GREY, linewidth=1.0)
        ax_top.add_patch(p)
        t = ax_top.text(x, -0.83, "", ha="center", va="center",
                        fontsize=10, color=GREY)
        h_patches.append(p); h_texts.append(t)
    # input arrows (token -> hidden)
    in_arrows = []
    for x in token_x:
        arr = ax_top.annotate("", xy=(x, -0.55), xytext=(x, 0.55),
                              arrowprops=dict(arrowstyle="-|>",
                                              color=GREY, lw=1.0, alpha=0.3))
        in_arrows.append(arr)
    # horizontal recurrent arrows (h_{t-1} -> h_t)
    rec_arrows = []
    for i in range(T - 1):
        arr = ax_top.annotate("", xy=(token_x[i+1] - 0.35, -0.83),
                              xytext=(token_x[i] + 0.35, -0.83),
                              arrowprops=dict(arrowstyle="-|>",
                                              color=GREY, lw=1.0, alpha=0.3))
        rec_arrows.append(arr)
    step_label_top = ax_top.text(0.5, 1.55, "",
                                 ha="center", va="center",
                                 fontsize=11, color=BLUE, fontweight="bold")

    # --- Attention panel -----------------------------------------------------
    ax_bot.set_xlim(0, T + 1); ax_bot.set_ylim(-1.8, 1.8); ax_bot.axis("off")
    for x, w in zip(token_x, words):
        p = FancyBboxPatch((x - 0.35, 0.55), 0.70, 0.55,
                           boxstyle="round,pad=0.02",
                           facecolor=DIM, edgecolor=GREY, linewidth=1.0)
        ax_bot.add_patch(p)
        ax_bot.text(x, 0.82, w, ha="center", va="center",
                    fontsize=10, color=DARK, fontweight="bold")
    out_patches = []
    for x in token_x:
        p = FancyBboxPatch((x - 0.35, -1.10), 0.70, 0.55,
                           boxstyle="round,pad=0.02",
                           facecolor="white", edgecolor=GREY, linewidth=1.0)
        ax_bot.add_patch(p)
        ax_bot.text(x, -0.83, "", ha="center", va="center",
                    fontsize=10, color=GREY)
        out_patches.append(p)
    # every-to-every attention beams
    att_beams = []
    for i in range(T):
        for j in range(T):
            arr = ax_bot.annotate("",
                                  xy=(token_x[j], -0.55),
                                  xytext=(token_x[i], 0.55),
                                  arrowprops=dict(arrowstyle="-",
                                                  color=ORANGE, lw=0.8, alpha=0.0))
            att_beams.append(arr)
    step_label_bot = ax_bot.text(0.5, 1.55, "",
                                 ha="center", va="center",
                                 fontsize=11, color=ORANGE, fontweight="bold")

    HOLD = 6
    # RNN takes T steps; attention takes 1. To show this visually,
    # we give each RNN step HOLD frames and the attention a single
    # "glow" phase of the same total duration.
    TOTAL = HOLD * (T + 1)

    def update(frame):
        # RNN: step through tokens
        rnn_step = min(frame // HOLD, T)
        for i, (tp, hp, ht, ia) in enumerate(
                zip(token_patches, h_patches, h_texts, in_arrows)):
            if i < rnn_step:
                tp.set_facecolor("#E8F6F3"); tp.set_edgecolor(GREEN)
                hp.set_facecolor("#E8F6F3"); hp.set_edgecolor(GREEN)
                ht.set_text(f"h{i+1}"); ht.set_color(DARK)
                ia.arrow_patch.set_alpha(1.0); ia.arrow_patch.set_color(GREEN)
            elif i == rnn_step and rnn_step < T:
                tp.set_facecolor("#FDEBD0"); tp.set_edgecolor(ORANGE)
                hp.set_facecolor("#FDEBD0"); hp.set_edgecolor(ORANGE)
                ht.set_text("…"); ht.set_color(ORANGE)
                ia.arrow_patch.set_alpha(1.0); ia.arrow_patch.set_color(ORANGE)
            else:
                tp.set_facecolor(DIM); tp.set_edgecolor(GREY)
                hp.set_facecolor("white"); hp.set_edgecolor(GREY)
                ht.set_text(""); ht.set_color(GREY)
                ia.arrow_patch.set_alpha(0.2); ia.arrow_patch.set_color(GREY)
        for i, ra in enumerate(rec_arrows):
            if i + 1 <= rnn_step:
                ra.arrow_patch.set_alpha(1.0); ra.arrow_patch.set_color(GREEN)
            else:
                ra.arrow_patch.set_alpha(0.2); ra.arrow_patch.set_color(GREY)
        step_label_top.set_text(f"RNN step {min(rnn_step, T)} / {T}"
                                if rnn_step < T else f"RNN done — {T} steps")

        # Attention: single big parallel flash.
        # Beam intensity ramps up in the first block and holds.
        att_phase = min(frame / (HOLD * 2), 1.0)
        for a in att_beams:
            a.arrow_patch.set_alpha(0.10 + 0.45 * att_phase)
            a.arrow_patch.set_linewidth(0.4 + 0.9 * att_phase)
        for p in out_patches:
            if att_phase >= 1.0:
                p.set_facecolor("#FDEBD0"); p.set_edgecolor(ORANGE)
            else:
                p.set_facecolor("white"); p.set_edgecolor(GREY)
        step_label_bot.set_text(
            "Attention: step 1 / 1 (all positions updated in parallel)"
            if att_phase >= 1.0 else "Attention step 1 / 1…"
        )
        return []

    anim = FuncAnimation(fig, update, frames=TOTAL, interval=130, blit=False)
    out = HERE / "rnn_vs_attention.gif"
    anim.save(out, writer=PillowWriter(fps=9))
    plt.close(fig)
    return out


# ═════════════════════════════════════════════════════════════════════════════
# 4.  Self-attention matrix filling in
# ═════════════════════════════════════════════════════════════════════════════
def animate_attention_matrix_build() -> Path:
    rng = np.random.default_rng(42)
    T, d = 8, 4
    X = rng.normal(size=(T, d))
    scores = X @ X.T / np.sqrt(d)
    A = softmax(scores, tau=1.0, axis=-1)   # (T, T), rows sum to 1

    names = ["MedInc", "HouseAge", "AveRooms", "AveBedrms",
             "Pop",    "AveOccup",  "Lat",     "Long"]

    fig, ax = plt.subplots(figsize=(7.5, 6.3), facecolor="white")
    fig.suptitle("Self-attention matrix  A = softmax(QKᵀ/√d)  —  filling in row by row",
                 fontsize=11.5, fontweight="bold", color=DARK)

    ax.set_xlim(-0.5, T - 0.5); ax.set_ylim(T - 0.5, -0.5)
    ax.set_xticks(range(T)); ax.set_xticklabels(names, rotation=40,
                                                ha="right", fontsize=9, color=DARK)
    ax.set_yticks(range(T)); ax.set_yticklabels(names, fontsize=9, color=DARK)
    ax.set_xlabel("key (attended-to)", fontsize=10.5, color=DARK)
    ax.set_ylabel("query (attending)", fontsize=10.5, color=DARK)

    img_data = np.full_like(A, np.nan)
    im = ax.imshow(img_data, cmap="viridis", vmin=0, vmax=float(A.max()))
    text_artists = [[None] * T for _ in range(T)]
    row_text = ax.text(0.02, 1.04, "", transform=ax.transAxes,
                       ha="left", va="bottom", fontsize=10.5, color=ORANGE,
                       fontweight="bold")

    HOLD_PER_ROW = 3
    TOTAL = HOLD_PER_ROW * T + 6  # extra hold at the end

    def update(frame):
        current_row = min(frame // HOLD_PER_ROW, T)
        data = np.full_like(A, np.nan)
        for i in range(current_row):
            data[i] = A[i]
        im.set_data(data)
        # annotate fully-revealed rows
        for i in range(T):
            for j in range(T):
                if text_artists[i][j] is None and i < current_row:
                    v = A[i, j]
                    text_artists[i][j] = ax.text(
                        j, i, f"{v:.2f}",
                        ha="center", va="center",
                        fontsize=7.5,
                        color="white" if v < float(A.max()) * 0.6 else "black",
                    )
        row_text.set_text(
            f"row {current_row} / {T} revealed — each row is a probability distribution over keys"
        )
        return [im, row_text]

    anim = FuncAnimation(fig, update, frames=TOTAL, interval=320, blit=False)
    out = HERE / "attention_matrix_build.gif"
    anim.save(out, writer=PillowWriter(fps=3))
    plt.close(fig)
    return out


# ═════════════════════════════════════════════════════════════════════════════
# 5.  Permutation equivariance (static)
# ═════════════════════════════════════════════════════════════════════════════
def render_permutation_equivariance() -> Path:
    words    = ["the", "cat", "bit",  "the", "man"]
    shuffled = ["man", "the", "cat", "bit", "the"]   # just a reordering
    perm     = [4, 0, 1, 2, 3]                       # index into original
    T = len(words)

    fig, axes = plt.subplots(2, 1, figsize=(10.0, 4.6), facecolor="white")
    fig.suptitle("Attention is permutation-equivariant — shuffle input ⇒ shuffle output (same vectors)",
                 fontsize=12, fontweight="bold", color=DARK)

    def draw_row(ax, inputs, outputs, title, perm_indices):
        ax.set_xlim(0, T + 1.5); ax.set_ylim(-1.3, 1.3); ax.axis("off")
        ax.set_title(title, fontsize=11, color=DARK, loc="left")
        for i, (w, p) in enumerate(zip(inputs, perm_indices), start=1):
            color = BLUE if p == 0 else (ORANGE if p == 4 else GREEN if p == 1 else PURPLE if p == 2 else RED)
            ax.add_patch(FancyBboxPatch((i - 0.4, 0.45), 0.80, 0.50,
                                         boxstyle="round,pad=0.02",
                                         facecolor=DIM, edgecolor=color, linewidth=1.4))
            ax.text(i, 0.70, w, ha="center", va="center",
                    fontsize=10, color=color, fontweight="bold")
            ax.add_patch(FancyBboxPatch((i - 0.4, -0.95), 0.80, 0.50,
                                         boxstyle="round,pad=0.02",
                                         facecolor="white", edgecolor=color, linewidth=1.4))
            ax.text(i, -0.70, f"y{p+1}", ha="center", va="center",
                    fontsize=10, color=color, fontweight="bold")
            ax.annotate("", xy=(i, -0.45), xytext=(i, 0.45),
                        arrowprops=dict(arrowstyle="-|>", color=color,
                                         lw=1.0, alpha=0.7))
        ax.text(T + 1.3, 0.70, "input", ha="left", va="center",
                fontsize=9.5, color=GREY, style="italic")
        ax.text(T + 1.3, -0.70, "output", ha="left", va="center",
                fontsize=9.5, color=GREY, style="italic")

    draw_row(axes[0], words,    None, "Original order",     [0, 1, 2, 3, 4])
    draw_row(axes[1], shuffled, None, "Shuffled input",     perm)

    fig.text(0.5, 0.02,
             "Same vectors. Just reordered. Attention has no idea which position is which "
             "→ Ch.18 injects position via positional encoding.",
             ha="center", fontsize=10, color=GREY, style="italic")

    out = HERE / "permutation_equivariance.png"
    fig.savefig(out, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


# ═════════════════════════════════════════════════════════════════════════════
# main
# ═════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    outs = [
        animate_hard_vs_soft(),
        animate_softmax_temperature(),
        animate_rnn_vs_attention(),
        animate_attention_matrix_build(),
        render_permutation_equivariance(),
    ]
    for p in outs:
        print(f"wrote {p}")
