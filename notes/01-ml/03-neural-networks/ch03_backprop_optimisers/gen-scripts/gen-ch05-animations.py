"""Generate ch05 backprop animation GIF.

Output (saved next to this script):
    backprop_neurons.gif — a small 2-3-2-1 network training on a toy point.
                           Each frame belongs to one of three phases:
                             (1) forward pass — neurons light up left→right,
                                 showing activations h^(l).
                             (2) backward pass — edges/neurons light up
                                 right→left in red, showing δ^(l).
                             (3) weight update + epoch tick — weights refresh
                                 and the epoch counter + loss value advance.
Run:  python gen_ch05_animations.py
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
GREY = "#95A5A6"
BLUE = "#2E86C1"          # forward activation glow
RED = "#C0392B"           # backward gradient glow
GREEN = "#27AE60"
DIM = "#D5DBDB"

HERE = Path(__file__).parent


# ──────────────────────────────────────────────────────────────────────────────
# Tiny network:  2 → 3 → 2 → 1  with ReLU hidden, linear output, MSE loss.
# We train on a single fixed (x, y) pair so the values are reproducible and
# evolve visibly across epochs.
# ──────────────────────────────────────────────────────────────────────────────
rng = np.random.default_rng(7)
LAYER_SIZES = [2, 3, 2, 1]

# one illustrative training sample
X_SAMPLE = np.array([0.9, -0.4])
Y_TARGET = 1.0
LR = 0.08
N_EPOCHS = 6


def init_params():
    Ws, bs = [], []
    for n_in, n_out in zip(LAYER_SIZES[:-1], LAYER_SIZES[1:]):
        Ws.append(rng.normal(0, 0.8, size=(n_in, n_out)))
        bs.append(np.zeros(n_out))
    return Ws, bs


def forward(Ws, bs, x):
    hs = [x]                 # activations per layer (h^0 = input)
    zs = []                  # pre-activations per hidden/output layer
    h = x
    for i, (W, b) in enumerate(zip(Ws, bs)):
        z = h @ W + b
        zs.append(z)
        if i < len(Ws) - 1:
            h = np.maximum(z, 0.0)   # ReLU
        else:
            h = z                    # linear output
        hs.append(h)
    return hs, zs


def backward(Ws, hs, zs, y):
    y_hat = hs[-1]
    # MSE on scalar output
    dL_dy = 2.0 * (y_hat - y)
    deltas = [np.atleast_1d(dL_dy)]
    for l in range(len(Ws) - 1, 0, -1):
        d_up = deltas[-1]
        # gradient back through linear: δ_prev = (W · δ) ⊙ ReLU'(z_{l-1})
        d_prev = (Ws[l] @ d_up) * (zs[l - 1] > 0).astype(float)
        deltas.append(d_prev)
    deltas.reverse()          # index l is the delta for layer l's pre-activation
    grads_W = []
    grads_b = []
    for l in range(len(Ws)):
        grads_W.append(np.outer(hs[l], deltas[l]))
        grads_b.append(deltas[l])
    return grads_W, grads_b


def sgd_step(Ws, bs, gW, gb, lr):
    return ([W - lr * g for W, g in zip(Ws, gW)],
            [b - lr * g for b, g in zip(bs, gb)])


# ── record every frame as a snapshot dict ───────────────────────────────────
def build_snapshots():
    Ws, bs = init_params()
    snaps = []
    for epoch in range(1, N_EPOCHS + 1):
        hs, zs = forward(Ws, bs, X_SAMPLE)
        y_hat = float(hs[-1][0])
        loss = (y_hat - Y_TARGET) ** 2
        gW, gb = backward(Ws, hs, zs, Y_TARGET)

        # forward frames: one per layer being "illuminated"
        for l_active in range(len(LAYER_SIZES)):
            snaps.append(dict(
                phase="forward", active_layer=l_active,
                epoch=epoch, loss=loss, y_hat=y_hat,
                Ws=[W.copy() for W in Ws], bs=[b.copy() for b in bs],
                hs=[h.copy() for h in hs],
                deltas=None,
            ))
        # backward frames: right-to-left per layer
        deltas_list = []
        dL_dy = 2.0 * (hs[-1] - Y_TARGET)
        deltas_list.append(dL_dy)
        for l in range(len(Ws) - 1, 0, -1):
            deltas_list.append((Ws[l] @ deltas_list[-1]) *
                               (zs[l - 1] > 0).astype(float))
        deltas_list.reverse()   # deltas_list[l] aligns with layer l+1's neurons
        for l_active in range(len(LAYER_SIZES) - 1, -1, -1):
            snaps.append(dict(
                phase="backward", active_layer=l_active,
                epoch=epoch, loss=loss, y_hat=y_hat,
                Ws=[W.copy() for W in Ws], bs=[b.copy() for b in bs],
                hs=[h.copy() for h in hs],
                deltas=deltas_list,
            ))
        # apply update, then a single "update" frame
        Ws, bs = sgd_step(Ws, bs, gW, gb, LR)
        hs_new, _ = forward(Ws, bs, X_SAMPLE)
        y_hat_new = float(hs_new[-1][0])
        loss_new = (y_hat_new - Y_TARGET) ** 2
        snaps.append(dict(
            phase="update", active_layer=None,
            epoch=epoch, loss=loss_new, y_hat=y_hat_new,
            Ws=[W.copy() for W in Ws], bs=[b.copy() for b in bs],
            hs=[h.copy() for h in hs_new],
            deltas=None,
        ))
    return snaps


# ── draw one frame ──────────────────────────────────────────────────────────
def neuron_positions():
    # x positions per layer, y positions centred vertically
    positions = []
    for l, n in enumerate(LAYER_SIZES):
        xs = np.full(n, l * 2.2)
        ys = np.linspace(-(n - 1) / 2, (n - 1) / 2, n) * 1.2
        positions.append(np.column_stack([xs, ys]))
    return positions


POS = neuron_positions()
LAYER_LABELS = ["Input", "Hidden 1\n(ReLU)", "Hidden 2\n(ReLU)", "Output\n(linear)"]


def animate_backprop() -> Path:
    snaps = build_snapshots()

    fig, ax = plt.subplots(figsize=(10.5, 5.4), facecolor="white")
    ax.set_xlim(-1.0, (len(LAYER_SIZES) - 1) * 2.2 + 1.6)
    ax.set_ylim(-2.6, 2.6)
    ax.set_axis_off()
    ax.set_title(
        "Backpropagation step-by-step on a single sample "
        "(2 → 3 → 2 → 1, ReLU hidden, MSE loss)",
        fontsize=12, color=DARK, fontweight="bold",
    )

    # layer headers
    for l, lab in enumerate(LAYER_LABELS):
        ax.text(l * 2.2, 2.3, lab, ha="center", fontsize=9.5,
                color=DARK, fontweight="bold")

    # pre-build all edge artists (one per weight)
    edge_artists = {}     # key: (l, i, j) -> Line2D
    weight_texts = {}
    for l in range(len(LAYER_SIZES) - 1):
        for i in range(LAYER_SIZES[l]):
            for j in range(LAYER_SIZES[l + 1]):
                p0 = POS[l][i]; p1 = POS[l + 1][j]
                (line,) = ax.plot([p0[0], p1[0]], [p0[1], p1[1]],
                                  color=DIM, lw=1.1, zorder=1)
                edge_artists[(l, i, j)] = line
                # small weight label near midpoint of a few representative edges only
                if l == 0 and i == 0:
                    tx = ax.text((p0[0] + p1[0]) / 2,
                                 (p0[1] + p1[1]) / 2 + 0.12,
                                 "", fontsize=7.2, color=GREY,
                                 family="monospace", ha="center")
                    weight_texts[(l, i, j)] = tx

    # pre-build neuron circles
    neuron_artists = {}
    neuron_texts = {}
    for l in range(len(LAYER_SIZES)):
        for i in range(LAYER_SIZES[l]):
            p = POS[l][i]
            (dot,) = ax.plot([p[0]], [p[1]], "o", ms=28,
                             markerfacecolor="white",
                             markeredgecolor=DARK, markeredgewidth=1.4,
                             zorder=3)
            neuron_artists[(l, i)] = dot
            tx = ax.text(p[0], p[1], "", ha="center", va="center",
                         fontsize=8.5, color=DARK, family="monospace",
                         zorder=4)
            neuron_texts[(l, i)] = tx

    # input value labels (shown left of input neurons)
    for i, val in enumerate(X_SAMPLE):
        p = POS[0][i]
        ax.text(p[0] - 0.7, p[1], f"x{i}={val:+.2f}",
                ha="right", va="center", fontsize=9,
                color=DARK, family="monospace")
    # target label
    p_out = POS[-1][0]
    ax.text(p_out[0] + 0.7, p_out[1] + 0.4, f"y target = {Y_TARGET:+.2f}",
            ha="left", va="center", fontsize=9, color=GREEN,
            family="monospace")

    status = ax.text(0.01, 0.02, "", transform=ax.transAxes,
                     fontsize=10, family="monospace", color=DARK)
    phase_banner = ax.text(0.99, 0.02, "", transform=ax.transAxes,
                           ha="right", fontsize=10.5, fontweight="bold",
                           color=DARK)

    def set_neuron(l, i, glow_color, glow_strength, text):
        dot = neuron_artists[(l, i)]
        # mix between white and glow_color by strength in [0, 1]
        dot.set_markerfacecolor(glow_color if glow_strength > 0.01 else "white")
        dot.set_alpha(1.0)
        dot.set_markeredgecolor(glow_color if glow_strength > 0.5 else DARK)
        dot.set_markeredgewidth(2.2 if glow_strength > 0.5 else 1.3)
        neuron_texts[(l, i)].set_text(text)
        neuron_texts[(l, i)].set_color("white" if glow_strength > 0.5 else DARK)

    def render(snap):
        phase = snap["phase"]
        active = snap["active_layer"]
        Ws = snap["Ws"]
        hs = snap["hs"]
        deltas = snap["deltas"]

        # reset all neurons to dim with their current activation text
        for l in range(len(LAYER_SIZES)):
            for i in range(LAYER_SIZES[l]):
                txt = f"{hs[l][i]:+.2f}"
                set_neuron(l, i, "white", 0.0, txt)

        # reset all edges to dim grey with current weight thickness
        for (l, i, j), line in edge_artists.items():
            w = Ws[l][i, j]
            line.set_color(DIM)
            line.set_linewidth(0.8 + 1.8 * min(abs(w), 1.6))
            line.set_alpha(0.75)
        for (l, i, j), tx in weight_texts.items():
            tx.set_text(f"w={Ws[l][i, j]:+.2f}")
            tx.set_color(GREY)

        if phase == "forward" and active is not None:
            # glow all neurons up to & including the active layer in blue
            for l in range(active + 1):
                for i in range(LAYER_SIZES[l]):
                    set_neuron(l, i, BLUE, 1.0, f"{hs[l][i]:+.2f}")
            # glow edges feeding the active layer
            if active > 0:
                l_edge = active - 1
                for i in range(LAYER_SIZES[l_edge]):
                    for j in range(LAYER_SIZES[active]):
                        edge_artists[(l_edge, i, j)].set_color(BLUE)
                        edge_artists[(l_edge, i, j)].set_alpha(0.95)
            phase_banner.set_text("▶ FORWARD PASS")
            phase_banner.set_color(BLUE)

        elif phase == "backward" and active is not None:
            # glow all neurons from output back through the active layer in red
            for l in range(len(LAYER_SIZES) - 1, active - 1, -1):
                for i in range(LAYER_SIZES[l]):
                    if deltas is not None and l < len(deltas) + 0 and l >= 0:
                        # align deltas: deltas[l-1] corresponds to layer l activations' upstream
                        # we stored deltas aligned to hidden/output pre-activations starting at l=1
                        idx = l - 1
                        if 0 <= idx < len(deltas):
                            text = f"δ={deltas[idx][i]:+.2f}"
                        else:
                            text = f"{hs[l][i]:+.2f}"
                    else:
                        text = f"{hs[l][i]:+.2f}"
                    set_neuron(l, i, RED, 1.0, text)
            # glow edges feeding INTO the active layer from the right (i.e. edges
            # between active and active+1 carrying the gradient back)
            if active < len(LAYER_SIZES) - 1:
                l_edge = active
                for i in range(LAYER_SIZES[l_edge]):
                    for j in range(LAYER_SIZES[l_edge + 1]):
                        edge_artists[(l_edge, i, j)].set_color(RED)
                        edge_artists[(l_edge, i, j)].set_alpha(0.95)
            phase_banner.set_text("◀ BACKWARD PASS")
            phase_banner.set_color(RED)

        else:   # update
            for l in range(len(LAYER_SIZES)):
                for i in range(LAYER_SIZES[l]):
                    set_neuron(l, i, "white", 0.0, f"{hs[l][i]:+.2f}")
            for (l, i, j), line in edge_artists.items():
                line.set_color(GREEN); line.set_alpha(0.95)
            phase_banner.set_text("⟳ WEIGHTS UPDATED")
            phase_banner.set_color(GREEN)

        status.set_text(
            f"epoch {snap['epoch']}/{N_EPOCHS}   "
            f"ŷ = {snap['y_hat']:+.3f}   "
            f"loss = {snap['loss']:.4f}"
        )

    def update(k):
        render(snaps[k])
        return []

    anim = FuncAnimation(
        fig, update, frames=len(snaps), interval=420,
        blit=False, repeat=True,
    )
    out = HERE / "backprop_neurons.gif"
    anim.save(out, writer=PillowWriter(fps=3))
    plt.close(fig)
    return out


if __name__ == "__main__":
    p = animate_backprop()
    print(f"wrote {p}")
