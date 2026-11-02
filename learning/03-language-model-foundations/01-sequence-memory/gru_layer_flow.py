"""Record and draw one real GRU layer at one token, including both output routes."""

from dataclasses import dataclass

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import torch
from torch import nn


BACKGROUND = "#1a1a2e"
TEXT = "#e7edf7"
INPUT = "#58d7d1"
STATE = "#76c6ff"
GATE = "#c5a3ff"
WRITE = "#ffbb78"
RESULT = "#8cdb9b"
LAYER_PHASES = (
    "Current input and previous hidden state",
    "Two learned Linear projections, with biases",
    "Split each projection; calculate reset and keep",
    "Propose content using reset-controlled projected memory",
    "Keep old content and admit complementary new content",
    "Add the two contributions",
    "Expose output and carry hidden: the SAME new values",
)


@dataclass(frozen=True)
class LayerStep:
    input_row: torch.Tensor
    old_state: torch.Tensor
    input_mix: torch.Tensor
    state_mix: torch.Tensor
    reset_gate: torch.Tensor
    update_gate: torch.Tensor
    candidate: torch.Tensor
    retained: torch.Tensor
    written: torch.Tensor
    new_state: torch.Tensor
    output: torch.Tensor
    carried: torch.Tensor


def record_layer_step(gru: nn.GRU | nn.GRUCell, inputs: torch.Tensor, previous: torch.Tensor) -> LayerStep:
    if not isinstance(gru, (nn.GRU, nn.GRUCell)):
        raise TypeError("The token-layer view requires a GRU or GRUCell.")
    if isinstance(gru, nn.GRU) and (gru.num_layers != 1 or gru.bidirectional):
        raise ValueError("This view records one unidirectional layer, not a stack.")
    if not gru.bias or inputs.shape != (1, gru.input_size) or previous.shape != (1, gru.hidden_size):
        raise ValueError("Supply one input row and one hidden row to a biased GRU.")
    with torch.no_grad():
        if isinstance(gru, nn.GRU):
            input_mix = nn.functional.linear(inputs, gru.weight_ih_l0, gru.bias_ih_l0)
            state_mix = nn.functional.linear(previous, gru.weight_hh_l0, gru.bias_hh_l0)
            outputs, carried = gru(inputs[:, None], previous[None])
            actual_output, actual_state = outputs.reshape(-1), carried.reshape(-1)
        else:
            input_mix = nn.functional.linear(inputs, gru.weight_ih, gru.bias_ih)
            state_mix = nn.functional.linear(previous, gru.weight_hh, gru.bias_hh)
            actual_state = gru(inputs, previous).reshape(-1)
            actual_output = actual_state
        xr, xz, xn = input_mix.chunk(3, -1)
        hr, hz, hn = state_mix.chunk(3, -1)
        reset, keep = (xr + hr).sigmoid(), (xz + hz).sigmoid()
        candidate = (xn + reset * hn).tanh()
        retained, written = keep * previous, (1 - keep) * candidate
        new_state = retained + written
        torch.testing.assert_close(new_state.reshape(-1), actual_state, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(actual_output, actual_state, rtol=0, atol=0)
        return LayerStep(*(
            value.detach().reshape(-1).clone() for value in (
                inputs, previous, input_mix, state_mix, reset, keep, candidate,
                retained, written, new_state, actual_output, actual_state,
            )
        ))


def excerpt(values: torch.Tensor) -> str:
    flat = values.reshape(-1)
    sample = ", ".join(f"{value:+.3f}" for value in flat[:2].tolist())
    return "[" + sample + (", ...]" if len(flat) > 2 else "]")


def draw_layer(
    axis, step: LayerStep, phase: int | None, *,
    input_label: str, old_label: str, new_label: str, output_use: str,
) -> None:
    axis.clear()
    axis.set(xlim=(0, 1), ylim=(0, 1), facecolor=BACKGROUND)
    axis.axis("off")
    width, hidden = step.input_row.numel(), step.old_state.numel()

    def box(x, y, title, detail, color, stages):
        active = phase is not None and phase in stages
        axis.add_patch(FancyBboxPatch(
            (x - 0.21, y - 0.043), 0.42, 0.086, boxstyle="round,pad=0.008",
            facecolor="#26364a" if active else BACKGROUND, edgecolor=color,
            linewidth=3 if active else 1,
        ))
        axis.text(x, y + 0.014, title, ha="center", va="center", color=TEXT, fontsize=11)
        axis.text(x, y - 0.021, detail, ha="center", va="center", color=color, fontsize=10)

    def arrow(start, end, color=STATE):
        axis.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="->", color=color, lw=1.5))

    def mix_detail(mix, prefix):
        r, z, n = mix.chunk(3)
        return f"{prefix}_r0={r[0]:+.3f}  {prefix}_z0={z[0]:+.3f}  {prefix}_n0={n[0]:+.3f}"

    box(0.25, 0.92, f"{input_label}  x_t [{width}]", excerpt(step.input_row), INPUT, {0})
    box(0.75, 0.92, f"old hidden {old_label} [{hidden}]", excerpt(step.old_state), STATE, {0})
    box(0.25, 0.78, f"Linear: {width} -> {3 * hidden}  (W_x + bias)", mix_detail(step.input_mix, "x"), INPUT, {1})
    box(0.75, 0.78, f"Linear: {hidden} -> {3 * hidden}  (W_h + bias)", mix_detail(step.state_mix, "m"), STATE, {1})
    arrow((0.25, 0.87), (0.25, 0.83), INPUT)
    arrow((0.75, 0.87), (0.75, 0.83))
    axis.plot([0.25, 0.25, 0.75, 0.75], [0.73, 0.70, 0.70, 0.73], color=GATE, lw=1.5)
    axis.text(0.5, 0.716, f"split BOTH mixes into r / z / n chunks [{hidden} each]",
              ha="center", color=GATE, fontsize=10)
    arrow((0.25, 0.70), (0.25, 0.65), GATE)
    arrow((0.75, 0.70), (0.75, 0.65), GATE)
    box(0.25, 0.60, "reset r = sigmoid(x_r + m_r)", excerpt(step.reset_gate), GATE, {2})
    box(0.75, 0.60, "keep z = sigmoid(x_z + m_z)", excerpt(step.update_gate), GATE, {2})
    box(0.25, 0.46, "proposal n = tanh(x_n + r * m_n)", excerpt(step.candidate), WRITE, {3})
    box(0.75, 0.46, "kept = z * old hidden", excerpt(step.retained), STATE, {4})
    arrow((0.25, 0.55), (0.25, 0.51), GATE)
    arrow((0.75, 0.55), (0.75, 0.51), GATE)
    axis.plot([0.97, 0.985, 0.985, 0.97], [0.92, 0.92, 0.46, 0.46], color=STATE, lw=1.5)
    axis.text(0.985, 0.66, "old h", color=STATE, fontsize=8, rotation=90, ha="right")
    axis.text(0.025, 0.53, "m_n = projected old state", color=WRITE, fontsize=8, rotation=90, va="center")
    box(0.25, 0.32, "written = (1-z) * proposal", excerpt(step.written), WRITE, {4})
    box(0.75, 0.32, f"{new_label} = kept + written [{hidden}]", excerpt(step.new_state), RESULT, {5})
    arrow((0.25, 0.41), (0.25, 0.37), WRITE)
    arrow((0.75, 0.41), (0.75, 0.37))
    arrow((0.47, 0.32), (0.53, 0.32), WRITE)
    axis.text(0.5, 0.34, "+", color=RESULT, fontsize=16, ha="center")
    axis.text(0.47, 0.42, "1-z", color=GATE, fontsize=10, ha="center")
    arrow((0.55, 0.55), (0.44, 0.37), GATE)
    box(0.25, 0.13, f"output_t [{hidden}]: {output_use}", excerpt(step.output), RESULT, {6})
    box(0.75, 0.13, f"hidden for next token [{hidden}]", excerpt(step.carried), STATE, {6})
    axis.plot([0.75, 0.75, 0.25, 0.25], [0.27, 0.23, 0.23, 0.18], color=RESULT, lw=1.5)
    arrow((0.25, 0.23), (0.25, 0.18), RESULT)
    arrow((0.75, 0.23), (0.75, 0.18), STATE)
    axis.text(0.5, 0.02, "Two routes, same h_t values. Neither route is a token ID or word scores.",
              ha="center", color=RESULT, fontsize=10)


def layer_figure(figsize=(12, 10)):
    figure, axis = plt.subplots(figsize=figsize)
    figure.patch.set_facecolor(BACKGROUND)
    figure.subplots_adjust(left=0.03, right=0.97, top=0.86, bottom=0.025)
    return figure, axis
