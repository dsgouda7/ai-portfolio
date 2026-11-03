"""Record an actual LSTM step; export the lesson's two-state walkthrough."""

import ast
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import FancyBboxPatch
import torch
from torch import nn

from gru_layer_flow import BACKGROUND, TEXT, INPUT, STATE, GATE, WRITE, RESULT, excerpt, layer_figure


HERE = Path(__file__).resolve().parent
NOTEBOOK = HERE / "01A-sequence-memory-theory.ipynb"
LSTM_PHASES = (
    "Read A3, previous hidden h, AND previous cell c",
    "Two learned Linear projections: input and previous h (not c)",
    "Split into input, forget, candidate, output; calculate controls",
    "Keep part of old c; write part of the candidate",
    "Add those contributions to make the new cell c",
    "SHOW a squashed view of new c as the new hidden h",
    "Expose h now; carry BOTH h and c to the next token",
)


@dataclass(frozen=True)
class LSTMLayerStep:
    input_row: torch.Tensor
    old_hidden: torch.Tensor
    old_cell: torch.Tensor
    input_mix: torch.Tensor
    state_mix: torch.Tensor
    input_gate: torch.Tensor
    forget_gate: torch.Tensor
    candidate: torch.Tensor
    output_gate: torch.Tensor
    retained: torch.Tensor
    written: torch.Tensor
    new_cell: torch.Tensor
    new_hidden: torch.Tensor
    output: torch.Tensor
    carried_hidden: torch.Tensor
    carried_cell: torch.Tensor


def record_lstm_layer_step(
    lstm: nn.LSTM | nn.LSTMCell, inputs: torch.Tensor,
    previous_hidden: torch.Tensor, previous_cell: torch.Tensor,
) -> LSTMLayerStep:
    if not isinstance(lstm, (nn.LSTM, nn.LSTMCell)):
        raise TypeError("The two-state view requires an LSTM or LSTMCell.")
    if isinstance(lstm, nn.LSTM) and (lstm.num_layers != 1 or lstm.bidirectional or lstm.proj_size):
        raise ValueError("This view records one unidirectional layer without projections, not a stack.")
    expected = (1, lstm.hidden_size)
    if (not lstm.bias or inputs.shape != (1, lstm.input_size)
            or previous_hidden.shape != expected or previous_cell.shape != expected):
        raise ValueError("Supply one input row and both (h, c) rows to a biased LSTM.")
    with torch.no_grad():
        if isinstance(lstm, nn.LSTM):
            input_mix = nn.functional.linear(inputs, lstm.weight_ih_l0, lstm.bias_ih_l0)
            state_mix = nn.functional.linear(previous_hidden, lstm.weight_hh_l0, lstm.bias_hh_l0)
            outputs, (hidden, cell) = lstm(inputs[:, None], (previous_hidden[None], previous_cell[None]))
            actual_output = outputs.reshape(-1)
        else:
            input_mix = nn.functional.linear(inputs, lstm.weight_ih, lstm.bias_ih)
            state_mix = nn.functional.linear(previous_hidden, lstm.weight_hh, lstm.bias_hh)
            hidden, cell = lstm(inputs, (previous_hidden, previous_cell))
            actual_output = hidden.reshape(-1)
        i, f, g, o = (input_mix + state_mix).chunk(4, -1)
        write, keep, proposal, show = i.sigmoid(), f.sigmoid(), g.tanh(), o.sigmoid()
        retained, written = keep * previous_cell, write * proposal
        new_cell = retained + written
        new_hidden = show * new_cell.tanh()
        torch.testing.assert_close(new_cell.reshape(-1), cell.reshape(-1), rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(new_hidden.reshape(-1), hidden.reshape(-1), rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(actual_output, hidden.reshape(-1), rtol=0, atol=0)
        return LSTMLayerStep(*(
            value.detach().reshape(-1).clone() for value in (
                inputs, previous_hidden, previous_cell, input_mix, state_mix,
                write, keep, proposal, show, retained, written, new_cell, new_hidden,
                actual_output, hidden, cell,
            )
        ))


def draw_lstm_layer(axis, step: LSTMLayerStep, phase: int) -> None:
    axis.clear()
    axis.set(xlim=(0, 1), ylim=(0, 1), facecolor=BACKGROUND)
    axis.axis("off")

    def box(x, y, title, values, color, stages, width=0.28):
        active = phase in stages
        axis.add_patch(FancyBboxPatch(
            (x - width / 2, y - 0.042), width, 0.084, boxstyle="round,pad=0.007",
            facecolor="#26364a" if active else BACKGROUND, edgecolor=color,
            linewidth=3 if active else 1,
        ))
        axis.text(x, y + 0.014, title, ha="center", color=TEXT, fontsize=11)
        axis.text(x, y - 0.023, values, ha="center", color=color, fontsize=10)

    def arrow(start, end, color):
        axis.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="->", color=color, lw=1.6))

    box(0.17, 0.94, "A3 input x [2]", excerpt(step.input_row), INPUT, {0})
    box(0.5, 0.94, "previous h [2]", excerpt(step.old_hidden), STATE, {0})
    box(0.83, 0.94, "previous c [2]", excerpt(step.old_cell), STATE, {0})
    box(0.17, 0.80, "Linear: input 2 -> 8", "W_x * input + bias", INPUT, {1})
    box(0.5, 0.80, "Linear: hidden 2 -> 8", "W_h * old h + bias", STATE, {1})
    arrow((0.17, 0.895), (0.17, 0.85), INPUT)
    arrow((0.5, 0.895), (0.5, 0.85), STATE)
    axis.plot([0.17, 0.17, 0.5, 0.5], [0.75, 0.72, 0.72, 0.75], color=GATE)
    axis.text(0.34, 0.685, "Add mixes; split i / f / g / o [2 each]", ha="center", color=GATE, fontsize=11)
    gates = (
        (0.13, "KEEP = sigmoid(f)", step.forget_gate, STATE),
        (0.38, "WRITE = sigmoid(i)", step.input_gate, WRITE),
        (0.63, "proposal = tanh(g)", step.candidate, WRITE),
        (0.88, "SHOW = sigmoid(o)", step.output_gate, GATE),
    )
    for x, label, values, color in gates:
        box(x, 0.60, label, excerpt(values), color, {2}, width=0.22)
        arrow((0.34, 0.67), (x, 0.65), GATE)
    box(0.27, 0.45, "kept = KEEP * old c", excerpt(step.retained), STATE, {3}, width=0.36)
    box(0.68, 0.45, "written = WRITE * proposal", excerpt(step.written), WRITE, {3}, width=0.39)
    arrow((0.13, 0.55), (0.27, 0.50), STATE)
    arrow((0.38, 0.55), (0.68, 0.50), WRITE)
    arrow((0.63, 0.55), (0.68, 0.50), WRITE)
    axis.plot([0.83, 0.83, 0.01, 0.01, 0.075],
              [0.895, 0.875, 0.875, 0.45, 0.45], color=STATE, lw=1.6)
    arrow((0.075, 0.45), (0.085, 0.45), STATE)
    axis.text(0.02, 0.69, "old c bypasses projections", color=STATE, fontsize=9, rotation=90, ha="left")
    box(0.47, 0.31, "new c = kept + written", excerpt(step.new_cell), RESULT, {4}, width=0.40)
    arrow((0.27, 0.40), (0.40, 0.36), STATE)
    arrow((0.68, 0.40), (0.55, 0.36), WRITE)
    box(0.47, 0.17, "new h = SHOW * tanh(new c)", excerpt(step.new_hidden), RESULT, {5}, width=0.43)
    arrow((0.47, 0.26), (0.47, 0.22), RESULT)
    axis.plot([0.88, 0.93, 0.93, 0.70], [0.55, 0.55, 0.17, 0.17], color=GATE)
    box(0.20, 0.055, "output now: h", excerpt(step.output), RESULT, {6}, width=0.30)
    box(0.72, 0.055, "next token receives (h, c)", "Both rows, not just output h", STATE, {6}, width=0.44)
    arrow((0.36, 0.12), (0.20, 0.105), RESULT)
    arrow((0.58, 0.12), (0.65, 0.105), RESULT)
    axis.plot([0.675, 0.95, 0.95, 0.81], [0.31, 0.31, 0.12, 0.12], color=STATE)
    arrow((0.81, 0.12), (0.81, 0.105), STATE)


def source_fingerprint(notebook: dict) -> str:
    cells = {cell["id"]: cell for cell in notebook["cells"]}
    sources = [ast.dump(ast.parse("".join(cells[name]["source"])))
               for name in ("eb7df241", "lstm-real-cell-parity")]
    return hashlib.sha256(("\n".join(sources)).encode() + Path(__file__).read_bytes()).hexdigest()


def export_lstm_layer(notebook: dict) -> None:
    from memory_gate_flows import responsive_animation_html

    cells = {cell["id"]: cell for cell in notebook["cells"]}
    namespace = {"__name__": "lstm_layer_recording"}
    for name in ("eb7df241", "lstm-real-cell-parity"):
        exec(compile("".join(cells[name]["source"]), name, "exec"), namespace)
    old = namespace["lstm_real_trace"][1]
    step = record_lstm_layer_step(
        namespace["lstm_real_cell"], namespace["lstm_piano_rows"][1:2], old["old_h"], old["old_c"],
    )
    torch.testing.assert_close(step.output, old["actual_h"].reshape(-1), rtol=0, atol=1e-12)
    torch.testing.assert_close(step.carried_cell, old["actual_c"].reshape(-1), rtol=0, atol=1e-12)
    figure, axis = layer_figure()

    def draw(phase):
        draw_lstm_layer(axis, step, phase)
        for text in list(figure.texts):
            text.remove()
        figure.text(0.5, 0.985, "ONE token through ONE LSTM layer", color=TEXT, ha="center",
                    va="top", fontsize=19, weight="bold")
        figure.text(0.5, 0.935, "A3 after D3 | actual initialized LSTMCell(2, 2), not fitted music",
                    color=TEXT, ha="center", fontsize=12)
        figure.text(0.5, 0.89, f"{phase + 1}/{len(LSTM_PHASES)}  {LSTM_PHASES[phase]}",
                    color=RESULT, ha="center", fontsize=12)

    prefix = HERE / "images" / "01a-lstm-token-layer-flow"
    draw(len(LSTM_PHASES) - 1)
    figure.savefig(prefix.with_suffix(".png"), dpi=120, facecolor=BACKGROUND, metadata={
        "SourceFingerprint": source_fingerprint(notebook),
        "Output": json.dumps(step.output.tolist()), "CarriedHidden": json.dumps(step.carried_hidden.tolist()),
        "CarriedCell": json.dumps(step.carried_cell.tolist()), "GateOrder": "input, forget, candidate, output",
    })
    animation = FuncAnimation(figure, draw, frames=len(LSTM_PHASES), interval=1800, repeat=False)
    plt.close(figure)
    html = responsive_animation_html(animation)
    prefix.with_suffix(".html").write_bytes(
        ("\n".join(line.rstrip() for line in html.splitlines()) + "\n").encode("utf-8"),
    )
    animation.save(prefix.with_suffix(".gif"), writer="pillow", fps=1000 / 1800)
    print(f"Saved {prefix.name}: {len(LSTM_PHASES)} actual-state stages.")


def export_access_preview() -> None:
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    figure.set_facecolor(BACKGROUND)
    titles = ("Encoder-decoder WITHOUT attention", "Optional source ATTENTION")
    for index, axis in enumerate(axes):
        axis.set(xlim=(0, 1), ylim=(0, 1), facecolor=BACKGROUND)
        axis.axis("off")
        axis.set_title(titles[index], color=TEXT, fontsize=16, pad=18)
        for position, note in enumerate(("D3", "A3", "D4")):
            x = 0.17 + position * 0.33
            axis.add_patch(FancyBboxPatch((x - 0.14, 0.72), 0.28, 0.17, boxstyle="round,pad=0.01",
                                         facecolor=BACKGROUND, edgecolor=STATE))
            axis.text(x, 0.80, f"after {note}\nsource state", color=TEXT, ha="center", va="center", fontsize=11)
            if position < 2:
                axis.annotate("", xy=(x + 0.18, 0.80), xytext=(x + 0.14, 0.80),
                              arrowprops=dict(arrowstyle="->", color=STATE))
        if index == 0:
            axis.annotate("", xy=(0.5, 0.43), xytext=(0.83, 0.71),
                          arrowprops=dict(arrowstyle="->", color=RESULT, lw=2))
            axis.text(0.30, 0.55, "Final summary only\nh for GRU; (h, c) for LSTM", ha="center", color=RESULT, fontsize=12)
            axis.text(0.5, 0.32, "Writer carries its own state\n+ reads the previous output token",
                      ha="center", color=TEXT, fontsize=12)
            axis.text(0.5, 0.12, "No direct look-back to earlier source positions", ha="center", color=WRITE, fontsize=11)
        else:
            axis.plot([0.03, 0.03, 0.97, 0.97], [0.70, 0.65, 0.65, 0.70], color=STATE, lw=2)
            axis.annotate("", xy=(0.5, 0.44), xytext=(0.5, 0.64),
                          arrowprops=dict(arrowstyle="<->", color=RESULT, lw=2))
            axis.text(0.5, 0.54, "Keep ALL source states available", ha="center", color=STATE, fontsize=12)
            axis.text(0.5, 0.32, "Writer asks what matters for THIS output\n-> relevance-weighted view -> next-note scores",
                      ha="center", color=TEXT, fontsize=12)
            axis.text(0.5, 0.12, "Access to every position != equal weight\nChoose a new view for each output",
                      ha="center", color=WRITE, fontsize=11)
    figure.text(0.5, 0.025, "Conceptual access diagram, not recorded attention weights. Either reader/writer can use LSTM or GRU.",
                ha="center", color=TEXT, fontsize=11)
    figure.subplots_adjust(left=0.02, right=0.98, top=0.86, bottom=0.13, wspace=0.10)
    figure.savefig(HERE / "images" / "01a-memory-vs-attention.png", dpi=120, facecolor=BACKGROUND,
                   metadata={"DiagramType": "conceptual access, not measured attention weights"})
    plt.close(figure)


if __name__ == "__main__":
    book = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    export_lstm_layer(book)
    export_access_preview()
    from memory_gate_flows import refresh_image_attachments
    refresh_image_attachments()
