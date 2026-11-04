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

from gru_layer_flow import BACKGROUND, TEXT, STATE, WRITE, RESULT
from memory_visuals import MOTION_FRAMES, INTERVAL_MS, draw_scene, lstm_scene


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


def draw_lstm_layer(axis, step: LSTMLayerStep, phase: int, *, progress=0.0) -> None:
    draw_scene(axis, lstm_scene(step), phase, native=True, progress=progress)


def source_fingerprint(notebook: dict) -> str:
    cells = {cell["id"]: cell for cell in notebook["cells"]}
    sources = [ast.dump(ast.parse("".join(cells[name]["source"])))
               for name in ("eb7df241", "lstm-real-cell-parity")]
    return hashlib.sha256(
        ("\n".join(sources)).encode() + Path(__file__).read_bytes()
        + (HERE / "memory_visuals.py").read_bytes(),
    ).hexdigest()


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
    figure, axis = plt.subplots(figsize=(12.8, 8))
    figure.patch.set_facecolor(BACKGROUND)
    figure.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.025)

    def draw(frame):
        draw_lstm_layer(axis, step, frame // MOTION_FRAMES,
                        progress=(frame % MOTION_FRAMES) / (MOTION_FRAMES - 1))

    prefix = HERE / "images" / "01a-lstm-token-layer-flow"
    draw_scene(axis, lstm_scene(step), None, native=True)
    figure.savefig(prefix.with_suffix(".png"), dpi=120, facecolor=BACKGROUND, metadata={
        "SourceFingerprint": source_fingerprint(notebook),
        "Output": json.dumps(step.output.tolist()), "CarriedHidden": json.dumps(step.carried_hidden.tolist()),
        "CarriedCell": json.dumps(step.carried_cell.tolist()), "GateOrder": "input, forget, candidate, output",
    })
    figure.savefig(prefix.with_suffix(".svg"), facecolor=BACKGROUND)
    animation = FuncAnimation(figure, draw, frames=len(LSTM_PHASES) * MOTION_FRAMES,
                              interval=INTERVAL_MS, repeat=False)
    plt.close(figure)
    html = responsive_animation_html(animation)
    prefix.with_suffix(".html").write_bytes(
        ("\n".join(line.rstrip() for line in html.splitlines()) + "\n").encode("utf-8"),
    )
    animation.save(prefix.with_suffix(".gif"), writer="pillow", fps=1000 / INTERVAL_MS)
    print(f"Saved {prefix.name}: {len(LSTM_PHASES)} actual-state stages.")


def export_access_preview() -> None:
    figure, axis = plt.subplots(figsize=(12.8, 8))
    figure.set_facecolor(BACKGROUND)
    axis.set(xlim=(0, 1), ylim=(0, 1), facecolor=BACKGROUND)
    axis.axis("off")
    axis.text(0.04, 0.955, "A final summary, or a source bank?", color=TEXT,
              fontsize=26, weight="bold", va="top")
    axis.text(0.04, 0.87, "Read in order; each exposed h already reflects its prefix.",
              color=STATE, fontsize=18)
    for position, note in enumerate(("D3", "A3", "D4")):
        x = 0.05 + position * 0.31
        axis.add_patch(FancyBboxPatch((x, 0.65), 0.28, 0.14, boxstyle="round,pad=0.008",
                                     facecolor="#20314c", edgecolor=STATE, linewidth=1.7))
        axis.text(x + 0.018, 0.755, f"After {note}", color=TEXT, fontsize=20, weight="bold", va="top")
        axis.text(x + 0.018, 0.68, "Source-position h", color=STATE, fontsize=18)
        if position < 2:
            axis.annotate("", xy=(x + 0.303, 0.72), xytext=(x + 0.291, 0.72),
                          arrowprops=dict(arrowstyle="->", color=STATE))
    for x, title, detail, color in (
        (0.05, "WITHOUT attention", "Final summary initializes writer.\nGRU: h. LSTM: usually (h, c).\nNo direct position-by-position revisit.", WRITE),
        (0.53, "WITH source attention", "Keep every source h available.\nAsk what matters for THIS output.\nBuild a relevance-weighted view.", RESULT),
    ):
        axis.add_patch(FancyBboxPatch((x, 0.24), 0.42, 0.32, boxstyle="round,pad=0.008",
                                     facecolor="#20314c", edgecolor=color, linewidth=1.7))
        axis.text(x + 0.018, 0.525, title, color=color, fontsize=20, weight="bold", va="top")
        axis.text(x + 0.018, 0.43, detail, color=TEXT, fontsize=18, va="top", linespacing=1.6)
    axis.text(0.04, 0.125, "Equal access does NOT mean equal weight. Choose a new view for each output.",
              color=RESULT, fontsize=18)
    axis.text(0.04, 0.055, "Conceptual preview, not measured attention weights. Either cell can be reader or writer.",
              color=TEXT, fontsize=18)
    figure.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.025)
    figure.savefig(HERE / "images" / "01a-memory-vs-attention.png", dpi=120, facecolor=BACKGROUND,
                   metadata={"DiagramType": "conceptual access, not measured attention weights"})
    plt.close(figure)


if __name__ == "__main__":
    book = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    export_lstm_layer(book)
    export_access_preview()
    from memory_gate_flows import refresh_image_attachments
    refresh_image_attachments()
