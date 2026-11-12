"""Compose a high-definition, verified handwritten BPTT sheet on MAI artwork."""

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Ellipse, FancyBboxPatch
from PIL import Image, ImageFilter, ImageOps, PngImagePlugin
import torch

import bptt_math
import error_feedback_flow
from error_feedback_flow import record_probe


HERE = Path(__file__).resolve().parent
PAPER = HERE / "images" / "01a-bptt-notebook-paper-mai.png"
MANIFEST = HERE / "images" / "01a-bptt-summary-prompts.json"
SHEET = HERE / "images" / "01a-bptt-memory-weight-explainer.png"
SIZE = (3200, 5600)
HEIGHT = 280
BACKGROUND = "#0d1b2e"
INK, TEAL, CORAL, GOLD = "#edf3ff", "#77dce5", "#ff997e", "#ffd172"
TAKEAWAY = (
    "Differentiate the error, trace it through prediction and memory, then expand earlier reads: "
    "one shared weight creates many paths whose multipliers compound."
)


@dataclass(frozen=True)
class WorkedExample:
    memory: float
    note_writes: tuple[float, ...]
    linear_states: tuple[float, ...]
    linear_response: float
    tanh_response: float
    ending_feedback: float
    recurrent_gradient: float
    backward_feedback: tuple[float, ...]
    long_feedback_ratio: float
    ending_probability: float
    probability_response: float


def worked_example(memory: float) -> WorkedExample:
    probe = record_probe(gain=memory)
    linear_states = [0.0]
    linear_response = tanh_response = 0.0
    for step in probe.steps:
        linear_response = linear_states[-1] + memory * linear_response
        linear_states.append(step.input_part + memory * linear_states[-1])
        tanh_response = step.slope * (step.previous + memory * tanh_response)
    longer = record_probe(gain=memory, repeats=7)
    final_hidden = torch.tensor(probe.steps[-1].state, dtype=torch.float64, requires_grad=True)
    head = torch.tensor(probe.weights.prediction, dtype=torch.float64)
    probability = (head * final_hidden).softmax(0)[
        error_feedback_flow.NOTES.index(error_feedback_flow.TARGET_NOTE)
    ]
    probability_response = torch.autograd.grad(probability, final_hidden)[0].item()
    return WorkedExample(
        memory, tuple(step.input_part for step in probe.steps), tuple(linear_states),
        linear_response, tanh_response, probe.steps[-1].feedback,
        probe.weight_gradients.memory, tuple(step.feedback for step in reversed(probe.steps)),
        longer.steps[0].feedback / longer.steps[-1].feedback,
        probability.item(), probability_response,
    )


def equation_rows() -> tuple[tuple[str, str], ...]:
    return (
        ("Linear forward step", r"h_t=m h_{t-1}+u_t"),
        ("Prediction and error",
         r"p=P(\mathrm{next=D3}\mid h_T),\qquad \mathrm{Error}=-\log p"),
        ("Differentiate the error",
         r"\frac{d\,\mathrm{Error}}{dm}=-\frac{1}{p}\,\frac{dp}{dm}"),
        ("Expand prediction through memory",
         r"\frac{dp}{dm}=\frac{dp}{dh_T}\,\frac{dh_T}{dm}"),
        ("Follow the whole route",
         r"\frac{d\,\mathrm{Error}}{dm}"
         r"=-\frac{1}{p}\,\frac{dp}{dh_T}\,\frac{dh_T}{dm}"),
        ("Product rule",
         r"\frac{dh_t}{dm}=h_{t-1}+m\left(\frac{dh_{t-1}}{dm}\right)"),
        ("One read earlier",
         r"\frac{dh_{t-1}}{dm}=h_{t-2}+m\left(\frac{dh_{t-2}}{dm}\right)"),
        ("Another read earlier",
         r"\frac{dh_{t-2}}{dm}=h_{t-3}+m\left(\frac{dh_{t-3}}{dm}\right)"),
        ("Substitute",
         r"\frac{dh_t}{dm}=h_{t-1}"
         r"+m\left[h_{t-2}+m\left(\frac{dh_{t-2}}{dm}\right)\right]"),
        ("Distribute",
         r"\frac{dh_t}{dm}=h_{t-1}+m h_{t-2}"
         r"+m^2\left[h_{t-3}+m\left(\frac{dh_{t-3}}{dm}\right)\right]"),
        ("Expand to the fixed start",
         r"\frac{dh_T}{dm}=h_{T-1}+m h_{T-2}+m^2 h_{T-3}"
         r"+\cdots+m^{T-1}h_0"),
        ("Exact tanh backward handoff",
         r"g_{t-1}=g_t\,m(1-h_t^2)"),
    )


def source_fingerprint() -> str:
    cases = json.dumps([asdict(worked_example(m)) for m in (0.8, 1.2)], sort_keys=True).encode()
    return hashlib.sha256(
        Path(__file__).read_bytes() + Path(bptt_math.__file__).read_bytes()
        + Path(error_feedback_flow.__file__).read_bytes() + PAPER.read_bytes()
        + MANIFEST.read_bytes() + cases
    ).hexdigest()


def sheet_figure():
    with Image.open(PAPER) as source:
        paper = ImageOps.fit(
            ImageOps.colorize(
                source.convert("L").filter(ImageFilter.GaussianBlur(1.2)),
                black="#243a54", white=BACKGROUND,
            ),
            SIZE, method=Image.Resampling.LANCZOS,
        )
    figure = plt.figure(figsize=(16, 28), dpi=200, facecolor=BACKGROUND)
    axis = figure.add_axes((0, 0, 1, 1))
    axis.imshow(paper, extent=(0, 160, HEIGHT, 0), aspect="auto", zorder=-10)
    axis.set(xlim=(0, 160), ylim=(HEIGHT, 0))
    axis.axis("off")

    def label(y, text, size=17, color=INK, x=28, math=False):
        return axis.text(
            x, y, f"${text}$" if math else text, fontsize=size,
            color=color, va="center", ha="left",
        )

    equations = dict(equation_rows())
    label(10, "BPTT: FOLLOW ONE WEIGHT'S INFLUENCE", 25, GOLD)
    label(17, "Differentiate -> expand -> substitute -> distribute", 18, TEAL)
    label(23, "One feature: m = recurrent weight; u = weighted note; h = saved memory", 14)

    label(31, "1  LINEAR SKETCH: ISOLATE THE MEMORY WEIGHT", 20, TEAL)
    label(38, equations["Linear forward step"], 25, math=True)
    label(46, equations["Prediction and error"], 23, math=True)
    label(52, "p is the chance of correct D3. Only the ending prediction is graded.", 15)

    label(60, "2  DIFFERENTIATE: HOW DOES m CHANGE THE ERROR?", 20, TEAL)
    label(68, equations["Differentiate the error"], 25, math=True)
    label(75, "More chance for D3 means less error; -1/p is that local response.", 15)
    label(83, equations["Expand prediction through memory"], 25, math=True)
    label(91, equations["Follow the whole route"], 23, math=True)
    label(98, "m changes memory, memory changes chance, chance changes error: multiply.", 15)
    label(103, "The first two factors give g: the ending error's response to final memory.", 13, GOLD)

    label(112, "3  THE CATCH: OLD MEMORY ALSO DEPENDS ON m", 20, TEAL)
    label(118, "BOTH multiplied pieces change. Product rule: count each effect, then add.", 15)
    label(127, equations["Product rule"], 26, math=True)
    label(135, "DIRECT: multiplier changes now", 13, GOLD)
    label(135, "INHERITED: earlier reads changed old h", 13, TEAL, x=88)

    label(145, "4  NOW EXPAND THE EARLIER MEMORY", 20, TEAL)
    label(154, equations["One read earlier"], 24, math=True)
    label(164, equations["Another read earlier"], 24, math=True)
    label(171, "Each earlier derivative is the SAME question, one read farther back.", 15)

    label(181, "5  SUBSTITUTE THE EARLIER RESULT, THEN DISTRIBUTE", 20, TEAL)
    label(190, equations["Substitute"], 22, math=True)
    label(200, equations["Distribute"], 22, math=True)
    label(206, "Keep expanding until you reach the fixed starting memory:", 13)
    label(214, equations["Expand to the fixed start"], 21, math=True)
    for x, power, color in ((50, "m", TEAL), (75, "m^2", GOLD),
                            (100, "m^3", CORAL), (125, "m^k", TEAL)):
        label(224, power, 21, color, x=x - 2, math=True)
        axis.add_patch(Ellipse((x, 224), 10, 8, fill=False, edgecolor=color, linewidth=1.6))
    label(231, "Each crossed link adds one m. Stop at fixed h0: its derivative is 0.", 13)

    label(239, "6  THIS IS WHY EARLY FEEDBACK CAN FADE OR GROW", 20, TEAL)
    for x, color, title, equation, meaning in (
        (28, TEAL, "Repeated x 0.8", r"0.8^{20}\approx0.01153", "Weak early guidance"),
        (90, CORAL, "Repeated x 1.2", r"1.2^{20}\approx38.34", "Amplified guidance"),
    ):
        axis.add_patch(FancyBboxPatch(
            (x - 1, 244), 57, 19, boxstyle="round,pad=0.8,rounding_size=1.2",
            facecolor="#172941",
            edgecolor=color, linewidth=1.4, alpha=0.95,
        ))
        label(248, title, 15, color, x=x + 2)
        label(254, equation, 20, color, x=x + 2, math=True)
        label(260, meaning, 12, color, x=x + 2)
    label(267, "RNN handoff: also multiply by the local squeeze response.", 15, TEAL)
    label(273, equations["Exact tanh backward handoff"], 17, math=True)
    label(278, "The squeeze can weaken feedback; memory weights can amplify it. Gradient is not error.", 12, GOLD)
    return figure, axis


def export(directory: Path = HERE / "images") -> None:
    font_manager.findfont(bptt_math.FONT, fallback_to_default=False)
    directory.mkdir(exist_ok=True)
    with plt.rc_context(bptt_math.HANDWRITING_STYLE):
        figure, axis = sheet_figure()
        try:
            figure.canvas.draw()
            renderer = figure.canvas.get_renderer()
            for index, text in enumerate(axis.texts):
                bounds = text.get_window_extent(renderer)
                if not (axis.bbox.contains(bounds.x0, bounds.y0) and axis.bbox.contains(bounds.x1, bounds.y1)):
                    raise RuntimeError(f"BPTT sheet label is clipped: {text.get_text()}")
                points = axis.transData.inverted().transform(((bounds.x0, bounds.y0), (bounds.x1, bounds.y1)))
                if min(points[:, 0]) < 26 or max(points[:, 0]) > 154 or max(points[:, 1]) > HEIGHT - 1:
                    raise RuntimeError(f"BPTT sheet label leaves the writing region: {text.get_text()}")
                for other in axis.texts[index + 1:]:
                    if bounds.overlaps(other.get_window_extent(renderer)):
                        raise RuntimeError(f"BPTT sheet labels overlap: {text.get_text()} / {other.get_text()}")
            image = Image.frombuffer("RGBA", SIZE, figure.canvas.buffer_rgba(), "raw", "RGBA", 0, 1).convert("RGB")
            image = image.quantize(colors=256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
            metadata = PngImagePlugin.PngInfo()
            metadata.add_text("SourceFingerprint", source_fingerprint())
            metadata.add_text("Generator", "MAI-Image-2.6 paper; verified local Segoe Print math overlay")
            metadata.add_text("Model", "MAI-Image-2.6")
            metadata.add_text("Compositing", f"{SIZE[0]}x{SIZE[1]} dark-theme code-rendered math; 256-color PNG")
            metadata.add_text("Theme", "dark navy; white, cyan, gold and coral handwriting")
            metadata.add_text("MathTranscript", json.dumps(equation_rows()))
            image.save(directory / SHEET.name, pnginfo=metadata, dpi=(200, 200), optimize=True)
        finally:
            plt.close(figure)
    print(f"Saved {SHEET.name}: {SIZE[0]}x{SIZE[1]}, {(directory / SHEET.name).stat().st_size:,} bytes.")


if __name__ == "__main__":
    export()
