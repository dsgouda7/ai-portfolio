"""Render intuition-first, handwritten-style math for the existing note RNN."""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager

import error_feedback_flow
from error_feedback_flow import ErrorFeedbackProbe, LEARNING_RATE, record_probe


HERE = Path(__file__).resolve().parent
FONT = "Segoe Print"
IMAGE_FILES = (
    "01a-bptt-forward-math.png",
    "01a-bptt-backward-math.png",
    "01a-bptt-recurrent-gradient.png",
)
HANDWRITING_STYLE = {
    "font.family": FONT,
    "mathtext.fontset": "custom",
    "mathtext.rm": FONT,
    "mathtext.it": FONT,
    "mathtext.bf": f"{FONT}:bold",
    "mathtext.sf": FONT,
    "mathtext.fallback": "stix",
}
FORWARD_REFERENCE = (
    ("Mix the current note with previous memory",
     r"\mathrm{mix}_t=W_x x_t+W_h h_{t-1}+b"),
    ("The cell makes new memory, not a note prediction",
     r"h_t=\tanh(\mathrm{mix}_t)"),
    ("A separate head turns that memory into next-note scores",
     r"\mathrm{scores}_t=W_{\mathrm{out}}h_t+b_{\mathrm{out}}"),
    ("Softmax: divide each exponentiated score by the total",
     r"p_t[j]=\frac{\exp(\mathrm{scores}_t[j])}"
     r"{\sum_k\exp(\mathrm{scores}_t[k])}"),
    ("Grade the probability assigned to the actual next note",
     r"\mathrm{loss}_t=-\log\!\left(p_t[\mathrm{actual\ next\ note}]\right)"),
)
BACKWARD_REFERENCE = (
    ("Start at a graded head; use zero here for a preview-only head",
     r"\mathrm{scoregrad}_t=\frac{\partial L}{\partial\mathrm{scores}_t}"
     r"=p_t-\mathrm{target}_t"),
    ("Local derivatives: tanh responsiveness, and one use of a recurrent weight",
     r"\frac{\partial h_t[i]}{\partial\mathrm{mix}_t[i]}=1-h_t[i]^2"
     r"\qquad\frac{\partial\mathrm{mix}_t[i]}{\partial W_h[i,j]}=h_{t-1}[j]"),
    ("Feedback to h: from its own head, plus the next read",
     r"g_t=W_{\mathrm{out}}^T\mathrm{scoregrad}_t"
     r"+W_h^T\mathrm{mixgrad}_{t+1}"),
    ("Pass that feedback through this read's tanh",
     r"\mathrm{mixgrad}_t=g_t\odot(1-h_t^2)"),
    ("For one recurrent weight: add contributions from every use",
     r"\frac{\partial L}{\partial W_h[i,j]}"
     r"=\sum_{t=1}^{T}g_t[i]\,(1-h_t[i]^2)\,h_{t-1}[j]"),
    ("The same result for the whole recurrent matrix",
     r"\frac{\partial L}{\partial W_h}"
     r"=\sum_{t=1}^{T}\mathrm{mixgrad}_t\,h_{t-1}^{T}"),
)
FORWARD_EQUATIONS = (
    ("Current note and old memory each make a contribution",
     r"\mathrm{mix}=\mathrm{note\ contribution}+m\times h_{\mathrm{incoming}}"),
    ("The RNN makes new memory; a separate head makes the scores",
     r"h_{\mathrm{new}}=\tanh(\mathrm{mix})"
     r"\qquad\mathrm{scores}=\mathrm{head\ weights}\times h_{\mathrm{new}}"),
    ("Let the three scores compete to become next-note probabilities",
     r"\mathrm{probabilities}=\operatorname{softmax}(\mathrm{scores})"),
    ("Grade the probability of the actual next note, not the guessed note label",
     r"\mathrm{loss}=-\log(\mathrm{probability\ of\ actual\ next\ note})"),
)
BACKWARD_EQUATIONS = (
    ("First ask what changing one saved memory would do to the ending error",
     r"\mathrm{error\ response}"
     r"=\frac{\mathrm{small\ change\ in\ ending\ error}}{\mathrm{small\ change\ in\ saved\ memory}}"),
    ("One handoff: how much does a tiny incoming-h nudge change new h?",
     r"\mathrm{handoff\ response}=m\times(1-h_{\mathrm{new}}^2)"),
    ("Along a route, each response rescales the nudge",
     r"\mathrm{earlier\ feedback}=\mathrm{later\ feedback}\times\mathrm{handoff\ response}"),
    ("For one use of m, hold incoming h fixed and ask how that multiplication reacts",
     r"\mathrm{one\ use's\ gradient}"
     r"=\mathrm{state\ feedback}\times(1-h_{\mathrm{new}}^2)\times h_{\mathrm{incoming}}"),
    ("Turning the shared dial changes all three reads, so collect every effect",
     r"\mathrm{shared\ gradient}=\mathrm{D3\ part}+\mathrm{A3\ part}+\mathrm{D4\ part}"),
    ("Response to m everywhere: direct = incoming h; inherited = m times earlier h's response",
     r"\mathrm{new\ h\ response}"
     r"=\mathrm{squash\ response}\times(\mathrm{direct}+\mathrm{inherited})"),
)


def worked_equations(probe: ErrorFeedbackProbe) -> tuple[tuple[str, str], ...]:
    _, middle, last = probe.steps
    return (
        ("Two actual reruns: higher ending error minus lower ending error",
         rf"\mathrm{{error\ change}}"
         rf"={last.loss_if_memory_higher:.12f}-{last.loss_if_memory_lower:.12f}"),
        ("Divide by the memory difference; this measures the gradient instead of assuming it",
         rf"\mathrm{{ending\ response}}"
         rf"=\frac{{{last.loss_if_memory_higher-last.loss_if_memory_lower:.12f}}}"
         rf"{{{2*error_feedback_flow.MEMORY_NUDGE:.8f}}}\approx{last.measured_response:.8f}"),
        ("To reach A3's saved h2, cross the D4 read's squash and memory multiplier",
         rf"\mathrm{{A3\ feedback}}\approx{last.feedback:.8f}"
         rf"\times{last.slope:.8f}\times{probe.gain:g}\approx{middle.feedback:.8f}"),
        ("D4's use of m: pass through its squash, then multiply by incoming h2",
         rf"\mathrm{{D4\ part}}\approx{last.feedback:.8f}"
         rf"\times{last.slope:.8f}\times{last.previous:.8f}"
         rf"\approx{last.memory_contribution:.8f}"),
        ("Add the effects of the shared dial at D3, A3 and D4",
         rf"\mathrm{{shared\ gradient}}"
         rf"\approx0+{middle.memory_contribution:.8f}+{last.memory_contribution:.8f}"
         rf"\approx{probe.weight_gradients.memory:.8f}"),
        ("Only now turn the one shared dial; reuse its new value on the next pass",
         rf"m_{{\mathrm{{new}}}}={probe.gain:g}-{LEARNING_RATE:g}"
         rf"\times{probe.weight_gradients.memory:.8f}"
         rf"\approx{probe.updated_weights.memory:.8f}"),
    )


def boards() -> tuple[tuple[str, tuple[tuple[str, str], ...]], ...]:
    return (
        ("BPTT: make a guess, then measure its mistake", FORWARD_EQUATIONS),
        ("BPTT: small changes, two bookkeeping rules", BACKWARD_EQUATIONS),
        ("D3, A3, D4: assemble one shared-weight gradient",
         worked_equations(record_probe(gain=0.8))),
    )


def math_figure(title: str, equations: tuple[tuple[str, str], ...]):
    height = 1.15 + 1.4 * len(equations)
    figure = plt.figure(figsize=(18, height), facecolor="#fffaf0")
    figure.text(0.045, 1 - 0.5 / height, title, fontsize=22, weight="bold",
                fontfamily="DejaVu Sans", color="#17334a", va="center")
    for index, (label, equation) in enumerate(equations):
        figure.text(0.045, 1 - (1.15 + index * 1.4) / height, label, fontsize=13,
                    fontfamily="DejaVu Sans", color="#516677", va="top")
        figure.text(0.045, 1 - (1.75 + index * 1.4) / height, f"${equation}$",
                    fontsize=26, color="#006d77", va="center")
    figure.text(0.045, 0.025, "Rounded readouts; full-precision arithmetic. Same weights at every read.",
                fontsize=11, fontfamily="DejaVu Sans", color="#516677")
    return figure


def source_fingerprint() -> str:
    model = json.dumps(asdict(record_probe(gain=0.8)), sort_keys=True).encode()
    return hashlib.sha256(
        Path(__file__).read_bytes() + Path(error_feedback_flow.__file__).read_bytes() + model
    ).hexdigest()


def export(directory: Path = HERE / "images") -> None:
    font_manager.findfont(FONT, fallback_to_default=False)
    directory.mkdir(exist_ok=True)
    metadata = {
        "Generator": "code-rendered BPTT equations; no image-model arithmetic",
        "SourceFingerprint": source_fingerprint(),
        "MathFont": FONT,
        "ToyGain": "0.8",
    }
    with plt.rc_context(HANDWRITING_STYLE):
        for filename, (title, equations) in zip(IMAGE_FILES, boards(), strict=True):
            figure = math_figure(title, equations)
            try:
                figure.savefig(directory / filename, dpi=140, metadata=metadata,
                               facecolor=figure.get_facecolor())
            finally:
                plt.close(figure)
    print("Saved three exact BPTT math cards using Segoe Print.")


if __name__ == "__main__":
    export()
