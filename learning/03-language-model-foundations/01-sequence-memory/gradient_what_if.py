"""A measured two-copy close-up of the saved RNN memory and its frozen head."""

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Ellipse, FancyBboxPatch
import torch
from torch.nn import functional as F

from backprop_memory_flow import NOTE_HEAD_WEIGHTS, NOTES
from gradient_memory import notecard_probe
from handwritten_notes import FONT, PAPER


IMAGE = Path(__file__).parent / "images" / "01a-gradient-what-if.png"
INK, MUTED, ACCENT = "#234875", "#596273", "#a5453e"


@dataclass(frozen=True)
class NudgeTrial:
    change: float
    memory: tuple[float, float]
    correct_probability: float
    loss: float
    guess: str


@dataclass(frozen=True)
class NudgeProbe:
    trials: tuple[NudgeTrial, ...]
    loss_difference: float
    memory_difference: float
    finite_difference: float
    exact_gradient: float


def record_nudge() -> NudgeProbe:
    saved = notecard_probe().new_hidden.detach().clone()
    head = torch.tensor(NOTE_HEAD_WEIGHTS, dtype=saved.dtype)
    target = torch.tensor([0])
    differentiable = saved.clone().requires_grad_()
    loss = F.cross_entropy((head @ differentiable).unsqueeze(0), target)
    exact = torch.autograd.grad(loss, differentiable)[0][0].item()
    trials = []
    for change in (-0.001, 0.0, 0.001):
        memory = saved.clone()
        memory[0] += change
        scores = head @ memory
        trials.append(NudgeTrial(
            change=change, memory=(memory[0].item(), memory[1].item()),
            correct_probability=scores.softmax(0)[0].item(),
            loss=F.cross_entropy(scores.unsqueeze(0), target).item(),
            guess=NOTES[int(scores.argmax())],
        ))
    error_change = trials[2].loss - trials[0].loss
    memory_change = trials[2].memory[0] - trials[0].memory[0]
    return NudgeProbe(tuple(trials), error_change, memory_change,
                      error_change / memory_change, exact)


def draw_nudge(probe: NudgeProbe) -> Figure:
    lower, original, higher = probe.trials
    font = FontProperties(fname=FONT)
    figure, axis = plt.subplots(figsize=(12, 9), facecolor="white")
    figure.subplots_adjust(left=.01, right=.99, top=.99, bottom=.01)
    axis.set(xlim=(0, 12), ylim=(0, 9))
    axis.axis("off")
    axis.imshow(plt.imread(PAPER), extent=(0, 12, 0, 9), aspect="auto")

    def text(x, y, value, *, size=19, color=INK, **kwargs):
        return axis.text(x, y, value, ha="center", va="center", fontsize=size,
                         color=color, fontproperties=font, **kwargs)

    def arrow(start, end):
        axis.annotate("", xy=end, xytext=start, arrowprops={
            "arrowstyle": "-|>", "color": INK, "lw": 1.8, "mutation_scale": 15,
        })

    text(6, 8.62, "Which memory number did we nudge?", size=27)
    text(6, 8.12, "Saved h3 after reading D4", color=MUTED)
    text(3.65, 7.64, "h3 = [", size=22, gid="memory-prefix")
    text(5.18, 7.64, f"{original.memory[0]:.6f}", size=22, gid="nudged-entry")
    text(7.23, 7.64, f", {original.memory[1]:.6f}]", size=22, gid="fixed-entry")
    axis.add_patch(Ellipse((5.18, 7.64), 1.90, .54, fill=False, edgecolor=ACCENT,
                           linewidth=2, gid="first-entry-circle"))
    text(6, 7.05, "Change only the circled entry, not a note or a weight.", color=ACCENT)
    text(6, 6.62, f"The second entry stays {original.memory[1]:.6f}.", color=MUTED)
    arrow((5.3, 6.35), (3, 6.08))
    arrow((6.7, 6.35), (9, 6.08))
    for x, trial, label in ((3, lower, "Smaller copy: subtract 0.001"),
                             (9, higher, "Larger copy: add 0.001")):
        text(x, 5.86, label, size=20)
        text(x, 5.40, f"[{trial.memory[0]:.6f}, {trial.memory[1]:.6f}]", size=21)
        arrow((x, 5.13), (x, 4.93))

    axis.add_patch(FancyBboxPatch(
        (.65, 4.12), 10.7, .78, boxstyle="round,pad=.03",
        facecolor="white", edgecolor=INK, linewidth=1.5,
    ))
    text(6, 4.64, "The SAME prediction head, with its weights fixed", size=21)
    text(6, 4.29, "Evaluate each copy separately; the correct answer is D3.", size=18, color=MUTED)
    for x, trial in ((3, lower), (9, higher)):
        arrow((x, 4.05), (x, 3.80))
        text(x, 3.52, f"Probability of D3 = {trial.correct_probability:.6f}", size=20)
        text(x, 3.06, f"Error = {trial.loss:.6f}", size=22)
        text(x, 2.64, f"Highest-scoring guess is still {trial.guess}.", size=18, color=MUTED)

    text(6, 1.99, "Error difference = larger-copy error - smaller-copy error", size=20)
    text(6, 1.52, f"{higher.loss:.6f} - {lower.loss:.6f} = {probe.loss_difference:.6f} (about)", size=21)
    text(6, 1.02, f"Memory difference = {probe.memory_difference:.3f}; "
         f"local rate = {probe.loss_difference:.6f} / {probe.memory_difference:.3f}"
         f" = {probe.finite_difference:.3f} (about)", size=18)
    text(6, .39, "The larger first entry helps D3, even before the guessed note changes.", size=20)
    return figure


def export(path: Path = IMAGE) -> Path:
    probe = record_nudge()
    record = json.dumps(asdict(probe), sort_keys=True)
    fingerprint = hashlib.sha256(
        Path(__file__).read_bytes() + record.encode() + FONT.read_bytes() + PAPER.read_bytes()
    ).hexdigest()
    figure = draw_nudge(probe)
    try:
        figure.savefig(path, dpi=150, facecolor="white", metadata={
            "NudgeProbe": record, "SourceFingerprint": fingerprint,
        })
    finally:
        plt.close(figure)
    return path


if __name__ == "__main__":
    print(f"Saved the measured two-copy comparison: {export()}")
