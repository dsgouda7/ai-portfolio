"""Follow the D3 prediction loss through its head, memory and shared weight."""

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties
import torch
from torch.nn import functional as F

from error_feedback_flow import LEARNING_RATE, record_probe
from gradient_cases import recorded_cases
from handwritten_notes import FONT, PAPER


IMAGE = Path(__file__).parent / "images" / "01a-rnn-gradient-discovery.png"
CHAIN_IMAGE = IMAGE.with_name("01a-rnn-gradient-chain.png")
INK, MUTED = "#234875", "#596273"
FONT_NAME = FontProperties(fname=FONT).get_name()
HANDWRITING_STYLE = {
    "font.family": FONT_NAME, "mathtext.fontset": "custom",
    "mathtext.rm": FONT_NAME, "mathtext.it": FONT_NAME, "mathtext.bf": FONT_NAME,
    "mathtext.fallback": "stix",
}
EQUATIONS = {
    "update": r"w_{\mathrm{new}}=w-\eta\frac{\partial L}{\partial w}",
    "error": r"L=-\log p_{\mathrm{D3}}",
    "scores": r"s=a h_3",
    "probability": r"p=\operatorname{softmax}(s)",
    "score_response": r"\frac{\partial L}{\partial s}=p-y",
    "head_response": r"\frac{\partial L}{\partial h_3}=a^T(p-y)",
    "loss_chain": r"\frac{\partial L}{\partial w}"
                  r"=\frac{\partial L}{\partial h_3}\frac{\partial h_3}{\partial w}",
    "linear": r"h_t=w h_{t-1}+u_t",
    "unrolled_forward": r"h_3(w)=w\left[w(wh_0+u_1)+u_2\right]+u_3",
    "product": r"\frac{\partial h_3}{\partial w}"
               r"=h_2+w\frac{\partial h_2}{\partial w}",
    "earlier": r"\frac{\partial h_2}{\partial w}=h_1+w h_0",
    "distribute": r"\frac{\partial h_3}{\partial w}"
                  r"=h_2+w(h_1+w h_0)=h_2+w h_1+w^2h_0",
    "assembled": r"\frac{\partial L}{\partial w}"
                 r"=\left[a^T(p-y)\right]\left(h_2+w h_1+w^2h_0\right)",
    "powers": r"\frac{dh_T}{dw}=h_{T-1}+w h_{T-2}+w^2 h_{T-3}"
              r"+\cdots+w^{T-1}h_0",
    "long_loss": r"\frac{\partial L}{\partial w}=\left[a^T(p-y)\right]"
                 r"\left(h_{T-1}+wh_{T-2}+w^2h_{T-3}+\cdots+w^{T-1}h_0\right)",
    "early_response": r"\frac{\partial L}{\partial h_{T-k}}"
                      r"=\frac{\partial L}{\partial h_T}w^k",
    "fading": r"|w|<1",
    "steady": r"|w|=1",
    "growing": r"|w|>1",
    "tanh": r"h_t=\tanh(wh_{t-1}+u_t)",
    "tanh_handoff": r"\frac{\partial h_t}{\partial h_{t-1}}=w(1-h_t^2)",
    "tanh_weight": r"\frac{\partial h_t}{\partial w}=(1-h_t^2)"
                   r"\left(h_{t-1}+w\frac{\partial h_{t-1}}{\partial w}\right)",
}
DISCOVERY_KEYS = (
    "update", "error", "scores", "probability", "score_response", "head_response",
    "loss_chain", "linear", "unrolled_forward", "product", "earlier", "distribute", "assembled",
)
CHAIN_KEYS = (
    "long_loss", "early_response", "fading", "steady", "growing",
    "tanh", "tanh_handoff", "tanh_weight",
)


@dataclass(frozen=True)
class LinearCase:
    title: str
    weight: float
    links: int
    linear_ratio: float
    tanh_ratio: float


@dataclass(frozen=True)
class WeightExample:
    weight: float
    step_size: float
    note_writes: tuple[float, ...]
    head_weights: tuple[float, ...]
    states: tuple[float, ...]
    scores: tuple[float, ...]
    probabilities: tuple[float, ...]
    score_response: tuple[float, ...]
    head_contributions: tuple[float, ...]
    memory_contributions: tuple[float, ...]
    loss: float
    head_response: float
    memory_response: float
    weight_gradient: float
    updated_weight: float
    updated_loss: float


def weight_example() -> WeightExample:
    probe = record_probe(gain=.8)
    if len(probe.steps) != 3 or tuple(probe.weights.prediction) != (-.5, 1., .25):
        raise ValueError("This derivation requires three reads and head weights [-0.5, 1, 0.25].")
    weight = torch.tensor(probe.gain, dtype=torch.float64, requires_grad=True)
    head = torch.tensor(probe.weights.prediction, dtype=torch.float64)
    writes = tuple(step.input_part for step in probe.steps)
    states = [torch.zeros((), dtype=torch.float64)]
    for write in writes:
        states.append(weight * states[-1] + write)
    scores = head * states[-1]
    loss = F.cross_entropy(scores.unsqueeze(0), torch.tensor([0]))
    score_response, head_response, gradient = torch.autograd.grad(
        loss, (scores, states[-1], weight), retain_graph=True,
    )
    memory_response = torch.autograd.grad(states[-1], weight)[0]
    updated_weight = weight.detach() - LEARNING_RATE * gradient
    updated_state = torch.zeros((), dtype=torch.float64)
    for write in writes:
        updated_state = updated_weight * updated_state + write
    updated_loss = F.cross_entropy((head * updated_state).unsqueeze(0), torch.tensor([0]))
    return WeightExample(
        weight=probe.gain, step_size=LEARNING_RATE, note_writes=writes,
        head_weights=tuple(probe.weights.prediction),
        states=tuple(state.detach().item() for state in states),
        scores=tuple(scores.detach().tolist()),
        probabilities=tuple(scores.detach().softmax(0).tolist()),
        score_response=tuple(score_response.tolist()),
        head_contributions=tuple((head * score_response).tolist()),
        memory_contributions=tuple(
            probe.gain ** (len(writes) - 1 - index) * states[index].detach().item()
            for index in range(len(writes))
        ),
        loss=loss.detach().item(), head_response=head_response.item(),
        memory_response=memory_response.item(), weight_gradient=gradient.item(),
        updated_weight=updated_weight.item(), updated_loss=updated_loss.item(),
    )


def linear_cases() -> tuple[LinearCase, ...]:
    """Keep the recorded notes/head but remove tanh and measure the same ratio."""
    result = []
    for case in recorded_cases():
        weight = torch.tensor(case["weight"], dtype=torch.float64, requires_grad=True)
        head = torch.tensor(case["probe"]["weights"]["prediction"], dtype=torch.float64)
        states = [torch.zeros((), dtype=torch.float64)]
        for step in case["probe"]["steps"]:
            states.append(step["input_part"] + weight * states[-1])
        loss = F.cross_entropy((head * states[-1]).unsqueeze(0), torch.tensor([0]))
        feedback = torch.autograd.grad(loss, states[1:])
        result.append(LinearCase(
            title=case["title"], weight=case["weight"], links=len(states) - 2,
            linear_ratio=(feedback[0] / feedback[-1]).item(), tanh_ratio=case["ratio"],
        ))
    return tuple(result)


def _paper(height: float) -> tuple[Figure, Axes]:
    font_manager.fontManager.addfont(FONT)
    figure, axis = plt.subplots(figsize=(14, height), facecolor="white")
    figure.subplots_adjust(left=.01, right=.99, top=.99, bottom=.01)
    axis.set(xlim=(0, 14), ylim=(0, height))
    axis.axis("off")
    axis.imshow(plt.imread(PAPER), extent=(0, 14, 0, height), aspect="auto")
    return figure, axis


def _line(axis: Axes, y: float, text: str, size: float = 20, color: str = INK,
          *, math: bool = False, x: float = 7) -> None:
    axis.text(x, y, f"${text}$" if math else text, ha="center", va="center",
              fontsize=size, color=color)


def discovery_figure() -> Figure:
    with plt.rc_context(HANDWRITING_STYLE):
        figure, axis = _paper(18)
        line = lambda *args, **kwargs: _line(axis, *args, **kwargs)
        line(17.48, "Follow the chain, not softmax's internal algebra.", 28)
        line(16.87, "D3, A3, D4; h3 is the final memory; w is the shared reader weight.", 20, MUTED)
        line(16.15, "1. Reduce the D3 error with ordinary gradient descent.", 22)
        line(15.47, EQUATIONS["update"], 28, math=True)
        line(14.65, EQUATIONS["error"], 28, math=True)
        line(14.03, "eta is the step size; L grades the probability of the correct note.", 19, MUTED)
        line(13.32, "2. Keep the prediction head as a named block.", 22)
        line(12.65, EQUATIONS["scores"], 28, math=True)
        line(11.95, EQUATIONS["probability"], 28, math=True)
        line(11.34, "a contains the fixed head weights; y is the one-hot correct-note target.", 19, MUTED)
        line(10.62, "3. Use the standard head response, then follow the chain.", 21)
        line(9.92, EQUATIONS["score_response"], 27, math=True)
        line(9.17, EQUATIONS["head_response"], 27, math=True)
        line(8.37, EQUATIONS["loss_chain"], 27, math=True)
        line(7.64, "4. Open how the same w made the memory.", 22)
        line(6.96, EQUATIONS["linear"], 28, math=True)
        line(6.23, EQUATIONS["unrolled_forward"], 26, math=True)
        line(5.64, "u is each note's fixed write; the starting h0 is fixed. No tanh yet.", 19, MUTED)
        line(4.93, "5. Product rule: current use plus the earlier uses.", 22)
        line(4.23, EQUATIONS["product"], 27, math=True)
        line(3.48, EQUATIONS["earlier"], 27, math=True)
        line(2.65, EQUATIONS["distribute"], 25, math=True)
        line(1.92, "6. Assemble the responses before updating the shared weight.", 21)
        line(1.12, EQUATIONS["assembled"], 25, math=True)
        line(.40, "The text below the image plugs in the values and checks the step.", 20)
    return figure


def chain_figure() -> Figure:
    with plt.rc_context(HANDWRITING_STYLE):
        figure, axis = _paper(15)
        line = lambda *args, **kwargs: _line(axis, *args, **kwargs)
        line(14.48, "The same error. A longer path back to the weight.", 27)
        line(13.84, "Keep softmax and the D3 loss; continue opening the recurrent reads.", 20, MUTED)
        line(13.10, "1. More reads give the shared weight more routes into the error.", 21)
        line(12.28, EQUATIONS["long_loss"], 24, math=True)
        line(11.54, "Each earlier use crosses more of the later multipliers.", 20, MUTED)
        line(10.81, "2. Follow one early memory's route through k later handoffs.", 21)
        line(10.04, EQUATIONS["early_response"], 28, math=True)
        for x, key, meaning in ((2.4, "fading", "Fades"), (7, "steady", "Same magnitude"),
                                 (11.6, "growing", "Amplifies")):
            line(9.03, EQUATIONS[key], 25, math=True, x=x)
            line(8.40, meaning, 20, MUTED, x=x)
        line(7.72, "These are backward responses, not memory amounts or prediction error.", 19, MUTED)
        line(6.97, "3. Only now restore the tanh in our actual RNN.", 22)
        line(6.25, EQUATIONS["tanh"], 28, math=True)
        line(5.34, EQUATIONS["tanh_handoff"], 28, math=True)
        line(4.30, EQUATIONS["tanh_weight"], 26, math=True)
        line(3.50, "One extra local response, 1 - h squared, rescales the existing chain.", 19, MUTED)
        line(2.74, "4. Check the same loss-to-weight route with the recorded values.", 21)
        line(1.98, "The text and player supply the values for this symbolic chain.", 19, MUTED)
        line(1.42, "Collect the shared weight's uses, then make one update.", 19, MUTED)
        line(.62, "A long route can be hard to teach even when the final error is clear.", 20)
    return figure


def _save(figure: Figure, path: Path, keys: tuple[str, ...]) -> Path:
    cases = json.dumps([asdict(case) for case in linear_cases()], sort_keys=True)
    example = json.dumps(asdict(weight_example()), sort_keys=True)
    transcript = json.dumps({key: EQUATIONS[key] for key in keys}, sort_keys=True)
    fingerprint = hashlib.sha256(
        Path(__file__).read_bytes() + cases.encode() + example.encode() + transcript.encode()
        + FONT.read_bytes() + PAPER.read_bytes()
    ).hexdigest()
    try:
        with plt.rc_context(HANDWRITING_STYLE):
            figure.savefig(path, dpi=150, facecolor="white", metadata={
                "LinearCases": cases, "WeightExample": example,
                "MathTranscript": transcript, "SourceFingerprint": fingerprint,
            })
    finally:
        plt.close(figure)
    return path


def export(path: Path = IMAGE) -> Path:
    return _save(discovery_figure(), path, DISCOVERY_KEYS)


def export_chain(path: Path = CHAIN_IMAGE) -> Path:
    return _save(chain_figure(), path, CHAIN_KEYS)


if __name__ == "__main__":
    print(f"Saved the loss-to-weight discovery: {export()}")
    print(f"Saved the longer-chain continuation: {export_chain()}")
