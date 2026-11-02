"""A measured cue-echo experiment and an offline, audio-clocked replay for 01A."""

import base64
from copy import deepcopy
from dataclasses import dataclass
import hashlib
from io import BytesIO
import json
from pathlib import Path
import uuid
import wave

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F


HERE = Path(__file__).resolve().parent
NOTES = ("D3", "A3", "D4")
MIDI = {"D3": 50, "A3": 57, "D4": 62}
MODES = ("No carry", "Before learning", "After learning", "Opening erased")
PAPER, INK = "#fffaf1", "#17324d"
COLORS = ("#b45309", "#945eaa", "#16765e", "#b73939")


def make_echo_questions():
    # Both openings have the same counts, followed by exactly the same distractions.
    answers = torch.tensor([[0, 1, 2], [2, 1, 0]], dtype=torch.long)
    distractions = torch.tensor([[1, 2, 1, 0] * 5] * 2, dtype=torch.long)
    play = torch.full((2, 1), 3, dtype=torch.long)
    prompts = torch.cat((answers, distractions, play), dim=1)
    practice = torch.cat((prompts, answers[:, :2]), dim=1)
    return prompts, practice, answers


class EchoGRU(nn.Module):
    # One shared GRUCell carries the note history; a separate head scores three notes.
    def __init__(self):
        super().__init__()
        self.cell = nn.GRUCell(4, 12)
        self.head = nn.Linear(12, 3)

    def forward(self, token_ids):
        profiles = F.one_hot(token_ids, num_classes=4).float()
        hidden = profiles.new_zeros((len(token_ids), 12))
        scores = []
        for profile in profiles.unbind(dim=1):
            hidden = self.cell(profile, hidden)
            scores.append(self.head(hidden))
        return torch.stack(scores, dim=1)


def fit_echo_model(model, practice, answers, steps=240):
    if steps < 1:
        raise ValueError("Use at least one optimizer step for the before/after comparison.")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    losses = []
    model.train()
    for _ in range(steps):
        logits = model(practice)[:, -3:]
        loss = F.cross_entropy(logits.reshape(-1, 3), answers.reshape(-1))
        losses.append(loss.item())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return losses


@dataclass(frozen=True)
class EchoStep:
    input_id: int
    old: torch.Tensor
    reset: torch.Tensor
    keep: torch.Tensor
    candidate: torch.Tensor
    retained: torch.Tensor
    written: torch.Tensor
    hidden: torch.Tensor
    logits: torch.Tensor


@dataclass(frozen=True)
class EchoTrace:
    prompt: tuple[EchoStep, ...]
    outputs: tuple[EchoStep, ...]
    generated: tuple[int, ...]


@dataclass(frozen=True)
class EchoComparison:
    traces: dict[str, tuple[EchoTrace, EchoTrace]]
    answers: torch.Tensor
    losses: tuple[float, ...]


def trace_echo(model, prompt, *, erase_opening=False, no_carry=False):
    copied = deepcopy(model).eval()
    hidden = torch.zeros((1, copied.cell.hidden_size))
    reading, outputs, generated = [], [], []

    def read(token_id):
        nonlocal hidden
        if no_carry:
            hidden = torch.zeros_like(hidden)
        profile = F.one_hot(torch.tensor([token_id]), num_classes=4).float()
        xr, xz, xn = F.linear(profile, copied.cell.weight_ih, copied.cell.bias_ih).chunk(3, -1)
        hr, hz, hn = F.linear(hidden, copied.cell.weight_hh, copied.cell.bias_hh).chunk(3, -1)
        reset, keep = torch.sigmoid(xr + hr), torch.sigmoid(xz + hz)
        candidate = torch.tanh(xn + reset * hn)
        retained, written = keep * hidden, (1 - keep) * candidate
        actual = copied.cell(profile, hidden)
        torch.testing.assert_close(retained + written, actual, rtol=1e-5, atol=1e-6)
        step = EchoStep(
            token_id, hidden.clone(), reset, keep, candidate, retained, written,
            actual, copied.head(actual),
        )
        hidden = actual
        return step

    with torch.inference_mode():
        for index, token in enumerate(prompt.tolist()):
            reading.append(read(token))
            if erase_opening and index == 2:
                hidden = torch.zeros_like(hidden)
        step = reading[-1]
        for index in range(3):
            outputs.append(step)
            guess = int(step.logits.argmax(-1).item())
            generated.append(guess)
            if index < 2:
                step = read(guess)
    return EchoTrace(tuple(reading), tuple(outputs), tuple(generated))


def compare_echo_models(before, after, prompts, answers, losses):
    traces = {}
    for mode in MODES:
        model = before if mode == "Before learning" else after
        traces[mode] = tuple(
            trace_echo(model, prompt, erase_opening=mode == "Opening erased", no_carry=mode == "No carry")
            for prompt in prompts
        )
    return EchoComparison(traces, answers.clone(), tuple(losses))


def run_echo_experiment():
    prompts, practice, answers = make_echo_questions()
    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(42)
            before = EchoGRU()
            after = deepcopy(before)
            losses = fit_echo_model(after, practice, answers)
    finally:
        torch.set_num_threads(threads)
    return compare_echo_models(before, after, prompts, answers, losses)


@dataclass(frozen=True)
class MusicFrame:
    title: str
    message: str
    note: str | None
    duration: float
    step: EchoStep | None = None
    case: int = 0
    mode: str = ""


def music_frames(comparison):
    frames = []
    for case, answer in enumerate(comparison.answers.tolist()):
        for note_id in answer:
            frames.append(MusicFrame(
                "The task: remember the opening, then echo it",
                "Two openings have identical note counts. After 20 shared distractions, the right echo still differs.",
                NOTES[note_id], 0.55, case=case,
            ))
    for mode in ("No carry", "Before learning"):
        frames.extend(_output_frames(comparison, mode))
    frames.append(MusicFrame(
        "Train on both openings; do not program the correct echo",
        f"{len(comparison.losses)} Adam steps. Practice loss: {comparison.losses[0]:.4f} -> {comparison.losses[-1]:.4f}. "
        "Only fitting uses correct previous answers. Every replay below feeds its own guesses back.",
        None, 2.4,
    ))
    for index, step in enumerate(comparison.traces["After learning"][0].prompt):
        role = "opening cue" if index < 3 else ("PLAY marker" if index == 23 else "shared distraction")
        frames.append(MusicFrame(
            f"Read {index + 1}/24: {role}",
            "Same GRU weights at every step. Current input + old hidden -> reset/keep controls -> new hidden. "
            "The 12 state coordinates are numbers, not named notes.",
            NOTES[step.input_id] if step.input_id < 3 else None, 0.38, step,
            mode="After learning",
        ))
    frames.extend(_output_frames(comparison, "After learning"))
    frames.extend(_output_frames(comparison, "Opening erased"))
    frames.append(MusicFrame(
        "The opening matters because the state carried it",
        "Erasing only the opening makes the two remaining inputs identical. "
        "This is fitted recall of two examples, not composition or an RNN-versus-GRU benchmark.",
        None, 3.0,
    ))
    return frames


def _output_frames(comparison, mode):
    messages = {
        "No carry": "Same fitted weights, but zero hidden before EVERY step. Both cues collapse to one answer.",
        "Before learning": "Initialized weights: these are actual guesses, not a deliberately scrambled target.",
        "After learning": "Read PLAY to score the first note; then feed each actual guess AND the state into the next step.",
        "Opening erased": "Same fitted weights. Zero h only after the third cue note; then read the same 20 distractions.",
    }
    return [
        MusicFrame(
            f"{mode}: opening {case + 1}, generated note {index + 1}/3",
            messages[mode], NOTES[trace.generated[index]], 0.7,
            trace.outputs[index], case, mode,
        )
        for case, trace in enumerate(comparison.traces[mode])
        for index in range(3)
    ]


def synthesize_timeline(frames, sample_rate=16000):
    chunks = []
    for frame in frames:
        time = np.arange(round(sample_rate * frame.duration)) / sample_rate
        tone = np.zeros_like(time)
        if frame.note is not None:
            frequency = 440 * 2 ** ((MIDI[frame.note] - 69) / 12)
            envelope = np.minimum(time / 0.012, 1) * np.minimum((frame.duration - time) / 0.05, 1)
            for harmonic, gain in ((1, 1.0), (2, 0.35), (3, 0.12)):
                tone += gain * np.sin(2 * np.pi * harmonic * frequency * time)
            tone *= 0.22 * np.exp(-4 * time) * envelope
        chunks.append(tone)
    return np.concatenate(chunks).astype(np.float32)


def wav_bytes(samples, sample_rate=16000):
    buffer = BytesIO()
    with wave.open(buffer, "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(sample_rate)
        stream.writeframes(np.rint(samples * 32767).astype("<i2").tobytes())
    return buffer.getvalue()


def draw_music_frame(figure, comparison, frame):
    figure.clear()
    figure.set_facecolor(PAPER)
    figure.suptitle(frame.title, color=INK, fontsize=18, weight="bold", y=0.97)
    from textwrap import fill
    figure.text(0.04, 0.84, fill(frame.message, 115), color=INK, fontsize=12, va="top")
    keyboard = figure.add_axes((0.055, 0.42, 0.39, 0.31))
    keyboard.set(xlim=(0, 8), ylim=(0, 1.45))
    keyboard.axis("off")
    white_names = ("D3", "E3", "F3", "G3", "A3", "B3", "C4", "D4")
    for index, note in enumerate(white_names):
        keyboard.add_patch(Rectangle(
            (index, 0), 0.96, 1, facecolor="#8cdb9b" if note == frame.note else "white",
            edgecolor=INK, linewidth=1.5,
        ))
        keyboard.text(index + 0.48, 0.1, note, ha="center", color=INK, fontsize=11)
    for position in (1, 3, 4, 5, 7):
        keyboard.add_patch(Rectangle((position - 0.26, 0.48), 0.48, 0.52, facecolor=INK))
    keyboard.text(0, 1.23, f"Sound: {frame.note or 'pause / PLAY marker (not a note)'}", color=INK, fontsize=12)
    if frame.note is not None:
        frequency = 440 * 2 ** ((MIDI[frame.note] - 69) / 12)
        keyboard.text(0, -0.18, f"Synthesized tone: {frequency:.2f} Hz", color=INK, fontsize=11)
    state_axis = figure.add_axes((0.59, 0.41, 0.36, 0.29))
    if frame.step is not None:
        step = frame.step
        rows = [step.old, step.reset, step.keep, step.candidate, step.retained, step.written, step.hidden]
        state_axis.imshow(torch.cat(rows).numpy(), aspect="auto", cmap="coolwarm", vmin=-1, vmax=1)
        state_axis.set_yticks(range(7), ["old h", "reset r", "keep z", "proposal", "kept", "written", "new h"])
        state_axis.set_xticks(range(12), range(12))
        state_axis.set_title("Actual GRU values: kept + written = new h", color=INK, fontsize=11)
        state_axis.tick_params(labelsize=9, colors=INK)
    else:
        state_axis.axis("off")
        state_axis.text(0, 0.9, "Opening 1: D3 A3 D4\nOpening 2: D4 A3 D3\n\nSame 20 distracting notes.\nDifferent three-note echoes.",
                        color=INK, fontsize=15, va="top", linespacing=1.5)
    figure.text(0.055, 0.29, f"Expected echo {frame.case + 1}: "
                + " ".join(NOTES[token] for token in comparison.answers[frame.case].tolist()),
                color=INK, fontsize=14, weight="bold")
    for index, mode in enumerate(MODES):
        trace = comparison.traces[mode][frame.case]
        correct = trace.generated == tuple(comparison.answers[frame.case].tolist())
        label = " ".join(NOTES[token] for token in trace.generated)
        figure.text(0.055 + (index % 2) * 0.47, 0.21 - (index // 2) * 0.075,
                    f"{mode}: {label}  [{'match' if correct else 'wrong'}]",
                    color=COLORS[index], fontsize=12,
                    weight="bold" if frame.mode == mode else "normal")
    figure.text(0.055, 0.035, "Piano = input while reading; actual predicted note while generating. "
                "Blue -1 / white 0 / red +1. Gates stay in [0,1].", color=INK, fontsize=10)


def scorecard_figure(comparison):
    figure, axes = plt.subplots(2, 1, figsize=(12, 7), gridspec_kw={"height_ratios": [1.4, 1]})
    figure.set_facecolor(PAPER)
    rows = []
    for mode in MODES:
        traces = comparison.traces[mode]
        first = sum(trace.generated[0] == int(comparison.answers[index, 0]) for index, trace in enumerate(traces))
        whole = sum(trace.generated == tuple(comparison.answers[index].tolist()) for index, trace in enumerate(traces))
        rows.append([" ".join(NOTES[token] for token in trace.generated) for trace in traces] + [f"{first}/2", f"{whole}/2"])
    axes[0].axis("off")
    axes[0].set_title("Same musical question; actual free-running guesses", color=INK, fontsize=17, pad=20)
    table = axes[0].table(
        cellText=rows, rowLabels=MODES,
        colLabels=["Echo D3 A3 D4", "Echo D4 A3 D3", "First note", "Whole echo"],
        loc="center", cellLoc="center", colWidths=[0.28, 0.28, 0.16, 0.16],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)
    for cell in table.get_celld().values():
        cell.set_text_props(color=INK)
        cell.set_facecolor(PAPER)
    axes[1].plot(range(1, len(comparison.losses) + 1), comparison.losses, color=COLORS[2])
    axes[1].set(xlabel="Optimizer step (both examples in every batch)", ylabel="Practice loss",
                title="Only training uses correct previous answers; every echo above uses its own guesses")
    axes[1].set_facecolor(PAPER)
    figure.text(0.05, 0.025, "Fitted recall of two openings, not held-out music quality. "
                "No carry and opening erased use exactly the trained weights.", color=INK, fontsize=11)
    figure.subplots_adjust(left=0.22, right=0.97, top=0.89, bottom=0.16, hspace=0.6)
    return figure


def render_replay(comparison):
    frames = music_frames(comparison)
    figure = plt.figure(figsize=(12, 7))
    pictures = []
    try:
        for frame in frames:
            draw_music_frame(figure, comparison, frame)
            buffer = BytesIO()
            figure.savefig(buffer, format="png", dpi=100, facecolor=PAPER)
            pictures.append(buffer.getvalue())
    finally:
        plt.close(figure)
    audio = wav_bytes(synthesize_timeline(frames))
    return frames, pictures, audio


def replay_html(frames, pictures, audio):
    identity = "echo-" + uuid.uuid4().hex
    images = [base64.b64encode(picture).decode("ascii") for picture in pictures]
    starts = np.cumsum([0] + [frame.duration for frame in frames[:-1]]).tolist()
    payload = json.dumps({"images": images, "starts": starts})
    audio_url = "data:audio/wav;base64," + base64.b64encode(audio).decode("ascii")
    return f"""<section id="{identity}" style="max-width:100%;color:#17324d">
<p>Press Play for sound. The audio clock advances the picture; pause or scrub to inspect a step.
Sound is synthesized locally, not a piano recording.</p>
<img src="data:image/png;base64,{images[0]}" alt="Musical GRU cue-echo walkthrough"
 style="display:block;width:100%;height:auto">
<audio controls preload="metadata" style="width:100%" src="{audio_url}"></audio>
<p><button type="button" data-action="back">Previous step</button>
<button type="button" data-action="next">Next step</button>
<label>Step <input type="range" min="0" max="{len(frames) - 1}" value="0" style="width:50%"></label>
<output>1 / {len(frames)}</output></p>
<noscript>Scripts are disabled. The audio still plays; use the notebook's static scorecard or saved GIF.</noscript>
<script>
(() => {{
  const root = document.getElementById("{identity}");
  const data = {payload};
  const audio = root.querySelector("audio"), image = root.querySelector("img");
  const slider = root.querySelector("input"), status = root.querySelector("output");
  let current = -1, timer = null;
  function show(index) {{
    if (index === current) return;
    current = index;
    image.src = "data:image/png;base64," + data.images[index];
    slider.value = index;
    status.textContent = `${{index + 1}} / ${{data.images.length}}`;
  }}
  function sync() {{
    let index = 0;
    while (index + 1 < data.starts.length && audio.currentTime >= data.starts[index + 1]) index++;
    show(index);
  }}
  function tick() {{ sync(); timer = audio.paused ? null : requestAnimationFrame(tick); }}
  function seek(index) {{
    audio.pause();
    index = Math.max(0, Math.min(data.starts.length - 1, index));
    audio.currentTime = data.starts[index];
    show(index);
  }}
  audio.addEventListener("play", () => {{ if (timer === null) tick(); }});
  audio.addEventListener("pause", () => {{ cancelAnimationFrame(timer); timer = null; sync(); }});
  audio.addEventListener("seeked", sync);
  audio.addEventListener("timeupdate", sync);
  slider.addEventListener("input", () => seek(Number(slider.value)));
  root.querySelector('[data-action="back"]').addEventListener("click", () => seek(current - 1));
  root.querySelector('[data-action="next"]').addEventListener("click", () => seek(current + 1));
  show(0);
}})();
</script></section>"""


def source_fingerprint():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def export_music(comparison):
    directory = HERE / "images"
    directory.mkdir(exist_ok=True)
    prefix = directory / "01a-musical-gru-memory"
    frames, pictures, audio = render_replay(comparison)
    prefix.with_suffix(".html").write_text(replay_html(frames, pictures, audio), encoding="utf-8")
    prefix.with_suffix(".wav").write_bytes(audio)
    metadata = {
        "SourceFingerprint": source_fingerprint(), "TrainingSteps": len(comparison.losses),
        "Distractions": 20, "SampleRate": 16000, "Loss": [comparison.losses[0], comparison.losses[-1]],
        "Predictions": {mode: [list(trace.generated) for trace in traces] for mode, traces in comparison.traces.items()},
        "Frames": [{"title": frame.title, "note": frame.note, "duration": frame.duration} for frame in frames],
        "Boundary": "fitted recall of two examples; no held-out music-quality claim",
    }
    images = [Image.open(BytesIO(picture)).convert("RGB") for picture in pictures]
    try:
        images[0].save(
            prefix.with_suffix(".gif"), save_all=True, append_images=images[1:],
            duration=[round(frame.duration * 1000) for frame in frames], loop=0,
            comment=json.dumps(metadata).encode("utf-8"),
        )
    finally:
        for image in images:
            image.close()
    figure = scorecard_figure(comparison)
    figure.savefig(prefix.with_suffix(".png"), dpi=120, facecolor=PAPER,
                   metadata={"SourceFingerprint": source_fingerprint()})
    plt.close(figure)
    print(json.dumps({key: value for key, value in metadata.items() if key != "Frames"}, indent=2))


if __name__ == "__main__":
    export_music(run_echo_experiment())
    from memory_gate_flows import refresh_image_attachments
    refresh_image_attachments()
