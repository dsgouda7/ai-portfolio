"""Trace the actual preview GRUs on copies, without training the main model."""

import ast
import base64
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import FancyBboxPatch
from PIL import Image, ImageSequence
import torch
from torch import nn


HERE = Path(__file__).resolve().parent
FOUNDATION = HERE.parent / "01-sequence-memory"
if str(FOUNDATION) not in sys.path:
    sys.path.insert(0, str(FOUNDATION))
from gru_layer_flow import LAYER_PHASES, LayerStep, draw_layer, excerpt, record_layer_step
from memory_gate_flows import responsive_animation_html
from language_validation import ModelProbeComparison, SourceOrderComparison, compare_source_order

IMAGE = HERE / "images" / "03a-encoder-decoder-embedding-feedback.png"
ANIMATION = HERE / "images" / "03a-encoder-decoder-state-flow.gif"
HTML_ANIMATION = ANIMATION.with_suffix(".html")
MECHANICS_IMAGE = HERE / "images" / "03a-gru-hidden-state-mechanics.png"
TOKEN_LAYERS_IMAGE = HERE / "images" / "03a-encoder-decoder-token-layers.png"
TEACHER_FORCING_IMAGE = HERE / "images" / "03a-correct-previous-word-vs-own-guess.png"
EMBEDDING_LEARNING_IMAGE = HERE / "images" / "03a-prediction-teaches-shared-table.png"
PROFILE_LEARNING_IMAGE = HERE / "images" / "03a-reader-writer-profiles-before-after.png"
ORDER_PRELUDE_IMAGE = HERE / "images" / "03a-order-to-attention.png"
BACKGROUND = "#1a1a2e"
TEXT = "#e7edf7"
FORWARD = "#58d7d1"
BACKWARD = "#ffbb78"
UPDATE = "#8cdb9b"
GENERATION = "#c5a3ff"


def save_replay(path: Path, animation: FuncAnimation) -> None:
    html = responsive_animation_html(animation)
    path.write_text("\n".join(line.rstrip() for line in html.splitlines()) + "\n", encoding="utf-8")


def refresh_saved_replay_attachment(notebook_path: Path, animation_path: Path) -> None:
    before = notebook_path.read_bytes()
    notebook = json.loads(before)
    if (json.dumps(notebook, ensure_ascii=False, indent=1) + "\n").encode("utf-8") != before:
        raise ValueError("Notebook formatting changed; update the replay attachment with a notebook-aware editor.")
    for cell in notebook["cells"]:
        bundle = cell.get("attachments", {}).get(animation_path.name)
        if bundle is not None and "image/gif" in bundle:
            bundle["image/gif"] = base64.b64encode(animation_path.read_bytes()).decode("ascii")
    if notebook_path.read_bytes() != before:
        raise RuntimeError("The notebook changed while its saved replay was being refreshed.")
    notebook_path.write_bytes((json.dumps(notebook, ensure_ascii=False, indent=1) + "\n").encode("utf-8"))


@dataclass
class GRUStepMechanics:
    # Every vector is computed from this step's actual input, previous state, and learned GRU weights.
    input_row: torch.Tensor
    old_state: torch.Tensor
    reset_gate: torch.Tensor
    update_gate: torch.Tensor
    input_proposal: torch.Tensor
    memory_proposal: torch.Tensor
    reset_memory: torch.Tensor
    candidate: torch.Tensor
    retained: torch.Tensor
    written: torch.Tensor
    new_state: torch.Tensor
    reset_from_input: torch.Tensor
    reset_from_state: torch.Tensor
    update_from_input: torch.Tensor
    update_from_state: torch.Tensor


def record_gru_step(gru: nn.GRU, input_row: torch.Tensor, old_state: torch.Tensor) -> GRUStepMechanics:
    if gru.num_layers != 1 or gru.bidirectional or not gru.bias:
        raise ValueError("The gate close-up needs a biased, one-layer, one-direction GRU.")
    if input_row.shape != (1, gru.input_size) or old_state.shape != (1, 1, gru.hidden_size):
        raise ValueError("The gate close-up needs one input row and one previous hidden state.")
    step = record_layer_step(gru, input_row, old_state[0])
    input_reset, input_update, input_proposal = step.input_mix.chunk(3)
    memory_reset, memory_update, memory_proposal = step.state_mix.chunk(3)
    return GRUStepMechanics(
        step.input_row, step.old_state, step.reset_gate, step.update_gate, input_proposal,
        memory_proposal, step.reset_gate * memory_proposal, step.candidate, step.retained,
        step.written, step.new_state, input_reset, memory_reset, input_update, memory_update,
    )


@dataclass
class FeedbackTrace:
    # One copied-model pass records temporary states, gradients, and a separate fresh-SGD update.
    logits: torch.Tensor
    loss: float
    encoder_states: tuple[torch.Tensor, ...]
    decoder_states: tuple[torch.Tensor, ...]
    encoder_gradient_norms: tuple[float, ...]
    decoder_gradient_norms: tuple[float, ...]
    embedding_gradient: torch.Tensor
    row_changes: torch.Tensor
    parameter_gradients: dict[str, torch.Tensor]
    source_rows: torch.Tensor
    writer_rows: torch.Tensor
    encoder_gradients: tuple[torch.Tensor | None, ...]
    decoder_gradients: tuple[torch.Tensor, ...]
    table_before: torch.Tensor
    table_after: torch.Tensor
    encoder_mechanics: tuple[GRUStepMechanics, ...]
    decoder_mechanics: tuple[GRUStepMechanics, ...]


@dataclass
class FeedbackComparison:
    # Connected and cut handoffs use identical weights and values; only the gradient route changes.
    connected: FeedbackTrace
    detached: FeedbackTrace
    source_ids: torch.Tensor
    target_ids: torch.Tensor
    fused_logit_error: float
    fused_gradient_error: float


def _layers(model: nn.Module) -> tuple[nn.Embedding, nn.GRU, nn.GRU, nn.Linear]:
    embedding = model.get_submodule("embedding")
    encoder = model.get_submodule("encoder")
    decoder = model.get_submodule("decoder")
    output = model.get_submodule("output")
    if not isinstance(embedding, nn.Embedding) or not isinstance(output, nn.Linear):
        raise TypeError("The trace needs the preview's embedding table and linear prediction head.")
    if not isinstance(encoder, nn.GRU) or not isinstance(decoder, nn.GRU):
        raise TypeError("The trace is for the preview's GRU reader and GRU writer.")
    if encoder.num_layers != 1 or decoder.num_layers != 1 or encoder.bidirectional or decoder.bidirectional:
        raise ValueError("This focused trace supports single-layer, one-direction GRUs.")
    if encoder.hidden_size != decoder.hidden_size:
        raise ValueError("The direct state handoff needs matching reader/writer hidden widths.")
    return embedding, encoder, decoder, output


def _trace(model: nn.Module, source: torch.Tensor, targets: torch.Tensor, detach: bool) -> FeedbackTrace:
    candidate = deepcopy(model).cpu().eval()
    candidate.zero_grad(set_to_none=True)
    embedding, encoder, decoder, output = _layers(candidate)
    before = embedding.weight.detach().clone()
    source_rows = embedding(source)
    decoder_rows = embedding(targets[:, :-1])
    encoder_states, encoder_mechanics = [], []
    state = None
    for position in range(source.shape[1]):
        previous = source_rows.new_zeros((1, 1, encoder.hidden_size)) if state is None else state
        mechanics = record_gru_step(encoder, source_rows[:, position], previous)
        _, state = encoder(source_rows[:, position:position + 1], state)
        torch.testing.assert_close(mechanics.new_state, state[0, 0].detach(), rtol=1e-5, atol=1e-6)
        encoder_mechanics.append(mechanics)
        state.retain_grad()
        encoder_states.append(state)
    if state is None:
        raise ValueError("The reader needs at least one source word.")
    decoder_state = state.detach() if detach else state
    decoder_states, decoder_mechanics, scores = [], [], []
    for position in range(decoder_rows.shape[1]):
        mechanics = record_gru_step(decoder, decoder_rows[:, position], decoder_state)
        _, decoder_state = decoder(decoder_rows[:, position:position + 1], decoder_state)
        torch.testing.assert_close(mechanics.new_state, decoder_state[0, 0].detach(), rtol=1e-5, atol=1e-6)
        decoder_mechanics.append(mechanics)
        decoder_state.retain_grad()
        decoder_states.append(decoder_state)
        scores.append(output(decoder_state[-1]).unsqueeze(1))
    logits = torch.cat(scores, dim=1)
    loss = nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), targets[:, 1:].reshape(-1),
    )
    loss.backward()
    if not detach and any(value.grad is None for value in encoder_states):
        raise RuntimeError("The connected reader states unexpectedly received no feedback.")
    if detach and any(value.grad is not None for value in encoder_states):
        raise RuntimeError("The detached handoff unexpectedly kept a reader gradient route.")
    if any(value.grad is None for value in decoder_states) or embedding.weight.grad is None:
        raise RuntimeError("The writer and shared embedding table must receive feedback.")
    gradients = {
        name: parameter.grad.detach().clone()
        for name, parameter in candidate.named_parameters() if parameter.grad is not None
    }
    embedding_gradient = embedding.weight.grad.detach().clone()
    encoder_norms = tuple(0.0 if value.grad is None else value.grad.norm().item() for value in encoder_states)
    decoder_norms = tuple(value.grad.norm().item() for value in decoder_states if value.grad is not None)
    torch.optim.SGD(candidate.parameters(), lr=0.1).step()
    return FeedbackTrace(
        logits.detach(), loss.item(),
        tuple(value.detach().clone() for value in encoder_states),
        tuple(value.detach().clone() for value in decoder_states),
        encoder_norms, decoder_norms, embedding_gradient,
        (embedding.weight.detach() - before).norm(dim=1), gradients,
        source_rows.detach().clone(), decoder_rows.detach().clone(),
        tuple(None if value.grad is None else value.grad.detach().clone() for value in encoder_states),
        tuple(value.grad.detach().clone() for value in decoder_states if value.grad is not None),
        before, embedding.weight.detach().clone(),
        tuple(encoder_mechanics), tuple(decoder_mechanics),
    )


def compare_feedback_routes(
    model: nn.Module, source_ids: torch.Tensor, target_ids: torch.Tensor, words: list[str],
) -> FeedbackComparison:
    if source_ids.ndim != 2 or source_ids.shape[0] != 1 or source_ids.shape[1] == 0:
        raise ValueError("The feedback trace needs one nonempty source sequence.")
    if target_ids.ndim != 2 or target_ids.shape[0] != 1 or target_ids.shape[1] < 2:
        raise ValueError("The feedback trace needs a start marker and checking answers.")
    if source_ids.dtype != torch.long or target_ids.dtype != torch.long:
        raise TypeError("Feedback trace inputs must use long word IDs.")
    embedding, _, _, _ = _layers(model)
    if len(words) != embedding.num_embeddings:
        raise ValueError("The trace vocabulary must match the model's embedding table.")
    source = source_ids.detach().cpu().clone()
    targets = target_ids.detach().cpu().clone()
    source_only = int(source[0, -1].item())
    if source_only in targets[0, :-1].tolist():
        raise ValueError("Use a final source word that is absent from the writer's input.")
    connected = _trace(model, source, targets, False)
    detached = _trace(model, source, targets, True)
    reference = deepcopy(model).cpu().eval()
    reference.zero_grad(set_to_none=True)
    fused_logits = reference(source, targets)
    fused_loss = nn.functional.cross_entropy(
        fused_logits.reshape(-1, fused_logits.shape[-1]), targets[:, 1:].reshape(-1),
    )
    fused_loss.backward()
    torch.testing.assert_close(connected.logits, fused_logits.detach(), rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(connected.logits, detached.logits, rtol=0, atol=0)
    gradient_error = 0.0
    for name, parameter in reference.named_parameters():
        if parameter.grad is None:
            raise RuntimeError(f"The fused preview parameter received no feedback: {name}")
        torch.testing.assert_close(
            connected.parameter_gradients[name], parameter.grad, rtol=1e-4, atol=1e-6,
        )
        gradient_error = max(
            gradient_error, (connected.parameter_gradients[name] - parameter.grad).abs().max().item(),
        )
    comparison = FeedbackComparison(
        connected, detached, source, targets,
        (connected.logits - fused_logits.detach()).abs().max().item(), gradient_error,
    )
    print("One question on copies: read", [words[index] for index in source[0].tolist()],
          "and check", [words[index] for index in targets[0, 1:].tolist()])
    print("Reader state gradient norms:", [round(value, 6) for value in connected.encoder_gradient_norms])
    print("Writer state gradient norms:", [round(value, 6) for value in connected.decoder_gradient_norms])
    print("Same forward scores after cutting the handoff:",
          (connected.logits - detached.logits).abs().max().item())
    print(f"{words[source_only]} input-row gradient: "
          f"{connected.embedding_gradient[source_only].norm().item():.6f} connected; "
          f"{detached.embedding_gradient[source_only].norm().item():.6f} cut")
    print(f"{words[source_only]} row change after fresh SGD: "
          f"{connected.row_changes[source_only].item():.6f} connected; "
          f"{detached.row_changes[source_only].item():.6f} cut")
    print("Cutting the note keeps its values but blocks feedback to the reader.")
    print("The main model is untouched. This checks a gradient route, not language quality.")
    return comparison


@dataclass
class GenerationStep:
    # Each step records the actual previous word, carried state, and scores from one decoder call.
    input_id: int
    state_before: torch.Tensor
    state_after: torch.Tensor
    logits: torch.Tensor
    predicted_id: int
    mechanics: GRUStepMechanics


@dataclass
class TokenLayerPair:
    encoder: LayerStep
    decoder: LayerStep
    head_weight: torch.Tensor
    head_bias: torch.Tensor | None
    logits: torch.Tensor


@dataclass
class StatePlayback:
    # Teacher-forced feedback and free-running generation use copies of the same model checkpoint.
    feedback: FeedbackComparison
    generation: tuple[GenerationStep, ...]
    zero_initial_logits: torch.Tensor
    omitted_state_error: float
    epochs: int
    token_layers: TokenLayerPair


def record_state_playback(
    model: nn.Module, source_ids: torch.Tensor, target_ids: torch.Tensor,
    words: list[str], *, epochs: int = 0,
) -> StatePlayback:
    if epochs < 0:
        raise ValueError("The recorded training epoch count cannot be negative.")
    if "<bos>" not in words or "<eos>" not in words:
        raise ValueError("State playback needs start and stop markers in the vocabulary.")
    feedback = compare_feedback_routes(model, source_ids, target_ids, words)
    if source_ids.shape[1] != 3 or target_ids.shape[1] != 4:
        raise ValueError("State playback follows the three-word cat probe and three writer steps.")
    if int(target_ids[0, 0]) != words.index("<bos>") or int(target_ids[0, -1]) != words.index("<eos>"):
        raise ValueError("The playback answer must start with <bos> and end with <eos>.")
    candidate = deepcopy(model).cpu().eval()
    embedding, encoder, decoder, output = _layers(candidate)
    source = feedback.source_ids
    targets = feedback.target_ids
    steps = []
    with torch.inference_mode():
        reader_step = record_layer_step(
            encoder, embedding(source)[:, -1],
            feedback.connected.encoder_mechanics[-1].old_state[None],
        )
        writer_step = record_layer_step(decoder, embedding(targets[:, 0]), reader_step.carried[None])
        first_logits = output(writer_step.output[None]).clone()
        torch.testing.assert_close(first_logits, feedback.connected.logits[:, 0], rtol=1e-5, atol=1e-6)
        token_layers = TokenLayerPair(
            reader_step, writer_step, output.weight.detach().clone(),
            None if output.bias is None else output.bias.detach().clone(), first_logits,
        )
        _, state = encoder(embedding(source))
        current = targets[:, :1]
        for _ in range(targets.shape[1] - 1):
            previous_state = state.clone()
            mechanics = record_gru_step(decoder, embedding(current)[:, 0], state)
            notes, state = decoder(embedding(current), state)
            torch.testing.assert_close(mechanics.new_state, state[0, 0], rtol=1e-5, atol=1e-6)
            logits = output(notes[:, -1])
            predicted = int(logits.argmax(dim=-1).item())
            steps.append(GenerationStep(
                int(current.item()), previous_state, state.clone(), logits.clone(), predicted, mechanics,
            ))
            if predicted == words.index("<eos>"):
                break
            current = torch.tensor([[predicted]], dtype=torch.long)
        first_row = embedding(targets[:, :1])
        omitted_notes, _ = decoder(first_row)
        zero_state = torch.zeros_like(feedback.connected.encoder_states[-1])
        zero_notes, _ = decoder(first_row, zero_state)
        omitted_state_error = (omitted_notes - zero_notes).abs().max().item()
        torch.testing.assert_close(omitted_notes, zero_notes, rtol=0, atol=0)
        zero_logits = output(zero_notes[:, -1]).clone()
    return StatePlayback(feedback, tuple(steps), zero_logits, omitted_state_error, epochs, token_layers)


def playback_stages(playback: StatePlayback, *, mechanics: bool = True) -> list[tuple[str, int]]:
    if not mechanics:
        return (
            [("overview", 0)] + [("encoder", index) for index in range(3)] + [("handoff", 0)]
            + [("teacher", index) for index in range(3)]
            + [("writer-backward", index) for index in range(2, -1, -1)]
            + [("handoff-backward", 0)]
            + [("reader-backward", index) for index in range(2, -1, -1)]
            + [("update", 0), ("detach", 0), ("zero", 0)]
            + [("generation", index) for index in range(len(playback.generation))]
            + [("finish", 0)]
        )
    return (
        [("overview", 0)]
        + [(phase, index) for index in range(3) for phase in ("encoder", "encoder-proposal", "encoder-blend")]
        + [("handoff", 0)]
        + [(phase, index) for index in range(3) for phase in ("teacher", "teacher-proposal", "teacher-blend")]
        + [("teacher-chain", 2)]
        + [("writer-backward", index) for index in range(2, -1, -1)]
        + [("handoff-backward", 0)]
        + [("reader-backward", index) for index in range(2, -1, -1)]
        + [("update", 0), ("detach", 0), ("zero", 0)]
        + [("generation", index) for index in range(len(playback.generation))]
        + [("generation-chain", len(playback.generation) - 1)]
        + [("finish", 0)]
    )


def _values(tensor: torch.Tensor) -> str:
    values = tensor.detach().reshape(-1)[:4].tolist()
    return "[" + ", ".join(f"{value:+.3f}" for value in values) + ", ...]"


def focus_coordinate(playback: StatePlayback) -> int:
    steps = playback.feedback.connected.encoder_mechanics + playback.feedback.connected.decoder_mechanics
    # Keep one numerical slot in view throughout; favor an observed keep/write mixture, not a semantic label.
    scores = torch.stack([
        step.update_gate * (1 - step.update_gate) * (step.old_state - step.candidate).abs()
        for step in steps
    ]).sum(dim=0)
    return int(scores.argmax().item())


def draw_hidden_mechanics(
    axis, playback: StatePlayback, words: list[str], phase: str, index: int, stage: int,
) -> None:
    trace = playback.feedback.connected
    is_encoder = phase.startswith("encoder")
    is_generation = phase == "generation"
    mechanics = (trace.encoder_mechanics[index] if is_encoder else
                 playback.generation[index].mechanics if is_generation else trace.decoder_mechanics[index])
    q = focus_coordinate(playback)
    role, prefix = ("ENCODER", "E") if is_encoder else ("DECODER", "D")
    word_id = (int(playback.feedback.source_ids[0, index]) if is_encoder else
               playback.generation[index].input_id if is_generation else
               int(playback.feedback.target_ids[0, index]))
    color = GENERATION if is_generation else FORWARD
    active_section = (0 if phase in {"encoder", "teacher"} else
                      1 if phase.endswith("proposal") else 2)
    previous = f"{prefix}{index}"
    current = f"{prefix}{index + 1}"
    if not is_encoder and index == 0:
        previous = "D0 = E3"
    title = ("Generation: construct a hidden state, then choose the next word" if is_generation else
             "Bring the old hidden state and current word together" if phase in {"encoder", "teacher"}
             else "Build proposed new content with the reset gate" if phase.endswith("proposal")
             else "Make the new hidden state: kept old + written proposal")
    axis.text(0.5, 0.975, title, ha="center", va="top", fontsize=18, color=color, weight="bold")
    axis.text(
        0.5, 0.923,
        f"{stage + 1}/{len(playback_stages(playback))} | {role} reads {words[word_id]!r} | "
        f"{previous} -> {current} | actual {playback.epochs}-epoch cat checkpoint",
        ha="center", fontsize=12, color=TEXT,
    )
    axis.text(0.5, 0.884, f"Follow slot {q} through the full state vector. This index has no assigned meaning.",
              ha="center", fontsize=11, color=TEXT)
    axis.plot([0.49, 0.49], [0.19, 0.85], color="#65758e", lw=1)
    lanes = [
        ("Current word profile x", mechanics.input_row, False, 0.79),
        (f"Old hidden state {previous}", mechanics.old_state, False, 0.68),
        ("Reset gate r", mechanics.reset_gate, True, 0.57),
        ("Proposed content n", mechanics.candidate, False, 0.46),
        ("Update gate z: keep fraction", mechanics.update_gate, True, 0.35),
        (f"NEW hidden state {current}", mechanics.new_state, False, 0.24),
    ]
    for label, values, gate, y in lanes:
        array = values.detach().cpu().numpy().reshape(1, -1)
        input_profile = label == "Current word profile x"
        bound = max(1.0, values.abs().max().item()) if input_profile else 1.0
        active = (y in (0.79, 0.68) if active_section == 0 else
                  y in (0.57, 0.46) if active_section == 1 else y in (0.35, 0.24))
        axis.text(0.035, y + 0.037, f"{label} [{values.numel()}]", color=TEXT, fontsize=11)
        axis.imshow(
            array, extent=(0.035, 0.46, y - 0.013, y + 0.016), aspect="auto",
            cmap="YlGn" if gate else "coolwarm", vmin=0 if gate else -bound, vmax=bound,
        )
        if active:
            axis.add_patch(FancyBboxPatch(
                (0.033, y - 0.016), 0.429, 0.035, boxstyle="square,pad=0",
                edgecolor=color, facecolor="none", linewidth=2,
            ))
        if not input_profile:
            slot_width = 0.425 / values.numel()
            axis.add_patch(FancyBboxPatch(
                (0.035 + q * slot_width, y - 0.016), slot_width, 0.035,
                boxstyle="square,pad=0", edgecolor="#ffd166", facecolor="none", linewidth=1.8,
            ))
            axis.text(0.035, y - 0.043, f"slot {q}: {values[q].item():+.4f}", color="#ffd166", fontsize=10)
    axis.text(0.52, 0.825, "1. Learned gates read BOTH x and old h",
              color=color if active_section == 0 else TEXT, fontsize=12, weight="bold")
    axis.text(
        0.52, 0.782,
        f"r = sigmoid({mechanics.reset_from_input[q]:+.3f} + ({mechanics.reset_from_state[q]:+.3f}))"
        f" = {mechanics.reset_gate[q]:.4f}", color=TEXT, fontsize=11,
    )
    axis.text(
        0.52, 0.744,
        f"z = sigmoid({mechanics.update_from_input[q]:+.3f} + ({mechanics.update_from_state[q]:+.3f}))"
        f" = {mechanics.update_gate[q]:.4f}", color=TEXT, fontsize=11,
    )
    axis.text(0.52, 0.677, "2. Propose a rewrite, do not overwrite yet",
              color=color if active_section == 1 else TEXT, fontsize=12, weight="bold")
    axis.text(
        0.52, 0.632,
        f"r * projected old-state contribution:\n"
        f"{mechanics.reset_gate[q]:.4f} * {mechanics.memory_proposal[q]:+.4f}"
        f" = {mechanics.reset_memory[q]:+.4f}", color=TEXT, fontsize=11, linespacing=1.5,
    )
    axis.text(
        0.52, 0.550,
        f"proposal = tanh(input contribution + reset memory)\n"
        f"tanh({mechanics.input_proposal[q]:+.4f} + ({mechanics.reset_memory[q]:+.4f}))"
        f" = {mechanics.candidate[q]:+.4f}", color=TEXT, fontsize=11, linespacing=1.5,
    )
    axis.text(0.52, 0.445, "3. Combine what is kept and what is written",
              color=color if active_section == 2 else TEXT, fontsize=12, weight="bold")
    axis.text(
        0.52, 0.400,
        f"KEPT: z * old h = {mechanics.update_gate[q]:.4f} * {mechanics.old_state[q]:+.4f}"
        f" = {mechanics.retained[q]:+.4f}", color=TEXT, fontsize=11,
    )
    axis.text(
        0.52, 0.355,
        f"WRITTEN: (1-z) * n = {1 - mechanics.update_gate[q]:.4f} * {mechanics.candidate[q]:+.4f}"
        f" = {mechanics.written[q]:+.4f}", color=TEXT, fontsize=11,
    )
    axis.text(
        0.52, 0.285,
        f"NEW h[{q}] = {mechanics.retained[q]:+.4f} + ({mechanics.written[q]:+.4f})"
        f" = {mechanics.new_state[q]:+.4f}", color=UPDATE, fontsize=14, weight="bold",
    )
    if is_encoder:
        next_use = (f"Carry the WHOLE {mechanics.new_state.numel()}-number {current} into "
                    + (f"the next source word {words[int(playback.feedback.source_ids[0, index + 1])]!r}."
                       if index < 2 else "the decoder as D0. It is not a predicted word."))
        detail = "The same reader weights compute new gates at each word; the state values change."
    else:
        logits = playback.generation[index].logits[0] if is_generation else trace.logits[0, index]
        predicted = int(logits.argmax())
        next_use = f"New {current} -> output head -> {len(words)} word scores -> highest {words[predicted]!r}."
        if is_generation:
            detail = ("The model chose <eos>: stop." if words[predicted] == "<eos>" else
                      f"Carry {current} AND feed the actual guess {words[predicted]!r} into the next writing step.")
        else:
            expected = words[int(playback.feedback.target_ids[0, index + 1])]
            detail = (f"Check {expected!r}; keep {current}. Next practice input is the correct {expected!r}, not the guess."
                      if index < 2 else f"Check {expected!r}; this is the last teacher-forced step.")
    axis.text(0.035, 0.130, next_use, color=color, fontsize=11, weight="bold")
    axis.text(0.035, 0.088, detail, color=TEXT, fontsize=10.5)
    axis.text(0.035, 0.038,
              "Hidden states are ACTIVATIONS, not parameter rows. Yellow marks the same numerical slot in each vector.",
              color=TEXT, fontsize=10.5)
    axis.text(0.035, 0.010, "Colors: gates 0..1; hidden/proposal -1..1; word profile scaled independently.",
              color=TEXT, fontsize=9)
    axis.set(xlim=(0, 1), ylim=(0, 1))


def draw_state_playback(
    axis, playback: StatePlayback, words: list[str], stage: int, *, mechanics: bool = True,
) -> None:
    stages = playback_stages(playback, mechanics=mechanics)
    if not 0 <= stage < len(stages):
        raise ValueError("The requested playback stage is outside the recorded sequence.")
    phase, index = stages[stage]
    trace = playback.feedback.connected
    source = playback.feedback.source_ids[0].tolist()
    writer = playback.feedback.target_ids[0, :-1].tolist()
    answers = playback.feedback.target_ids[0, 1:].tolist()
    state_width = trace.encoder_states[0].shape[-1]
    embedding_width = trace.source_rows.shape[-1]
    color = (BACKWARD if "backward" in phase or phase == "detach" else UPDATE
             if phase == "update" else GENERATION if phase in {"generation", "generation-chain", "zero"} else FORWARD)
    axis.clear()
    axis.set(xlim=(0, 1), ylim=(0, 1), facecolor=BACKGROUND)
    axis.axis("off")
    if mechanics and phase in {
        "encoder", "encoder-proposal", "encoder-blend",
        "teacher", "teacher-proposal", "teacher-blend", "generation",
    }:
        draw_hidden_mechanics(axis, playback, words, phase, index, stage)
        return
    axis.text(0.5, 0.975, "Live cat probe: profiles in, states carried, word scores out",
              ha="center", va="top", fontsize=19, color=TEXT, weight="bold")
    axis.text(0.5, 0.925,
              f"Recorded after {playback.epochs} epochs | cyan: values  orange: gradients  "
              "green: one copied SGD step  purple: generation", ha="center", fontsize=11, color=TEXT)
    axis.text(0.5, 0.89, "Solid arrows carry state vectors. Dashed arrows select the next word input.",
              ha="center", fontsize=10, color=TEXT)
    xs = (0.25, 0.51, 0.77)
    active_encoder = index if phase in {"encoder", "reader-backward"} else None
    active_decoder = index if phase in {
        "teacher", "teacher-chain", "writer-backward", "generation", "generation-chain",
    } else None

    def arrow(start, end, selected=False, arrow_color=color):
        axis.annotate("", xy=end, xytext=start, arrowprops=dict(
            arrowstyle="->", color=arrow_color if selected else "#65758e",
            linewidth=3 if selected else 1.4,
        ))

    def box(x, y, label, active):
        axis.add_patch(FancyBboxPatch(
            (x - 0.1, y - 0.047), 0.20, 0.094, boxstyle="round,pad=0.01",
            edgecolor=color if active else "#65758e",
            facecolor="#25364b" if active else BACKGROUND, linewidth=3 if active else 1.2,
        ))
        axis.text(x, y, label, color=TEXT, ha="center", va="center", fontsize=12)

    axis.text(0.025, 0.765, "ENCODER\nsource word\n+ old state\n-> new state", color=FORWARD, fontsize=11, va="center")
    axis.text(0.025, 0.515, "DECODER\nprevious word\n+ old state\n-> new state\n-> word scores", color=FORWARD, fontsize=11, va="center")
    axis.text(0.14, 0.70, "E0 = zeros", color=TEXT, fontsize=10)
    axis.text(0.14, 0.45, "D0 = zeros" if phase == "zero" else "D0 = E3",
              color=color if "handoff" in phase or phase == "zero" else TEXT, fontsize=10)
    for step, x in enumerate(xs):
        box(x, 0.765, f"{words[source[step]]} -> E{step + 1}", active_encoder == step)
        if phase in {"generation", "generation-chain"}:
            label = (f"{words[playback.generation[step].input_id]} -> D{step + 1}"
                     if step < len(playback.generation) else "not reached")
        else:
            label = f"{words[writer[step]]} -> D{step + 1}"
        box(x, 0.515, label, active_decoder == step)
        axis.text(x, 0.84, f"input: row {words[source[step]]!r}", ha="center", color=TEXT, fontsize=10)
        score_label = f"check: {words[answers[step]]}"
        if phase in {"generation", "generation-chain"}:
            score_label = (f"guess: {words[playback.generation[step].predicted_id]}"
                           if step < len(playback.generation) else "not reached")
        axis.text(x, 0.425, score_label, ha="center", color=color, fontsize=11)
        if step:
            backwards = "backward" in phase
            endpoints = ((xs[step - 1] + 0.115, 0.765), (x - 0.115, 0.765))
            arrow(*(endpoints[::-1] if backwards else endpoints),
                  selected=active_encoder == step or (backwards and active_encoder == step - 1))
            endpoints = ((xs[step - 1] + 0.115, 0.515), (x - 0.115, 0.515))
            arrow(*(endpoints[::-1] if backwards else endpoints),
                  selected=active_decoder == step or (backwards and active_decoder == step - 1))
            axis.text((xs[step - 1] + x) / 2, 0.548, "state", ha="center",
                      color=color if active_decoder == step else TEXT, fontsize=9)
            if phase in {"teacher", "teacher-chain", "generation", "generation-chain"} and (
                phase in {"teacher", "teacher-chain"} or step < len(playback.generation)
            ):
                axis.plot(
                    [xs[step - 1], xs[step - 1], x, x],
                    [0.41, 0.385, 0.385, 0.447],
                    linestyle="--", color=color, linewidth=1.5,
                )
                arrow((x, 0.447), (x, 0.458), selected=True)
                axis.text((xs[step - 1] + x) / 2, 0.37,
                          "correct word" if phase in {"teacher", "teacher-chain"} else "own guess",
                          ha="center", color=color, fontsize=9)
    handoff_color = color if "handoff" in phase else "#65758e"
    if phase != "zero":
        axis.plot([0.88, 0.94, 0.94, 0.105, 0.105], [0.765, 0.765, 0.65, 0.65, 0.515],
                  color=handoff_color, linewidth=3 if "handoff" in phase else 1.3)
        if "backward" in phase:
            arrow((0.94, 0.765), (0.88, 0.765), selected="handoff" in phase, arrow_color=handoff_color)
        else:
            arrow((0.105, 0.515), (0.135, 0.515), selected="handoff" in phase, arrow_color=handoff_color)
    axis.text(
        0.52, 0.665,
        "No encoder note supplied: D0 starts at zeros" if phase == "zero" else
        "state handoff: E3 -> D0 (not a word, not vocabulary scores)",
        ha="center", color=color if phase == "zero" else handoff_color, fontsize=11,
    )

    title, details, takeaway = "", [], ""
    if phase == "overview":
        title = "Two different chains: a numerical note and a word input"
        details = [
            f"Read: {' '.join(words[word] for word in source)} | Practice answers: "
            f"{' '.join(words[word] for word in answers)}",
            f"Word profile: {embedding_width} stored numbers. Working state: {state_width} temporary numbers.",
        ]
        takeaway = "The encoder reads source words. The decoder reads previous output words and scores the next word."
    elif phase == "encoder":
        previous = torch.zeros_like(trace.encoder_states[0]) if index == 0 else trace.encoder_states[index - 1]
        title = f"Encoder reads {words[source[index]]!r}: current profile + carried E{index} -> E{index + 1}"
        details = [f"input profile [{embedding_width}]: {_values(trace.source_rows[0, index])}",
                   f"old state E{index} [{state_width}]: {_values(previous)}",
                   f"new state E{index + 1} [{state_width}]: {_values(trace.encoder_states[index])}"]
        takeaway = "One encoder GRU reuses its weights; this computed state, not a predicted word, goes to the next read."
    elif phase == "handoff":
        title = "Finish reading: E3 initializes the decoder's D0"
        details = [f"E3 = D0 [1, 1, {state_width}] {_values(trace.encoder_states[-1])}",
                   "The decoder has different recurrent weights, but uses the same embedding table.",
                   "The first decoder word is <bos>; E3 is a second, separate input to that step."]
        takeaway = "The handoff is a vector of state values, not the predicted word 'cat'."
    elif phase == "teacher-chain":
        title = "Practice carries TWO things: the new hidden state and the correct previous word"
        details = ["Solid arrows: the complete D1/D2 state vector goes to the next GRU call.",
                   "Dashed arrows: sat and on are supplied from the training answer, not the model's guesses."]
        takeaway = "The output head's vocabulary scores are not passed as the next hidden state."
    elif phase == "teacher":
        logits = trace.logits[0, index]
        predicted = int(logits.argmax())
        previous = trace.encoder_states[-1] if index == 0 else trace.decoder_states[index - 1]
        title = f"Decoder practice: input {words[writer[index]]!r} + D{index} -> D{index + 1} -> word scores"
        details = [f"input profile {_values(trace.writer_rows[0, index])}; old state {_values(previous)}",
                   f"new state D{index + 1}: {_values(trace.decoder_states[index])}",
                   f"head's highest guess {words[predicted]!r}; checked answer {words[answers[index]]!r}"]
        takeaway = (f"Carry D{index + 1}; supply the correct {words[answers[index]]!r} as the next practice input."
                    if index < 2 else "The checked stop answer ends this practice sequence; it was not an input lookup.")
    elif phase in {"writer-backward", "reader-backward"}:
        writer_phase = phase == "writer-backward"
        gradients = trace.decoder_gradients if writer_phase else trace.encoder_gradients
        gradient = gradients[index]
        if gradient is None:
            raise RuntimeError("The connected recorded state lost its gradient.")
        name = f"{'D' if writer_phase else 'E'}{index + 1}"
        token = writer[index] if writer_phase else source[index]
        title = f"Backward through {name}: teaching feedback, not another forward state"
        details = [f"gradient at {name}: {_values(gradient)} | norm {gradient.norm().item():.6f}",
                   f"shared input row {words[token]!r} gradient: {_values(trace.embedding_gradient[token])}",
                   "Recorded gradients combine contributions from all three checking losses."]
        takeaway = "Backward fills gradient buffers. The embedding values have not been updated yet."
    elif phase == "handoff-backward":
        final_gradient = trace.encoder_gradients[-1]
        if final_gradient is None:
            raise RuntimeError("The connected reader's final state lost its gradient.")
        title = "The same E3 -> D0 link also carries feedback backward"
        details = [f"feedback into final reader state E3: {_values(final_gradient)}",
                   "Decoder loss reaches reader weights and source-word rows through this connection."]
        takeaway = "cat is never a decoder input here; its row learns through the reader."
    elif phase == "update":
        cat = source[-1]
        title = "Only now: one fresh-SGD step updates the copied model's embedding values"
        details = [f"cat row before {_values(trace.table_before[cat])}",
                   f"cat gradient  {_values(trace.embedding_gradient[cat])}",
                   f"cat row after  {_values(trace.table_after[cat])} | row change {trace.row_changes[cat].item():.6f}"]
        takeaway = "Weights and rows update; temporary hidden states are rebuilt on the next pass. Main training is untouched."
    elif phase == "detach":
        cat = source[-1]
        title = "Counterexample: pass the same state numbers, but cut the backward handoff"
        details = [f"Forward-score difference: {(trace.logits - playback.feedback.detached.logits).abs().max().item():.1f}",
                   f"cat gradient: {trace.embedding_gradient[cat].norm().item():.6f} connected / "
                   f"{playback.feedback.detached.embedding_gradient[cat].norm().item():.6f} cut"]
        takeaway = "Detaching removes a learning route; it does not remove the note's forward values."
    elif phase == "zero":
        title = "Can this decoder start without an encoder? Yes. Without a state? No."
        difference = (playback.zero_initial_logits - trace.logits[:, 0]).abs().max().item()
        details = ["Omit the GRU state argument -> PyTorch supplies an all-zero initial state.",
                   f"Omitted versus explicit-zero state output difference: {playback.omitted_state_error:.1f}",
                   f"Zero-start versus encoded-source word-score difference: {difference:.6f}"]
        takeaway = "A zero-start decoder still carries a state; it simply has no encoded cat-source information."
    elif phase in {"generation", "generation-chain"}:
        step = playback.generation[index]
        title = "Generation carries the new hidden state AND feeds back its own chosen word"
        details = [f"carried state D{index}: {_values(step.state_before)}",
                   f"new state D{index + 1}: {_values(step.state_after)}",
                   f"highest-scoring word: {words[step.predicted_id]!r} "
                   f"({step.logits.softmax(-1)[0, step.predicted_id].item():.2%})"]
        takeaway = ("The model chose <eos>: stop." if words[step.predicted_id] == "<eos>" else
                    f"Feed its own guess {words[step.predicted_id]!r} into the next step, together with the carried state.")
    else:
        title = "Same cat probe, different roles"
        details = [f"Actual generated words: {' '.join(words[step.predicted_id] for step in playback.generation)}",
                   "Encoder: source word + old state -> new state.",
                   "Decoder: previous word + old state -> new state -> vocabulary scores."]
        takeaway = "An input profile, a carried hidden state, and a guessed next word are three different objects."
    axis.text(0.04, 0.355, f"{stage + 1}/{len(stages)}  {title}", color=color,
              fontsize=15, weight="bold", va="top")
    for row, text in enumerate(details):
        axis.text(0.04, 0.295 - row * 0.052, text, color=TEXT, fontsize=12, va="top")
    axis.text(0.04, 0.065, takeaway, color=color, fontsize=11, va="top")


def animate_states(
    playback: StatePlayback, words: list[str], *, mechanics: bool = True,
) -> FuncAnimation:
    figure, axis = plt.subplots(figsize=(16, 9))
    figure.patch.set_facecolor(BACKGROUND)
    figure.subplots_adjust(left=0.02, right=0.98, bottom=0.01, top=0.99)
    stages = playback_stages(playback, mechanics=mechanics)
    animation = FuncAnimation(
        figure, lambda frame: draw_state_playback(axis, playback, words, frame // 2, mechanics=mechanics),
        frames=len(stages) * 2, interval=1100, repeat=False,
    )
    plt.close(figure)
    return animation


def token_layer_stages() -> list[tuple[str, int]]:
    return (
        [("overview", 0)] + [("encoder-layer", phase) for phase in range(len(LAYER_PHASES))]
        + [("handoff", 0)] + [("decoder-layer", phase) for phase in range(len(LAYER_PHASES))]
        + [("head", 0), ("next-token", 0), ("feedback", 0)]
    )


def draw_token_layers(figure, playback: StatePlayback, words: list[str], stage: int) -> None:
    phase, index = token_layer_stages()[stage]
    figure.clear()
    figure.set_facecolor(BACKGROUND)
    pair = playback.token_layers
    encoder_axis = figure.add_axes((0.015, 0.19, 0.47, 0.67))
    decoder_axis = figure.add_axes((0.515, 0.19, 0.47, 0.67))
    draw_layer(
        encoder_axis, pair.encoder, index if phase == "encoder-layer" else None,
        input_label="'cat' profile", old_label="E2", new_label="E3", output_use="ignored by encode()",
    )
    draw_layer(
        decoder_axis, pair.decoder, index if phase == "decoder-layer" else None,
        input_label="'<bos>' profile", old_label="D0 = E3", new_label="D1", output_use="to prediction head",
    )
    figure.text(0.5, 0.99, "Open ONE encoder token and ONE decoder token", ha="center",
                va="top", color=TEXT, fontsize=21, weight="bold")
    title = (LAYER_PHASES[index] if phase.endswith("-layer") else
             "One recurrent layer per block; time steps are repeated uses, not extra layers" if phase == "overview" else
             "Final encoder hidden E3 initializes decoder hidden D0" if phase == "handoff" else
             "Only the decoder's separate prediction head produces vocabulary scores" if phase == "head" else
             "Next token: carry the state AND supply a word input" if phase == "next-token" else
             "The loss teaches both the GRU weights and the shared input profiles")
    figure.text(0.5, 0.94, f"{stage + 1}/{len(token_layer_stages())}  {title}", ha="center",
                color=UPDATE if phase == "handoff" else FORWARD, fontsize=15)
    figure.text(0.25, 0.89, "ENCODER: cat + E2 -> E3", ha="center", color=FORWARD, fontsize=15, weight="bold")
    figure.text(0.75, 0.89, "DECODER: <bos> + D0 -> D1", ha="center", color=GENERATION, fontsize=15, weight="bold")
    figure.text(0.5, 0.864, "E3 = D0: a state-vector handoff, NOT a word passed to the decoder",
                ha="center", color=UPDATE, fontsize=11)
    vocabulary, hidden = pair.head_weight.shape
    predicted = int(pair.logits.argmax(-1).item())
    probability = pair.logits.softmax(-1)[0, predicted].item()
    head = figure.add_axes((0.53, 0.035, 0.45, 0.145))
    head.set(xlim=(0, 1), ylim=(0, 1))
    head.axis("off")
    head.add_patch(FancyBboxPatch(
        (0.01, 0.18), 0.98, 0.75, boxstyle="round,pad=0.01", facecolor=BACKGROUND,
        edgecolor=GENERATION, linewidth=3 if phase == "head" else 1,
    ))
    head.text(0.5, 0.75, f"OUTSIDE GRU: Linear {hidden} -> {vocabulary} (W_out + bias)",
              ha="center", color=GENERATION, fontsize=12)
    head.text(0.5, 0.51, f"logits {excerpt(pair.logits)} -> softmax readout",
              ha="center", color=TEXT, fontsize=11)
    head.text(0.5, 0.28, f"highest guess {words[predicted]!r}: {probability:.2%}; checked answer 'sat'",
              ha="center", color=TEXT, fontsize=11)
    if phase == "next-token":
        note = ("Next practice input: correct 'sat'.\n"
                f"Next generation input: own guess {words[predicted]!r}.\n"
                "Both also carry D1; a word is not a hidden state.")
    elif phase == "feedback":
        cat = int(playback.feedback.source_ids[0, -1])
        gradient = playback.feedback.connected.embedding_gradient[cat].norm().item()
        note = (f"Measured cat input-row gradient: {gradient:.6f}.\n"
                "Loss -> head -> D1 -> D0/E3 -> reader -> cat row.\n"
                "States are temporary; the optimizer updates weights.")
    elif phase == "handoff":
        note = (f"Final encoder state: {excerpt(pair.encoder.carried)}\n"
                f"Initial decoder state: {excerpt(pair.decoder.old_state)}\n"
                "Shared embedding table; separate encoder/decoder weights.")
    else:
        note = ("Both GRUs compute the SAME kind of cell update.\n"
                "Encoder output exists; this model keeps only its final hidden.\n"
                "Decoder output goes to a head; hidden goes to the next step.")
    figure.text(0.035, 0.14, note, color=TEXT, fontsize=11, linespacing=1.6, va="top")
    figure.text(0.5, 0.008, f"Actual {playback.epochs}-epoch cat checkpoint. "
                "One layer, one direction: output_t = carried h_t. Extra stacked layers are not used here.",
                ha="center", color=UPDATE, fontsize=10)


def animate_token_layers(playback: StatePlayback, words: list[str]) -> FuncAnimation:
    figure = plt.figure(figsize=(12, 10))
    animation = FuncAnimation(
        figure, lambda stage: draw_token_layer_zoom(figure, playback, words, stage),
        frames=len(token_layer_stages()), interval=1800, repeat=False,
    )
    plt.close(figure)
    return animation


def draw_token_layer_zoom(figure, playback: StatePlayback, words: list[str], stage: int) -> None:
    from textwrap import fill
    phase, index = token_layer_stages()[stage]
    encoder = phase in {"overview", "encoder-layer", "handoff"}
    pair = playback.token_layers
    step = pair.encoder if encoder else pair.decoder
    figure.clear()
    figure.set_facecolor(BACKGROUND)
    axis = figure.add_axes((0.015, 0.18, 0.73, 0.69))
    draw_layer(
        axis, step, index if phase.endswith("-layer") else None,
        input_label="'cat' profile" if encoder else "'<bos>' profile",
        old_label="E2" if encoder else "D0 = E3", new_label="E3" if encoder else "D1",
        output_use="not scored here" if encoder else "to Linear head",
    )
    title = ("ENCODER: one 'cat' input" if encoder else "DECODER: one '<bos>' input")
    figure.text(0.5, 0.985, title, ha="center", color=FORWARD if encoder else GENERATION,
                va="top", fontsize=20, weight="bold")
    message = (LAYER_PHASES[index] if phase.endswith("-layer") else
               "The last source word; E2 already summarizes 'the sleepy'" if phase == "overview" else
               "Copy the final encoder state E3 into the decoder's initial D0" if phase == "handoff" else
               "The separate Linear head converts GRU output into word scores" if phase == "head" else
               "The next writer step needs both a word input and D1" if phase == "next-token" else
               "Backward feedback teaches the reader too, through E3 = D0")
    figure.text(0.5, 0.935, f"{stage + 1}/{len(token_layer_stages())}  {message}",
                ha="center", color=UPDATE, fontsize=12)
    side = figure.add_axes((0.765, 0.22, 0.22, 0.65))
    side.axis("off")
    if encoder:
        sections = [
            ("Time, not depth", "the -> sleepy -> CAT. These are three uses of ONE encoder GRU layer."),
            ("Inside this layer", "Two trainable Linear projections, gates, nonlinearities and one state blend."),
            ("Output exists", "encode() discards the output sequence. It keeps final hidden E3 for decoder D0."),
        ]
    else:
        sections = [
            ("Same GRU rule", "Different recurrent weights. Same shared embedding table. D0 starts with E3."),
            ("Two outputs?", "output_t and carried h_t have identical values here. They are two uses, not two predictions."),
            ("Next time step", "Carry D1 plus correct 'sat' during practice, or the model's actual guess during generation."),
        ]
    for row, (heading, text) in enumerate(sections):
        y = 0.97 - row * 0.32
        side.text(0, y, heading, color=UPDATE, fontsize=12, weight="bold", va="top")
        side.text(0, y - 0.06, fill(text, 24), color=TEXT, fontsize=11, va="top", linespacing=1.5)
    if encoder:
        bottom = ("E3 is a hidden vector, not the word 'cat'.\n"
                  f"E3 {excerpt(pair.encoder.carried)} = D0 {excerpt(pair.decoder.old_state)}\n"
                  "The decoder separately looks up <bos> as its current input.")
    else:
        predicted = int(pair.logits.argmax(-1).item())
        vocabulary, hidden = pair.head_weight.shape
        bottom = (f"OUTSIDE GRU: output_t [{hidden}] -> Linear({hidden}, {vocabulary}) -> logits -> softmax readout\n"
                  f"Actual highest guess: {words[predicted]!r}; checked answer: 'sat'. "
                  "A guessed word is NOT the hidden state.")
        if phase == "feedback":
            cat = int(playback.feedback.source_ids[0, -1])
            gradient = playback.feedback.connected.embedding_gradient[cat].norm().item()
            bottom += f"\nLoss -> head -> D1 -> D0/E3 -> reader -> cat input row. Measured row-gradient norm {gradient:.6f}."
    figure.text(0.04, 0.115, bottom, color=GENERATION if not encoder else FORWARD,
                fontsize=11, va="top", linespacing=1.6)
    figure.text(0.5, 0.018, f"Measured {playback.epochs}-epoch checkpoint | exactly one recurrent layer per block | "
                "weights reused across tokens", ha="center", color=TEXT, fontsize=10)


def teacher_forcing_stages(playback: StatePlayback) -> list[tuple[str, int]]:
    return [("practice", index) for index in range(3)] + [
        ("generation", index) for index in range(len(playback.generation))
    ]


def draw_teacher_forcing(axis, playback: StatePlayback, words: list[str], stage: int | None = None) -> None:
    active = None if stage is None else teacher_forcing_stages(playback)[stage]
    trace = playback.feedback.connected
    inputs = playback.feedback.target_ids[0, :-1].tolist()
    answers = playback.feedback.target_ids[0, 1:].tolist()
    axis.clear()
    axis.set(xlim=(0, 1), ylim=(0, 1), facecolor=BACKGROUND)
    axis.axis("off")
    axis.text(0.5, 0.975, "What should the decoder receive AFTER a wrong guess?",
              ha="center", va="top", fontsize=20, color=TEXT, weight="bold")
    source = " ".join(words[index] for index in playback.feedback.source_ids[0].tolist())
    axis.text(0.5, 0.92, f"Source: {source!r} | same {playback.epochs}-epoch weights in both rows",
              ha="center", color=TEXT, fontsize=12)
    positions = (0.05, 0.37, 0.69)
    for mode, y, color, label in (
        ("practice", 0.59, UPDATE, "PRACTICE: supply the correct previous word (teacher forcing)"),
        ("generation", 0.23, GENERATION, "GENERATE: supply the model's own previous guess"),
    ):
        axis.text(0.04, y + 0.265, label, color=color, fontsize=14, weight="bold")
        for index, x in enumerate(positions):
            reached = mode == "practice" or index < len(playback.generation)
            axis.add_patch(FancyBboxPatch(
                (x, y), 0.26, 0.20, boxstyle="round,pad=0.01",
                edgecolor=color if reached else "#65758e",
                facecolor="#26364a" if active == (mode, index) else BACKGROUND,
                linewidth=3 if active == (mode, index) else 1.2,
            ))
            if not reached:
                axis.text(x + 0.13, y + 0.11, "Not reached:\nmodel already stopped",
                          ha="center", va="center", color=TEXT, fontsize=11)
                continue
            token = inputs[index] if mode == "practice" else playback.generation[index].input_id
            logits = trace.logits[0, index] if mode == "practice" else playback.generation[index].logits[0]
            guess = int(logits.argmax().item())
            expected = answers[index]
            axis.text(x + 0.13, y + 0.16, f"{index + 1}. Shown now: {words[token]!r}",
                      ha="center", color=color, fontsize=13, weight="bold")
            axis.text(x + 0.13, y + 0.115, f"Highest guess: {words[guess]!r}",
                      ha="center", color=TEXT, fontsize=12)
            axis.text(x + 0.13, y + 0.07,
                      f"{'Check answer' if mode == 'practice' else 'Reference only'}: {words[expected]!r}",
                      ha="center", color=UPDATE if mode == "practice" else "#94a3b8", fontsize=12)
            if mode == "practice":
                share = logits.softmax(-1)[expected].item()
                axis.text(x + 0.13, y + 0.025, f"Correct-word share: {share:.2%}",
                          ha="center", color=TEXT, fontsize=10)
            else:
                axis.text(x + 0.13, y + 0.025, "No reference word is supplied",
                          ha="center", color=TEXT, fontsize=10)
            if index < 2 and (mode == "practice" or index + 1 < len(playback.generation)):
                start_y = y + (0.07 if mode == "practice" else 0.115)
                next_x = positions[index + 1]
                axis.plot([x + 0.27, x + 0.29, next_x - 0.02],
                          [start_y, start_y, y + 0.16], color=color, lw=2, linestyle="--")
                axis.annotate("", xy=(next_x - 0.01, y + 0.16), xytext=(next_x - 0.02, y + 0.16),
                              arrowprops=dict(arrowstyle="->", color=color, lw=2))
                axis.annotate("", xy=(next_x, y + 0.225), xytext=(x + 0.26, y + 0.225),
                              arrowprops=dict(arrowstyle="->", color="#76c6ff", lw=1.5))
        axis.text(0.5, y - 0.045,
                  "Dashed: choose the NEXT word input. Blue: also carry the new hidden state; do not reset it.",
                  ha="center", color=TEXT, fontsize=10)
    axis.text(0.5, 0.105, "Giving 'sat' next does NOT turn an earlier 'and' guess into a correct answer.",
              ha="center", color=UPDATE, fontsize=13, weight="bold")
    axis.text(0.5, 0.065, "Top: check each guess, then use the real previous word. "
              "Bottom: a mistaken word becomes the next input.",
              ha="center", color=TEXT, fontsize=11)
    axis.text(0.5, 0.025, "This replays two input rules with fixed fitted weights. "
              "Training updates weights separately after the batch loss.",
              ha="center", color=TEXT, fontsize=10)


def save_teacher_forcing(playback: StatePlayback, words: list[str]) -> None:
    figure, axis = plt.subplots(figsize=(14, 9))
    figure.patch.set_facecolor(BACKGROUND)
    figure.subplots_adjust(left=0.015, right=0.985, top=0.99, bottom=0.01)
    draw_teacher_forcing(axis, playback, words)
    trace = playback.feedback.connected
    metadata = {
        "SourceFingerprint": source_fingerprint(), "TrainingEpochs": str(playback.epochs),
        "TeacherInputs": json.dumps([words[index] for index in playback.feedback.target_ids[0, :-1].tolist()]),
        "TeacherGuesses": json.dumps([words[index] for index in trace.logits[0].argmax(-1).tolist()]),
        "CheckingAnswers": json.dumps([words[index] for index in playback.feedback.target_ids[0, 1:].tolist()]),
        "GenerationInputs": json.dumps([words[step.input_id] for step in playback.generation]),
        "GenerationGuesses": json.dumps([words[step.predicted_id] for step in playback.generation]),
        "CorrectWordShares": json.dumps(trace.logits[0].softmax(-1).gather(
            1, playback.feedback.target_ids[0, 1:, None],
        ).flatten().tolist()),
    }
    figure.savefig(TEACHER_FORCING_IMAGE, dpi=110, facecolor=BACKGROUND, metadata=metadata)
    animation = FuncAnimation(
        figure, lambda stage: draw_teacher_forcing(axis, playback, words, stage),
        frames=len(teacher_forcing_stages(playback)), interval=2400, repeat=False,
    )
    plt.close(figure)
    save_replay(TEACHER_FORCING_IMAGE.with_suffix(".html"), animation)
    animation.save(TEACHER_FORCING_IMAGE.with_suffix(".gif"), writer="pillow", fps=1000 / 2400)
    print("Saved correct-previous-word versus own-guess views:", metadata["TeacherInputs"],
          "/", metadata["GenerationInputs"])


def save_hidden_mechanics(playback: StatePlayback, words: list[str]) -> None:
    figure, axis = plt.subplots(figsize=(16, 9))
    figure.patch.set_facecolor(BACKGROUND)
    figure.subplots_adjust(left=0.02, right=0.98, bottom=0.01, top=0.99)
    stage = playback_stages(playback).index(("encoder-blend", 1))
    draw_state_playback(axis, playback, words, stage)
    step = playback.feedback.connected.encoder_mechanics[1]
    q = focus_coordinate(playback)
    figure.savefig(MECHANICS_IMAGE, dpi=110, facecolor=BACKGROUND, metadata={
        "SourceFingerprint": source_fingerprint(), "TrainingEpochs": str(playback.epochs),
        "FocusCoordinate": str(q), "ResetGate": str(step.reset_gate[q].item()),
        "UpdateGate": str(step.update_gate[q].item()), "OldState": str(step.old_state[q].item()),
        "Candidate": str(step.candidate[q].item()), "Retained": str(step.retained[q].item()),
        "Written": str(step.written[q].item()), "NewState": str(step.new_state[q].item()),
        "ReconstructionError": str((
            step.new_state - playback.feedback.connected.encoder_states[1].reshape(-1)
        ).abs().max().item()),
        "Capture": "actual GRU gates and old/new state after the live cat probe reads sleepy",
    })
    plt.close(figure)
    print(f"Saved {MECHANICS_IMAGE.name}; hidden slot {q} uses measured gate values.")


def source_fingerprint() -> str:
    notebook = json.loads((HERE / "03A-training-skeleton-and-word-objectives.ipynb").read_text(encoding="utf-8"))
    sources = [
        "".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"
    ] + [
        path.read_text(encoding="utf-8") for path in (
            Path(__file__), HERE / "language_data.py", HERE / "language_model.py",
            HERE / "language_validation.py", HERE / "real_text_embeddings.py",
            FOUNDATION / "gru_layer_flow.py", FOUNDATION / "memory_gate_flows.py",
        )
    ]
    syntax = "\n".join(ast.dump(ast.parse(source)) for source in sources)
    return hashlib.sha256(syntax.encode("utf-8")).hexdigest()


def draw_feedback(comparison: FeedbackComparison, words: list[str], *, epochs: int = 0) -> None:
    if comparison.source_ids.shape[1] != 3 or comparison.target_ids.shape[1] != 4:
        raise ValueError("The saved storyboard is for three source and three writer steps.")
    figure, axes = plt.subplots(2, 1, figsize=(15, 9))
    figure.patch.set_facecolor(BACKGROUND)
    source = [words[index] for index in comparison.source_ids[0].tolist()]
    writer = [words[index] for index in comparison.target_ids[0, :-1].tolist()]
    answers = [words[index] for index in comparison.target_ids[0, 1:].tolist()]
    hidden_width = comparison.connected.encoder_states[0].shape[-1]

    def box(axis, x, y, label, color):
        axis.add_patch(FancyBboxPatch(
            (x, y), 0.19, 0.17, boxstyle="round,pad=0.015",
            edgecolor=color, facecolor=BACKGROUND, linewidth=2,
        ))
        axis.text(x + 0.095, y + 0.085, label, ha="center", va="center",
                  color=TEXT, fontsize=12)

    def arrow(axis, start, end, color):
        axis.annotate("", xy=end, xytext=start,
                      arrowprops=dict(arrowstyle="->", color=color, lw=2))

    for axis in axes:
        axis.set(xlim=(-0.08, 1.1), ylim=(-0.24, 1.06))
        axis.axis("off")
    forward = axes[0]
    forward.set_title("Forward: a word profile rewrites the running note", color=TEXT, fontsize=17)
    positions = [0.08, 0.39, 0.70]
    for index, x in enumerate(positions):
        box(forward, x, 0.60, f"E{index + 1}: after {source[index]}\n{hidden_width} state values", FORWARD)
        forward.text(x + 0.095, 0.87, f"lookup {source[index]!r}", color=FORWARD, ha="center", fontsize=12)
        arrow(forward, (x + 0.095, 0.84), (x + 0.095, 0.78), FORWARD)
        if index:
            arrow(forward, (positions[index - 1] + 0.21, 0.68), (x - 0.015, 0.68), FORWARD)
        box(forward, x, 0.10, f"D{index + 1}: input {writer[index]}\ncheck {answers[index]}", FORWARD)
        if index:
            arrow(forward, (positions[index - 1] + 0.21, 0.18), (x - 0.015, 0.18), FORWARD)
    arrow(forward, (0.90, 0.68), (1.03, 0.68), FORWARD)
    forward.plot([1.03, 1.03, -0.035, -0.035], [0.68, 0.43, 0.43, 0.18], color=FORWARD, lw=2)
    arrow(forward, (-0.035, 0.18), (0.055, 0.18), FORWARD)
    forward.text(0.48, 0.45, "E3 becomes the decoder's starting note D0; no averaging",
                 color=FORWARD, fontsize=12, ha="center", va="bottom")
    forward.text(0.48, -0.08, "Same encoder weights reused for E1-E3; separate decoder weights reused for D1-D3.",
                 color=TEXT, fontsize=11, ha="center")

    backward = axes[1]
    backward.set_title("Backward: each checked answer teaches earlier steps and input rows", color=TEXT, fontsize=17)
    for index, x in enumerate(positions):
        e_gradient = comparison.connected.encoder_gradient_norms[index]
        d_gradient = comparison.connected.decoder_gradient_norms[index]
        box(backward, x, 0.60, f"E{index + 1} feedback\nnorm {e_gradient:.4f}", BACKWARD)
        box(backward, x, 0.10, f"D{index + 1} feedback\nnorm {d_gradient:.4f}", BACKWARD)
        arrow(backward, (x + 0.095, 0.785), (x + 0.095, 0.86), BACKWARD)
        backward.text(x + 0.095, 0.89, f"row gradient: {source[index]!r}",
                      color=BACKWARD, ha="center", fontsize=11)
        arrow(backward, (x + 0.095, 0.285), (x + 0.095, 0.34), BACKWARD)
        backward.text(x + 0.095, 0.365, f"row gradient: {writer[index]!r}",
                      color=BACKWARD, ha="center", fontsize=11)
        arrow(backward, (x + 0.095, -0.05), (x + 0.095, 0.075), BACKWARD)
        backward.text(x + 0.095, -0.11, f"loss for {answers[index]}", color=BACKWARD, ha="center", fontsize=12)
        if index:
            arrow(backward, (x - 0.015, 0.68), (positions[index - 1] + 0.21, 0.68), BACKWARD)
            arrow(backward, (x - 0.015, 0.18), (positions[index - 1] + 0.21, 0.18), BACKWARD)
    backward.plot([0.065, -0.035, -0.035, 1.03, 1.03], [0.18, 0.18, 0.43, 0.43, 0.68],
                  color=BACKWARD, lw=2)
    arrow(backward, (1.03, 0.68), (0.90, 0.68), BACKWARD)
    backward.text(0.48, 0.45, "Feedback through D0 reaches E3, then E2 and E1", color=BACKWARD,
                  fontsize=12, ha="center", va="bottom")
    cat = int(comparison.source_ids[0, -1])
    figure.text(
        0.5, 0.04, f"Measured {words[cat]!r} row change: "
        f"{comparison.connected.row_changes[cat]:.6f} connected / "
        f"{comparison.detached.row_changes[cat]:.6f} with a detached handoff. Forward scores stay identical.",
        color=BACKWARD, fontsize=12, ha="center",
    )
    figure.text(0.5, 0.012, "Hidden states are temporary values. Gradients pass through them; SGD updates weights and embedding rows.",
                color=TEXT, fontsize=11, ha="center")
    figure.tight_layout(rect=(0, 0.08, 1, 1))
    figure.savefig(IMAGE, dpi=140, facecolor=BACKGROUND, bbox_inches="tight", metadata={
        "SourceFingerprint": source_fingerprint(),
        "SourceOnlyWord": words[cat],
        "ConnectedRowChange": str(comparison.connected.row_changes[cat].item()),
        "DetachedRowChange": str(comparison.detached.row_changes[cat].item()),
        "HiddenWidth": str(hidden_width),
        "TrainingEpochs": str(epochs),
        "Capture": "real single-example GRU states and gradients on copies of the preview model",
    })
    plt.close(figure)
    print(f"Saved {IMAGE.name}")


EMBEDDING_PHASES = (
    "Look up input profiles in ONE shared trainable table",
    "The reader carries the source words in order",
    "The writer uses previous words and the reader's final state",
    "Check the next-word scores; no similarity labels were supplied",
    "Feedback reaches the writer's fetched rows",
    "Feedback also crosses the reader/writer handoff to source rows",
    "One update changes the selected input rows on a COPY",
    "Cut only backward feedback: same scores, no source-row update",
)


def draw_embedding_learning(axis, comparison: FeedbackComparison, words: list[str], stage: int) -> None:
    trace = comparison.connected
    axis.clear()
    axis.set(xlim=(0, 1), ylim=(0, 1), facecolor=BACKGROUND)
    axis.axis("off")
    axis.text(0.5, 0.975, "Predict words. The input profiles learn as a side effect.",
              ha="center", va="top", fontsize=20, color=TEXT, weight="bold")
    axis.text(0.5, 0.92, f"{stage + 1}/{len(EMBEDDING_PHASES)}  {EMBEDDING_PHASES[stage]}",
              ha="center", fontsize=13, color=UPDATE if stage == 6 else BACKWARD if stage >= 4 else FORWARD)

    def box(x, y, width, title, detail, active, color=FORWARD):
        axis.add_patch(FancyBboxPatch(
            (x, y), width, 0.105, boxstyle="round,pad=0.01", edgecolor=color,
            facecolor="#26364a" if active else BACKGROUND, linewidth=3 if active else 1.2,
        ))
        axis.text(x + width / 2, y + 0.071, title, ha="center", color=TEXT, fontsize=12)
        axis.text(x + width / 2, y + 0.027, detail, ha="center", color=color, fontsize=11)

    def arrow(start, end, color=FORWARD, active=False):
        axis.annotate("", xy=end, xytext=start, arrowprops=dict(
            arrowstyle="->", color=color, lw=3 if active else 1.4,
        ))

    source = " ".join(words[index] for index in comparison.source_ids[0].tolist())
    writer = " ".join(words[index] for index in comparison.target_ids[0, :-1].tolist())
    targets = " ".join(words[index] for index in comparison.target_ids[0, 1:].tolist())
    box(0.27, 0.73, 0.46, "ONE input embedding table", "Stored word rows, shared by both lookups", stage in {0, 6})
    box(0.04, 0.48, 0.26, "Reader: source profiles", source, stage in {1, 5})
    box(0.37, 0.48, 0.22, "Final hidden state", f"{trace.encoder_states[-1].numel()} temporary values", stage in {1, 5})
    box(0.68, 0.48, 0.28, "Writer: previous-word profiles", writer, stage in {2, 4})
    box(0.68, 0.25, 0.28, "Separate output head", "One next-word score list per step", stage in {2, 3, 4})
    box(0.35, 0.25, 0.25, "Check the real next words", targets, stage in {3, 4}, BACKWARD)
    arrow((0.31, 0.73), (0.17, 0.60), active=stage == 0)
    arrow((0.69, 0.73), (0.82, 0.60), active=stage == 0)
    arrow((0.31, 0.53), (0.36, 0.53), active=stage == 1)
    arrow((0.60, 0.53), (0.67, 0.53), active=stage in {1, 2})
    arrow((0.82, 0.47), (0.82, 0.37), active=stage == 2)
    arrow((0.67, 0.30), (0.61, 0.30), active=stage == 3)
    if stage >= 4:
        arrow((0.60, 0.34), (0.67, 0.34), BACKWARD, True)
        arrow((0.74, 0.37), (0.74, 0.47), BACKWARD, True)
        arrow((0.91, 0.60), (0.70, 0.72), BACKWARD, True)
    if stage in {5, 6}:
        arrow((0.67, 0.57), (0.60, 0.57), BACKWARD, True)
        arrow((0.36, 0.57), (0.31, 0.57), BACKWARD, True)
        arrow((0.09, 0.60), (0.30, 0.72), BACKWARD, True)
    cat = int(comparison.source_ids[0, -1])
    if stage == 6:
        note = (f"cat row before: {excerpt(trace.table_before[cat])}\n"
                f"cat feedback: {excerpt(trace.embedding_gradient[cat])}\n"
                f"cat row after copied update: {excerpt(trace.table_after[cat])}")
    elif stage == 7:
        note = (f"Forward-score change after cutting feedback: "
                f"{(trace.logits - comparison.detached.logits).abs().max().item():.1f}\n"
                f"cat row movement: {trace.row_changes[cat].item():.6f} connected / "
                f"{comparison.detached.row_changes[cat].item():.6f} cut\n"
                "The source word does not need to be an answer to receive teaching feedback.")
    else:
        note = ("IDs select stored rows; hidden states carry this example's context.\n"
                "The loss trains the fetched values, both GRUs and the separate prediction head.\n"
                "No instruction says 'make cat and dog similar'. Useful organization is an empirical outcome.")
    axis.text(0.045, 0.19, note, va="top", color=UPDATE if stage == 6 else TEXT,
              fontsize=12, linespacing=1.5)
    axis.text(0.5, 0.025, "Fresh pre-training model; one question checks sat / on / stop. "
              "The diagnostic SGD step uses a COPY; the real model is untouched.",
              ha="center", color=TEXT, fontsize=10)


def save_embedding_learning(comparison: FeedbackComparison, words: list[str]) -> None:
    figure, axis = plt.subplots(figsize=(13, 9))
    figure.patch.set_facecolor(BACKGROUND)
    figure.subplots_adjust(left=0.015, right=0.985, top=0.99, bottom=0.01)
    draw_embedding_learning(axis, comparison, words, 5)
    cat = int(comparison.source_ids[0, -1])
    figure.savefig(EMBEDDING_LEARNING_IMAGE, dpi=120, facecolor=BACKGROUND, metadata={
        "SourceFingerprint": source_fingerprint(), "Checkpoint": "fresh model before training",
        "ConnectedCatRowChange": str(comparison.connected.row_changes[cat].item()),
        "DetachedCatRowChange": str(comparison.detached.row_changes[cat].item()),
    })
    animation = FuncAnimation(
        figure, lambda stage: draw_embedding_learning(axis, comparison, words, stage),
        frames=len(EMBEDDING_PHASES), interval=2100, repeat=False,
    )
    plt.close(figure)
    save_replay(EMBEDDING_LEARNING_IMAGE.with_suffix(".html"), animation)
    animation.save(EMBEDDING_LEARNING_IMAGE.with_suffix(".gif"), writer="pillow", fps=1000 / 2100)


def save_profile_learning(comparison: ModelProbeComparison, words: list[str], epochs: int) -> None:
    visible = ("cat", "dog", "mat", "sat", "<bos>", "<eos>")
    ids = [words.index(word) for word in visible]
    before, after = comparison.before.embeddings[ids], comparison.after.embeddings[ids]
    bound = max(before.abs().max().item(), after.abs().max().item())
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    figure.patch.set_facecolor(BACKGROUND)
    for axis, rows, title in zip(axes, (before, after), ("Before training", f"After {epochs} Adam epochs")):
        axis.imshow(rows.numpy(), cmap="coolwarm", vmin=-bound, vmax=bound, aspect="auto")
        axis.set_yticks(range(len(visible)), visible)
        axis.set_xlabel("Stored feature coordinate (no hand-assigned meaning)", color=TEXT)
        axis.set_title(title, color=TEXT)
        axis.tick_params(colors=TEXT)
    pair_values = {}
    for word in ("dog", "mat"):
        index = words.index(word)
        scores = []
        for rows in (comparison.before.embeddings, comparison.after.embeddings):
            scores.append(nn.functional.cosine_similarity(
                rows[words.index("cat")][None], rows[index][None],
            ).item())
        pair_values[f"cat/{word}"] = scores
    figure.suptitle("Actual input rows from the SAME six-epoch run", color=TEXT, fontsize=17)
    figure.text(0.5, 0.10, "Full-row similarity: " + " | ".join(
        f"{pair} {scores[0]:+.3f} -> {scores[1]:+.3f}" for pair, scores in pair_values.items()
    ), ha="center", color=UPDATE, fontsize=12)
    figure.text(0.5, 0.025, "Changed values prove updates, not general semantic quality. "
                "Compare related words with story-context words; do not promise every pair moves closer.",
                ha="center", color=TEXT, fontsize=10)
    figure.tight_layout(rect=(0.01, 0.16, 0.99, 0.93))
    figure.savefig(PROFILE_LEARNING_IMAGE, dpi=120, facecolor=BACKGROUND, metadata={
        "SourceFingerprint": source_fingerprint(), "TrainingEpochs": str(epochs),
        "PairSimilarities": json.dumps(pair_values),
        "EOSRowChange": str(comparison.row_changes[words.index("<eos>")].item()),
    })
    plt.close(figure)


def save_order_prelude(order: SourceOrderComparison, words: list[str]) -> None:
    if order.source_ids.shape[1] != 3:
        raise ValueError("The order storyboard follows the three-word cat source.")
    figure, axis = plt.subplots(figsize=(14, 8))
    figure.patch.set_facecolor(BACKGROUND)
    axis.set(xlim=(0, 1), ylim=(0, 1), facecolor=BACKGROUND)
    axis.axis("off")
    axis.text(0.5, 0.98, "Order is already in the recurrent states. Attention changes access.",
              ha="center", va="top", color=TEXT, fontsize=19, weight="bold")
    positions = (0.08, 0.39, 0.70)
    for y, ids, states, label in (
        (0.69, order.source_ids, order.ordered_states, "Read in order"),
        (0.46, order.reversed_ids, order.reversed_states, "Reverse the same rows"),
    ):
        axis.text(0.03, y + 0.14, label, color=FORWARD, fontsize=13, weight="bold")
        for index, x in enumerate(positions):
            axis.add_patch(FancyBboxPatch(
                (x, y), 0.22, 0.10, boxstyle="round,pad=0.01",
                edgecolor=FORWARD, facecolor=BACKGROUND, linewidth=1.5,
            ))
            axis.text(x + 0.11, y + 0.065, words[int(ids[0, index])], ha="center", color=TEXT, fontsize=14)
            axis.text(x + 0.11, y + 0.02, excerpt(states[0, index]), ha="center", color=FORWARD, fontsize=10)
            if index:
                axis.annotate("", xy=(x - 0.015, y + 0.05), xytext=(positions[index - 1] + 0.235, y + 0.05),
                              arrowprops=dict(arrowstyle="->", color=FORWARD, lw=2))
    axis.text(0.5, 0.37,
              f"Same table, no updates | mean-row difference {order.mean_difference:.2e} (rounding) | "
              f"final-state difference {order.hidden_difference:.4f}",
              ha="center", color=UPDATE, fontsize=11)
    axis.text(0.5, 0.31, f"First word-score difference: {order.score_difference:.4f}. "
              "This proves order affects computation, not correct understanding.",
              ha="center", color=TEXT, fontsize=11)
    axis.text(0.06, 0.20, "Current writer: receives only the final summary.", color=TEXT, fontsize=13)
    axis.text(0.06, 0.13, "Attention idea: keep every source-position state and let each writer step look back.",
              color=UPDATE, fontsize=13, weight="bold")
    axis.text(0.06, 0.055, "Attention is not implemented here. In this GRU model, recurrence supplies order; "
              "attention would provide selective access to the already ordered notes.", color=TEXT, fontsize=10)
    figure.savefig(ORDER_PRELUDE_IMAGE, dpi=120, facecolor=BACKGROUND, metadata={
        "SourceFingerprint": source_fingerprint(), "MeanDifference": str(order.mean_difference),
        "HiddenDifference": str(order.hidden_difference), "ScoreDifference": str(order.score_difference),
        "AttentionImplemented": "False",
    })
    plt.close(figure)


def main() -> None:
    if not (HERE / "data" / "tinystories" / "TinyStories-valid.txt").is_file():
        raise FileNotFoundError("Run the lesson once to obtain its verified TinyStories cache.")
    notebook = json.loads((HERE / "03A-training-skeleton-and-word-objectives.ipynb").read_text(encoding="utf-8"))
    cells = notebook["cells"]
    shared_end = next(index for index, cell in enumerate(cells) if cell["id"] == "06c-16")
    selected = [
        cell for cell in cells[:shared_end] if cell["cell_type"] == "code"
    ] + [
        next(cell for cell in cells if cell["id"] == cell_id)
        for cell_id in (
            "ch03a-training-base", "ch03a-word-model-base", "06c-17",
            "ch03a-probe-helpers", "06c-s2s-02", "06c-s2s-04",
            "ch03a-sequence-probe-setup",
        )
    ]
    namespace = {"__name__": "__main__", "__file__": str(HERE / "03A-training-skeleton-and-word-objectives.py")}
    for cell in selected:
        exec(compile("".join(cell["source"]), cell["id"], "exec"), namespace)
    training_source = "".join(next(cell["source"] for cell in cells if cell["id"] == "ch03a-sequence-training"))
    exec(compile(training_source, "actual-preview-training", "exec"), namespace)
    playback = record_state_playback(
        namespace["sequence_model"], namespace["source_sequences"][-3:-2],
        namespace["sequence_targets"][-3:-2], namespace["sequence_itos"],
        epochs=len(namespace["sequence_losses"]),
    )
    words = namespace["sequence_itos"]
    save_embedding_learning(namespace["embedding_feedback"], words)
    save_profile_learning(namespace["sequence_after_record"], words, playback.epochs)
    order = compare_source_order(namespace["sequence_model"], namespace["cat_sources"][:1], namespace["BOS_ID"])
    save_order_prelude(order, words)
    draw_feedback(playback.feedback, words, epochs=playback.epochs)
    save_hidden_mechanics(playback, words)
    save_teacher_forcing(playback, words)
    figure = plt.figure(figsize=(18, 11))
    draw_token_layers(figure, playback, words, token_layer_stages().index(("head", 0)))
    figure.savefig(TOKEN_LAYERS_IMAGE, dpi=110, facecolor=BACKGROUND, metadata={
        "SourceFingerprint": source_fingerprint(), "TrainingEpochs": str(playback.epochs),
        "EncoderToken": "cat", "DecoderToken": "<bos>", "RecurrentLayers": "1",
        "OutputEqualsCarried": str(torch.equal(
            playback.token_layers.decoder.output, playback.token_layers.decoder.carried,
        )),
    })
    plt.close(figure)
    animation = animate_token_layers(playback, words)
    save_replay(HTML_ANIMATION, animation)
    animation.save(ANIMATION, writer="pillow", fps=1000 / 1800)
    metadata = {
        "SourceFingerprint": source_fingerprint(), "TrainingEpochs": playback.epochs,
        "Stages": token_layer_stages(),
        "Source": [words[index] for index in playback.feedback.source_ids[0].tolist()],
        "CheckingAnswers": [words[index] for index in playback.feedback.target_ids[0, 1:].tolist()],
        "Generated": [words[step.predicted_id] for step in playback.generation],
        "OmittedStateError": playback.omitted_state_error,
        "FocusCoordinate": focus_coordinate(playback),
        "GateMechanics": "PyTorch GRU: reset projected memory; keep z*old; write (1-z)*candidate",
        "PlaybackView": "one-token-through-each-layer",
        "EncoderToken": "cat", "DecoderToken": "<bos>", "RecurrentLayers": 1,
        "GRUOutputEqualsCarriedState": all(
            torch.equal(step.output, step.carried)
            for step in (playback.token_layers.encoder, playback.token_layers.decoder)
        ),
    }
    with Image.open(ANIMATION) as image:
        durations = [frame.info["duration"] for frame in ImageSequence.Iterator(image)]
        frames = [frame.copy().convert("RGB") for frame in ImageSequence.Iterator(image)]
    frames[0].save(
        ANIMATION, save_all=True, append_images=frames[1:], duration=durations,
        loop=0, comment=json.dumps(metadata).encode("utf-8"),
    )
    print(f"Saved {ANIMATION.name}: {len(frames)} recorded stages, "
          f"generated words {metadata['Generated']}; {HTML_ANIMATION.name} has replay controls.")
    refresh_saved_replay_attachment(HERE / "03A-training-skeleton-and-word-objectives.ipynb", ANIMATION)


if __name__ == "__main__":
    main()
