"""Self-contained, fixed-board LSTM/GRU replays for 01A.

Public API: ``record_story(kind)`` returns JSON-safe, full-precision records;
``render_player(kind)`` returns standalone HTML; ``companion_narration(kind)``
returns notebook-ready prose; ``export(directory=HERE / "images")`` writes only
the four new sequence-storyboard assets. Run this file to export both stories.

Each fresh phrase starts from zero h (and zero c for LSTM). The compact SVG
keeps every saved card in place; a separate CSS-sized readout shows active math.
Inputs reuse D3=[1, 0], A3=[0, 1], D4=[1, 1] from the RNN example.
Weights are hand-picked, not trained. Predictions use a three-note vocabulary,
and only the final prediction is graded against D3. Native torch cells supply
the forward pass and boundary VJPs. An equivalent local equation replay exposes
LSTM's internal c gradient (the native fused output c has a separate boundary
gradient); every replay is checked against the native cell.
"""

from __future__ import annotations

import html
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import torch
from torch import nn
from torch.nn import functional as F

from handwritten_notes import paper_style


HERE = Path(__file__).resolve().parent
NOTES = ("D3", "A3", "D4")
BG, PANEL, TEXT, MUTED = "#081426", "#142b42", "#f4f1e8", "#a9bad1"
GOLD, CYAN, VIOLET = "#ffcd76", "#75e1e3", "#c5a6ff"
BACK_H, BACK_C = "#ff9c8d", "#f28dce"
WIDTH, HEIGHT = 1200, 662
__all__ = ["record_story", "render_player", "companion_narration", "export"]


def _kind(kind: str) -> str:
    value = kind.lower()
    if value not in ("lstm", "gru"):
        raise ValueError("kind must be 'lstm' or 'gru'")
    return value


def _model(kind):
    # Construction is isolated so recording never changes the caller's RNG.
    with torch.random.fork_rng(devices=[]):
        cell = (nn.LSTMCell if kind == "lstm" else nn.GRUCell)(2, 2).double()
        head = nn.Linear(2, 3).double()
    if kind == "lstm":
        weights = {
            "weight_ih": [[.35, -.30], [-.25, .40],
                          [.45, -.20], [-.35, .25],
                          [.80, -.45], [-.60, .75],
                          [.25, -.40], [-.45, .30]],
            "weight_hh": [[.20, -.10], [.15, .25], [-.15, .20], [.30, -.10],
                          [.40, -.25], [-.20, .35], [.15, .30], [-.25, .20]],
            "bias_ih": [-.10, .15, .65, .25, .10, -.05, .20, -.10],
            "bias_hh": [.05, -.10, .10, -.05, -.05, .10, -.10, .05],
        }
    else:
        weights = {
            "weight_ih": [[.40, -.35], [-.30, .45],
                          [.35, -.25], [-.20, .40],
                          [.70, -.50], [-.55, .65]],
            "weight_hh": [[.25, -.15], [.20, .30], [-.20, .25],
                          [.30, -.10], [.45, -.25], [-.30, .40]],
            "bias_ih": [.10, -.15, .45, .05, .05, -.10],
            "bias_hh": [-.05, .10, .10, -.05, .15, -.10],
        }
    with torch.no_grad():
        for name, value in weights.items():
            getattr(cell, name).copy_(torch.tensor(value, dtype=torch.float64))
        head.weight.copy_(torch.tensor([[.65, -.40], [-.30, .80], [.45, .35]],
                                       dtype=torch.float64))
        head.bias.copy_(torch.tensor([-.15, .10, .20], dtype=torch.float64))
    return cell, head


def _equations(kind, cell, x, old_h, old_c=None):
    a = F.linear(x, cell.weight_ih, cell.bias_ih)
    b = F.linear(old_h, cell.weight_hh, cell.bias_hh)
    values = {"input_projection": a, "hidden_projection": b}
    if kind == "lstm":
        i, f, g, o = (a + b).chunk(4)
        write, keep, proposal, show = i.sigmoid(), f.sigmoid(), g.tanh(), o.sigmoid()
        retained, written = keep * old_c, write * proposal
        c = retained + written
        h = show * c.tanh()
        values.update(keep=keep, write=write, proposal=proposal, show=show,
                      retained=retained, written=written, c=c, h=h)
    else:
        xr, xz, xn = a.chunk(3)
        hr, hz, hn = b.chunk(3)
        reset, keep = (xr + hr).sigmoid(), (xz + hz).sigmoid()
        reset_branch = reset * hn
        proposal = (xn + reset_branch).tanh()
        retained, written = keep * old_h, (1 - keep) * proposal
        h = retained + written
        values.update(reset=reset, keep=keep, write=1 - keep,
                      proposal_hidden_projection=hn, reset_branch=reset_branch,
                      proposal=proposal, retained=retained, written=written, h=h)
    return values


def _list(tensor):
    return tensor.detach().tolist()


def _pair(values, places=3):
    return "[" + ", ".join(f"{v:+.{places}f}" for v in values) + "]"


def _frames(kind, steps, loss):
    two = kind == "lstm"
    result = [{
        "phase": "overview", "column": -1, "focus": "overview",
        "label": "Start: three reads, one fixed board",
        "narration": ("Start a fresh phrase with zero h0 and zero c0. " if two else
                      "Start a fresh phrase with zero h0. ") +
                     "Dim rows preview recorded work. Follow the readable arithmetic below; "
                     "all three columns and saved cards stay in place.",
    }]
    for t, step in enumerate(steps):
        old = f"h{t} {_pair(step['old_h'])}"
        if two:
            old += f" and c{t} {_pair(step['old_c'])}"
        details = [
            ("input", "read", f"Read {step['note']} with {old}. "
             "The same cell weights are reused in every column."),
            ("gates", "per-number controls",
             ("KEEP, WRITE and SHOW each have two controls, not one universal dial. "
              "The proposal supplies content; it is not a gate. Gates read the note and old h."
              if two else
              "RESET changes only the projected old-h branch inside the proposal. "
              "Each number has its own RESET and KEEP z; WRITE is 1-z.")),
            ("blend", "retain and write",
             ("KEEP times old c gives retained content. WRITE times the proposal gives "
              "written content. Add them to get the new c."
              if two else
              "KEEP times old h plus WRITE times the proposal makes the new h. "
              "There is no tanh after this blend.")),
            ("save", "save the state",
             (f"Save c{t + 1} {_pair(step['c'])}. SHOW times tanh(c) makes "
              f"h{t + 1} {_pair(step['h'])}. BOTH continue into the next read."
              if two else
              f"Save the ONE h{t + 1} {_pair(step['h'])}. The very same two numbers "
              "continue into the next read and enter the note head.")),
            ("head", "guess, not yet graded",
             f"The head reads only h{t + 1}; it guesses {step['guess']}. "
             + ("No c is sent to the head. " if two else "")
             + ("This earlier guess is not graded." if t < 2 else
                "Only this last guess will be graded against D3.")),
        ]
        for focus, label, narration in details:
            result.append({"phase": "forward", "column": t, "focus": focus,
                           "label": f"Forward {t + 1}/3: {label}", "narration": narration})
    result.append({
        "phase": "grade", "column": 2, "focus": "grade", "label": "Grade only the final guess",
        "narration": f"The target after D4 is D3. Cross-entropy is {loss:.4f}. "
                     "Only the last head contributes feedback. Loss, probability error, "
                     "and state gradient are different quantities.",
    })
    for t in (2, 1, 0):
        detail = f"The final grade sends feedback to saved h{t + 1}. "
        if two:
            detail += (
                "Feedback to c combines the SHOW route and the later c route. "
                "KEEP scales that total along the direct old-c route. "
                "Feedback to old h also follows the gates."
            )
        else:
            detail += ("Backprop follows both the direct KEEP route and the proposal/gate "
                       "routes into old h. The full derivative is not just KEEP.")
        result.append({"phase": "backward", "column": t, "focus": "feedback",
                       "label": f"Backward {3 - t}/3: revisit the saved {steps[t]['note']} read",
                       "narration": detail})
    result.append({
        "phase": "done", "column": -1, "focus": "done", "label": "Complete: one grade, all three reads",
        "narration": "Feedback reached the initial state through the saved computation. "
                     "No weights changed. Hand-picked weights show the mechanism; "
                     "these gates and guesses are not learned music.",
    })
    for frame in result:
        frame["readout"] = _active_values(kind, steps, loss, frame)
    return result


def _active_values(kind, steps, loss, frame):
    """Four short, actual-CSS-sized readouts; the board is the spatial overview."""
    two = kind == "lstm"
    focus = frame["focus"]
    step = steps[max(0, frame["column"])]

    def item(label, value, tone="h"):
        return {"label": label, "value": value, "tone": tone}

    if focus == "overview":
        return [
            item("Fresh h0", _pair(steps[0]["old_h"])),
            item("Fresh c0", _pair(steps[0]["old_c"]), "c") if two else
            item("Saved state", "ONE h; no separate c"),
            item("Input order", "D3, A3, D4", "plain"),
            item("Grade", "Last guess only; target D3", "head"),
        ]
    if focus == "input":
        return [
            item("Current note and input code",
                 f"{step['note']} [{step['x'][0]:g}, {step['x'][1]:g}]", "plain"),
            item(f"Incoming h{step['index']}", _pair(step["old_h"])),
            item(f"Incoming c{step['index']}", _pair(step["old_c"]), "c") if two else
            item("Same h", "From the preceding saved card"),
            item("Cell weights", "Reused in all three reads", "plain"),
        ]
    if focus == "gates":
        controls = ([("KEEP", "keep", "c"), ("WRITE", "write", "c"),
                     ("proposal", "proposal", "plain"), ("SHOW", "show", "h")] if two else
                    [("RESET", "reset", "head"), ("KEEP z", "keep", "h"),
                     ("WRITE 1-z", "write", "h"), ("proposal", "proposal", "plain")])
        return [item(f"{label}: numbers #0 and #1", _pair(step[key]), tone)
                for label, key, tone in controls]
    if focus == "blend":
        old = step["old_c" if two else "old_h"]
        rows = []
        for n in (0, 1):
            rows.extend([
                item(f"Retained #{n}: KEEP × old " + ("c" if two else "h"),
                     f"{step['keep'][n]:.3f} × {old[n]:.3f} = {step['retained'][n]:.3f}",
                     "c" if two else "h"),
                item(f"Written #{n}: WRITE × proposal",
                     f"{step['write'][n]:.3f} × {step['proposal'][n]:.3f} = {step['written'][n]:.3f}",
                     "plain"),
            ])
        return rows
    if focus == "save":
        if two:
            return [
                item(f"Saved c{step['index'] + 1}: retained + written", _pair(step["c"]), "c"),
                item("New h #0: SHOW × tanh(c)",
                     f"{step['show'][0]:.3f} × tanh({step['c'][0]:.3f}) = {step['h'][0]:.3f}"),
                item("New h #1: SHOW × tanh(c)",
                     f"{step['show'][1]:.3f} × tanh({step['c'][1]:.3f}) = {step['h'][1]:.3f}"),
                item("Two channels continue", "Both h and c; only h to head", "plain"),
            ]
        return [
            *[item(f"New h #{n}: retained + written",
                   f"{step['retained'][n]:.3f} + {step['written'][n]:.3f} = {step['h'][n]:.3f}")
              for n in (0, 1)],
            item(f"Saved h{step['index'] + 1}", _pair(step["h"])),
            item("Same h, two uses", "Next read + note head; no extra tanh", "plain"),
        ]
    if focus == "head":
        return [
            item("Head reads only h", _pair(step["head_input"])),
            item("Note probabilities",
                 "  ".join(f"{note} {p:.3f}" for note, p in zip(NOTES, step["probabilities"])), "head"),
            item("Next-note guess", step["guess"], "head"),
            item("Grade", "Not graded" if step["index"] < 2 else "Next: grade against D3", "plain"),
        ]
    if focus == "grade":
        return [item("Target after D4", "D3", "plain"),
                item("Actual guess", steps[-1]["guess"], "head"),
                item("Probability of D3", f"{steps[-1]['probabilities'][0]:.4f}", "head"),
                item("Loss, not state feedback", f"{loss:.4f}", "back_h")]
    if focus == "feedback":
        fb = step["feedback"]
        if two:
            return [
                item(f"Feedback to h{step['index'] + 1}", _pair(fb["dh"], 4), "back_h"),
                item(f"Feedback to c{step['index'] + 1}", _pair(fb["dc"], 4), "back_c"),
                item(f"Feedback to old h{step['index']}", _pair(fb["to_previous_h"], 4), "back_h"),
                item(f"Feedback to old c{step['index']}: KEEP × total",
                     _pair(fb["to_previous_c"], 4), "back_c"),
            ]
        return [
            item(f"Feedback to h{step['index'] + 1}", _pair(fb["dh"], 4), "back_h"),
            item(f"Feedback to old h{step['index']}", _pair(fb["to_previous_h"], 4), "back_h"),
            item("Direct KEEP route", _pair(fb["direct_h_carry"], 4), "back_h"),
            item("Proposal + gate routes", _pair(fb["proposal_and_gate_dh"], 4), "back_h"),
        ]
    first = steps[0]["feedback"]
    return [
        item("Feedback reached initial h0", _pair(first["to_previous_h"], 4), "back_h"),
        item("Feedback reached initial c0", _pair(first["to_previous_c"], 4), "back_c") if two else
        item("Saved state", "ONE h; no separate c"),
        item("Final loss", f"{loss:.4f}", "plain"),
        item("Weight updates", "None; weights were hand-picked", "plain"),
    ]


def record_story(kind: str) -> dict:
    """Return deterministic JSON-safe records of native forward and autograd.

    ``feedback.dh`` is the saved h output's boundary gradient.
    LSTM ``feedback.future_dc`` is the native c output's boundary gradient;
    ``feedback.dc`` joins it with the local SHOW route. ``to_previous_*`` are
    native-cell VJPs, not estimates or copies of the prediction error.
    All values are float64 CPU results; no optimizer or notebook loader runs.
    """
    kind = _kind(kind)
    cell, head = _model(kind)
    x = torch.tensor([[1., 0.], [0., 1.], [1., 1.]],
                     dtype=torch.float64, requires_grad=True)
    h0 = torch.zeros(2, dtype=torch.float64, requires_grad=True)
    c0 = (torch.zeros(2, dtype=torch.float64, requires_grad=True)
          if kind == "lstm" else None)
    h, c = h0, c0
    graph, steps = [], []
    for t, note in enumerate(NOTES):
        old_h, old_c = h, c
        if kind == "lstm":
            h, c = cell(x[t], (old_h, old_c))
            c.retain_grad()
        else:
            h = cell(x[t], old_h)
        h.retain_grad()
        logits = head(h)
        logits.retain_grad()
        with torch.no_grad():
            values = _equations(kind, cell, x[t], old_h, old_c)
            torch.testing.assert_close(values["h"], h, rtol=0, atol=1e-14)
            if kind == "lstm":
                torch.testing.assert_close(values["c"], c, rtol=0, atol=1e-14)
        step = {"index": t, "note": note, "x": _list(x[t]), "old_h": _list(old_h),
                **{name: _list(value) for name, value in values.items()},
                "h": _list(h),
                "logits": _list(logits), "probabilities": _list(logits.softmax(0)),
                "guess": NOTES[int(logits.argmax())], "graded": t == 2,
                "head_input": _list(h), "carry_h": _list(h)}
        if kind == "lstm":
            step.update(old_c=_list(old_c), c=_list(c), carry_c=_list(c))
        graph.append((h, c, logits))
        steps.append(step)
    loss = F.cross_entropy(graph[-1][2].unsqueeze(0), torch.tensor([0]))
    loss.backward()
    for t, ((h, c, logits), step) in enumerate(zip(graph, steps)):
        dh = h.grad
        dlogits = logits.grad if logits.grad is not None else torch.zeros(3, dtype=torch.float64)
        head_dh = head.weight.detach().T @ dlogits
        old_h = torch.tensor(step["old_h"], dtype=torch.float64, requires_grad=True)
        local_x = torch.tensor(step["x"], dtype=torch.float64, requires_grad=True)
        feedback = {"dh": _list(dh), "dlogits": _list(dlogits),
                    "head_dh": _list(head_dh), "future_dh": _list(dh - head_dh)}
        if kind == "lstm":
            old_c = torch.tensor(step["old_c"], dtype=torch.float64, requires_grad=True)
            boundary_dc = c.grad if c.grad is not None else torch.zeros_like(c)
            native_h, native_c = cell(local_x, (old_h, old_c))
            previous = torch.autograd.grad((native_h, native_c), (old_h, old_c, local_x),
                                           (dh, boundary_dc))
            local = _equations(kind, cell, local_x, old_h, old_c)
            show_dc = torch.autograd.grad(local["h"], local["c"], dh, retain_graph=True)[0]
            dc = show_dc + boundary_dc
            replay_previous = torch.autograd.grad((local["h"], local["c"]),
                                                  (old_h, old_c, local_x), (dh, boundary_dc))
            for native, replay in zip(previous, replay_previous):
                torch.testing.assert_close(native, replay, rtol=1e-12, atol=1e-14)
            direct_c = local["keep"].detach() * dc
            torch.testing.assert_close(direct_c, previous[1], rtol=1e-12, atol=1e-14)
            feedback.update(future_dc=_list(boundary_dc), through_show_dc=_list(show_dc),
                            dc=_list(dc), direct_c_carry=_list(direct_c),
                            to_previous_c=_list(previous[1]))
            dx = previous[2]
        else:
            native_h = cell(local_x, old_h)
            previous = torch.autograd.grad(native_h, (old_h, local_x), dh)
            direct_h = torch.tensor(step["keep"], dtype=torch.float64) * dh
            feedback.update(direct_h_carry=_list(direct_h),
                            proposal_and_gate_dh=_list(previous[0] - direct_h))
            dx = previous[1]
        feedback.update(to_previous_h=_list(previous[0]), dx=_list(dx))
        step["feedback"] = feedback
    for t in (1, 2):
        torch.testing.assert_close(
            torch.tensor(steps[t]["feedback"]["to_previous_h"]),
            torch.tensor(steps[t - 1]["feedback"]["dh"]),
        )
        if kind == "lstm":
            torch.testing.assert_close(
                torch.tensor(steps[t]["feedback"]["to_previous_c"]),
                torch.tensor(steps[t - 1]["feedback"]["future_dc"]),
            )
    loss_value = float(loss.detach())
    return {
        "schema_version": 1, "kind": kind, "notes": list(NOTES), "target": "D3",
        "target_index": 0, "dtype": "torch.float64", "input_size": 2, "hidden_size": 2,
        "source": f"torch.nn.{type(cell).__name__} forward; torch.autograd backward",
        "weight_origin": "Hand-picked initial weights; no training and no weight update.",
        "gate_origin": "Computed separately for each number from the note and old h.",
        "loss_scope": "Only the final head after D4 is graded against D3.",
        "loss": loss_value, "initial_h": _list(h0),
        "initial_c": _list(c0) if kind == "lstm" else None,
        "cell_parameters": {name: _list(value) for name, value in cell.named_parameters()},
        "head_parameters": {name: _list(value) for name, value in head.named_parameters()},
        "initial_feedback": {"dh": _list(h0.grad),
                             "dc": _list(c0.grad) if kind == "lstm" else None},
        "parameter_gradients": {name: _list(value.grad) for name, value in cell.named_parameters()},
        "head_parameter_gradients": {name: _list(value.grad) for name, value in head.named_parameters()},
        "input_gradients": _list(x.grad), "steps": steps,
        "frames": _frames(kind, steps, loss_value),
        "gradient_convention": (
            "dh is the saved h boundary gradient. future_dc is the native saved c boundary "
            "gradient; joined dc = future_dc + through_show_dc. Native-cell VJPs include "
            "both output routes; direct old-c feedback = KEEP * joined dc."
            if kind == "lstm" else
            "dh is the saved h gradient; old-h feedback includes the direct KEEP route "
            "plus the proposal and gate routes."
        ),
    }


def companion_narration(kind: str) -> str:
    """Return concise notebook-ready instructions; no notebook access."""
    kind = _kind(kind)
    mechanism = (
        "Each number gets its own KEEP, WRITE and SHOW control. KEEP scales old c; "
        "WRITE scales the proposal. Add those pieces to save new c. SHOW exposes "
        "a squashed view as h. Carry BOTH h and c onward, but send only h to the note head."
        if kind == "lstm" else
        "There is ONE saved h. RESET changes the old-h projection inside the proposal, "
        "not the carry lane. KEEP z retains old h; WRITE 1-z admits the proposal. "
        "Add them without another tanh. The same h goes to the next read and the note head."
    )
    feedback = (
        "At each c card, join feedback arriving through SHOW with feedback from the later "
        "c route. KEEP scales this joined feedback along the direct old-c route; "
        "old-h feedback also travels through the gates."
        if kind == "lstm" else
        "Old-h feedback includes both the direct KEEP route and the proposal/gate routes; "
        "it is not just KEEP times the incoming feedback."
    )
    return (
        "**Watch one fixed board, not three changing screens.**\n\n"
        "1. Start a fresh phrase with zero h0" + (" and zero c0" if kind == "lstm" else "") +
        ". Read D3, A3, then D4. Each column keeps its controls and saved cards. "
        "The full-sized readout below the board shows the active arithmetic.\n"
        f"2. {mechanism}\n"
        "3. Grade only the final guess against D3. The earlier guesses are visible, not graded.\n"
        f"4. Walk backward across the same three columns. {feedback} "
        "Signed gradients describe loss sensitivity, not prediction error.\n\n"
        "The two-number states come from actual torch cells and autograd. "
        "Weights were hand-picked; neither the gates nor the music were learned here."
    )


class _Scene:
    """One primitive list drives both the SVG player and the PNG fallback."""

    def __init__(self):
        self.ops = []
        self.owner, self.reveal, self.role = None, 0, "overview"

    def scope(self, owner=None, reveal=0, role="overview"):
        self.owner, self.reveal, self.role = owner, reveal, role

    def add(self, tag, **attrs):
        self.ops.append((self.owner, self.reveal, self.role, tag, attrs))

    def text(self, x, y, value, color=TEXT, size=17, weight="normal", anchor="start"):
        self.add("text", x=x, y=y, value=value, fill=color, size=size,
                 weight=weight, anchor=anchor)

    def box(self, x, y, w, h, color=CYAN, fill=PANEL, width=1.5):
        self.add("rect", x=x, y=y, width=w, height=h, stroke=color,
                 fill=fill, line_width=width, rx=10)

    def line(self, points, color, width=2):
        self.add("polyline", points=points, fill="none", stroke=color, line_width=width)

    def arrow(self, points, color, width=2):
        self.line(points, color, width)
        x, y = points[-1]
        px, py = points[-2]
        angle = math.atan2(y - py, x - px)
        base_x, base_y = x - 9 * math.cos(angle), y - 9 * math.sin(angle)
        corners = [(x, y), (base_x - 4 * math.sin(angle), base_y + 4 * math.cos(angle)),
                   (base_x + 4 * math.sin(angle), base_y - 4 * math.cos(angle))]
        self.add("polygon", points=corners, fill=color, stroke=color, line_width=1)

    def card(self, x, y, w, h, color):
        fold = 15
        self.add("polygon", points=[(x, y), (x + w - fold, y), (x + w, y + fold),
                                    (x + w, y + h), (x, y + h)],
                 fill=PANEL, stroke=color, line_width=2)
        self.line([(x + w - fold, y), (x + w - fold, y + fold),
                   (x + w, y + fold)], color, 1)

    def keyboard(self, x, y, note):
        white = [m for m in range(48, 73) if m % 12 not in (1, 3, 6, 8, 10)]
        active = {"D3": 50, "A3": 57, "D4": 62}[note]
        for n, pitch in enumerate(white):
            self.add("rect", x=x + n * 9, y=y, width=8.5, height=28,
                     fill=GOLD if pitch == active else "#e2e8ec",
                     stroke="#324050", line_width=.5, rx=0)
            if pitch % 12 in (0, 2, 5, 7, 9) and pitch < 72:
                self.add("rect", x=x + n * 9 + 6, y=y, width=5, height=16,
                         fill=BG, stroke="#67768b", line_width=.3, rx=0)

    def svg(self, story):
        root = ET.Element("svg", {
            "xmlns": "http://www.w3.org/2000/svg", "viewBox": f"0 0 {WIDTH} {HEIGHT}",
            "role": "img", "aria-labelledby": "board-title board-description",
            "preserveAspectRatio": "xMidYMin meet", "id": "sequence-board",
        })
        ET.SubElement(root, "title", id="board-title").text = (
            f"{story['kind'].upper()}: D3, A3, D4 forward and backward on one board")
        ET.SubElement(root, "desc", id="board-description").text = (
            "All three note columns stay in place. Gold c, cyan h, violet predictions, "
            "coral h feedback and pink c feedback. Values are recorded torch results.")
        ET.SubElement(root, "rect", width=str(WIDTH), height=str(HEIGHT), fill=BG)
        owners, groups = {}, {}
        for owner, reveal, role, tag, attrs in self.ops:
            if owner not in owners:
                owners[owner] = ET.SubElement(root, "g", {
                    "id": "board-routes" if owner is None else f"note-column-{owner}",
                    "class": "routes" if owner is None else "note-column",
                    **({} if owner is None else {"data-note": story["notes"][owner]}),
                })
            key = owner, reveal, role
            if key not in groups:
                groups[key] = ET.SubElement(owners[owner], "g", {
                    "class": "stage-group", "data-reveal": str(reveal), "data-role": role,
                    "data-column": str(owner if owner is not None else -1),
                })
            a = dict(attrs)
            value = a.pop("value", None)
            if "points" in a:
                a["points"] = " ".join(f"{x},{y}" for x, y in a["points"])
            if "line_width" in a:
                a["stroke-width"] = a.pop("line_width")
            if tag == "text":
                a["font-size"] = a.pop("size")
                a["font-weight"] = a.pop("weight")
                a["text-anchor"] = a.pop("anchor")
                a["font-family"] = "DejaVu Sans,Arial,sans-serif"
            element = ET.SubElement(groups[key], tag, {k: str(v) for k, v in a.items()})
            if value is not None:
                element.text = value
        # Draw wires over the column backgrounds, never underneath the inputs.
        root.remove(owners[None])
        root.append(owners[None])
        return ET.tostring(root, encoding="unicode")

    def png(self, path, story):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon, Rectangle

        figure, axis = plt.subplots(figsize=(WIDTH / 90, HEIGHT / 90), dpi=120, facecolor=BG)
        figure.subplots_adjust(left=0, right=1, top=1, bottom=0)
        axis.set(xlim=(0, WIDTH), ylim=(HEIGHT, 0))
        axis.axis("off")
        for _, _, _, tag, a in self.ops:
            if tag == "text":
                axis.text(a["x"], a["y"], a["value"], color=a["fill"],
                          fontsize=a["size"] * .8, fontweight=a["weight"], va="baseline",
                          ha={"start": "left", "middle": "center", "end": "right"}[a["anchor"]])
            elif tag == "rect":
                axis.add_patch(Rectangle((a["x"], a["y"]), a["width"], a["height"],
                                         facecolor=a["fill"], edgecolor=a["stroke"],
                                         linewidth=a["line_width"]))
            elif tag == "polygon":
                axis.add_patch(Polygon(a["points"], facecolor=a["fill"], edgecolor=a["stroke"],
                                       linewidth=a["line_width"]))
            else:
                axis.plot(*zip(*a["points"]), color=a["stroke"], linewidth=a["line_width"])
        figure.savefig(path, facecolor=BG, metadata={
            "Description": "Static overview; all recorded forward and backward stages are visible.",
            "StoryKind": story["kind"], "Source": story["source"],
            "Loss": str(story["loss"]), "WeightOrigin": story["weight_origin"],
            "InitialFeedback": json.dumps(story["initial_feedback"], separators=(",", ":")),
        })
        plt.close(figure)


def _board(story):
    s = _Scene()
    two = story["kind"] == "lstm"
    name = story["kind"].upper()
    s.text(16, 29, f"{name} | Three reads. One fixed board.", size=26, weight="bold")
    s.text(1184, 29, "Final guess only: target D3.", size=18, anchor="end")
    s.text(16, 86, "Fresh phrase: h0 = c0 = 0" if two else "Fresh phrase: h0 = 0", MUTED, 18)
    positions = [16, 416, 816]
    for t, (left, step) in enumerate(zip(positions, story["steps"])):
        s.scope(t)
        s.box(left, 110, 368, 480, "#314b68", BG)
        s.text(left + 18, 139, f"{t + 1}   {step['note']}", GOLD, 26, "bold")
        start = 1 + t * 5
        s.scope(t, start, "input")
        s.keyboard(left + 154, 117, step["note"])
        incoming = f"Read with h{t}" + (f" and c{t}" if two else " (one state)")
        s.text(left + 18, 165, incoming, MUTED, 18)
        s.scope(t, start + 1, "gates")
        s.text(left + 180, 187, "#0", MUTED, 16)
        s.text(left + 278, 187, "#1", MUTED, 16)
        labels = ([("KEEP", "keep", GOLD), ("WRITE", "write", GOLD),
                   ("proposal", "proposal", TEXT), ("SHOW", "show", CYAN)] if two else
                  [("RESET", "reset", VIOLET), ("KEEP z", "keep", CYAN),
                   ("WRITE 1-z", "write", CYAN), ("proposal", "proposal", TEXT)])
        for row, (label, key, color) in enumerate(labels):
            y = 210 + 23 * row
            s.text(left + 20, y, label, color, 19)
            s.text(left + 176, y, f"{step[key][0]:.3f}", color, 20)
            s.text(left + 274, y, f"{step[key][1]:.3f}", color, 20)
        s.scope(t, start + 2, "blend")
        s.text(left + 20, 302, "KEEP × old + WRITE × proposal", MUTED, 17)
        s.scope(t, start + 3, "save")
        if two:
            s.card(left + 20, 313, 326, 57, GOLD)
            s.text(left + 34, 335, f"Saved c{t + 1} = retained + written", GOLD, 19)
            s.text(left + 34, 360, _pair(step["c"]), GOLD, 24, "bold")
            s.arrow([(left + 148, 372), (left + 148, 391)], CYAN)
            s.text(left + 165, 386, "SHOW × tanh(c)", CYAN, 17)
        else:
            s.box(left + 20, 313, 326, 57, VIOLET)
            s.text(left + 34, 336, "RESET: old-h proposal branch only", VIOLET, 18)
            s.text(left + 34, 359, "Blend once; no extra tanh.", TEXT, 18)
            s.arrow([(left + 148, 372), (left + 148, 391)], CYAN)
        s.card(left + 20, 395, 326, 57, CYAN)
        s.text(left + 34, 417, f"Saved h{t + 1}", CYAN, 20)
        s.text(left + 34, 442, _pair(step["h"]), CYAN, 24, "bold")
        s.scope(t, start + 4, "head")
        s.arrow([(left + 148, 454), (left + 148, 477)], VIOLET)
        s.text(left + 165, 472, "only h to head" if two else "same h to head", VIOLET, 17)
        s.box(left + 20, 480, 326, 40, VIOLET)
        s.text(left + 34, 507, f"Guess: {step['guess']}", VIOLET, 23, "bold")
        s.scope(t, 16 if t == 2 else start + 4, "grade" if t == 2 else "head")
        s.text(left + 20, 541, f"Target D3 | loss {story['loss']:.4f}" if t == 2 else "Not graded",
               BACK_H if t == 2 else MUTED, 17)
        s.scope(t, 17 + 2 - t, "feedback")
        fb = step["feedback"]
        s.text(left + 20, 562, f"feedback to h{t + 1}  {_pair(fb['dh'], 4)}", BACK_H, 16)
        if two:
            s.arrow([(left + 20, 421), (left + 10, 421), (left + 10, 341),
                     (left + 20, 341)], BACK_C)
            s.text(left + 20, 584, f"feedback to c{t + 1}  {_pair(fb['dc'], 4)}", BACK_C, 16)
        else:
            s.text(left + 20, 584, "KEEP route + proposal/gate routes", BACK_H, 16)
        if t == 2:
            s.arrow([(left + 346, 500), (left + 352, 500),
                     (left + 352, 423), (left + 346, 423)], BACK_H)
    for t, left in enumerate(positions):
        start = 1 + t * 5
        s.scope(None, start + 3, "carry")
        if t < 2:
            nxt = positions[t + 1]
            if two:
                s.arrow([(left + 346, 341), (left + 377, 341), (left + 377, 93),
                         (nxt + 360, 93), (nxt + 360, 168), (nxt + 345, 168)], GOLD)
                s.text(nxt + 10, 87, f"carry c{t + 1}", GOLD, 17)
            s.arrow([(left + 346, 423), (left + 386, 423), (left + 386, 72),
                     (nxt + 350, 72), (nxt + 350, 151), (nxt + 340, 151)], CYAN)
            s.text(nxt + 10, 67, f"carry h{t + 1}", CYAN, 17)
        else:
            s.text(948, 89, "h3 + c3 continue" if two else "h3 continues", GOLD if two else CYAN, 17)
            if two:
                s.arrow([(left + 346, 341), (1195, 341)], GOLD)
            s.arrow([(left + 346, 423), (1195, 423)], CYAN)
        s.scope(None, 17 + 2 - t, "feedback")
        source = left + 180
        destination = positions[t - 1] + 352 if t else 30
        h_points = [(source, 592), (source, 612), (destination, 612)]
        if t:
            h_points += [(destination, 423), (destination - 6, 423)]
        s.arrow(h_points, BACK_H, 2.5)
        if two:
            c_points = [(source + 15, 592), (source + 15, 635), (destination + 8, 635)]
            if t:
                c_points += [(destination + 8, 341), (destination - 6, 341)]
            s.arrow(c_points, BACK_C, 2.5)
    s.scope(None)
    s.text(16, 656, "Hand-picked weights. Actual torch values. No training.", MUTED, 17)
    s.text(1184, 656, "Active arithmetic below the player", MUTED, 17, anchor="end")
    return s


_STYLE = """
*{box-sizing:border-box}html,body{margin:0;width:100%;background:#081426;color:#f4f1e8;
font-family:Arial,sans-serif}body{min-height:100vh;display:flex;flex-direction:column}
main{flex:0 1 auto;min-height:0;width:100%;aspect-ratio:1200/662}
svg{width:100%;height:100%;display:block}
.stage-group{transition:opacity .25s}.current polyline{stroke-dasharray:8 4;animation:flow 1s linear infinite}
.current[data-role="gates"] text{font-size:24px;font-weight:600}
@keyframes flow{to{stroke-dashoffset:-24}}.note-column.active>g:first-child>rect{stroke:#ffcd76;stroke-width:3}
body.backward .note-column.active>g:first-child>rect{stroke:#ff9c8d}
footer{flex:none;padding:20px 24px}
#phase{font:24px/1.4 LessonHand,cursive;color:#23334a;margin:0 0 16px}
#active-values{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:18px;margin:0 0 18px}
.value-cell{min-height:82px;padding:4px 8px}
.value-label{display:block;color:#596273;font:18px/1.5 LessonHand,cursive;margin-bottom:6px}
.active-value{display:block;font:22px/1.65 LessonHand,cursive;overflow-wrap:anywhere;font-weight:400}
[data-tone="c"]{color:#80521c}[data-tone="h"]{color:#234875}[data-tone="head"]{color:#60428b}
[data-tone="back_h"]{color:#9e3a32}[data-tone="back_c"]{color:#8e3569}
#narration{font:18px/1.6 LessonHand,cursive;margin:0 0 16px;color:#23334a}
nav{display:flex;align-items:center;gap:8px;flex-wrap:wrap}
button{font-size:14px;background:#fff;color:#234875;border:1px solid #8794a2;border-radius:6px;
padding:5px 13px;cursor:pointer}button:disabled{opacity:.4;cursor:default}button:focus-visible{outline:2px solid #75e1e3}
#progress{color:#596273;font:13px Arial,sans-serif;margin-left:8px}
@media(max-width:850px){#active-values{grid-template-columns:repeat(2,minmax(0,1fr))}}
@media(prefers-reduced-motion:reduce){.stage-group{transition:none}.current polyline{animation:none}}
"""

_SCRIPT = """
const story=JSON.parse(document.getElementById('story-data').textContent);
const frames=story.frames, phase=document.getElementById('phase'), narration=document.getElementById('narration');
const play=document.getElementById('play'), next=document.getElementById('next'), back=document.getElementById('back');
const groups=Array.from(document.querySelectorAll('.stage-group'));
const columns=Array.from(document.querySelectorAll('.note-column'));
const values=Array.from({length:4},(_,i)=>document.getElementById('active-value-'+i));
const valueLabels=Array.from({length:4},(_,i)=>document.getElementById('value-label-'+i));
let index=0, timer=null;
function pause(){if(timer!==null){clearInterval(timer);timer=null;}play.textContent='Play';play.setAttribute('aria-pressed','false');}
function show(){
 const f=frames[index], done=f.phase==='done', overview=f.phase==='overview';
 phase.textContent=f.label;narration.textContent=f.narration;
 f.readout.forEach((v,i)=>{
  valueLabels[i].textContent=v.label;values[i].textContent=v.value;
  values[i].setAttribute('data-tone',v.tone);
 });
 document.body.classList.toggle('backward',f.phase==='backward');
 columns.forEach((c,i)=>c.classList.toggle('active',i===f.column));
 groups.forEach(g=>{
  const reveal=Number(g.dataset.reveal), role=g.dataset.role;
  const current=(Number(g.dataset.column)===f.column&&role===f.focus)||
   (Number(g.dataset.column)===-1&&reveal===index&&(role==='carry'||role==='feedback'));
  const future=reveal>index, feedback=role==='feedback';
  g.style.opacity=current?'1':role==='save'?((future||overview)?'0.50':'1'):
   feedback?((future||overview)?'0.08':'0.65'):role==='gates'?'0.42':
   ((future||overview)?'0.35':'0.75');
  if(done&&role==='save')g.style.opacity='1';
  if(reveal===0)g.style.opacity='1';
  g.classList.toggle('current',current);
 });
 back.disabled=index===0;next.disabled=index===frames.length-1;
 document.getElementById('progress').textContent='Step '+(index+1)+' / '+frames.length+' | no weights updated';
 if(index===frames.length-1)pause();
}
play.addEventListener('click',()=>{
 if(timer!==null){pause();return;}
 if(index===frames.length-1)index=0;
 play.textContent='Pause';play.setAttribute('aria-pressed','true');show();
 timer=setInterval(()=>{index=Math.min(index+1,frames.length-1);show();},4500);
});
next.addEventListener('click',()=>{pause();index=Math.min(index+1,frames.length-1);show();});
back.addEventListener('click',()=>{pause();index=Math.max(index-1,0);show();});
document.getElementById('reset').addEventListener('click',()=>{pause();index=0;show();});
document.addEventListener('keydown',e=>{
 if(e.target.tagName==='BUTTON')return;
 if(e.key==='ArrowRight'){e.preventDefault();next.click();}
 if(e.key==='ArrowLeft'){e.preventDefault();back.click();}
 if(e.code==='Space'){e.preventDefault();play.click();}
});
show();
"""


def _render(story):
    data = json.dumps(story, ensure_ascii=True, separators=(",", ":"), allow_nan=False)
    # Avoid an HTML end-tag sequence if future narration gains user-provided text.
    data = data.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    svg = _board(story).svg(story)
    title = html.escape(story["kind"].upper() + " fixed sequence storyboard")
    first = story["frames"][0]
    readout = "".join(
        f'<div class="value-cell"><span class="value-label" id="value-label-{i}">'
        f'{html.escape(value["label"])}</span>'
        f'<strong class="active-value" id="active-value-{i}" data-tone="{value["tone"]}">'
        f'{html.escape(value["value"])}</strong></div>'
        for i, value in enumerate(first["readout"])
    )
    return (
        f'<!doctype html><html lang="en" data-player="{story["kind"]}-sequence-storyboard">'
        '<head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>{title}</title><style>{_STYLE}{paper_style()}</style></head><body>"
        f'<main>{svg}</main><footer class="paper"><p id="phase" aria-live="polite">{html.escape(first["label"])}</p>'
        f'<section id="active-values" aria-label="Active recorded values" aria-live="polite">{readout}</section>'
        f'<p id="narration" aria-live="polite">{html.escape(first["narration"])}</p>'
        '<nav aria-label="Playback">'
        '<button id="back" type="button">Back</button>'
        '<button id="play" type="button" aria-pressed="false">Play</button>'
        '<button id="next" type="button">Next</button>'
        '<button id="reset" type="button">Reset</button><span id="progress"></span>'
        '</nav></footer><noscript>Playback needs JavaScript. All recorded cards remain visible; '
        'the PNG companion shows the same overview.</noscript>'
        f'<script id="story-data" type="application/json">{data}</script>'
        f"<script>{_SCRIPT}</script></body></html>\n"
    )


def render_player(kind: str) -> str:
    """Return a standalone board with embedded MAI paper and offline handwriting."""
    return _render(record_story(kind))


def export(directory: str | Path = HERE / "images") -> dict:
    """Write ONLY ``01a-{lstm,gru}-sequence-storyboard.{html,png}``."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    result = {}
    for kind in ("lstm", "gru"):
        story = record_story(kind)
        prefix = directory / f"01a-{kind}-sequence-storyboard"
        player, image = prefix.with_suffix(".html"), prefix.with_suffix(".png")
        player.write_text(_render(story), encoding="utf-8", newline="\n")
        _board(story).png(image, story)
        result[kind] = {"html": str(player), "png": str(image), "loss": story["loss"],
                        "frames": len(story["frames"]), "html_bytes": player.stat().st_size,
                        "png_bytes": image.stat().st_size}
    return result


if __name__ == "__main__":
    print(json.dumps(export(), indent=2))
