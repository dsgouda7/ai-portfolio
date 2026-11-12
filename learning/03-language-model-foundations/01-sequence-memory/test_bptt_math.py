"""Check the displayed BPTT equations against scalar and matrix autograd."""

import ast
import base64
import contextlib
from io import StringIO
import json
import math
from pathlib import Path
import re
from tempfile import TemporaryDirectory
import unittest

import matplotlib.pyplot as plt
from matplotlib import font_manager
from PIL import Image
import torch
from torch.nn import functional as F

from backprop_memory_flow import NOTEBOOK, NOTE_HEAD_WEIGHTS, _inputs_and_weights
from bptt_math import (
    BACKWARD_EQUATIONS, FONT, FORWARD_EQUATIONS, HANDWRITING_STYLE, HERE, IMAGE_FILES,
    boards, export, math_figure, source_fingerprint,
)
from error_feedback_flow import record_probe


class BpttMathTests(unittest.TestCase):
    def test_generic_matrix_recurrence_and_outer_products_match_autograd(self):
        for grade_all in (False, True):
            with self.subTest(grade_all=grade_all):
                inputs, wx, wh = _inputs_and_weights(NOTEBOOK)
                wx.requires_grad_()
                wh.requires_grad_()
                wo = torch.tensor(NOTE_HEAD_WEIGHTS, dtype=torch.float64, requires_grad=True)
                bias = torch.tensor((0.05, -0.02), dtype=torch.float64, requires_grad=True)
                head_bias = torch.tensor((0.03, -0.01, 0.02), dtype=torch.float64, requires_grad=True)
                states = [torch.tensor((0.1, -0.2), dtype=torch.float64, requires_grad=True)]
                scores = []
                for current in inputs:
                    states.append(torch.tanh(wx @ current + wh @ states[-1] + bias))
                    scores.append(wo @ states[-1] + head_bias)
                targets = (1, 2, 0)
                loss = sum(
                    F.cross_entropy(score.unsqueeze(0), torch.tensor([target]))
                    for index, (score, target) in enumerate(zip(scores, targets, strict=True))
                    if grade_all or index == 2
                )
                native = torch.autograd.grad(loss, (wx, wh, wo, bias, head_bias, *states))
                manual = [torch.zeros_like(value) for value in (wx, wh, wo, bias, head_bias)]
                next_mix = torch.zeros_like(states[0])
                with torch.no_grad():
                    for index in reversed(range(3)):
                        score_grad = torch.zeros_like(scores[index])
                        if grade_all or index == 2:
                            score_grad = scores[index].softmax(0)
                            score_grad[targets[index]] -= 1
                        hidden_grad = wo.T @ score_grad + wh.T @ next_mix
                        torch.testing.assert_close(hidden_grad, native[6 + index], rtol=1e-12, atol=1e-14)
                        mix_grad = hidden_grad * (1 - states[index + 1] ** 2)
                        manual[0] += torch.outer(mix_grad, inputs[index])
                        manual[1] += torch.outer(mix_grad, states[index])
                        manual[2] += torch.outer(score_grad, states[index + 1])
                        manual[3] += mix_grad
                        manual[4] += score_grad
                        next_mix = mix_grad
                    torch.testing.assert_close(wh.T @ next_mix, native[5], rtol=1e-12, atol=1e-14)
                    for actual, expected in zip(manual, native[:5], strict=True):
                        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-14)

    def test_toy_recurrent_gradient_matches_finite_differences(self):
        for gain in (0.6, 0.8, 1.0, 1.2, 1.3):
            probe = record_probe(gain=gain)
            score_grad = list(probe.probabilities)
            score_grad[0] -= 1
            feedback = sum(w * e for w, e in zip(probe.weights.prediction, score_grad, strict=True))
            contributions = []
            for step in reversed(probe.steps):
                self.assertAlmostEqual(feedback, step.feedback, places=14)
                mix_grad = feedback * (1 - step.state**2)
                contributions.append(mix_grad * step.previous)
                feedback = probe.gain * mix_grad
            total = sum(contributions)
            self.assertAlmostEqual(total, probe.weight_gradients.memory, places=14)
            epsilon = 1e-5

            def loss_at(memory):
                state = 0.0
                for step in probe.steps:
                    state = math.tanh(step.input_part + memory * state)
                scores = torch.tensor(probe.weights.prediction, dtype=torch.float64) * state
                return -scores.log_softmax(0)[0].item()

            finite = (loss_at(gain + epsilon) - loss_at(gain - epsilon)) / (2 * epsilon)
            self.assertAlmostEqual(total, finite, delta=2e-11)
        default = record_probe()
        self.assertEqual(default.steps[0].memory_contribution, 0)
        self.assertGreater(default.steps[0].feedback, 0)
        self.assertAlmostEqual(default.weight_gradients.memory, 0.0008253043369809912, places=16)

    def test_tiny_nudges_explain_local_and_inherited_responses(self):
        epsilon = 1e-6
        for gain in (0.8, 1.2):
            probe = record_probe(gain=gain)
            inherited_response = 0.0
            for step in probe.steps:
                def read(previous, memory):
                    return math.tanh(step.input_part + memory * previous)

                handoff = (read(step.previous + epsilon, gain)
                           - read(step.previous - epsilon, gain)) / (2 * epsilon)
                self.assertAlmostEqual(handoff, step.factor, delta=1e-10)
                local_weight_response = (read(step.previous, gain + epsilon)
                                         - read(step.previous, gain - epsilon)) / (2 * epsilon)
                self.assertAlmostEqual(local_weight_response, step.slope * step.previous, delta=1e-10)
                inherited_response = step.slope * (step.previous + gain * inherited_response)
            feedback = probe.steps[-1].feedback
            self.assertAlmostEqual(feedback * inherited_response,
                                   probe.weight_gradients.memory, places=14)
            self.assertNotAlmostEqual(local_weight_response * feedback,
                                      probe.weight_gradients.memory, places=7)
        default = record_probe()
        head = torch.tensor(default.weights.prediction, dtype=torch.float64)
        h3 = default.steps[-1].state
        loss_before = -(head * h3).log_softmax(0)[0].item()
        loss_after = -(head * (h3 + 1e-5)).log_softmax(0)[0].item()
        self.assertAlmostEqual(loss_after - loss_before, 0.000007502793748, places=14)
        self.assertAlmostEqual((loss_after - loss_before) / 1e-5,
                               default.steps[-1].feedback, delta=2e-6)

    def test_main_reading_explains_the_rules_without_matrix_notation(self):
        book = json.loads(NOTEBOOK.read_bytes())
        cells = {cell["id"]: cell for cell in book["cells"]}
        visible = "\n".join(
            re.sub(r"<details>.*?</details>", "", "".join(cells[cell_id]["source"]), flags=re.S)
            for cell_id in ("gradient-native-intro", "gradient-vanishing-intuition",
                            "gradient-exploding-intuition")
        )
        for term in ("tiny", "multiply", "add", "direct", "earlier", "two handoffs"):
            self.assertIn(term, visible.lower())
        for notation in ("mixgrad", "scoregrad", "\\sum", "W_h", "\\odot"):
            self.assertNotIn(notation, visible)
            self.assertNotIn(notation, str(FORWARD_EQUATIONS + BACKWARD_EQUATIONS))
        detailed = "".join(cells["gradient-vanishing-intuition"]["source"])
        self.assertIn("dh3/dm", detailed)
        self.assertIn("<details>", detailed)

    def test_notebook_places_generic_then_worked_math_after_the_player(self):
        book = json.loads(NOTEBOOK.read_bytes())
        ids = [cell["id"] for cell in book["cells"]]
        ordered = (
            "embedded-01a-error-feedback", "gradient-vanishing-intuition",
            "gradient-exploding-intuition", "gradient-technical-appendix",
            "bptt-forward-math", "bptt-backward-math",
            "bptt-worked-example", "bptt-manual-check", "bptt-learning-bridge",
        )
        self.assertEqual(sorted(ids.index(cell_id) for cell_id in ordered),
                         [ids.index(cell_id) for cell_id in ordered])
        cells = {cell["id"]: cell for cell in book["cells"]}
        self.assertIn("### BPTT in RNNs", "".join(cells["bptt-forward-math"]["source"]))
        self.assertIn("preview", "".join(cells["bptt-backward-math"]["source"]))
        self.assertIn("0.00082530", "".join(cells["bptt-worked-example"]["source"]))
        source = "".join(cells["bptt-manual-check"]["source"])
        ast.parse(source)
        namespace = {}
        with contextlib.redirect_stdout(StringIO()):
            exec(compile(source, "bptt-manual-check", "exec"), namespace)
            for gain in (0.8, 1.2):
                probe = record_probe(gain=gain)
                self.assertAlmostEqual(namespace["manual_memory_gradient"](probe),
                                       probe.weight_gradients.memory, places=14)

    def test_cards_are_fingerprinted_embedded_and_have_equation_transcripts(self):
        book = json.loads(NOTEBOOK.read_bytes())
        for filename in IMAGE_FILES:
            owner = next(cell for cell in book["cells"]
                         if f"attachment:{filename}" in "".join(cell["source"]))
            self.assertEqual(base64.b64decode(owner["attachments"][filename]["image/png"]),
                             (HERE / "images" / filename).read_bytes())
            source = "".join(owner["source"])
            self.assertIn("<details>", source)
            self.assertIn("$$", source)
            with Image.open(HERE / "images" / filename) as image:
                self.assertEqual(image.info["SourceFingerprint"], source_fingerprint())
                self.assertEqual(image.info["MathFont"], FONT)

    @unittest.skipUnless(any(font.name == FONT for font in font_manager.fontManager.ttflist),
                         "Regenerating handwritten cards requires Windows Segoe Print.")
    def test_handwritten_equations_are_readable_unclipped_and_deterministic(self):
        with plt.rc_context(HANDWRITING_STYLE):
            for title, equations in boards():
                figure = math_figure(title, equations)
                try:
                    figure.canvas.draw()
                    renderer = figure.canvas.get_renderer()
                    for index, text in enumerate(figure.texts):
                        bounds = text.get_window_extent(renderer)
                        self.assertTrue(figure.bbox.contains(bounds.x0, bounds.y0), text.get_text())
                        self.assertTrue(figure.bbox.contains(bounds.x1, bounds.y1), text.get_text())
                        self.assertLessEqual(bounds.x1, figure.bbox.width * 0.955, text.get_text())
                        if text.get_text().startswith("$"):
                            self.assertGreaterEqual(text.get_fontsize(), 24)
                        for other in figure.texts[index + 1:]:
                            self.assertFalse(bounds.overlaps(other.get_window_extent(renderer)),
                                             (text.get_text(), other.get_text()))
                finally:
                    plt.close(figure)
        with TemporaryDirectory() as directory:
            export(Path(directory))
            self.assertEqual({path.name for path in Path(directory).iterdir()}, set(IMAGE_FILES))
            for filename in IMAGE_FILES:
                self.assertEqual((Path(directory) / filename).read_bytes(),
                                 (HERE / "images" / filename).read_bytes())


if __name__ == "__main__":
    unittest.main()
