"""Check the companion's RNN gradients against autograd and finite differences."""

import ast
import contextlib
from io import StringIO
import json
import math
import re
import unittest

import torch
from torch.nn import functional as F

from backprop_memory_flow import NOTEBOOK, NOTE_HEAD_WEIGHTS, _inputs_and_weights
from error_feedback_flow import record_probe


class RnnGradientTests(unittest.TestCase):
    def test_notebook_what_if_checks_both_feature_signs_without_changing_saved_memory(self):
        book = json.loads(NOTEBOOK.read_bytes())
        cells = {cell["id"]: cell for cell in book["cells"]}
        source = "".join(cells["gradient-meaning-probe"]["source"])
        for feature in (0, 1):
            tree = ast.parse(source)
            setting = next(node for node in tree.body if isinstance(node, ast.Assign)
                           and any(isinstance(target, ast.Name) and target.id == "nudge_feature"
                                   for target in node.targets))
            setting.value = ast.Constant(feature)
            namespace = {"torch": torch, "F": F}
            with contextlib.redirect_stdout(StringIO()):
                exec(compile(ast.fix_missing_locations(tree), "memory_what_if", "exec"), namespace)
            self.assertAlmostEqual(namespace["nudge_rate"],
                                   namespace["nudge_exact"][feature].item(), delta=1e-8)
            torch.testing.assert_close(namespace["nudge_h"].detach(),
                                       namespace["nudge_trace"].states[-1], rtol=0, atol=0)
            if feature == 0:
                self.assertLess(namespace["nudge_rate"], 0)
            else:
                self.assertGreater(namespace["nudge_rate"], 0)
            if feature == 0:
                reading = "".join(cells["gradient-meaning-reading"]["source"])
                for memory, loss in namespace["nudge_trials"]:
                    self.assertIn(f"{memory:.10f}", reading)
                    self.assertIn(f"{loss:.10f}", reading)

    def test_restored_manual_check_matches_short_and_long_cases_and_local_matrix_detail(self):
        book = json.loads(NOTEBOOK.read_bytes())
        cells = {cell["id"]: cell for cell in book["cells"]}
        namespace = {}
        with contextlib.redirect_stdout(StringIO()):
            exec(compile("".join(cells["bptt-manual-check"]["source"]), "manual_bptt", "exec"), namespace)
        for gain in (0.8, 1.0, 1.2):
            for repeats in (1, 7):
                probe = record_probe(gain=gain, repeats=repeats)
                self.assertAlmostEqual(namespace["manual_recurrent_gradient"](probe),
                                       probe.weight_gradients.memory, places=12)
        detail = "".join(cells["bptt-vector-detail"]["source"])
        for parameter in (r"W_h}", r"W_x}", r"W_{\mathrm{out}}}", r"b}", r"b_{\mathrm{out}}}"):
            self.assertIn(r"\frac{\partial L}{\partial " + parameter, detail)
        for expression in (
            r"\frac{dh_t}{dW_h}=h_{t-1}+W_h\frac{dh_{t-1}}{dW_h}",
            r"\frac{dh_{t-2}}{dW_h}=h_{t-3}+W_h\frac{dh_{t-3}}{dW_h}",
            r"+W_h^2\left[",
            r"+\cdots+W_h^{T-1}h_0",
        ):
            self.assertIn(expression, detail)
        self.assertIn("preview", detail)
        from gradient_memory import piano_probe
        for delay, digits in ((5, 8), (20, 10)):
            self.assertIn(f"{piano_probe(delay).feedback[0]:.{digits}f}", detail)
        ids = [cell["id"] for cell in book["cells"]]
        self.assertLess(ids.index("gradient-meaning-probe"), ids.index("gradient-native-intro"))
        self.assertLess(ids.index("bptt-manual-check"), ids.index("bptt-vector-detail"))
        self.assertLess(ids.index("bptt-vector-detail"), ids.index("gated-memory-demand"))
        self.assertNotIn("gradient-technical-appendix", ids)

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
            for cell_id in ("gradient-meaning-reading", "gradient-native-intro",
                            "gradient-exploding-intuition")
        )
        for term in ("multiply", "add", "h1", "h2", "h3", "shared"):
            self.assertIn(term, visible.lower())
        for notation in ("mixgrad", "scoregrad", "\\sum", "\\odot"):
            self.assertNotIn(notation, visible)
        self.assertIn("W_x/W_h/W_out", visible)

    def test_shared_weight_check_stays_reproducible_without_a_teaching_appendix(self):
        book = json.loads(NOTEBOOK.read_bytes())
        ids = [cell["id"] for cell in book["cells"]]
        ordered = ("gradient-native-intro", "embedded-01a-error-feedback",
                   "gradient-exploding-intuition", "gated-memory-demand")
        self.assertEqual(sorted(ids.index(cell_id) for cell_id in ordered),
                         [ids.index(cell_id) for cell_id in ordered])
        self.assertNotIn("gradient-technical-appendix", ids)
        for gain in (0.8, 1.2):
            probe = record_probe(gain=gain)
            feedback = probe.steps[-1].feedback
            total = 0.0
            for step in reversed(probe.steps):
                total += feedback * step.slope * step.previous
                feedback *= step.factor
            self.assertAlmostEqual(total, probe.weight_gradients.memory, places=14)

if __name__ == "__main__":
    unittest.main()
