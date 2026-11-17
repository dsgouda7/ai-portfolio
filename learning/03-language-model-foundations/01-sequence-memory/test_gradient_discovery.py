"""Verify the loss-to-weight proof and its recorded numerical example."""

import importlib.util
import json
import math
from pathlib import Path
import unittest

import matplotlib.pyplot as plt
import torch
from torch.nn import functional as F

from error_feedback_flow import record_probe
from gradient_discovery import (
    CHAIN_KEYS, DISCOVERY_KEYS, EQUATIONS, HANDWRITING_STYLE as DISCOVERY_STYLE,
    chain_figure, discovery_figure, linear_cases, weight_example,
)


class GradientDiscoveryTests(unittest.TestCase):
    def test_notepad_values_are_rounded_autograd_path_gains(self):
        manifest = Path(__file__).parent / "images" / "01a-rnn-notepad-prompts.json"
        entries = json.loads(manifest.read_text(encoding="utf-8"))
        values = next(entry["calculation_provenance"] for entry in entries
                      if "calculation_provenance" in entry)
        for case in values["cases"]:
            for hops, displayed in zip(values["handoffs"], case["display"], strict=True):
                with self.subTest(weight=case["weight"], handoffs=hops):
                    opening = torch.tensor(.25, dtype=torch.float64, requires_grad=True)
                    memory = opening
                    for index in range(hops):
                        memory = case["weight"] * memory + .001 * (index + 1)
                    response = torch.autograd.grad(memory, opening)[0].item()
                    self.assertAlmostEqual(response, case["weight"] ** hops, delta=1e-10)
                    feedback = values["ending_feedback"] * response
                    places = len(displayed.partition(".")[2])
                    self.assertEqual(f"{feedback:.{places}f}", displayed)

    def test_linear_feedback_vanishes_or_explodes_without_a_squash(self):
        cases = linear_cases()
        for case in cases:
            self.assertEqual(case.links, 20)
            self.assertAlmostEqual(case.linear_ratio, case.weight ** case.links, delta=1e-12)
            native = record_probe(gain=case.weight, repeats=7)
            self.assertEqual(case.tanh_ratio, native.steps[0].feedback / native.steps[-1].feedback)
        by_weight = {case.weight: case for case in cases}
        self.assertLess(by_weight[.8].linear_ratio, .02)
        self.assertEqual(by_weight[1.].linear_ratio, 1.)
        self.assertGreater(by_weight[1.2].linear_ratio, 30)
        self.assertGreater(by_weight[1.2].linear_ratio, by_weight[1.2].tanh_ratio)

    def test_three_read_derivative_discovery_matches_the_ending_loss(self):
        for initial in (0., .2):
            weight = torch.tensor(.8, dtype=torch.float64, requires_grad=True)
            states = [torch.tensor(initial, dtype=torch.float64)]
            for write in weight_example().note_writes:
                states.append(weight * states[-1] + write)
            head = torch.tensor((-.5, 1., .25), dtype=torch.float64)
            probabilities = (head * states[-1]).softmax(0)
            loss = F.cross_entropy((head * states[-1]).unsqueeze(0), torch.tensor([0]))
            actual = torch.autograd.grad(loss, weight)[0]
            memory_response = states[2] + weight * states[1] + weight ** 2 * states[0]
            head_response = .5 + (head * probabilities).sum()
            expected = head_response * memory_response
            torch.testing.assert_close(actual, expected, rtol=1e-14, atol=1e-15)
        self.assertIn("w^{T-1}h_0", EQUATIONS["powers"])

    def test_loss_first_example_expands_prediction_and_takes_a_downhill_step(self):
        example = weight_example()

        def loss_at(weight):
            memory = 0.
            for write in example.note_writes:
                memory = weight * memory + write
            exponentials = [math.exp(value * memory) for value in example.head_weights]
            return -math.log(exponentials[0] / sum(exponentials))

        self.assertAlmostEqual(example.loss, loss_at(example.weight), places=14)
        for index, (score, response, contribution) in enumerate(zip(
            example.scores, example.score_response, example.head_contributions, strict=True,
        )):
            self.assertAlmostEqual(score, example.head_weights[index] * example.states[-1], places=14)
            self.assertAlmostEqual(response, example.probabilities[index] - (index == 0), places=14)
            self.assertAlmostEqual(contribution, example.head_weights[index] * response, places=14)
        self.assertAlmostEqual(sum(example.head_contributions), example.head_response, places=14)
        self.assertAlmostEqual(sum(example.memory_contributions), example.memory_response, places=14)
        expected_head = .5 + sum(
            weight * probability
            for weight, probability in zip(example.head_weights, example.probabilities)
        )
        self.assertAlmostEqual(example.head_response, expected_head, places=14)
        self.assertAlmostEqual(example.memory_response,
                               example.states[2] + example.weight * example.states[1], places=14)
        self.assertAlmostEqual(example.weight_gradient,
                               example.head_response * example.memory_response, places=14)
        epsilon = 1e-5
        finite = (loss_at(example.weight + epsilon) - loss_at(example.weight - epsilon)) / (2 * epsilon)
        self.assertAlmostEqual(example.weight_gradient, finite, delta=1e-10)
        self.assertEqual(example.updated_weight,
                         example.weight - example.step_size * example.weight_gradient)
        self.assertAlmostEqual(example.updated_loss, loss_at(example.updated_weight), places=14)
        self.assertLess(example.updated_loss, example.loss)

    def test_discovery_sheet_keeps_the_chain_and_expansion_legible(self):
        self.assertIn(r"\operatorname{softmax}", EQUATIONS["probability"])
        self.assertFalse(any("e^{" in EQUATIONS[key] or r"\exp" in EQUATIONS[key]
                             for key in DISCOVERY_KEYS + CHAIN_KEYS))
        for render, keys in ((discovery_figure, DISCOVERY_KEYS), (chain_figure, CHAIN_KEYS)):
            with self.subTest(page=render.__name__):
                figure = render()
                try:
                    with plt.rc_context(DISCOVERY_STYLE):
                        figure.canvas.draw()
                    axis = figure.axes[0]
                    renderer = figure.canvas.get_renderer()
                    labels = [text.get_text() for text in axis.texts]
                    positions = [labels.index(f"${EQUATIONS[key]}$") for key in keys]
                    self.assertEqual(positions, sorted(positions))
                    for index, text in enumerate(axis.texts):
                        bounds = text.get_window_extent(renderer)
                        self.assertTrue(figure.bbox.contains(bounds.x0, bounds.y0), text.get_text())
                        self.assertTrue(figure.bbox.contains(bounds.x1, bounds.y1), text.get_text())
                        for other in axis.texts[index + 1:]:
                            self.assertFalse(bounds.overlaps(other.get_window_extent(renderer)),
                                             f"{text.get_text()} / {other.get_text()}")
                finally:
                    plt.close(figure)

    def test_linear_expansion_counts_the_powers_and_all_shared_uses(self):
        for memory in (0.8, 1.2):
            for initial in (0.0, 0.2):
                for length in (3, 6):
                    weight = torch.tensor(memory, dtype=torch.float64, requires_grad=True)
                    states = [torch.tensor(initial, dtype=torch.float64)]
                    for note_write in weight_example().note_writes * (length // 3):
                        states.append(note_write + weight * states[-1])
                    derivative = torch.autograd.grad(states[-1], weight)[0]
                    expanded = sum(weight ** (length - 1 - index) * states[index] for index in range(length))
                    torch.testing.assert_close(derivative, expanded, rtol=1e-14, atol=1e-15)

    def test_image_helper_preserves_defaults_and_validates_portrait_limits(self):
        path = Path(__file__).parent.parent / "03-language-models" / "generate_illustrations.py"
        spec = importlib.util.spec_from_file_location("curriculum_mai_helper", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertEqual(module.image_dimensions({}), (1536, 864))
        self.assertEqual(module.image_dimensions({"width": 1216, "height": 1792}), (1216, 1792))
        for entry in ({"width": True}, {"height": 767}, {"width": 2048, "height": 2048},
                      {"width": 1200.5}):
            with self.assertRaises(ValueError):
                module.image_dimensions(entry)

if __name__ == "__main__":
    unittest.main()
