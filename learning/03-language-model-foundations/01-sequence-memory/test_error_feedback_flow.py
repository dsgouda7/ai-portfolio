"""Check the recorded model and the code-free error-feedback player."""

import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from error_feedback_flow import (
    GAINS, INPUT, PLAYER, PREDICTION_WEIGHT, REWRITES, SCALE_GAINS, TARGET, WRITING_WEIGHT,
    export_player, read_recording, record_probe, render_player,
)


class ErrorFeedbackFlowTests(unittest.TestCase):
    def test_weights_generate_nonzero_output_and_error(self):
        for gain in GAINS:
            probe = record_probe(gain=gain)
            expected = math.tanh(WRITING_WEIGHT * INPUT)
            self.assertEqual(len(probe.states), REWRITES + 1)
            self.assertAlmostEqual(probe.states[0], expected, places=14)
            for state in probe.states[1:]:
                expected = math.tanh(gain * expected)
                self.assertAlmostEqual(state, expected, places=14)
            self.assertTrue(all(0 < state < 0.05 for state in probe.states))
            self.assertEqual(probe.prediction, PREDICTION_WEIGHT * probe.states[-1])
            self.assertEqual(probe.error, probe.prediction - TARGET)
            self.assertAlmostEqual(probe.loss, 0.5 * probe.error**2)

    def test_backward_feedback_and_actual_weight_gradients(self):
        for gain in GAINS:
            probe = record_probe(gain=gain)
            self.assertEqual(probe.feedback[-1], probe.error * PREDICTION_WEIGHT)
            for index in range(REWRITES):
                expected = probe.feedback[index + 1] * gain * (1 - probe.states[index + 1]**2)
                self.assertAlmostEqual(probe.feedback[index], expected, places=12)
            self.assertAlmostEqual(
                probe.writing_gradient, probe.feedback[0] * (1 - probe.states[0]**2) * INPUT, places=12,
            )
            previous_states = (0.0,) + probe.states[:-1]
            expected_memory = sum(
                feedback * (1 - state**2) * previous
                for feedback, state, previous in zip(probe.feedback, probe.states, previous_states, strict=True)
            )
            self.assertAlmostEqual(probe.memory_gradient, expected_memory, places=12)
            self.assertAlmostEqual(probe.prediction_gradient, probe.error * probe.states[-1], places=12)

    def test_toggle_demonstrates_vanishing_and_exploding_early_weight_gradients(self):
        fading, growing = (record_probe(gain=gain) for gain in GAINS)
        self.assertEqual(fading.states[0], growing.states[0])
        self.assertLess(abs(fading.feedback[0] / fading.feedback[-1]), 0.02)
        self.assertGreater(abs(growing.feedback[0] / growing.feedback[-1]), 30)
        self.assertLess(abs(fading.writing_gradient), 0.01)
        self.assertGreater(abs(growing.writing_gradient), 10)
        self.assertGreater(abs(growing.writing_gradient / fading.writing_gradient), 3000)

    def test_zero_error_still_produces_zero_gradients(self):
        for gain in GAINS:
            output = record_probe(gain=gain).prediction
            probe = record_probe(gain=gain, target=output)
            self.assertEqual(probe.loss, 0)
            self.assertEqual(probe.feedback, (0.0,) * (REWRITES + 1))
            self.assertEqual((probe.writing_gradient, probe.memory_gradient, probe.prediction_gradient), (0, 0, 0))

    def test_recorded_routes_match_fresh_autograd(self):
        data = read_recording()
        self.assertEqual(data["version"], 3)
        self.assertEqual(data["target"], TARGET)
        self.assertEqual(data["rewrites"], REWRITES)
        self.assertEqual([route["gain"] for route in data["routes"]], list(SCALE_GAINS))
        for route in data["routes"]:
            probe = record_probe(gain=route["gain"])
            self.assertEqual(route["states"], list(probe.states))
            self.assertEqual(route["feedback"], list(probe.feedback))
            self.assertEqual(route["prediction"], probe.prediction)
            self.assertEqual(route["error"], probe.error)
            self.assertEqual(route["weight_gradients"]["writing"], probe.writing_gradient)
            self.assertEqual(route["weight_gradients"]["memory"], probe.memory_gradient)
            self.assertEqual(route["weight_gradients"]["prediction"], probe.prediction_gradient)

    def test_export_regenerates_the_player_and_is_idempotent(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / PLAYER.name
            original = PLAYER.read_bytes()
            path.write_bytes(original)
            export_player(path)
            self.assertEqual(path.read_bytes(), original)
            source = original.decode("utf-8")
            recording = read_recording(path)
            encoded = json.dumps(recording, separators=(",", ":"))
            self.assertIn(encoded, source)
            path.write_bytes(source.replace(encoded, "{}", 1).encode())
            export_player(path)
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(original.decode(), render_player())

    def test_bad_probe_inputs_are_reported(self):
        for target in (math.nan, math.inf, True, "wrong"):
            with self.assertRaises(ValueError):
                record_probe(target=target)
        with self.assertRaises(ValueError):
            record_probe(gain=2)
        with self.assertRaises(ValueError):
            record_probe(target=1e200)

    def test_player_has_a_weight_scale_and_two_clean_presets(self):
        source = PLAYER.read_text(encoding="utf-8")
        self.assertEqual(source.count('type="range"'), 1)
        self.assertNotIn('type="checkbox"', source)
        self.assertIn('id="weight-scale"', source)
        self.assertIn('id="preset-vanishing"', source)
        self.assertIn('id="preset-exploding"', source)
        self.assertNotIn('type="number"', source)
        self.assertNotIn('id="prediction-slider"', source)

    def test_scale_covers_fading_balanced_and_growing_feedback(self):
        fading = record_probe(gain=0.6)
        balanced = record_probe(gain=1.0)
        growing = record_probe(gain=1.3)
        self.assertLess(abs(fading.writing_gradient / fading.feedback[-1]), 0.0001)
        self.assertAlmostEqual(abs(balanced.writing_gradient / balanced.feedback[-1]), 1, places=4)
        self.assertGreater(abs(growing.writing_gradient / growing.feedback[-1]), 100)
        self.assertTrue(all(0 < state < 1 for state in growing.states))


if __name__ == "__main__":
    unittest.main()
