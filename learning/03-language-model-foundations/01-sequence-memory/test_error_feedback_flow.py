"""Verify the note chain, all local factors, and the actual optimizer preview."""

import base64
from dataclasses import asdict
from io import BytesIO
import json
import math
from pathlib import Path
import re
from tempfile import TemporaryDirectory
import unittest

from PIL import Image

from backprop_memory_flow import _inputs_and_weights, NOTEBOOK, NOTES
from error_feedback_flow import (
    CARD_ART, GAINS, LEARNING_RATE, MEMORY_NUDGE, NOTE_INPUTS, PLAYER, PREDICTION_WEIGHTS,
    PROBABILITY_DECIMALS, REPETITIONS, SCALE_GAINS, STORYBOARD, TARGET_NOTE, WRITING_WEIGHTS,
    Parameters, export_player, export_storyboard, read_recording, record_probe, render_player,
)


def forward_loss(weights, repeats):
    states = [0.0]
    for inputs in NOTE_INPUTS * repeats:
        states.append(math.tanh(
            sum(w * x for w, x in zip(weights.writing, inputs, strict=True)) + weights.memory * states[-1]
        ))
    scores = [weight * states[-1] for weight in weights.prediction]
    exponentials = [math.exp(score - max(scores)) for score in scores]
    probabilities = tuple(value / sum(exponentials) for value in exponentials)
    return -math.log(probabilities[NOTES.index(TARGET_NOTE)]), states, probabilities


class ErrorFeedbackFlowTests(unittest.TestCase):
    def test_note_encodings_match_the_existing_reader(self):
        inputs, _, _ = _inputs_and_weights(NOTEBOOK)
        self.assertEqual(inputs.tolist(), [list(row) for row in NOTE_INPUTS])
        self.assertEqual(NOTES, ("D3", "A3", "D4"))

    def test_every_note_has_its_own_saved_cell_but_weights_are_shared(self):
        for repeats in REPETITIONS:
            for gain in (0.6, 0.8, 1.0, 1.2, 1.3):
                probe = record_probe(gain=gain, repeats=repeats)
                self.assertEqual(len(probe.steps), 3 * repeats)
                self.assertEqual(tuple(step.note for step in probe.steps), NOTES * repeats)
                self.assertEqual(tuple(step.inputs for step in probe.steps), NOTE_INPUTS * repeats)
                self.assertEqual(probe.steps[0].previous, 0)
                old = 0
                for index, step in enumerate(probe.steps, 1):
                    self.assertEqual(step.index, index)
                    self.assertEqual(step.previous, old)
                    self.assertEqual(step.memory_part, gain * old)
                    self.assertAlmostEqual(step.input_part, sum(w * x for w, x in zip(WRITING_WEIGHTS, step.inputs)), places=14)
                    self.assertAlmostEqual(step.combined, step.input_part + step.memory_part, places=14)
                    self.assertAlmostEqual(step.state, math.tanh(step.combined), places=14)
                    old = step.state
                loss, states, probabilities = forward_loss(probe.weights, repeats)
                self.assertAlmostEqual(probe.loss, loss, places=13)
                for actual, expected in zip(probe.probabilities, probabilities):
                    self.assertAlmostEqual(actual, expected, places=14)
                self.assertEqual(probe.prediction, NOTES[max(range(3), key=probabilities.__getitem__)])
                for step, expected in zip(probe.steps, states[1:]):
                    self.assertAlmostEqual(step.state, expected, places=14)

    def test_one_final_note_loss_seeds_the_whole_backward_chain(self):
        for repeats in REPETITIONS:
            probe = record_probe(repeats=repeats)
            self.assertAlmostEqual(sum(probe.score_errors), 0, places=14)
            self.assertAlmostEqual(probe.loss, -math.log(probe.probabilities[0]), places=14)
            self.assertEqual(probe.score_errors[0], probe.probabilities[0] - 1)
            for error, probability in zip(probe.score_errors[1:], probe.probabilities[1:]):
                self.assertEqual(error, probability)
            seed = sum(error * weight for error, weight in zip(probe.score_errors, PREDICTION_WEIGHTS))
            self.assertAlmostEqual(probe.steps[-1].feedback, seed, places=14)
            for score, weight in zip(probe.scores, PREDICTION_WEIGHTS):
                self.assertEqual(score, weight * probe.steps[-1].state)
            self.assertEqual(probe.steps[-1].scores, probe.scores)
            self.assertEqual(probe.steps[-1].probabilities, probe.probabilities)
            self.assertEqual(probe.steps[-1].local_loss, probe.loss)
            self.assertEqual([step.graded for step in probe.steps], [False] * (len(probe.steps) - 1) + [True])
            self.assertNotEqual(probe.loss, sum(step.local_loss for step in probe.steps))

    def test_notebook_replaces_the_slider_with_the_three_case_calculation(self):
        book = json.loads(NOTEBOOK.read_bytes())
        cells = {cell["id"]: cell for cell in book["cells"]}
        numeric = "".join(cells["gradient-exploding-intuition"]["source"])
        for gain in (1.0, 1.2, 0.8):
            longer = record_probe(gain=gain, repeats=7)
            ratio = longer.steps[0].feedback / longer.steps[-1].feedback
            self.assertIn(f"{ratio:.6f}", numeric)
        intro = "".join(cells["gradient-native-intro"]["source"])
        self.assertIn("W_x/W_h/W_out", intro)
        self.assertNotIn("Optional static snapshot of the animation", numeric)
        self.assertEqual(cells["embedded-01a-error-feedback"]["metadata"]["saved_replay"]["file"],
                         "01a-gradient-three-cases.html")
        for cell in book["cells"]:
            self.assertNotIn("attachment:01a-note-gradient-chain.png", "".join(cell["source"]))
            self.assertNotIn("01a-note-gradient-chain.png", cell.get("attachments", {}))

    def test_tiny_memory_change_readout_is_a_loss_response_not_a_weight_update(self):
        epsilon = 1e-6
        for gain in (SCALE_GAINS[0], *GAINS, SCALE_GAINS[-1]):
            for repeats in REPETITIONS:
                probe = record_probe(gain=gain, repeats=repeats)
                for index, step in enumerate(probe.steps):
                    def ending_loss(state):
                        for later in probe.steps[index + 1:]:
                            state = math.tanh(later.input_part + gain * state)
                        scores = [weight * state for weight in probe.weights.prediction]
                        maximum = max(scores)
                        return maximum + math.log(sum(math.exp(value - maximum) for value in scores)) - scores[0]

                    measured = (ending_loss(step.state + epsilon) - ending_loss(step.state - epsilon)) / (2 * epsilon)
                    self.assertTrue(math.isclose(step.feedback, measured, rel_tol=2e-6, abs_tol=1e-8))
                    self.assertAlmostEqual(step.loss_if_memory_lower,
                                           ending_loss(step.state - MEMORY_NUDGE), places=13)
                    self.assertAlmostEqual(step.loss_if_memory_higher,
                                           ending_loss(step.state + MEMORY_NUDGE), places=13)
                    self.assertEqual(step.measured_response,
                                     (step.loss_if_memory_higher - step.loss_if_memory_lower) / (2 * MEMORY_NUDGE))
                    self.assertTrue(math.isclose(step.feedback, step.measured_response,
                                                 rel_tol=2e-6, abs_tol=1e-8))
        default = record_probe()
        self.assertAlmostEqual(default.steps[-1].loss_if_memory_lower, 1.099159888320054, places=14)
        self.assertAlmostEqual(default.steps[-1].loss_if_memory_higher, 1.099174893870051, places=14)
        self.assertAlmostEqual(default.steps[-1].measured_response, default.steps[-1].feedback, places=10)

    def test_each_head_predicts_the_actual_next_note_and_records_its_own_loss(self):
        for repeats in REPETITIONS:
            for gain in GAINS:
                probe = record_probe(gain=gain, repeats=repeats)
                for step in probe.steps:
                    self.assertEqual(step.next_note, NOTES[step.index % len(NOTES)])
                    scores = [weight * step.state for weight in PREDICTION_WEIGHTS]
                    exponentials = [math.exp(score - max(scores)) for score in scores]
                    probabilities = [value / sum(exponentials) for value in exponentials]
                    self.assertEqual(step.scores, tuple(scores))
                    for actual, expected in zip(step.probabilities, probabilities):
                        self.assertAlmostEqual(actual, expected, places=14)
                    self.assertAlmostEqual(step.local_loss, -math.log(step.probabilities[NOTES.index(step.next_note)]), places=14)
                    self.assertEqual(step.prediction, NOTES[max(range(3), key=probabilities.__getitem__)])

    def test_preview_predictions_are_causal_and_do_not_change_the_training_scope(self):
        short, long = record_probe(), record_probe(repeats=7)
        for a, b in zip(short.steps, long.steps[:3]):
            self.assertEqual((a.state, a.scores, a.probabilities, a.local_loss),
                             (b.state, b.scores, b.probabilities, b.local_loss))
        self.assertTrue(short.steps[2].graded)
        self.assertFalse(long.steps[2].graded)

    def test_precision_exposes_tiny_middle_head_changes_and_first_head_stays_fixed(self):
        trials = [record_probe(gain=gain) for gain in SCALE_GAINS]
        self.assertEqual(len({trial.steps[0].probabilities for trial in trials}), 1)
        self.assertEqual(len({trial.steps[0].local_loss for trial in trials}), 1)
        probabilities = [f"{trial.steps[1].probabilities[2]:.{PROBABILITY_DECIMALS}f}" for trial in trials]
        losses = [f"{trial.steps[1].local_loss:.{PROBABILITY_DECIMALS}f}" for trial in trials]
        self.assertEqual(len(set(probabilities)), len(SCALE_GAINS))
        self.assertEqual(len(set(losses)), len(SCALE_GAINS))

    def test_each_local_factor_and_backward_substitution_matches_autograd(self):
        for repeats in REPETITIONS:
            for gain in GAINS:
                probe = record_probe(gain=gain, repeats=repeats)
                for offset, step in enumerate(probe.steps):
                    self.assertEqual(step.slope, 1 - step.state**2)
                    self.assertEqual(step.factor, gain * step.slope)
                    self.assertAlmostEqual(step.delta, step.feedback * step.slope, places=13)
                    self.assertAlmostEqual(step.previous_feedback, step.feedback * step.factor, places=12)
                    if offset:
                        self.assertEqual(step.previous_feedback, probe.steps[offset - 1].feedback)
                product = math.prod(step.factor for step in probe.steps[1:])
                self.assertAlmostEqual(probe.steps[0].feedback / probe.steps[-1].feedback, product, places=11)

    def test_weight_gradients_sum_all_cells_instead_of_updating_per_note(self):
        for repeats in REPETITIONS:
            for gain in GAINS:
                probe = record_probe(gain=gain, repeats=repeats)
                for feature in range(2):
                    total = sum(step.writing_contribution[feature] for step in probe.steps)
                    self.assertAlmostEqual(total, probe.weight_gradients.writing[feature], places=11)
                    for step in probe.steps:
                        self.assertAlmostEqual(step.writing_contribution[feature], step.delta * step.inputs[feature], places=13)
                self.assertEqual(probe.steps[0].memory_contribution, 0)
                self.assertAlmostEqual(sum(step.memory_contribution for step in probe.steps), probe.weight_gradients.memory, places=12)
                for step in probe.steps:
                    self.assertAlmostEqual(step.memory_contribution, step.delta * step.previous, places=13)
                for gradient, error in zip(probe.weight_gradients.prediction, probe.score_errors):
                    self.assertAlmostEqual(gradient, error * probe.steps[-1].state, places=14)

    def test_gradients_match_independent_finite_differences(self):
        epsilon = 1e-6
        for repeats in REPETITIONS:
            for gain in GAINS:
                probe = record_probe(gain=gain, repeats=repeats)
                for field in ("writing", "memory", "prediction"):
                    baseline = getattr(probe.weights, field)
                    baseline = (baseline,) if isinstance(baseline, float) else baseline
                    gradients = getattr(probe.weight_gradients, field)
                    gradients = (gradients,) if isinstance(gradients, float) else gradients
                    for index, expected in enumerate(gradients):
                        losses = []
                        for sign in (-1, 1):
                            values = list(baseline)
                            values[index] += sign * epsilon
                            kwargs = asdict(probe.weights)
                            kwargs[field] = values[0] if field == "memory" else tuple(values)
                            losses.append(forward_loss(Parameters(**kwargs), repeats)[0])
                        numerical = (losses[1] - losses[0]) / (2 * epsilon)
                        self.assertTrue(math.isclose(numerical, expected, rel_tol=2e-6, abs_tol=1e-8),
                                        (repeats, gain, field, index, numerical, expected))

    def test_new_weights_and_after_prediction_are_a_real_sgd_step(self):
        for repeats in REPETITIONS:
            for gain in GAINS:
                probe = record_probe(gain=gain, repeats=repeats)
                for field in ("writing", "memory", "prediction"):
                    before, gradient, after = (getattr(values, field) for values in (
                        probe.weights, probe.weight_gradients, probe.updated_weights,
                    ))
                    if field == "memory":
                        self.assertAlmostEqual(after, before - LEARNING_RATE * gradient, places=13)
                    else:
                        for old, grad, new in zip(before, gradient, after):
                            self.assertAlmostEqual(new, old - LEARNING_RATE * grad, places=13)
                loss, states, probabilities = forward_loss(probe.updated_weights, repeats)
                self.assertAlmostEqual(probe.after_loss, loss, places=13)
                for actual, expected in zip(probe.after_states, states):
                    self.assertAlmostEqual(actual, expected, places=14)
                for actual, expected in zip(probe.after_probabilities, probabilities):
                    self.assertAlmostEqual(actual, expected, places=14)
                self.assertEqual(probe.after_prediction, NOTES[max(range(3), key=probabilities.__getitem__)])
                self.assertLess(probe.after_loss, probe.loss)
                self.assertEqual(probe.prediction, "A3")
                self.assertEqual(probe.after_prediction, "D3")

    def test_three_notes_explain_the_factors_and_extension_compounds_them(self):
        for gain in GAINS:
            short, long = (record_probe(gain=gain, repeats=value) for value in REPETITIONS)
            short_ratio = abs(short.steps[0].feedback / short.steps[-1].feedback)
            long_ratio = abs(long.steps[0].feedback / long.steps[-1].feedback)
            self.assertAlmostEqual(short_ratio, gain**2, places=5)
            if gain < 1:
                self.assertLess(long_ratio, 0.02)
                self.assertLess(long_ratio, short_ratio / 50)
            else:
                self.assertGreater(long_ratio, 30)
                self.assertGreater(long_ratio, short_ratio * 20)
        balanced = record_probe(gain=1, repeats=7)
        self.assertAlmostEqual(balanced.steps[0].feedback / balanced.steps[-1].feedback, 1, places=3)

    def test_slider_trials_are_deterministic_not_fresh_random_initializations(self):
        self.assertEqual(record_probe(), record_probe())
        self.assertEqual(record_probe().weights.writing, record_probe(gain=1.2).weights.writing)
        self.assertEqual(record_probe().weights.prediction, record_probe(gain=1.2).weights.prediction)

    def test_bad_controls_are_reported(self):
        for gain in (math.nan, math.inf, True, "wrong", 0.59, 1.31):
            with self.assertRaises(ValueError):
                record_probe(gain=gain)
        for repeats in (False, 0, 2, 7.0):
            with self.assertRaises(ValueError):
                record_probe(repeats=repeats)

    def test_all_142_recorded_trials_match_fresh_autograd(self):
        data = read_recording()
        self.assertEqual(data["version"], 6)
        self.assertEqual(data["memory_nudge"], MEMORY_NUDGE)
        self.assertEqual(data["target_note"], TARGET_NOTE)
        self.assertEqual(data["learning_rate"], LEARNING_RATE)
        self.assertEqual(data["probability_decimals"], PROBABILITY_DECIMALS)
        self.assertEqual([(r["repeats"], r["gain"]) for r in data["routes"]],
                         [(repeats, gain) for repeats in REPETITIONS for gain in SCALE_GAINS])
        for route in data["routes"]:
            expected = asdict(record_probe(gain=route["gain"], repeats=route["repeats"]))
            self.assertEqual(route, json.loads(json.dumps(expected)))

    def test_player_export_is_complete_and_idempotent(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / PLAYER.name
            export_player(path)
            self.assertEqual(path.read_bytes(), PLAYER.read_bytes())
            self.assertEqual(path.read_text(encoding="utf-8"), render_player())
            path.write_text("<p>stale</p>", encoding="utf-8")
            export_player(path)
            self.assertEqual(path.read_bytes(), PLAYER.read_bytes())

    def test_storyboard_is_real_labelled_evidence_and_reproducible(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / STORYBOARD.name
            export_storyboard(path)
            with Image.open(STORYBOARD) as saved, Image.open(path) as rebuilt:
                self.assertEqual(saved.size, rebuilt.size)
                self.assertEqual(saved.tobytes(), rebuilt.tobytes())
                self.assertIn("D3/A3/D4", saved.info["TraceOrigin"])
                self.assertIn("all labels outside artwork", saved.info["DiagramLayout"])

    def test_player_controls_note_cells_formulas_and_update_are_present(self):
        source = PLAYER.read_text(encoding="utf-8")
        self.assertEqual(source.count('type="range"'), 1)
        self.assertEqual(source.count('type="checkbox"'), 1)
        for element_id in ("weight-scale", "long-phrase", "replay", "pause", "cells",
                           "forward-formula", "backward-formula", "factor-formula",
                           "seed-formula", "contribution-formula", "update-rows", "update-formula",
                           "memory-nudge", "lower-memory", "higher-memory", "response-arithmetic",
                           "response-comparison"):
            self.assertIn(f'id="{element_id}"', source)
        self.assertIn("Previous-memory multiplier", source)
        self.assertIn("Previous-memory multiplier (recurrent weight)", source)
        self.assertIn("Input multipliers", source)
        self.assertIn('"Error=" + probability(s.local_loss)', source)
        self.assertIn('"Ending-error"', source)
        self.assertIn('"response at h" + s.index', source)
        self.assertIn("Error response = ending-error change / saved-memory change", source)
        self.assertIn("Freeze the weights", source)
        self.assertIn("Positive response", source)
        self.assertIn("Negative response", source)
        self.assertNotIn("s.feedback * memoryNudge", source)
        self.assertNotIn('"w=["', source)
        self.assertNotIn('"L" + s.index + "="', source)
        self.assertNotIn('"g" + s.index', source)
        self.assertIn("Only the final prediction is graded", source)
        self.assertIn("not fresh random weights per note", source)
        self.assertIn("one hidden feature instead of two", source)
        self.assertIn('class:"memory-product"', source)
        self.assertIn('class:"head-probability"', source)
        self.assertIn('class:"prediction-block"', source)
        self.assertIn('class:"loss-seed-wire"', source)
        self.assertIn("preview only", source)
        self.assertNotIn('aria-label="Final next-note error"', source)
        self.assertNotIn("2. One final prediction", source)
        self.assertNotIn("preset-vanishing", source)
        self.assertNotIn("preset-exploding", source)
        self.assertNotIn("First</text>", source)
        self.assertNotIn("Middle</text>", source)
        self.assertNotIn("Last</text>", source)

    def test_mai_card_art_is_embedded_without_generated_measurements(self):
        source = PLAYER.read_text(encoding="utf-8")
        encoded = re.search(r'data:image/webp;base64,([A-Za-z0-9+/=]+)', source)[1]
        with Image.open(BytesIO(base64.b64decode(encoded, validate=True))) as image:
            self.assertEqual(image.format, "WEBP")
            self.assertEqual(image.size, (512, 288))
        manifest = json.loads(CARD_ART.with_name("01a-gradient-card-prompts.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest[0]["model"], "MAI-Image-2.6")
        self.assertEqual(manifest[0]["filename"], CARD_ART.name)

    def test_diagram_has_real_blocks_and_artwork_is_not_a_text_background(self):
        source = PLAYER.read_text(encoding="utf-8")
        self.assertIn('svgNode("rect"', source)
        self.assertIn('class:"rnn-block"', source)
        self.assertIn('class:"memory-icon"', source)
        self.assertIn('class:"backward-wire"', source)
        self.assertIn('"aria-label":"Weighted RNN blocks; memory flows right and ending-error responses flow left"', source)
        self.assertNotIn("var(--card-art)", source)
        self.assertIn('svgNode("use", {href:"#memory-card-art"', source)
        self.assertIn('event.key === "Enter" || event.key === " "', source)


if __name__ == "__main__":
    unittest.main()
