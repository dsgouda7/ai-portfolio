"""Verify gradient-led motivation without treating gated memory as a guarantee."""

import base64
import contextlib
from io import StringIO
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from PIL import Image
import torch

from backprop_memory_flow import _inputs_and_weights
from gradient_memory import (
    HERE, IMAGE_FILES, NOTEBOOK, clipping_probe, draw_cell_feedback,
    draw_gru_feedback, draw_saturation, draw_unsquashed_total, export,
    gru_carry, lstm_carry, native_gated_probe, piano_probe, recurrent_probe,
    refresh_notebook_captures, source_fingerprint, draw_notecard, notecard_probe,
    unsquashed_total,
)
from gradient_what_if import draw_nudge, record_nudge
from lstm_state_roles import _figure
from notebook_replays import ERROR_FEEDBACK_FILE


class GradientMemoryTests(unittest.TestCase):
    def test_nudge_copies_only_the_first_saved_entry_and_reuses_the_frozen_head(self):
        from backprop_memory_flow import NOTE_HEAD_WEIGHTS, NOTES
        from torch.nn import functional as F

        saved = notecard_probe().new_hidden.clone()
        probe = record_nudge()
        head = torch.tensor(NOTE_HEAD_WEIGHTS, dtype=saved.dtype)
        for trial in probe.trials:
            self.assertEqual(trial.memory[1], saved[1].item())
            self.assertEqual(trial.memory[0], saved[0].item() + trial.change)
            memory = torch.tensor(trial.memory, dtype=saved.dtype)
            scores = head @ memory
            self.assertEqual(trial.correct_probability, scores.softmax(0)[0].item())
            self.assertEqual(trial.loss, F.cross_entropy(scores.unsqueeze(0), torch.tensor([0])).item())
            self.assertEqual(trial.guess, NOTES[int(scores.argmax())])
        torch.testing.assert_close(notecard_probe().new_hidden, saved, rtol=0, atol=0)

    def test_nudge_exposes_a_useful_gradient_before_the_guessed_note_changes(self):
        probe = record_nudge()
        lower, original, higher = probe.trials
        self.assertEqual([trial.guess for trial in probe.trials], ["A3"] * 3)
        self.assertLess(lower.correct_probability, original.correct_probability)
        self.assertLess(original.correct_probability, higher.correct_probability)
        self.assertGreater(lower.loss, original.loss)
        self.assertGreater(original.loss, higher.loss)
        self.assertEqual(probe.loss_difference, higher.loss - lower.loss)
        self.assertEqual(probe.memory_difference, higher.memory[0] - lower.memory[0])
        self.assertAlmostEqual(probe.finite_difference, probe.exact_gradient, delta=1e-6)
        self.assertLess(probe.finite_difference, 0)

    def test_nudge_image_circles_the_changed_entry_and_shows_the_measured_comparison(self):
        probe = record_nudge()
        figure = draw_nudge(probe)
        try:
            figure.canvas.draw()
            axis = figure.axes[0]
            renderer = figure.canvas.get_renderer()
            first = next(text for text in axis.texts if text.get_gid() == "nudged-entry")
            fixed = next(text for text in axis.texts if text.get_gid() == "fixed-entry")
            prefix = next(text for text in axis.texts if text.get_gid() == "memory-prefix")
            circle = next(patch for patch in axis.patches if patch.get_gid() == "first-entry-circle")
            circle_bounds = circle.get_window_extent(renderer)
            first_bounds = first.get_window_extent(renderer)
            self.assertTrue(circle_bounds.contains(first_bounds.x0, first_bounds.y0))
            self.assertTrue(circle_bounds.contains(first_bounds.x1, first_bounds.y1))
            self.assertFalse(circle_bounds.overlaps(fixed.get_window_extent(renderer)))
            self.assertFalse(circle_bounds.overlaps(prefix.get_window_extent(renderer)))
            labels = " ".join(text.get_text() for text in axis.texts)
            for trial in (probe.trials[0], probe.trials[2]):
                for value in (trial.memory[0], trial.correct_probability, trial.loss):
                    self.assertIn(f"{value:.6f}", labels)
            for text in axis.texts:
                bounds = text.get_window_extent(renderer)
                self.assertTrue(figure.bbox.contains(bounds.x0, bounds.y0), text.get_text())
                self.assertTrue(figure.bbox.contains(bounds.x1, bounds.y1), text.get_text())
        finally:
            plt.close(figure)

    def test_notecard_contains_the_actual_d4_step_and_shared_head_scores(self):
        probe = notecard_probe()
        torch.testing.assert_close(probe.input_part, probe.input_weights @ probe.current_input, rtol=0, atol=0)
        torch.testing.assert_close(probe.memory_part, probe.memory_weights @ probe.old_hidden, rtol=0, atol=0)
        torch.testing.assert_close(probe.new_hidden, torch.tanh(probe.combined), rtol=0, atol=0)
        torch.testing.assert_close(probe.new_hidden,
                                   torch.tensor([0.594478, 0.892802], dtype=torch.float64), atol=1e-6, rtol=0)
        self.assertEqual(probe.prediction, "A3")

    def test_erasing_only_the_incoming_card_changes_context_not_the_note_or_weights(self):
        original, erased = notecard_probe(), notecard_probe(erase_incoming=True)
        for field in ("current_input", "input_weights", "memory_weights", "input_part"):
            torch.testing.assert_close(getattr(original, field), getattr(erased, field), rtol=0, atol=0)
        self.assertTrue(torch.equal(erased.memory_part, torch.zeros_like(erased.memory_part)))
        self.assertFalse(torch.equal(original.new_hidden, erased.new_hidden))
        self.assertFalse(torch.equal(original.scores, erased.scores))
        with self.assertRaises(ValueError):
            notecard_probe(erase_incoming="yes")

    def test_notecard_figure_displays_actual_values_without_invented_note_meanings(self):
        figure, axis = _figure()
        try:
            probe = notecard_probe()
            draw_notecard(axis, probe)
            labels = " ".join(text.get_text() for text in axis.texts)
            for tensor in (probe.old_hidden, probe.input_part, probe.memory_part, probe.new_hidden, probe.scores):
                for value in tensor:
                    self.assertIn(f"{value:.4f}", labels)
            self.assertIn("computed features", labels)
            self.assertIn("not learned music", labels)
            self.assertTrue(any(patch.get_gid() == "notecard-carry" for patch in axis.patches))
            self.assertTrue(any(patch.get_gid() == "notecard-expose" for patch in axis.patches))
            figure.canvas.draw()
            renderer = figure.canvas.get_renderer()
            score_label = next(text for text in axis.texts if text.get_gid() == "notecard-score-label")
            score_card = next(patch for patch in axis.patches if patch.get_gid() == "notecard-score-card")
            bounds = score_label.get_window_extent(renderer)
            card_bounds = score_card.get_window_extent(renderer)
            self.assertTrue(card_bounds.contains(bounds.x0, bounds.y0))
            self.assertTrue(card_bounds.contains(bounds.x1, bounds.y1))
        finally:
            plt.close(figure)

    def test_scalar_rnn_feedback_includes_gain_and_every_tanh_response(self):
        probe = recurrent_probe()
        expected = math.prod(0.8 * (1 - state[0] ** 2) for state in probe.states[1:])
        self.assertAlmostEqual(probe.feedback[0], expected, places=14)
        self.assertLess(probe.feedback[0], 0.8**20)
        self.assertEqual(probe.feedback[-1], 1)

    def test_bounded_states_do_not_prevent_exploding_feedback(self):
        probe = recurrent_probe(1.2, opening=0)
        self.assertEqual(probe.states, ((0.0,),) * 21)
        self.assertAlmostEqual(probe.feedback[0], 1.2**20, places=10)
        self.assertGreater(probe.feedback[0], 30)

    def test_saturation_weakens_a_change_even_with_unit_recurrent_gain(self):
        ordinary = recurrent_probe(1.0, opening=0.8, delay=1)
        saturated = recurrent_probe(1.0, opening=0.8, drive=4.0, delay=1)
        self.assertLess(saturated.feedback[0], ordinary.feedback[0] / 1000)
        self.assertTrue(all(abs(state[0]) < 1 for state in saturated.states[1:]))

    def test_saturation_comparison_measures_both_regions_in_one_static_figure(self):
        figure, axis = _figure()
        try:
            draw_saturation(axis)
            labels = " ".join(text.get_text() for text in axis.texts)
            for start in (0.0, 3.0):
                change = math.tanh(start + 0.0001) - math.tanh(start)
                self.assertIn(f"+{change:.10f}", labels)
                self.assertIn(f"{change / 0.0001:.0%}", labels)
            curve = axis.lines[0]
            inputs = (curve.get_xdata() - 4.5) / 0.8
            expected = 5.6 + torch.as_tensor(inputs).tanh().numpy() * 1.8
            torch.testing.assert_close(torch.as_tensor(curve.get_ydata()), torch.as_tensor(expected))
        finally:
            plt.close(figure)
        book = json.loads(NOTEBOOK.read_bytes())
        cell = next(cell for cell in book["cells"] if cell["id"] == "gradient-saturation-intro")
        self.assertEqual(cell["cell_type"], "markdown")
        self.assertNotIn("outputs", cell)
        self.assertIn("01a-tanh-saturation.png", cell["attachments"])
        self.assertFalse(any(cell["id"] == "gradient-saturation-probe" for cell in book["cells"]))

    def test_native_piano_reader_has_two_features_and_a_connected_weak_opening_route(self):
        short, long = piano_probe(5), piano_probe(50)
        self.assertEqual(len(long.states), 51)
        self.assertTrue(all(len(state) == 2 for state in long.states))
        self.assertTrue(all(abs(value) < 1 for state in long.states for value in state))
        self.assertGreater(long.feedback[0], 0)
        self.assertLess(long.feedback[0], short.feedback[0] / 1000)
        self.assertAlmostEqual(long.feedback[-1], math.sqrt(1.25))

    def test_clipping_caps_an_actual_writer_weight_update(self):
        probe = clipping_probe(1.2, opening=0)
        self.assertGreater(abs(probe.raw_gradient), 30)
        self.assertAlmostEqual(probe.raw_gradient, -(1.2**20), places=10)
        self.assertAlmostEqual(abs(probe.clipped_gradient), 1, places=6)
        self.assertAlmostEqual(probe.raw_update, -0.01 * probe.raw_gradient)
        self.assertAlmostEqual(probe.clipped_update, -0.01 * probe.clipped_gradient)
        self.assertGreater(abs(probe.raw_update), 30 * abs(probe.clipped_update))

    def test_clipping_does_not_increase_a_vanished_writer_gradient(self):
        probe = clipping_probe(0.8, opening=0.8)
        self.assertLess(abs(probe.raw_gradient), 0.01)
        self.assertEqual(probe.raw_gradient, probe.clipped_gradient)
        self.assertEqual(probe.raw_update, probe.clipped_update)

    def test_unsquashed_totals_can_lose_order_with_perfect_gradient_carry(self):
        first, other = unsquashed_total("D3"), unsquashed_total("D4")
        self.assertNotEqual(first.stored[0], other.stored[0])
        self.assertEqual(first.notes, ("D3", "A3", "D4", "A3"))
        self.assertEqual(other.notes, ("D4", "A3", "D3", "A3"))
        self.assertEqual(first.inputs, ((1, 0), (0, 1), (1, 1), (0, 1)))
        self.assertEqual(other.inputs, ((1, 1), (0, 1), (1, 0), (0, 1)))
        for actual, expected in (
            (first.stored, ((0.70, 0.30), (0.50, 1.10), (1.00, 2.20), (0.80, 3.00))),
            (other.stored, ((0.50, 1.10), (0.30, 1.90), (1.00, 2.20), (0.80, 3.00))),
        ):
            torch.testing.assert_close(torch.tensor(actual, dtype=torch.float64),
                                       torch.tensor(expected, dtype=torch.float64), rtol=0, atol=1e-6)
        self.assertEqual(first.stored[-1], other.stored[-1])
        _, input_weights, _ = _inputs_and_weights(NOTEBOOK)
        for probe in (first, other):
            writes = torch.tensor(probe.inputs, dtype=torch.float64) @ input_weights.T
            torch.testing.assert_close(torch.tensor(probe.writes, dtype=torch.float64), writes, rtol=0, atol=0)
            torch.testing.assert_close(torch.tensor(probe.stored, dtype=torch.float64),
                                       writes.cumsum(0), rtol=0, atol=1e-14)
            self.assertEqual(probe.stored_feedback, (1, 1))
            self.assertEqual(probe.first_write_jacobian, ((1, 0), (0, 1)))

    def test_additive_figure_shows_the_actual_fractional_writes_and_totals(self):
        figure, axis = _figure()
        try:
            draw_unsquashed_total(axis)
            labels = " ".join(text.get_text() for text in axis.texts)
            probe = unsquashed_total("D3")
            for values in (*probe.writes, *probe.stored):
                self.assertIn("[" + ", ".join(f"{value:.2f}" for value in values) + "]", labels)
            self.assertIn("[0.80, 3.00]", labels)
            self.assertIn("100% carry-through", labels)
            self.assertNotIn("add [1, 0]", labels)
            self.assertNotIn("add [0, 1]", labels)
        finally:
            plt.close(figure)

    def test_linear_remixing_can_keep_order_without_a_squash(self):
        endings = []
        for opening in ("D3", "D4"):
            state = 0.0
            for write in unsquashed_total(opening).writes:
                state = 0.8 * state + write[0]
            endings.append(state)
        self.assertAlmostEqual(endings[0], 0.4304, places=6)
        self.assertAlmostEqual(endings[1], 0.4880, places=6)
        self.assertGreater(abs(endings[0] - endings[1]), 0.05)

    def test_additive_first_write_nudges_survive_in_the_matching_feature(self):
        probe = unsquashed_total("D3")
        writes = torch.tensor(probe.writes, dtype=torch.float64)
        for feature in range(2):
            nudged = writes.clone()
            nudged[0, feature] += 0.01
            expected = torch.zeros(2, dtype=torch.float64)
            expected[feature] = 0.01
            torch.testing.assert_close(nudged.sum(0) - writes.sum(0), expected, rtol=0, atol=1e-14)

    def test_additive_notebook_check_reloads_the_helper_and_has_current_outputs(self):
        book = json.loads(NOTEBOOK.read_bytes())
        cell = next(cell for cell in book["cells"] if cell["id"] == "unsquashed-total-probe")
        source = "".join(cell["source"])
        self.assertIn("importlib.reload", source)
        with contextlib.redirect_stdout(StringIO()) as captured:
            exec(compile(source, "unsquashed-total-probe", "exec"), {})
        text = captured.getvalue()
        for value in ("[0.70, 0.30]", "[-0.20, 0.80]", "[0.50, 1.10]", "[0.80, 3.00]"):
            self.assertIn(value, text)
        self.assertTrue(all(output["output_type"] != "error" for output in cell.get("outputs", [])))
        if cell.get("outputs"):
            saved = "".join(
                "".join(output["text"]) if isinstance(output["text"], list) else output["text"]
                for output in cell["outputs"] if output["output_type"] == "stream"
            )
            self.assertEqual(saved, text)

    def test_lstm_c_is_not_bounded_but_h_is_and_the_readout_can_saturate(self):
        probe = lstm_carry(1, opening=0, write=1, delay=5)
        self.assertAlmostEqual(probe.stored[-1], 4)
        self.assertLess(probe.exposed[-1], 1)
        self.assertEqual(probe.stored_feedback, 1)
        self.assertLess(probe.exposed_feedback, 0.002)

    def test_constant_lstm_keep_controls_determine_only_the_isolated_direct_lane(self):
        for keep in (1, 0.99, 0.9):
            probe = lstm_carry(keep)
            self.assertAlmostEqual(probe.stored_feedback, keep**20)
            self.assertAlmostEqual(probe.exposed_feedback,
                                   keep**20 * (1 - probe.exposed[-1]**2))
        self.assertGreater(lstm_carry(0.99).stored_feedback, 0.8)
        self.assertLess(lstm_carry(0.9).stored_feedback, 0.13)

    def test_gru_protects_bounded_h_without_an_outer_tanh(self):
        for keep in (1, 0.99, 0.9):
            probe = gru_carry(keep)
            self.assertIs(probe.stored, probe.exposed)
            expected = math.tanh(4) + keep**20 * (0.8 - math.tanh(4))
            self.assertAlmostEqual(probe.stored[-1], expected)
            self.assertAlmostEqual(probe.stored_feedback, keep**20)
            self.assertTrue(all(-1 <= value <= 1 for value in probe.stored))
        self.assertEqual(gru_carry(1).stored, (0.8,) * 21)

    def test_gru_reset_and_keep_change_different_gradient_routes(self):
        closed = gru_carry(0, reset=0, drive=0.2, delay=1)
        open_reset = gru_carry(0, reset=1, drive=0.2, delay=1)
        self.assertEqual(closed.stored_feedback, 0)
        self.assertGreater(open_reset.stored_feedback, 0)
        first, other = gru_carry(0.99), gru_carry(0.99, opening=-0.8)
        self.assertAlmostEqual(first.stored[-1] - other.stored[-1], 1.6 * 0.99**20)

    def test_real_lstm_and_gru_can_amplify_feedback_despite_a_protected_keep_lane(self):
        for kind, gain in (("lstm", 8), ("gru", 400)):
            calm = native_gated_probe(kind, 0)
            aggressive = native_gated_probe(kind, gain)
            self.assertTrue(math.isclose(calm.stored_feedback, 0.99**20, rel_tol=1e-12))
            self.assertTrue(math.isclose(aggressive.stored_feedback, 2.99**20, rel_tol=1e-12))
            self.assertGreater(aggressive.exposed_feedback, 1e8)
            self.assertEqual(aggressive.stored, (0.0,) * 21)
            self.assertEqual(aggressive.exposed, (0.0,) * 21)

    def test_probes_leave_global_rng_and_thread_settings_unchanged(self):
        rng, threads = torch.random.get_rng_state().clone(), torch.get_num_threads()
        recurrent_probe(); piano_probe(); clipping_probe(1.2, opening=0)
        lstm_carry(); gru_carry(); unsquashed_total("D3")
        native_gated_probe("lstm", 8); native_gated_probe("gru", 400)
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        self.assertEqual(threads, torch.get_num_threads())

    def test_invalid_controls_and_delays_raise_explicit_errors(self):
        for call in (
            lambda: recurrent_probe(delay=0), lambda: piano_probe(True),
            lambda: recurrent_probe(gain=float("nan")), lambda: lstm_carry(-0.1),
            lambda: lstm_carry(proposal=2), lambda: gru_carry(opening=2),
            lambda: gru_carry(reset=1.1), lambda: clipping_probe(1.2, opening=0, max_norm=0),
            lambda: unsquashed_total("A3"), lambda: unsquashed_total(None),
        ):
            with self.assertRaises(ValueError):
                call()

    def test_static_labels_fit_and_connectors_do_not_cross_them(self):
        for draw in (draw_saturation, draw_unsquashed_total, draw_cell_feedback, draw_gru_feedback,
                     lambda axis: draw_notecard(axis, notecard_probe())):
            figure, axis = _figure()
            try:
                draw(axis)
                figure.canvas.draw()
                renderer = figure.canvas.get_renderer()
                texts = [text for text in axis.texts if text.get_text()]
                for index, text in enumerate(texts):
                    bounds = text.get_window_extent(renderer)
                    self.assertTrue(axis.bbox.contains(bounds.x0, bounds.y0), text.get_text())
                    self.assertTrue(axis.bbox.contains(bounds.x1, bounds.y1), text.get_text())
                    for other in texts[index + 1:]:
                        self.assertFalse(bounds.overlaps(other.get_window_extent(renderer)),
                                         (text.get_text(), other.get_text()))
                    for arrow in axis.patches:
                        if isinstance(arrow, FancyArrowPatch):
                            path = arrow.get_transform().transform_path(arrow.get_path())
                            if path.get_extents().overlaps(bounds):
                                self.assertFalse(path.intersects_bbox(bounds, filled=False), text.get_text())
            finally:
                plt.close(figure)

    def test_export_writes_only_the_retained_static_evidence(self):
        with TemporaryDirectory() as directory:
            target = Path(directory)
            export(directory=target)
            expected = set(IMAGE_FILES) | {
                "01a-backprop-saved-cards.png", "01a-backprop-saved-cards-preview.png",
            }
            self.assertEqual({path.name for path in target.iterdir()}, expected)

    def test_gru_diagram_separates_one_update_from_twenty_step_feedback(self):
        figure, axis = _figure()
        try:
            draw_gru_feedback(axis)
            labels = " ".join(text.get_text() for text in axis.texts)
            self.assertIn(f"new h: {gru_carry(delay=1).stored[-1]:+.4f}", labels)
            self.assertIn(f"{gru_carry().stored_feedback:.6f}", labels)
            write = next(text for text in axis.texts if text.get_gid() == "gru-write-label")
            x, y = write.get_position()
            self.assertLess(x, 12.15)
            self.assertGreater(y, 5.6)
            self.assertLess(y, 7.25)
            self.assertTrue(any(patch.get_gid() == "gru-proposal-write" for patch in axis.patches))
        finally:
            plt.close(figure)

    def test_saved_evidence_is_fingerprinted_and_embedded(self):
        book = json.loads(NOTEBOOK.read_bytes())
        for filename in IMAGE_FILES:
            owner = next(cell for cell in book["cells"]
                         if f"attachment:{filename}" in "".join(cell["source"]))
            self.assertEqual(base64.b64decode(owner["attachments"][filename]["image/png"]),
                             (HERE / "images" / filename).read_bytes())
            with Image.open(HERE / "images" / filename) as image:
                self.assertEqual(image.info["SourceFingerprint"], source_fingerprint())

    def test_notebook_builds_gradient_motivation_before_cell_roles_and_gru_controls(self):
        book = json.loads(NOTEBOOK.read_bytes())
        ids = [cell["id"] for cell in book["cells"]]
        self.assertLess(ids.index("gradient-native-intro"), ids.index("embedded-01a-error-feedback"))
        start = ids.index("gradient-native-intro")
        self.assertEqual(ids[start:start + 3], [
            "gradient-native-intro", "embedded-01a-error-feedback",
            "gradient-exploding-intuition",
        ])
        intro = next(cell for cell in book["cells"] if cell["id"] == "gradient-native-intro")
        self.assertTrue("".join(intro["source"]).startswith(
            "## 4 - One calculation; three feedback outcomes\n"
        ))
        self.assertFalse(any("Small numbers on the card can still produce large feedback"
                             in "".join(cell["source"]) for cell in book["cells"]))
        player = next(cell for cell in book["cells"] if cell["id"] == "embedded-01a-error-feedback")
        self.assertEqual(player["cell_type"], "markdown")
        self.assertTrue("".join(player["source"]).startswith("<iframe "))
        self.assertEqual(player["metadata"]["saved_replay"]["file"], ERROR_FEEDBACK_FILE)
        self.assertLess(ids.index("gradient-exploding-intuition"), ids.index("5001b8f0"))
        self.assertLess(ids.index("bptt-manual-check"), ids.index("5001b8f0"))
        for removed in ("gradient-native-probe", "gradient-technical-appendix", "9df9ac84"):
            self.assertNotIn(removed, ids)
        self.assertLess(ids.index("gradient-saturation-intro"), ids.index("133e7838"))
        self.assertLess(ids.index("gradient-clipping-probe"), ids.index("133e7838"))
        self.assertLess(ids.index("unsquashed-total-probe"), ids.index("133e7838"))
        self.assertLess(ids.index("lstm-real-cell-parity"), ids.index("lstm-gradient-carry-probe"))
        self.assertLess(ids.index("gru-real-cell-parity"), ids.index("gru-gradient-carry-probe"))
        self.assertLess(ids.index("gru-real-cell-parity"), ids.index("gated-gradient-limits"))
        self.assertLess(ids.index("gated-gradient-limits"), ids.index("lstm-echo-intro"))
        self.assertLess(ids.index("lstm-echo-training"), ids.index("musical-echo-training"))

    def test_notebook_names_model_changes_and_distinguishes_values_from_parameters(self):
        book = json.loads(NOTEBOOK.read_bytes())
        sources = {cell["id"]: " ".join("".join(cell["source"]).split()) for cell in book["cells"]}
        checks = {
            "7d99e444": ("fourth note", "first note", "same last two notes",
                         "hand-picked weights"),
            "gradient-native-intro": ("one number", "[0.001,-0.0005]",
                                     "only the final prediction", "starting memory weight W_h"),
            "5001b8f0": ("20 steps", "not 20 memory features", "1.0",
                        "0.01", "same forward values and error"),
            "gated-memory-demand": ("same total whichever order", "without tanh can still use order", "(100%)",
                                   "[0.70,0.30]", "[0.71,0.30]", "[0.80,3.00]", "[0.81,3.00]"),
            "133e7838": ("one number inside c", "chosen memory writes", "not input codes"),
            "lstm-readout-after-memory": ("computed values", "trainable weights",
                                         "`g` means **proposal**", "`dL/dh`"),
            "lstm-native-intro": ("two input numbers", "not trained them yet",
                                 "same PyTorch cell", "Two `Linear` calculations"),
            "lstm-echo-intro": ("four-number code", "twelve numbers",
                               "three echo answers", "one final guess", "silent request",
                               "A3 D4 A3 D3 repeated five times", "twenty note events"),
            "gru-learned-gates-intro": ("two memory numbers", "not been trained yet",
                                      "larger memory before training"),
        }
        for cell_id, phrases in checks.items():
            for phrase in phrases:
                with self.subTest(cell=cell_id, phrase=phrase):
                    self.assertIn(phrase, sources[cell_id])

    def test_opening_puzzle_requires_the_first_note_not_the_shared_suffix(self):
        book = json.loads(NOTEBOOK.read_bytes())
        opening = "".join(next(cell for cell in book["cells"] if cell["id"] == "7d99e444")["source"])
        examples = []
        for line in opening.splitlines():
            if line.startswith("| ") and line.count("|") == 3:
                phrase, answer = (column.strip() for column in line.split("|")[1:-1])
                notes = phrase.split()
                if len(notes) == 3 and all(note in ("D3", "A3", "D4") for note in notes):
                    examples.append((notes, answer))
        self.assertEqual(len(examples), 2)
        self.assertEqual(examples[0][0][1:], examples[1][0][1:])
        self.assertNotEqual(examples[0][1], examples[1][1])
        for notes, answer in examples:
            self.assertEqual(answer, notes[0])
        self.assertNotIn("PLAY", opening)
        self.assertNotIn("twenty", opening.lower())

    def test_notebook_bridges_link_to_the_actual_walkthrough_and_appendix_checks(self):
        book = json.loads(NOTEBOOK.read_bytes())
        cells = {cell["id"]: cell for cell in book["cells"]}
        prose = "\n".join("".join(cell["source"]) for cell in book["cells"]
                          if cell["cell_type"] == "markdown"
                          and "saved_replay" not in cell["metadata"])
        for anchor in ("memory-challenge", "rnn-gradient-microscope", "lstm-complete-read",
                       "backprop-horizon"):
            self.assertEqual(prose.count(f'<a id="{anchor}"></a>'), 1)
        for cell_id, anchor in (
            ("lstm-echo-intro", "memory-challenge"),
            ("lstm-native-intro", "lstm-complete-read"),
        ):
            self.assertIn(f"](#{anchor})", "".join(cells[cell_id]["source"]))
        ids = [cell["id"] for cell in book["cells"]]
        self.assertLess(ids.index("lstm-gradient-carry-intro"), ids.index("lstm-gradient-carry-probe"))
        self.assertLess(ids.index("lstm-gradient-carry-probe"), ids.index("lstm-gradient-carry-reading"))
        self.assertIn("**What you measured:**", "".join(cells["lstm-gradient-carry-reading"]["source"]))
        self.assertIn("not a note-prediction error or the size of a weight update",
                      " ".join("".join(cells["gated-gradient-limits-intro"]["source"]).split()))
        self.assertFalse(any("fixed-fraction-retention" in "".join(cell["source"])
                             for cell in book["cells"]))

    def test_regeneration_refresh_preserves_experiments_and_is_idempotent(self):
        before = json.loads(NOTEBOOK.read_bytes())
        with TemporaryDirectory() as directory:
            path = Path(directory) / NOTEBOOK.name
            path.write_bytes(NOTEBOOK.read_bytes())
            refresh_notebook_captures(path)
            first = path.read_bytes()
            refresh_notebook_captures(path)
            self.assertEqual(first, path.read_bytes())
            after = json.loads(first)
        for original, updated in zip(before["cells"], after["cells"]):
            self.assertEqual(original["source"], updated["source"])
            if original["cell_type"] == "code":
                self.assertEqual(original, updated)


if __name__ == "__main__":
    unittest.main()
