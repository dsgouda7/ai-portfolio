"""Verify two-state LSTM mechanics, actual audible recall, and the teaching boundary."""

import base64
from copy import deepcopy
from io import BytesIO
import json
from pathlib import Path
import re
import unittest

import matplotlib.pyplot as plt
from PIL import Image
import torch
from torch import nn

import musical_memory
from lstm_layer_flow import LSTM_PHASES, draw_lstm_layer, record_lstm_layer_step, source_fingerprint
from memory_visuals import MOTION_FRAMES
from musical_memory import (
    EchoLSTM, NOTES, echo_audio_panel, fit_echo_model, make_echo_questions,
    music_frames, note_audio, run_echo_experiment, source_fingerprint as music_fingerprint,
    synthesize_timeline, trace_echo, wav_bytes,
)


HERE = Path(__file__).parent


class LSTMLayerTests(unittest.TestCase):
    def setUp(self):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(42)
            self.cell = nn.LSTMCell(2, 2).double()
        self.x = torch.tensor([[0., 1.]], dtype=torch.float64)
        self.h = torch.tensor([[0.2, -0.3]], dtype=torch.float64)
        self.c = torch.tensor([[0.7, -0.4]], dtype=torch.float64)

    def test_native_cell_both_states_and_separate_output_role(self):
        step = record_lstm_layer_step(self.cell, self.x, self.h, self.c)
        h, c = self.cell(self.x, (self.h, self.c))
        torch.testing.assert_close(step.new_cell, c.flatten(), rtol=0, atol=1e-12)
        torch.testing.assert_close(step.new_hidden, h.flatten(), rtol=0, atol=1e-12)
        torch.testing.assert_close(step.output, step.carried_hidden, rtol=0, atol=0)
        torch.testing.assert_close(step.carried_cell, c.flatten(), rtol=0, atol=0)
        self.assertFalse(torch.equal(step.carried_hidden, step.carried_cell))
        torch.testing.assert_close(step.retained, step.forget_gate * self.c.flatten())
        torch.testing.assert_close(step.written, step.input_gate * step.candidate)
        torch.testing.assert_close(step.new_cell, step.retained + step.written)
        torch.testing.assert_close(step.new_hidden, step.output_gate * step.new_cell.tanh())

    def test_cell_state_bypasses_the_standard_gate_projections(self):
        first = record_lstm_layer_step(self.cell, self.x, self.h, self.c)
        other = record_lstm_layer_step(self.cell, self.x, self.h, self.c + 1)
        for field in ("input_gate", "forget_gate", "candidate", "output_gate"):
            torch.testing.assert_close(getattr(first, field), getattr(other, field), rtol=0, atol=0)
        self.assertFalse(torch.equal(first.new_cell, other.new_cell))

    def test_close_only_show_on_one_real_step_without_erasing_cell(self):
        closed = deepcopy(self.cell)
        with torch.no_grad():
            closed.weight_ih[6:].zero_()
            closed.weight_hh[6:].zero_()
            closed.bias_ih[6:].fill_(-1e9)
            closed.bias_hh[6:].zero_()
        original = record_lstm_layer_step(self.cell, self.x, self.h, self.c)
        hidden = record_lstm_layer_step(closed, self.x, self.h, self.c)
        torch.testing.assert_close(original.carried_cell, hidden.carried_cell, rtol=0, atol=0)
        self.assertGreater(float(hidden.carried_cell.abs().max()), 0)
        self.assertEqual(float(hidden.output.abs().max()), 0)

    def test_native_sequence_layer_and_read_only_capture(self):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(42)
            layer = nn.LSTM(2, 2, batch_first=True).double()
        weights = deepcopy(layer.state_dict())
        rng = torch.random.get_rng_state().clone()
        old_h, old_c = self.h.clone(), self.c.clone()
        step = record_lstm_layer_step(layer, self.x, self.h, self.c)
        out, (h, c) = layer(self.x[:, None], (self.h[None], self.c[None]))
        for actual, expected in ((step.output, out), (step.carried_hidden, h), (step.carried_cell, c)):
            torch.testing.assert_close(actual, expected.flatten(), rtol=0, atol=0)
        self.assertTrue(layer.training)
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        self.assertTrue(torch.equal(old_h, self.h) and torch.equal(old_c, self.c))
        for key, value in layer.state_dict().items():
            self.assertTrue(torch.equal(value, weights[key]))
        self.assertFalse(step.output.requires_grad)

    def test_reject_unsupported_cells_stacks_projections_and_missing_state_rows(self):
        with self.assertRaisesRegex(TypeError, "LSTM"):
            record_lstm_layer_step(nn.GRUCell(2, 2), self.x, self.h, self.c)
        for layer in (nn.LSTM(2, 2, num_layers=2), nn.LSTM(2, 2, bidirectional=True),
                      nn.LSTM(2, 2, proj_size=1)):
            with self.assertRaisesRegex(ValueError, "one unidirectional"):
                record_lstm_layer_step(layer, self.x, self.h, self.c)
        with self.assertRaisesRegex(ValueError, "both"):
            record_lstm_layer_step(self.cell, self.x, self.h, self.c.flatten())

    def test_draw_has_both_states_and_no_pretend_predicted_notes(self):
        step = record_lstm_layer_step(self.cell, self.x, self.h, self.c)
        figure, axis = plt.subplots()
        try:
            seen = []
            for phase in range(len(LSTM_PHASES)):
                draw_lstm_layer(axis, step, phase)
                seen.extend(text.get_text() for text in axis.texts)
            labels = " ".join(seen).replace("\n", " ")
            for label in ("Previous h", "Previous c", "Output now: h", "Next token: (h, c)",
                          "KEEP", "WRITE", "SHOW", "old c bypasses"):
                self.assertIn(label, labels)
        finally:
            plt.close(figure)


class LSTMEchoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.comparison = run_echo_experiment(kind="LSTM")
        cls.book = json.loads((HERE / "01A-sequence-memory-theory.ipynb").read_text(encoding="utf-8"))

    def test_actual_before_after_and_both_state_interventions(self):
        comparison = self.comparison
        self.assertEqual(comparison.kind, "LSTM")
        self.assertEqual(len(comparison.losses), 480)
        self.assertLess(comparison.losses[-1], 0.001)
        expected = {
            "Before learning": [(2, 2, 2)] * 2,
            "After learning": [(0, 1, 2), (2, 1, 0)],
            "Opening erased": [(2, 1, 2)] * 2,
            "No carry": [(0, 1, 2)] * 2,
        }
        for mode, traces in comparison.traces.items():
            self.assertEqual([trace.generated for trace in traces], expected[mode])
            for trace in traces:
                for index in range(1, 3):
                    self.assertEqual(trace.outputs[index].input_id, trace.generated[index - 1])
                for step in (*trace.prompt, *trace.outputs):
                    assert isinstance(step, musical_memory.LSTMEchoStep)
                    torch.testing.assert_close(step.cell, step.retained + step.written)
                    torch.testing.assert_close(step.retained, step.forget_gate * step.old_cell)
                    torch.testing.assert_close(step.written, step.input_gate * step.candidate)
                    torch.testing.assert_close(step.hidden, step.output_gate * step.cell.tanh())
                    if mode == "No carry":
                        self.assertEqual(float(step.old.norm() + step.old_cell.norm()), 0)
        for case in range(2):
            erased = comparison.traces["Opening erased"][case].prompt[3]
            assert isinstance(erased, musical_memory.LSTMEchoStep)
            self.assertEqual(float(erased.old.norm() + erased.old_cell.norm()), 0)
            kept = comparison.traces["After learning"][case].prompt[3]
            assert isinstance(kept, musical_memory.LSTMEchoStep)
            self.assertGreater(float(kept.old.norm()), 0)
            self.assertGreater(float(kept.old_cell.norm()), 0)

    def test_shorter_fit_publishes_a_real_failure_not_a_guarantee(self):
        prompts, practice, answers = make_echo_questions()
        threads = torch.get_num_threads()
        try:
            torch.set_num_threads(1)
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(42)
                short = EchoLSTM()
                fit_echo_model(short, practice, answers, steps=240)
                self.assertEqual([trace_echo(short, prompt).generated for prompt in prompts], [(2, 1, 0)] * 2)
        finally:
            torch.set_num_threads(threads)

    def test_recording_does_not_change_weights_mode_rng_or_live_gradients(self):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(42)
            model = EchoLSTM()
        prompts, practice, answers = make_echo_questions()
        nn.functional.cross_entropy(model(practice)[:, -3:].reshape(-1, 3), answers.flatten()).backward()
        weights = deepcopy(model.state_dict())
        gradients = [value.grad.clone() for value in model.parameters()]
        rng = torch.random.get_rng_state().clone()
        threads = torch.get_num_threads()
        trace_echo(model, prompts[0], erase_opening=True)
        self.assertTrue(model.training)
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        self.assertEqual(threads, torch.get_num_threads())
        for key, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, weights[key]))
        for value, gradient in zip(model.parameters(), gradients):
            self.assertTrue(torch.equal(value.grad, gradient))
        run_echo_experiment(kind="LSTM")
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        self.assertEqual(threads, torch.get_num_threads())

    def test_audio_is_the_actual_generated_notes_not_hidden_state_sonification(self):
        html = echo_audio_panel(self.comparison).data
        clips = re.findall(r'src="data:audio/wav;base64,([^"]+)"', html)
        expected = [tuple(NOTES[index] for index in answer) for answer in self.comparison.answers.tolist()]
        for mode in ("Before learning", "After learning", "Opening erased", "No carry"):
            expected.extend(tuple(NOTES[index] for index in trace.generated) for trace in self.comparison.traces[mode])
        self.assertEqual(len(clips), 10)
        for encoded, notes in zip(clips, expected):
            self.assertEqual(base64.b64decode(encoded), note_audio(notes))
        self.assertIn("BOTH h and c", html)
        self.assertNotRegex(html, r"<audio\b[^>]*\bautoplay")
        with self.assertRaisesRegex(ValueError, "LSTM or GRU"):
            run_echo_experiment(kind="attention")

    def test_notebook_teaches_complete_lstm_before_gru_and_keeps_attention_a_preview(self):
        sources = {cell["id"]: "".join(cell["source"]) for cell in self.book["cells"]}
        order = list(sources)
        self.assertLess(order.index("lstm-echo-training"), order.index("gru-mechanics-intro"))
        self.assertLess(order.index("lstm-real-cell-parity"), order.index("lstm-echo-training"))
        self.assertIn("echo_audio_panel(lstm_echo_comparison)", sources["lstm-echo-training"])
        self.assertIn("lstm_steps = 480", sources["lstm-echo-training"])
        boundary = sources["memory-wiring-attention"]
        preview = boundary.split("### Attention preview:", 1)[1]
        self.assertLessEqual(len(preview.split()), 140)
        for phrase in ("Equal access does **not** mean equal weight", "without attention",
                       "features inside memory", "source positions", "order", "Both cell types"):
            self.assertIn(phrase.lower(), boundary.lower())
        self.assertIn("not six stacked layers", sources["lstm-native-reading"])
        self.assertIn("same current state pair", sources["lstm-native-reading"])

    def test_saved_native_layer_and_music_captures_match_actual_sources_and_notes(self):
        prefix = HERE / "images" / "01a-lstm-token-layer-flow"
        with Image.open(prefix.with_suffix(".png")) as image:
            self.assertEqual(image.info["SourceFingerprint"], source_fingerprint(self.book))
            self.assertEqual(image.info["Output"], image.info["CarriedHidden"])
            self.assertNotEqual(image.info["Output"], image.info["CarriedCell"])
        with Image.open(prefix.with_suffix(".gif")) as image:
            self.assertEqual(image.n_frames, len(LSTM_PHASES) * MOTION_FRAMES)
        html = prefix.with_suffix(".html").read_text(encoding="utf-8")
        self.assertNotIn("https:", html)
        self.assertIn(">Next frame</button>", html)
        prefix = HERE / "images" / "01a-musical-lstm-memory"
        frames = music_frames(self.comparison)
        with Image.open(prefix.with_suffix(".gif")) as image:
            metadata = json.loads(image.info["comment"])
            self.assertEqual(metadata["SourceFingerprint"], music_fingerprint())
            self.assertEqual(metadata["TrainingSteps"], 480)
            self.assertEqual(metadata["ErasedStates"], ["h", "c"])
            self.assertEqual(image.n_frames, len(frames))
            for mode, traces in self.comparison.traces.items():
                self.assertEqual(metadata["Predictions"][mode], [list(trace.generated) for trace in traces])
        self.assertEqual(prefix.with_suffix(".wav").read_bytes(), wav_bytes(synthesize_timeline(frames)))
        native_cell = next(cell for cell in self.book["cells"] if cell["id"] == "lstm-native-intro")
        data = base64.b64decode(native_cell["attachments"]["01a-lstm-token-layer-flow.png"]["image/png"])
        with Image.open(BytesIO(data)) as image:
            self.assertGreater(image.width, 500)


if __name__ == "__main__":
    unittest.main()
