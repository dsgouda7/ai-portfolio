"""Verify that the token-layer diagrams show native GRU computations and output semantics."""

from contextlib import redirect_stdout
from dataclasses import asdict
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import matplotlib.pyplot as plt
from PIL import Image, ImageChops, ImageSequence
import torch
from torch import nn

from gru_layer_flow import LAYER_PHASES, draw_layer, record_layer_step
from memory_gate_flows import fingerprint, recorded_foundation_layer
from memory_visuals import MOTION_FRAMES
import gru_forward_flow as forward
from gated_sequence_storyboards import record_story
from lstm_forward_flow import GIF as LSTM_GIF, export_gif as export_lstm_gif


class GRULayerFlowTests(unittest.TestCase):
    def setUp(self):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(42)
            self.cell = nn.GRUCell(2, 2).double()
        self.inputs = torch.tensor([[0., 1.]], dtype=torch.float64)
        self.old = torch.tensor([[0.3, -0.2]], dtype=torch.float64)

    def test_projections_gates_and_state_match_the_native_cell(self):
        step = record_layer_step(self.cell, self.inputs, self.old)
        self.assertEqual(step.input_mix.shape, (6,))
        self.assertEqual(step.state_mix.shape, (6,))
        torch.testing.assert_close(
            step.input_mix, nn.functional.linear(self.inputs, self.cell.weight_ih, self.cell.bias_ih).reshape(-1),
        )
        torch.testing.assert_close(step.retained + step.written, self.cell(self.inputs, self.old).reshape(-1))
        torch.testing.assert_close(step.output, step.carried, rtol=0, atol=0)

    def test_gru_sequence_output_and_terminal_hidden_are_different_axes_not_different_values(self):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(42)
            gru = nn.GRU(2, 3, batch_first=True).double()
        step = record_layer_step(gru, self.inputs, torch.zeros((1, 3), dtype=torch.float64))
        output, hidden = gru(self.inputs[:, None], torch.zeros((1, 1, 3), dtype=torch.float64))
        self.assertEqual(output.shape, (1, 1, 3))
        self.assertEqual(hidden.shape, (1, 1, 3))
        torch.testing.assert_close(step.output, output[0, 0], rtol=0, atol=0)
        torch.testing.assert_close(step.carried, hidden[0, 0], rtol=0, atol=0)
        torch.testing.assert_close(step.output, step.carried, rtol=0, atol=0)

    def test_stacked_layers_have_per_layer_hidden_states_and_are_not_claimed_by_this_view(self):
        with torch.random.fork_rng(devices=[]):
            gru = nn.GRU(2, 3, num_layers=2, batch_first=True).double()
        output, hidden = gru(self.inputs[:, None])
        self.assertEqual(hidden.shape, (2, 1, 3))
        torch.testing.assert_close(output[0, 0], hidden[-1, 0], rtol=0, atol=0)
        with self.assertRaisesRegex(ValueError, "not a stack"):
            record_layer_step(gru, self.inputs, torch.zeros((1, 3), dtype=torch.float64))

    def test_read_only_recording_preserves_weights_gradients_mode_and_rng(self):
        self.cell.train()
        for parameter in self.cell.parameters():
            parameter.grad = torch.ones_like(parameter)
        before = {name: parameter.clone() for name, parameter in self.cell.named_parameters()}
        state = torch.random.get_rng_state().clone()
        record_layer_step(self.cell, self.inputs, self.old)
        for name, parameter in self.cell.named_parameters():
            torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)
            torch.testing.assert_close(parameter.grad, torch.ones_like(parameter), rtol=0, atol=0)
        self.assertTrue(self.cell.training)
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))

    def test_drawing_exposes_both_linear_projections_and_both_output_routes(self):
        step = record_layer_step(self.cell, self.inputs, self.old)
        figure, axis = plt.subplots(figsize=(12, 10))
        try:
            draw_layer(axis, step, 6, input_label="A3", old_label="after D3",
                       new_label="h_t", output_use="to caller")
            text = "\n".join(item.get_text() for item in axis.texts)
            for term in ("Linear: 2 -> 6", "sigmoid", "tanh", "kept", "written",
                         "output_t [2]", "hidden for next token [2]", "same h_t values"):
                self.assertIn(term, text)
        finally:
            plt.close(figure)

    def test_saved_foundation_focuses_one_real_note_and_preserves_output_state_equality(self):
        here = Path(__file__).parent
        notebook = json.loads((here / "01A1-sequence-memory-worked-details.ipynb").read_text(encoding="utf-8"))
        with redirect_stdout(StringIO()):
            step = recorded_foundation_layer(notebook)
        prefix = here / "images" / "01a-gru-token-layer-flow"
        with Image.open(prefix.with_suffix(".png")) as image:
            self.assertEqual(image.info["SourceFingerprint"], fingerprint(notebook))
            self.assertEqual(image.info["Input"], "A3")
            self.assertEqual(image.info["OutputEqualsCarried"], "True")
            torch.testing.assert_close(torch.tensor(json.loads(image.info["Output"])), step.output.float())
            self.assertEqual(image.info["Output"], image.info["Carried"])
        with Image.open(prefix.with_suffix(".gif")) as image:
            self.assertEqual(image.n_frames, len(LAYER_PHASES) * MOTION_FRAMES)
        html = prefix.with_suffix(".html").read_text(encoding="utf-8")
        self.assertNotIn("https:", html)
        self.assertIn(">Next frame</button>", html)


class GRUForwardFlowTests(unittest.TestCase):
    def native_cell(self):
        story = record_story("gru")
        with torch.random.fork_rng(devices=[]):
            cell = nn.GRUCell(2, 2).double()
        cell.load_state_dict({name: torch.tensor(value, dtype=torch.float64)
                              for name, value in story["cell_parameters"].items()})
        return cell, story

    def test_recorded_reads_match_native_state_head_and_complementary_controls(self):
        cell, story = self.native_cell()
        rng = torch.random.get_rng_state().clone()
        reads = forward.record_reads()
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        for read in reads:
            actual = cell(torch.tensor(read.inputs, dtype=torch.float64),
                          torch.tensor(read.old_hidden, dtype=torch.float64)).detach()
            torch.testing.assert_close(actual, torch.tensor(read.new_hidden, dtype=torch.float64),
                                       rtol=0, atol=1e-14)
            for keep, write, old, proposal, kept, written, new in zip(
                read.keep, read.write, read.old_hidden, read.proposal, read.retained,
                read.written, read.new_hidden, strict=True,
            ):
                self.assertEqual(write, 1 - keep)
                self.assertAlmostEqual(kept, keep * old, places=14)
                self.assertAlmostEqual(written, write * proposal, places=14)
                self.assertAlmostEqual(new, kept + written, places=14)
            head = story["head_parameters"]
            scores = nn.functional.linear(actual, torch.tensor(head["weight"], dtype=torch.float64),
                                           torch.tensor(head["bias"], dtype=torch.float64))
            self.assertEqual(tuple(scores.tolist()), read.readout.scores)
        self.assertEqual(reads[2].old_hidden, reads[1].new_hidden)

    def test_reset_zero_changes_only_proposal_not_the_separately_retained_old_amount(self):
        cell, _ = self.native_cell()
        with torch.no_grad():
            cell.weight_ih[:2].zero_()
            cell.weight_hh[:2].zero_()
            cell.bias_ih[:2].fill_(-1000.)
            cell.bias_hh[:2].zero_()
        normal, closed = forward.record_reads(), forward.record_reads(close_reset=True)
        for old, changed in zip(normal, closed, strict=True):
            self.assertEqual(changed.keep, old.keep)
            self.assertEqual(changed.write, old.write)
            self.assertEqual(changed.retained, old.retained)
            self.assertEqual(changed.old_hidden, old.old_hidden)
            self.assertEqual(changed.reset_context, (0., 0.))
            actual = cell(torch.tensor(changed.inputs, dtype=torch.float64),
                          torch.tensor(changed.old_hidden, dtype=torch.float64)).detach()
            torch.testing.assert_close(actual, torch.tensor(changed.new_hidden, dtype=torch.float64),
                                       rtol=0, atol=1e-14)
            self.assertNotEqual(changed.proposal, old.proposal)
        self.assertNotEqual(closed[1].new_hidden, (0., 0.))

    def test_spaced_diagram_routes_one_state_and_keeps_visited_paths_bright(self):
        reads = forward.record_reads()
        boxes, wires, _ = forward.scene(reads[1])
        incoming = {}
        for wire in wires:
            incoming.setdefault(wire.target, set()).add(wire.source)
            for start, end in zip(wire.points, wire.points[1:]):
                for box in boxes:
                    if box.name in (wire.source, wire.target):
                        continue
                    x, y, width, height = box.bounds
                    if start[1] == end[1]:
                        blocked = y - 10 < start[1] < y + height + 10 and (
                            max(min(start[0], end[0]), x - 10) < min(max(start[0], end[0]), x + width + 10))
                    else:
                        self.assertEqual(start[0], end[0])
                        blocked = x - 10 < start[0] < x + width + 10 and (
                            max(min(start[1], end[1]), y - 10) < min(max(start[1], end[1]), y + height + 10))
                    self.assertFalse(blocked, (wire.name, box.name))
        self.assertEqual(incoming["kept"], {"old-hidden", "controls"})
        self.assertEqual(incoming["proposal"], {"input", "reset"})
        self.assertEqual(incoming["head"], {"new-hidden"})
        self.assertEqual(incoming["prediction"], {"head"})
        self.assertEqual(incoming["next-hidden"], {"new-hidden"})
        self.assertFalse(any("cell" in box.name or "show" in box.name for box in boxes))
        bright = set()
        for frame in forward.gif_frames():
            if frame.progress != 1.:
                continue
            read = forward.record_reads(close_reset=frame.close_reset)[frame.read_index - 1]
            figure = forward.gif_figure(read, frame)
            try:
                axis = figure.axes[0]
                artists = list(axis.patches) + list(axis.lines)
                artists += [text.arrow_patch for text in axis.texts
                            if getattr(text, "arrow_patch", None) is not None]
                styles = {artist.get_gid(): artist.get_alpha() for artist in artists
                          if artist.get_gid() and artist.get_gid().startswith(("node-", "wire-"))}
                self.assertTrue(all(styles[name] == 1 for name in bright))
                bright.update(name for name, alpha in styles.items() if alpha == 1)
                figure.canvas.draw()
                renderer = figure.canvas.get_renderer()
                labels = [text for text in axis.texts if text.get_text()]
                for index, text in enumerate(labels):
                    bounds = text.get_window_extent(renderer)
                    self.assertTrue(figure.bbox.contains(bounds.x0, bounds.y0), text.get_text())
                    self.assertTrue(figure.bbox.contains(bounds.x1, bounds.y1), text.get_text())
                    for other in labels[index + 1:]:
                        self.assertFalse(bounds.overlaps(other.get_window_extent(renderer)),
                                         (text.get_text(), other.get_text()))
            finally:
                plt.close(figure)

    def test_linear_head_shows_real_biased_scores_before_softmax_choice(self):
        read = forward.record_reads()[1]
        for progress in (.25, .5, 1.):
            frame = forward.GRUFrame(2, 7, progress, 120)
            figure = forward.gif_figure(read, frame)
            try:
                texts = [text.get_text() for text in figure.axes[0].texts]
                self.assertEqual(forward.vector(read.readout.scores) in texts, progress >= .5)
                self.assertEqual("scores = W_out h2 + b" in texts, progress >= .5)
                self.assertEqual(f"Next note: {read.readout.prediction} (max)" in texts, progress == 1.)
            finally:
                plt.close(figure)
        self.assertEqual({name for name, _ in forward.flow_signals(7, .25)}, {"hidden-to-head"})
        self.assertEqual({name for name, _ in forward.flow_signals(7, .75)}, {"head-to-prediction"})

    def test_gif_has_native_data_visible_motion_and_static_fallback(self):
        with Image.open(forward.GIF) as image:
            metadata = json.loads(image.info["comment"])
            self.assertEqual(metadata["state_count"], 1)
            self.assertEqual(metadata["head_input"], "new_hidden")
            self.assertEqual(image.n_frames, len(forward.gif_frames()))
            self.assertLessEqual(image.n_frames, 64)
            self.assertEqual(metadata["steps"], json.loads(json.dumps([
                asdict(read) for read in forward.record_reads()
            ])))
            duration = sum(frame.info["duration"] for frame in ImageSequence.Iterator(image))
            self.assertGreaterEqual(duration, 20_000)
            self.assertLessEqual(duration, 35_000)
            image.seek(0)
            start = image.convert("RGB")
            image.seek(3)
            self.assertIsNotNone(ImageChops.difference(start, image.convert("RGB")).getbbox())
        self.assertLess(forward.GIF.stat().st_size, 4_000_000)
        with Image.open(forward.STORYBOARD) as image:
            self.assertEqual(image.info["StateCount"], "1")
            self.assertEqual(image.info["HeadInput"], "new_hidden")

    def test_shared_renderer_preserves_every_existing_lstm_gif_frame(self):
        with TemporaryDirectory() as directory:
            rebuilt = Path(directory) / LSTM_GIF.name
            export_lstm_gif(rebuilt)
            with Image.open(LSTM_GIF) as original, Image.open(rebuilt) as current:
                self.assertEqual(current.n_frames, original.n_frames)
                self.assertEqual(current.info["comment"], original.info["comment"])
                for old, new in zip(ImageSequence.Iterator(original), ImageSequence.Iterator(current), strict=True):
                    self.assertEqual(new.convert("RGB").tobytes(), old.convert("RGB").tobytes())


if __name__ == "__main__":
    unittest.main()
