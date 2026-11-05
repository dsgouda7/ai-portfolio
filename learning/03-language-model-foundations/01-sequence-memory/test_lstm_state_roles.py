"""Verify that c supplies contents, h supplies control context, and SHOW is not erasure."""

from copy import deepcopy
import base64
import json
import unittest

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.patches import FancyArrowPatch
from matplotlib.path import Path as PlotPath
from PIL import Image
import torch
from torch import nn

from lstm_state_roles import (
    CONTROLS, HERE, NOTEBOOK, draw_cell_memory, draw_state_roles, draw_two_state_step, load_state_roles,
    record_state_roles, source_fingerprint, trace_cell_memory, _figure as role_figure,
)
from sequence_studio import BACKGROUND, CYAN, GOLD


class LSTMStateRoleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.probe = load_state_roles()

    def test_closed_show_keeps_the_same_current_cell_but_exposes_zero_hidden(self):
        original, closed = self.probe.baseline, self.probe.show_closed
        torch.testing.assert_close(original.new_cell, closed.new_cell, rtol=0, atol=0)
        self.assertGreater(float(closed.new_cell.abs().max()), 0)
        self.assertEqual(torch.count_nonzero(closed.new_hidden), 0)
        torch.testing.assert_close(closed.output, closed.carried_hidden, rtol=0, atol=0)

    def test_memory_number_retains_the_opening_when_later_writes_are_closed(self):
        first, reversed_opening = trace_cell_memory("D3"), trace_cell_memory("D4")
        self.assertEqual(first[-1].note, reversed_opening[-1].note)
        self.assertEqual(first[-1].note, "A3")
        self.assertEqual([step.value for step in first], [0.8] * 4)
        self.assertEqual([step.value for step in reversed_opening], [-0.8] * 4)
        for step in first[1:]:
            self.assertEqual((step.keep, step.write), (1.0, 0.0))
            self.assertEqual(step.previous, step.value)

    def test_overwriting_the_coordinate_removes_its_opening_distinction(self):
        first = trace_cell_memory("D3", overwrite_last=True)
        other = trace_cell_memory("D4", overwrite_last=True)
        self.assertEqual(first[-1].value, 0.2)
        self.assertEqual(first[-1].value, other[-1].value)
        self.assertEqual((first[-1].keep, first[-1].write), (0.0, 1.0))
        with self.assertRaisesRegex(ValueError, "D3 or D4"):
            trace_cell_memory("unknown")

    def test_memory_figure_keeps_numbers_and_note_inputs_in_the_studio_style(self):
        figure, axis = role_figure()
        try:
            draw_cell_memory(axis)
            texts = " ".join(text.get_text() for text in axis.texts)
            self.assertIn("numeric workspace", texts)
            self.assertIn("[+0.80]", texts)
            self.assertIn("[-0.80]", texts)
            self.assertIn("ONE current coordinate", texts)
            self.assertFalse(axis.images)
            self.assertEqual(figure.get_facecolor(), to_rgba(BACKGROUND))
            for opening in ("D3", "D4"):
                for index, step in enumerate(trace_cell_memory(opening)):
                    text = next(text for text in axis.texts
                                if text.get_gid() == f"memory-input-{opening}-{index}")
                    self.assertEqual(text.get_text(), f"read {step.note}")
                    self.assertEqual(text.get_color(), GOLD)
        finally:
            plt.close(figure)

    def test_labels_are_not_crossed_by_arrow_paths(self):
        for draw in (draw_cell_memory, lambda axis: draw_two_state_step(axis, self.probe),
                     lambda axis: draw_state_roles(axis, self.probe)):
            figure, axis = role_figure()
            try:
                draw(axis)
                figure.canvas.draw()
                for text in axis.texts:
                    if not text.get_text():
                        continue
                    bounds = text.get_window_extent(figure.canvas.get_renderer())
                    for arrow in axis.patches:
                        if isinstance(arrow, FancyArrowPatch):
                            path = arrow.get_transform().transform_path(arrow.get_path())
                            if path.get_extents().overlaps(bounds):
                                self.assertFalse(path.intersects_bbox(bounds, filled=False), text.get_text())
            finally:
                plt.close(figure)

    def test_state_routes_keep_the_rnn_palette_and_separate_the_prediction_head(self):
        figure, axis = role_figure()
        try:
            draw_two_state_step(axis, self.probe)
            routes = {patch.get_gid(): patch for patch in axis.patches
                      if isinstance(patch, FancyArrowPatch)}
            self.assertEqual(routes["carry-c2"].get_edgecolor(), to_rgba(GOLD))
            self.assertEqual(routes["carry-h2"].get_edgecolor(), to_rgba(CYAN))
            self.assertIn("expose-h2", routes)
            self.assertTrue(any(patch.get_gid() == "prediction-head" for patch in axis.patches))
        finally:
            plt.close(figure)

    def test_figure_labels_are_contained_and_do_not_overlap(self):
        for draw in (draw_cell_memory, lambda axis: draw_two_state_step(axis, self.probe),
                     lambda axis: draw_state_roles(axis, self.probe)):
            figure, axis = role_figure()
            try:
                draw(axis)
                figure.canvas.draw()
                renderer = figure.canvas.get_renderer()
                labels = [text for text in axis.texts if text.get_text()]
                for index, text in enumerate(labels):
                    bounds = text.get_window_extent(renderer)
                    self.assertTrue(axis.bbox.contains(bounds.x0, bounds.y0), text.get_text())
                    self.assertTrue(axis.bbox.contains(bounds.x1, bounds.y1), text.get_text())
                    for other in labels[index + 1:]:
                        self.assertFalse(bounds.overlaps(other.get_window_extent(renderer)),
                                         (text.get_text(), other.get_text()))
            finally:
                plt.close(figure)

    def test_every_connector_has_a_visible_arrowhead_including_vertical_show_routes(self):
        for draw in (draw_cell_memory, lambda axis: draw_two_state_step(axis, self.probe),
                     lambda axis: draw_state_roles(axis, self.probe)):
            figure, axis = role_figure()
            try:
                draw(axis)
                figure.canvas.draw()
                for arrow in axis.patches:
                    if not isinstance(arrow, FancyArrowPatch):
                        continue
                    path = arrow.get_transform().transform_path(arrow.get_path())
                    start = max(index for index, code in enumerate(path.codes) if code == PlotPath.MOVETO)
                    head = path.vertices[start:]
                    area = abs(sum(a[0] * b[1] - b[0] * a[1]
                                   for a, b in zip(head, list(head[1:]) + [head[0]]))) / 2
                    self.assertGreater(area, 8, arrow.get_gid())
            finally:
                plt.close(figure)

    def test_old_cell_changes_contents_without_changing_the_controls(self):
        original, erased = self.probe.baseline, self.probe.no_old_cell
        for field in CONTROLS:
            torch.testing.assert_close(getattr(original, field), getattr(erased, field), rtol=0, atol=0)
        self.assertFalse(torch.equal(original.new_cell, erased.new_cell))
        self.assertFalse(torch.equal(original.new_hidden, erased.new_hidden))

    def test_old_hidden_changes_the_controls_while_old_cell_is_held_fixed(self):
        original, erased = self.probe.baseline, self.probe.no_old_hidden
        torch.testing.assert_close(original.old_cell, erased.old_cell, rtol=0, atol=0)
        self.assertFalse(torch.equal(original.state_mix, erased.state_mix))
        self.assertTrue(any(not torch.equal(getattr(original, field), getattr(erased, field))
                            for field in CONTROLS))

    def test_probe_preserves_live_weights_mode_and_rng(self):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(42)
            cell = nn.LSTMCell(2, 2).double()
        weights = deepcopy(cell.state_dict())
        rng = torch.random.get_rng_state().clone()
        x = torch.tensor([[0.0, 1.0]], dtype=torch.float64)
        record_state_roles(cell, x, torch.ones_like(x), torch.ones_like(x))
        self.assertTrue(cell.training)
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        for name, value in cell.state_dict().items():
            torch.testing.assert_close(value, weights[name], rtol=0, atol=0)

    def test_role_chart_shows_both_vectors_and_both_carried_states(self):
        figure, axis = role_figure()
        try:
            draw_state_roles(axis, self.probe)
            texts = " ".join(text.get_text() for text in axis.texts)
            for label in ("c2: memory contents", "h2: exposed readout", "BOTH (h2, c2)", "SHOW closed"):
                self.assertIn(label, texts)
            figure.canvas.draw()
        finally:
            plt.close(figure)

    def test_saved_role_trace_matches_the_actual_state_vectors(self):
        with Image.open(HERE / "images" / "01a-lstm-state-roles-trace.png") as image:
            self.assertEqual(image.info["SourceFingerprint"], source_fingerprint())
            self.assertEqual(image.info["OriginalCell"], image.info["ClosedCell"])
            self.assertEqual(json.loads(image.info["ClosedHidden"]), [0.0, 0.0])

    def test_role_illustrations_and_measured_trace_are_embedded_in_the_notebook(self):
        book = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        for filename in (
            "01a-lstm-cell-vs-hidden.png", "01a-lstm-two-state-step.png",
            "01a-lstm-state-roles-trace.png",
        ):
            cell = next(cell for cell in book["cells"]
                        if f"attachment:{filename}" in "".join(cell.get("source", [])))
            encoded = cell["attachments"][filename]["image/png"]
            self.assertEqual(base64.b64decode(encoded), (HERE / "images" / filename).read_bytes())

    def test_numeric_memory_is_introduced_before_the_hidden_readout_comparison(self):
        book = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        order = [cell["id"] for cell in book["cells"]]
        self.assertLess(order.index("133e7838"), order.index("lstm-cell-memory-demo"))
        self.assertLess(order.index("lstm-cell-memory-demo"), order.index("lstm-readout-after-memory"))
        intro = next(cell for cell in book["cells"] if cell["id"] == "133e7838")
        self.assertIn("fixed-length row of memory numbers", "".join(intro["source"]))
        self.assertIn("not another set of trained weights", "".join(intro["source"]))


if __name__ == "__main__":
    unittest.main()
