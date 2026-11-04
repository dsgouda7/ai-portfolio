"""Check rendered spacing and motion, not merely that image files exist."""

from io import BytesIO
from itertools import combinations
import unittest

import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.transforms import Bbox
import torch
from torch import nn

from gru_layer_flow import record_layer_step
from lstm_layer_flow import record_lstm_layer_step
from memory_visuals import draw_scene, gru_scene, lstm_scene


class MemoryVisualTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(42)
            x = torch.tensor([[0., 1.]], dtype=torch.float64)
            old = torch.tensor([[0.4, -0.2]], dtype=torch.float64)
            cls.gru = record_layer_step(nn.GRUCell(2, 2).double(), x, old)
            cls.lstm = record_lstm_layer_step(nn.LSTMCell(2, 2).double(), x, old, old + 0.3)

    def test_adapters_preserve_actual_states_controls_and_projection_values(self):
        lstm = lstm_scene(self.lstm)
        gru = gru_scene(self.gru)
        self.assertEqual(lstm.hidden, tuple(self.lstm.output.tolist()))
        self.assertEqual(lstm.cell, tuple(self.lstm.carried_cell.tolist()))
        self.assertEqual(lstm.keep, tuple(self.lstm.forget_gate.tolist()))
        self.assertEqual(lstm.input_mix, tuple(self.lstm.input_mix.tolist()))
        self.assertEqual(gru.hidden, tuple(self.gru.output.tolist()))
        self.assertEqual(gru.reset, tuple(self.gru.reset_gate.tolist()))
        self.assertEqual(gru.write, tuple(1 - value for value in self.gru.update_gate.tolist()))

    def test_every_phase_has_readable_contained_text_and_no_text_overlap(self):
        figure, axis = plt.subplots(figsize=(12.8, 8), dpi=100)
        figure.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.025)
        try:
            for scene in (lstm_scene(self.lstm), gru_scene(self.gru)):
                for phase in (None, *range(7)):
                    with self.subTest(kind=scene.kind, phase=phase):
                        draw_scene(axis, scene, phase, native=True, progress=0.5)
                        figure.canvas.draw()
                        renderer = figure.canvas.get_renderer()
                        boxes = {patch.get_gid(): patch.get_window_extent(renderer)
                                 for patch in axis.patches if (patch.get_gid() or "").startswith("card-")}
                        content = [text for text in axis.texts if ":" in (text.get_gid() or "")]
                        for text in content:
                            bounds = text.get_window_extent(renderer)
                            box = boxes[text.get_gid().split(":")[0]]
                            self.assertTrue(box.contains(bounds.x0, bounds.y0), text.get_text())
                            self.assertTrue(box.contains(bounds.x1, bounds.y1), text.get_text())
                            css_size = text.get_fontsize() * 100 / 72 * 640 / 1280
                            self.assertGreaterEqual(css_size, 12)
                        for first, second in combinations(content, 2):
                            intersection = Bbox.intersection(
                                first.get_window_extent(renderer), second.get_window_extent(renderer),
                            )
                            self.assertTrue(intersection is None or intersection.width <= 0 or intersection.height <= 0,
                                            f"{first.get_text()} overlaps {second.get_text()}")
                        connectors = [patch for patch in axis.patches if patch.get_gid() == "connector"]
                        self.assertLessEqual(len(connectors), 2)
                        for connector in connectors:
                            path = connector.get_path().transformed(connector.get_transform())
                            vertices = path.vertices.copy()
                            start = vertices[0]
                            for index, code in enumerate(path.codes):
                                if code == Path.MOVETO:
                                    start = vertices[index]
                                elif code == Path.CLOSEPOLY:
                                    # CLOSEPOLY's dummy coordinate is not part of the rendered stroke.
                                    vertices[index] = start
                            path = Path(vertices, path.codes)
                            for text in content:
                                self.assertFalse(path.intersects_bbox(text.get_window_extent(renderer), filled=False),
                                                 f"Connector crosses {text.get_text()}")
        finally:
            plt.close(figure)

    def test_motion_changes_the_picture_without_interpolating_recorded_numbers(self):
        scene = lstm_scene(self.lstm)
        original = repr(scene)
        figure, axis = plt.subplots(figsize=(12.8, 8), dpi=100)
        try:
            pictures, labels = [], []
            for progress in (0.0, 1.0):
                draw_scene(axis, scene, 4, native=True, progress=progress)
                labels.append([text.get_text() for text in axis.texts])
                buffer = BytesIO()
                figure.savefig(buffer, format="png")
                pictures.append(buffer.getvalue())
            self.assertEqual(labels[0], labels[1])
            self.assertNotEqual(pictures[0], pictures[1])
            self.assertEqual(repr(scene), original)
        finally:
            plt.close(figure)


if __name__ == "__main__":
    unittest.main()
