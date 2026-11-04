"""Check actual saved-state feedback, shared-weight accumulation, and the stop boundary."""

import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import unittest

import matplotlib.pyplot as plt
from PIL import Image, ImageSequence
import torch

from backprop_memory_flow import (
    FRAMES_PER_STAGE, HERE, INTERVAL_MS, NOTEBOOK, PREFIX, STAGES, draw_stage,
    record_trace, refresh_notebook_captures, source_fingerprint,
)
from generate_memory_diagrams import canvas
from notebook_replays import NOTEBOOK_SIZE_LIMIT


class BackpropMemoryFlowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.full = record_trace()
        cls.truncated = record_trace(truncated=True)

    def test_full_backward_reaches_every_saved_card_and_accumulates_shared_weights(self):
        self.assertEqual(self.full.guess, "A3")
        self.assertEqual(len(self.full.states), 4)
        self.assertTrue(all(signal is not None for signal in self.full.feedback))
        torch.testing.assert_close(
            torch.stack(self.full.memory_contributions).sum(0), self.full.memory_gradient,
            rtol=0, atol=1e-12,
        )
        self.assertLess(self.full.loss_after, self.full.loss_before)

    def test_detach_preserves_every_card_and_guess_but_stops_opening_feedback(self):
        for full, truncated in zip(self.full.states, self.truncated.states):
            torch.testing.assert_close(full, truncated, rtol=0, atol=0)
        self.assertEqual(self.full.guess, self.truncated.guess)
        self.assertEqual(self.full.loss_before, self.truncated.loss_before)
        self.assertIsNone(self.truncated.feedback[0])
        self.assertTrue(all(signal is not None for signal in self.truncated.feedback[1:]))
        torch.testing.assert_close(
            self.truncated.memory_contributions[0], torch.zeros(2, 2, dtype=torch.float64),
            rtol=0, atol=0,
        )

    def test_all_stages_render_with_the_requested_inspection_order(self):
        self.assertIn("h3", STAGES[4])
        self.assertIn("h2", STAGES[5])
        self.assertIn("h1", STAGES[6])
        self.assertIn("ONCE", STAGES[7])
        figure, axis = canvas("", "")
        try:
            for stage in range(len(STAGES)):
                draw_stage(axis, self.full, self.truncated, stage, 0.5)
                figure.canvas.draw()
                for text in axis.texts:
                    bounds = text.get_window_extent(figure.canvas.get_renderer())
                    self.assertGreaterEqual(bounds.x0, 0)
                    self.assertLessEqual(bounds.x1, figure.bbox.width)
            draw_stage(axis, self.full, self.truncated, 4)
            self.assertEqual(sum("feedback size:" in text.get_text() for text in axis.texts), 1)
            draw_stage(axis, self.full, self.truncated, 6)
            self.assertEqual(sum("feedback size:" in text.get_text() for text in axis.texts), 3)
        finally:
            plt.close(figure)
        with self.assertRaises(ValueError):
            draw_stage(axis, self.full, self.truncated, len(STAGES))

    def test_saved_captures_match_the_actual_trace_and_timing(self):
        with Image.open(HERE / "images" / (PREFIX + "-trace.png")) as image:
            self.assertEqual(image.info["SourceFingerprint"], source_fingerprint())
        with Image.open(HERE / "images" / (PREFIX + ".gif")) as image:
            metadata = json.loads(image.info["comment"])
            duration = sum(frame.info["duration"] for frame in ImageSequence.Iterator(image))
        self.assertEqual(metadata["LossBefore"], self.full.loss_before)
        self.assertEqual(metadata["LossAfter"], self.full.loss_after)
        self.assertEqual(metadata["Stages"], list(STAGES))
        self.assertEqual(duration, len(STAGES) * FRAMES_PER_STAGE * INTERVAL_MS)

    def test_capture_refresh_preserves_sources_and_is_idempotent(self):
        with TemporaryDirectory() as directory:
            copy = Path(directory) / NOTEBOOK.name
            shutil.copyfile(NOTEBOOK, copy)
            before = copy.read_bytes()
            refresh_notebook_captures(copy, HERE / "images")
            self.assertEqual(copy.read_bytes(), before)

    def test_new_player_is_persisted_without_changing_existing_outputs(self):
        book = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        player = next(cell for cell in book["cells"] if cell["id"] == "backprop-saved-card-replay")
        self.assertEqual(len(player["outputs"]), 1)
        html = player["outputs"][0]["data"]["text/html"]
        self.assertIn(f'data-saved-replay="{PREFIX}.html"', html)
        self.assertIn('srcdoc="', html)
        self.assertLess(NOTEBOOK.stat().st_size, NOTEBOOK_SIZE_LIMIT)


if __name__ == "__main__":
    unittest.main()
