"""Verify the reference-inspired sheet without trusting generated mathematical text."""

import base64
import importlib.util
import json
import math
from pathlib import Path
import re
from tempfile import TemporaryDirectory
import unittest

import matplotlib.pyplot as plt
from matplotlib import font_manager
from PIL import Image
import torch

import bptt_math
from bptt_summary import (
    HERE, MANIFEST, PAPER, SHEET, SIZE, TAKEAWAY, equation_rows, export,
    sheet_figure, source_fingerprint, worked_example,
)
from error_feedback_flow import record_probe


class BpttSummaryTests(unittest.TestCase):
    def test_linear_expansion_counts_the_powers_and_all_shared_uses(self):
        for memory in (0.8, 1.2):
            for initial in (0.0, 0.2):
                for length in (3, 6):
                    weight = torch.tensor(memory, dtype=torch.float64, requires_grad=True)
                    states = [torch.tensor(initial, dtype=torch.float64)]
                    for note_write in worked_example(memory).note_writes * (length // 3):
                        states.append(note_write + weight * states[-1])
                    derivative = torch.autograd.grad(states[-1], weight)[0]
                    expanded = sum(weight ** (length - 1 - index) * states[index] for index in range(length))
                    torch.testing.assert_close(derivative, expanded, rtol=1e-14, atol=1e-15)
        self.assertIn("m^{T-1}h_0", dict(equation_rows())["Expand to the fixed start"])

    def test_sheet_keeps_the_full_derivation_instead_of_hiding_it_in_shorthand(self):
        equations = dict(equation_rows())
        for key in ("Differentiate the error", "Expand prediction through memory",
                    "Follow the whole route", "Product rule", "One read earlier",
                    "Another read earlier", "Substitute", "Distribute", "Expand to the fixed start"):
            self.assertIn(key, equations)
        self.assertIn(r"\frac{dp}{dh_T}", equations["Follow the whole route"])
        self.assertIn(r"\frac{dh_{t-2}}{dm}", equations["One read earlier"])
        self.assertIn(r"+m^2\left[", equations["Distribute"])
        self.assertNotIn("S_t", str(equation_rows()))

    def test_compact_values_use_original_notes_and_exact_tanh_gradients(self):
        for memory, linear_response in ((0.8, 0.0011), (1.2, 0.0019)):
            example = worked_example(memory)
            probe = record_probe(gain=memory)
            self.assertEqual(example.note_writes, (0.001, -0.0005, 0.0005))
            self.assertAlmostEqual(example.linear_response, linear_response, places=14)
            self.assertAlmostEqual(example.tanh_response * example.ending_feedback,
                                   probe.weight_gradients.memory, places=14)
            self.assertEqual(example.recurrent_gradient, probe.weight_gradients.memory)
            self.assertEqual(example.ending_probability, probe.probabilities[0])
            self.assertAlmostEqual(-example.probability_response / example.ending_probability,
                                   example.ending_feedback, places=14)
            longer = record_probe(gain=memory, repeats=7)
            product = math.prod(step.factor for step in longer.steps[1:])
            self.assertAlmostEqual(example.long_feedback_ratio, product, places=12)
            self.assertLess(example.long_feedback_ratio, memory**20)

    def test_generated_art_and_exact_overlay_have_honest_provenance(self):
        entry = json.loads(MANIFEST.read_bytes())[0]
        self.assertEqual(entry["model"], "MAI-Image-2.6")
        for key in ("semantic_anchor", "aspect_ratio", "prompt", "acceptance_criteria",
                    "prohibited_content", "alt_text", "postprocessing"):
            self.assertTrue(entry[key])
        with Image.open(PAPER) as art:
            self.assertEqual(art.size, (entry["width"], entry["height"]))
        with Image.open(SHEET) as image:
            self.assertEqual(image.size, SIZE)
            self.assertEqual(image.mode, "P")
            self.assertEqual(image.info["SourceFingerprint"], source_fingerprint())
            self.assertIn("verified local", image.info["Generator"])
            self.assertIn("dark navy", image.info["Theme"])
            histogram = image.convert("L").histogram()
            self.assertGreater(sum(histogram[:80]), SIZE[0] * SIZE[1] * 0.8)
            self.assertEqual(json.loads(image.info["MathTranscript"]),
                             [list(row) for row in equation_rows()])
        self.assertLess(SHEET.stat().st_size, 5_000_000)

    def test_notebook_leads_with_one_image_and_a_short_verified_example(self):
        book = json.loads((HERE / "01A-sequence-memory-theory.ipynb").read_bytes())
        cells = {cell["id"]: cell for cell in book["cells"]}
        intro = "".join(cells["gradient-native-intro"]["source"])
        self.assertIn(TAKEAWAY, intro)
        self.assertIn(f"attachment:{SHEET.name}", intro)
        self.assertEqual(base64.b64decode(cells["gradient-native-intro"]["attachments"][SHEET.name]["image/png"]),
                         SHEET.read_bytes())
        primary = "\n".join(
            re.sub(r"<details>.*?</details>", "", "".join(cells[cell_id]["source"]), flags=re.S)
            for cell_id in ("gradient-native-intro", "gradient-vanishing-intuition",
                            "gradient-exploding-intuition")
        )
        self.assertLessEqual(len(primary.split()), 450)
        for memory in (0.8, 1.2):
            example = worked_example(memory)
            self.assertIn(f"{example.long_feedback_ratio:.6f}", primary)
            detail = "".join(cells["gradient-exploding-intuition"]["source"])
            self.assertIn(f"{example.tanh_response:.12f}", detail)
            self.assertIn(f"{example.recurrent_gradient:.12f}", detail)
            self.assertNotIn(f"{example.tanh_response:.12f}", primary)
        self.assertIn("<details>", "".join(cells["gradient-vanishing-intuition"]["source"]))
        self.assertNotIn("S3", primary)
        ids = [cell["id"] for cell in book["cells"]]
        appendix = ids.index("gradient-technical-appendix")
        for cell_id in ("bptt-forward-math", "bptt-backward-math", "bptt-worked-example",
                        "bptt-manual-check"):
            self.assertGreater(ids.index(cell_id), appendix)
        for cell_id in ("bptt-forward-math", "bptt-backward-math", "bptt-worked-example"):
            self.assertTrue("".join(cells[cell_id]["source"]).startswith("<details>"))

    def test_image_helper_preserves_defaults_and_validates_portrait_limits(self):
        path = HERE.parent / "03-language-models" / "generate_illustrations.py"
        spec = importlib.util.spec_from_file_location("curriculum_mai_helper", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertEqual(module.image_dimensions({}), (1536, 864))
        self.assertEqual(module.image_dimensions({"width": 1216, "height": 1792}), (1216, 1792))
        for entry in ({"width": True}, {"height": 767}, {"width": 2048, "height": 2048},
                      {"width": 1200.5}):
            with self.assertRaises(ValueError):
                module.image_dimensions(entry)

    @unittest.skipUnless(any(font.name == bptt_math.FONT for font in font_manager.fontManager.ttflist),
                         "Regeneration requires the existing Windows handwriting font.")
    def test_high_definition_sheet_is_legible_and_reproducible(self):
        with plt.rc_context(bptt_math.HANDWRITING_STYLE):
            figure, axis = sheet_figure()
            try:
                self.assertGreaterEqual(len(axis.texts), 30)
                self.assertTrue(all(text.get_fontsize() >= 12 for text in axis.texts))
                printed = [text.get_text() for text in axis.texts]
                for _, equation in equation_rows():
                    self.assertIn(f"${equation}$", printed)
                self.assertIn("BOTH multiplied pieces change. Product rule: count each effect, then add.", printed)
                self.assertIn("Each earlier derivative is the SAME question, one read farther back.", printed)
            finally:
                plt.close(figure)
        with TemporaryDirectory() as directory:
            export(Path(directory))
            self.assertEqual((Path(directory) / SHEET.name).read_bytes(), SHEET.read_bytes())


if __name__ == "__main__":
    unittest.main()
