"""Verify the concise route uses the shown models and preserves a runnable closer look."""

import ast
import base64
from html.parser import HTMLParser
import inspect
from io import BytesIO
import json
from pathlib import Path
import unittest

from notebook_replays import FORWARD_STEP_FILES
from PIL import Image
from sequence_music_comparison import RNNSequenceModel, LSTMSequenceModel, GRUSequenceModel


HERE = Path(__file__).parent


class AudioSources(HTMLParser):
    def __init__(self):
        super().__init__()
        self.sources = []

    def handle_starttag(self, tag, attrs):
        if tag in {"audio", "source"}:
            self.sources.extend(value for key, value in attrs if key == "src")


class SequenceNotebookFlowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.main = json.loads((HERE / "01A-sequence-memory-theory.ipynb").read_bytes())
        cls.details = json.loads((HERE / "01A1-sequence-memory-worked-details.ipynb").read_bytes())
        cls.cells = {cell["id"]: cell for cell in cls.main["cells"]}

    def test_prediction_problem_motivates_count_failure_and_state_before_the_metaphor(self):
        counts = " ".join("".join(self.cells["main-counting-and-order"]["source"]).split("![", 1)[0].split())
        self.assertIn("same counts", counts)
        self.assertIn("next note repeats the first", counts)
        self.assertIn("which of those two answers", counts)
        memory = " ".join("".join(self.cells["main-rnn-intuition"]["source"]).split())
        self.assertLess(memory.index("clue from the past"), memory.index("running summary"))
        self.assertLess(memory.index("running summary"), memory.index("notecard"))
        self.assertLess(memory.index("hidden state h"), memory.index("notecard"))
        self.assertIn("which first note **this phrase** had", memory)

    def test_six_applied_cells_use_the_visible_generic_models(self):
        codes = [cell for cell in self.main["cells"] if cell["cell_type"] == "code"]
        self.assertEqual(len(codes), 6)
        for kind, model in (("rnn", RNNSequenceModel), ("lstm", LSTMSequenceModel), ("gru", GRUSequenceModel)):
            generic = ast.parse("".join(self.cells[f"generic-{kind}-model"]["source"]))
            shown = next(node for node in generic.body if isinstance(node, ast.ClassDef))
            expected = ast.parse(inspect.getsource(model)).body[0]
            self.assertEqual(ast.dump(shown), ast.dump(expected))
            application = ast.parse("".join(self.cells[f"music-{kind}-application"]["source"]))
            calls = [node for node in ast.walk(application) if isinstance(node, ast.Call)
                     and isinstance(node.func, ast.Name) and node.func.id == "run_music_experiment"]
            self.assertEqual(len(calls), 1)
            self.assertIsInstance(calls[0].args[0], ast.Name)
            self.assertEqual(calls[0].args[0].id, model.__name__)
            self.assertEqual(calls[0].args[1].id, "music_data")

    def test_cached_results_share_the_same_data_prompt_and_budget(self):
        reports = [
            json.loads((HERE / "images" / f"01a-sequence-music-comparison-{kind}.json").read_bytes())
            for kind in ("rnn", "lstm", "gru")
        ]
        for report in reports[1:]:
            self.assertEqual(report["dataset"], reports[0]["dataset"])
            self.assertEqual(report["config"], reports[0]["config"])
            self.assertEqual(report["example"], reports[0]["example"])
        for kind, report in zip(("rnn", "lstm", "gru"), reports, strict=True):
            cell = self.cells[f"music-{kind}-application"]
            self.assertEqual(cell["metadata"]["saved_music_report"]["dataset_fingerprint"],
                             report["dataset"]["fingerprint"])
            html = "".join(next(output["data"]["text/html"] for output in cell["outputs"]
                               if "text/html" in output.get("data", {})))
            media = AudioSources()
            media.feed(html)
            self.assertGreaterEqual(len(media.sources), 3)
            self.assertTrue(all(source.startswith("data:audio/wav;base64,") for source in media.sources))
            self.assertLess(report["after"]["heldout"]["loss"], report["before"]["heldout"]["loss"])

    def test_companion_keeps_the_context_and_original_executable_checks(self):
        ids = {cell["id"] for cell in self.details["cells"]}
        self.assertTrue({
            "worked-details-reading-guide", "ac9aca71", "gradient-meaning-probe",
            "bptt-manual-check", "lstm-copying-vs-squashing-demo",
            "lstm-real-cell-parity", "gru-real-cell-parity", "gated-gradient-limits",
            "worked-details-return",
        } <= ids)
        self.assertEqual(sum(cell["cell_type"] == "code" for cell in self.details["cells"]), 29)
        guide = "".join(self.details["cells"][0]["source"])
        self.assertIn("D3, A3 and D4", guide)
        self.assertIn("01A-sequence-memory-theory.ipynb", guide)
        for cell in self.details["cells"]:
            if cell["cell_type"] == "code":
                ast.parse("".join(cell["source"]))

    def test_main_manual_steps_are_embedded_with_static_fallbacks_and_full_companion_replays(self):
        players = {
            cell["metadata"]["saved_replay"]["file"]: cell
            for cell in self.main["cells"] if "saved_replay" in cell.get("metadata", {})
        }
        lstm = "01a-lstm-forward-walkthrough.html"
        self.assertNotIn(lstm, players)
        gru = "01a-gru-sequence-storyboard.html"
        self.assertNotIn(gru, players)
        companion_players = {
            cell["metadata"]["saved_replay"]["file"]: cell
            for cell in self.details["cells"] if "saved_replay" in cell.get("metadata", {})
        }
        for name in (lstm, gru):
            self.assertIn(name, companion_players)
            self.assertIn(f'data-saved-replay="{name}"', "".join(companion_players[name]["source"]))
        for name in FORWARD_STEP_FILES:
            self.assertIn(name, players)
            self.assertIn(f'data-saved-replay="{name}"', "".join(players[name]["source"]))
            self.assertEqual(players[name]["cell_type"], "markdown")
        text = "\n".join("".join(cell["source"]) for cell in self.main["cells"]
                         if cell["cell_type"] == "markdown" and "saved_replay" not in cell["metadata"])
        self.assertIn("not one architecture's states literally glued together", text)
        self.assertIn("not raw notes", text)
        intro = self.cells["main-lstm-story-intro"]
        source = "".join(intro["source"])
        self.assertIn("01A1-sequence-memory-worked-details.ipynb#lstm-complete-read", source)
        self.assertIn("Play", source)
        self.assertNotIn("](attachment:01a-lstm-forward-walkthrough.gif)", source)
        gru_intro = self.cells["main-gru-intuition"]
        gru_source = "".join(gru_intro["source"])
        self.assertIn("Next stage", gru_source)
        self.assertNotIn("](attachment:01a-gru-forward-walkthrough.gif)", gru_source)
        self.assertIn("01A1-sequence-memory-worked-details.ipynb#gru-forward-details",
                      "".join(self.cells["main-gru-steps"]["source"]))
        order = [cell["id"] for cell in self.main["cells"]]
        for kind, filename in zip(("lstm", "gru"), FORWARD_STEP_FILES, strict=True):
            fallback = self.cells[f"main-{kind}-steps"]
            image_name = f"01a-{kind}-forward-walkthrough.png"
            self.assertIn(f"](attachment:{image_name})", "".join(fallback["source"]))
            raw = base64.b64decode(fallback["attachments"][image_name]["image/png"])
            self.assertEqual(raw, (HERE / "images" / image_name).read_bytes())
            with Image.open(BytesIO(raw)) as image:
                self.assertEqual(image.format, "PNG")
            self.assertLess(order.index(players[filename]["id"]), order.index(fallback["id"]))


if __name__ == "__main__":
    unittest.main()
