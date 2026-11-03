"""Source-aware checks for the paired LSTM/GRU foundation and saved flow evidence."""

import ast
from collections import Counter
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import re
import unittest

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch

from memory_gate_flows import PHASES, draw_gate_flow, fingerprint, recorded_evidence, recorded_steps


class MemoryGateFlowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebook = json.loads((Path(__file__).parent / "01A-sequence-memory-theory.ipynb").read_text(encoding="utf-8"))
        cls.cells = {cell["id"]: cell for cell in cls.notebook["cells"]}
        cls.namespace = {}
        with redirect_stdout(StringIO()):
            for cell_id in (
                "eb7df241", "f6985b8c", "gru-controlled-recording",
                "gru-control-interventions", "gru-keep-drill", "gru-real-cell-parity",
            ):
                exec(compile("".join(cls.cells[cell_id]["source"]), cell_id, "exec"), cls.namespace)
            cls.recordings = recorded_steps(cls.notebook)

    def test_counting_uses_the_existing_illustration_not_an_inline_one_hot_detour(self):
        self.assertNotIn("c3f86706", self.cells)
        introduction = "".join(self.cells["856af87b"]["source"])
        self.assertIn("attachment:01a-01-order-is-not-a-bag.png", introduction)
        self.assertIn("carrying only a small memory?", introduction)
        self.assertNotIn("The next cell checks both", introduction)
        notes = self.namespace["melody"]
        self.assertEqual(Counter(notes), Counter(reversed(notes)))
        self.assertNotEqual(notes, list(reversed(notes)))
        recurrence = "".join(self.cells["ac9aca71"]["source"])
        self.assertIn("state = torch.tanh(W_x @ x_t + W_h @ state)", recurrence)
        self.assertIn("Reversed phrase's final card", recurrence)
        self.assertIn("| Explained or illustrated | Ordered versus pooled inputs",
                      "".join(self.cells["9b40ba08"]["source"]))

    def test_lstm_retained_and_written_parts_reproduce_the_existing_state_trace(self):
        for index, step in enumerate(self.recordings["lstm"]):
            if step.show is None:
                self.fail("LSTM flow must include its show gate.")
            self.assertAlmostEqual(step.retained + step.written, step.carried)
            self.assertAlmostEqual(step.exposed, step.show * np.tanh(step.carried))
            self.assertAlmostEqual(step.carried, self.namespace["cell_history"][index])
            self.assertAlmostEqual(step.exposed, self.namespace["hidden_history"][index])
        self.assertEqual(self.namespace["hidden_if_show_closed"], 0)
        self.assertNotEqual(self.namespace["cell_history"][2], 0)

    def test_gru_reset_proposal_and_complementary_write_reproduce_the_recorded_updates(self):
        for index, step in enumerate(self.recordings["gru"]):
            if step.reset is None or step.input_proposal is None or step.memory_proposal is None:
                self.fail("GRU flow must include the reset and proposal contributions.")
            self.assertAlmostEqual(step.write, 1 - step.keep)
            self.assertAlmostEqual(step.proposal, np.tanh(step.input_proposal + step.reset * step.memory_proposal))
            self.assertAlmostEqual(step.retained + step.written, step.carried)
            self.assertAlmostEqual(step.exposed, step.carried)
            if index:
                self.assertAlmostEqual(step.old, self.recordings["gru"][index - 1].carried)
        self.assertEqual(self.recordings["gru"][0].old, 0)

    def test_interventions_distinguish_reset_from_erasing_the_carry_route(self):
        cases = self.namespace["gru_interventions"]
        self.assertEqual(cases["reset closed"]["retained"], cases["baseline"]["retained"])
        self.assertEqual(cases["reset closed"]["feedback"], 0.8)
        self.assertAlmostEqual(cases["keep everything"]["state"], 0.7)
        self.assertEqual(cases["keep everything"]["feedback"], 1)
        self.assertEqual(cases["replace everything"]["retained"], 0)
        self.assertGreater(cases["replace everything"]["feedback"], 0)
        self.assertEqual(self.namespace["gru_trial_feedback"], self.namespace["gru_keep_to_try"])

    def test_the_one_variable_drill_changes_retention_and_old_state_feedback(self):
        for keep in (0.8, 0.2):
            old = torch.tensor(0.7, dtype=torch.float64, requires_grad=True)
            step = self.namespace["gru_note_step"](old, 0.2, 0, keep)
            feedback = torch.autograd.grad(step["carried"], old)[0].item()
            self.assertAlmostEqual(step["retained"].item(), 0.7 * keep)
            self.assertAlmostEqual(feedback, keep)

    def test_real_initialized_gru_matches_visible_gate_arithmetic_at_every_note(self):
        self.assertLess(self.namespace["gru_real_max_error"], 1e-12)
        self.assertEqual(len(self.namespace["gru_real_trace"]), 6)
        for trace in self.namespace["gru_real_trace"]:
            torch.testing.assert_close(trace["manual"], trace["actual"], rtol=0, atol=1e-12)
            for key in ("reset", "update"):
                self.assertTrue(bool(((trace[key] >= 0) & (trace[key] <= 1)).all()))
        self.assertEqual(sum(value.numel() for value in self.namespace["gru_real_cell"].parameters()), 36)
        self.assertEqual(sum(value.numel() for value in self.namespace["gru_size_match_lstm"].parameters()), 48)

    def test_real_cell_bridge_does_not_change_the_existing_rng_state(self):
        namespace = {"torch": torch, "melody": self.namespace["melody"]}
        before = torch.random.get_rng_state().clone()
        with redirect_stdout(StringIO()):
            exec(compile("".join(self.cells["gru-real-cell-parity"]["source"]), "real-gate-bridge", "exec"), namespace)
        self.assertTrue(torch.equal(torch.random.get_rng_state(), before))

    def test_notebook_references_saved_visuals_and_sections_remain_sequential(self):
        for cell_id in ("ce06d8e3", "955076cb", "lstm-inline-replay", "gru-inline-replay", "musical-echo-replay"):
            self.assertEqual(self.cells[cell_id]["cell_type"], "markdown")
            self.assertRegex("".join(self.cells[cell_id]["source"]), r"!\[[^\]]+\]\(")
        code = "\n".join(
            "".join(cell["source"]) for cell in self.notebook["cells"] if cell["cell_type"] == "code"
        )
        self.assertNotRegex(
            code, r"matplotlib|FuncAnimation|to_jshtml|savefig|plt\.|render_replay|replay_html|display\(|IPython",
        )
        headings = [
            match[1] for cell in self.notebook["cells"] if cell["cell_type"] == "markdown"
            for match in re.finditer(r"(?m)^## (\d+) - ", "".join(cell["source"]))
        ]
        self.assertEqual(headings, list("123456"))
        self.assertIn('<a id="gru-hidden-state-mechanics"></a>', "".join(self.cells["gru-mechanics-intro"]["source"]))
        for cell in self.notebook["cells"]:
            if cell["cell_type"] == "code":
                ast.parse("".join(cell["source"]))

    def test_saved_charts_match_the_notebooks_recorded_measurements(self):
        with redirect_stdout(StringIO()):
            values = recorded_evidence(self.notebook)
        for name, expected in (
            ("feedback-decay", {"distances": values["distances"], "gradient": values["vanilla_gradient"]}),
            ("lstm-state-trace", {key: values[key] for key in (
                "forget", "input_gate", "output", "cell_history", "hidden_history",
            )}),
            ("feedback-retention-comparison", {
                "distances": values["distances"], "vanilla": values["vanilla_gradient"],
                "lstm": values["lstm_gradient"],
            }),
        ):
            with Image.open(Path(__file__).parent / "images" / f"01a-{name}.png") as image:
                self.assertEqual(image.info["SourceFingerprint"], fingerprint(self.notebook))
                measured = json.loads(image.info["MeasuredValues"])
                for key, expected_values in expected.items():
                    np.testing.assert_array_equal(measured[key], expected_values)

    def test_matching_flow_images_are_bounded_and_match_the_same_recorded_phrase(self):
        expected_hash = fingerprint(self.notebook)
        for kind in ("lstm", "gru"):
            prefix = Path(__file__).parent / "images" / f"01a-{kind}-gate-flow"
            with Image.open(prefix.with_suffix(".png")) as image:
                self.assertGreaterEqual(image.width, 1300)
                self.assertGreaterEqual(image.height, 750)
                self.assertEqual(image.info["SourceFingerprint"], expected_hash)
            with Image.open(prefix.with_suffix(".gif")) as image:
                metadata = json.loads(image.info["comment"])
                self.assertEqual(metadata["Notes"], ["D3", "A3", "D4"] * 2)
                self.assertEqual(metadata["SourceFingerprint"], expected_hash)
                self.assertEqual(metadata["StateCount"], 2 if kind == "lstm" else 1)
                self.assertEqual(image.n_frames, len(PHASES) * 6)
                self.assertLessEqual(image.n_frames, 60)
                self.assertAlmostEqual(metadata["FinalCarried"], self.recordings[kind][-1].carried)
                self.assertIn("not trained", metadata["TraceOrigin"])
            html = prefix.with_suffix(".html").read_text(encoding="utf-8")
            self.assertIn("new Animation", html)
            self.assertIn("frames[29]", html)
            self.assertNotIn("frames[30]", html)
            self.assertNotIn("https:", html)
            self.assertNotIn('class="fa ', html)
            self.assertIn(">Next frame</button>", html)
            self.assertIn("max-width:100%", html)
            self.assertRegex(html, r'<img\b[^>]+src="data:image/png;base64,')

    def test_flow_rendering_shows_the_control_paths_and_measured_values(self):
        figure, axis = plt.subplots(figsize=(14, 8))
        try:
            for kind, required in (
                ("lstm", ("KEEP f", "WRITE i", "SHOW o", "new CELL c", "exposed HIDDEN h")),
                ("gru", ("RESET r", "KEEP z", "WRITTEN (1-z)*n", "new HIDDEN h")),
            ):
                draw_gate_flow(axis, kind, self.recordings[kind], 13)
                texts = "\n".join(item.get_text() for item in axis.texts)
                for text in required:
                    self.assertIn(text, texts)
                self.assertIn(f"{self.recordings[kind][2].carried:+.4f}", texts)
                self.assertIn("Chosen scalar controls", texts)
        finally:
            plt.close(figure)


if __name__ == "__main__":
    unittest.main()
