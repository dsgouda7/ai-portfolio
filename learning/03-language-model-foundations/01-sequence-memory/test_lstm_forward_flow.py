"""Verify that the visual LSTM states and prediction routes are real and distinct."""

import base64
from copy import deepcopy
from dataclasses import asdict
from html import unescape
from io import BytesIO
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory
import unittest
from xml.etree import ElementTree

from PIL import Image
import torch

from backprop_memory_flow import NOTES, NOTE_HEAD_WEIGHTS, _inputs_and_weights
from lstm_forward_flow import (
    ART, ART_MANIFEST, NOTEBOOK, PHASES, PLAYER, STORYBOARD,
    export_player, export_storyboard, load_cell_and_inputs, note_readout,
    read_recording, record_reads, render_player, role_images, scene,
)
from lstm_layer_flow import record_lstm_layer_step
from lstm_state_roles import load_state_roles
from notebook_replays import LSTM_FORWARD_FILE, refresh_embedded_replays, saved_replay


class LSTMForwardFlowTests(unittest.TestCase):
    def test_native_values_and_both_carried_states_match_each_read(self):
        cell, inputs = load_cell_and_inputs()
        original_inputs, _, _ = _inputs_and_weights(NOTEBOOK)
        torch.testing.assert_close(inputs, original_inputs, rtol=0, atol=0)
        hidden = torch.zeros((1, 2), dtype=torch.float64)
        memory = torch.zeros_like(hidden)
        reads = record_reads()
        self.assertEqual(tuple(read.note for read in reads), NOTES)
        for offset, read in enumerate(reads):
            with torch.no_grad():
                actual_h, actual_c = cell(inputs[offset:offset + 1], (hidden, memory))
            self.assertEqual(read.inputs, tuple(inputs[offset].tolist()))
            self.assertEqual(read.old_hidden, tuple(hidden[0].tolist()))
            self.assertEqual(read.old_cell, tuple(memory[0].tolist()))
            self.assertEqual(read.new_hidden, tuple(actual_h[0].tolist()))
            self.assertEqual(read.new_cell, tuple(actual_c[0].tolist()))
            self.assertEqual(len(read.new_hidden), len(read.new_cell))
            self.assertEqual(len(read.new_cell), 2)
            self.assertEqual(read.next_note, NOTES[(offset + 1) % 3])
            if offset:
                self.assertEqual(read.old_hidden, reads[offset - 1].new_hidden)
                self.assertEqual(read.old_cell, reads[offset - 1].new_cell)
            hidden, memory = actual_h, actual_c

    def test_gate_arithmetic_and_head_read_the_correct_values(self):
        for read in record_reads():
            for index in range(2):
                self.assertAlmostEqual(read.retained[index], read.keep[index] * read.old_cell[index], places=14)
                self.assertAlmostEqual(read.written[index], read.write[index] * read.proposal[index], places=14)
                self.assertAlmostEqual(read.new_cell[index], read.retained[index] + read.written[index], places=14)
                expected_h = read.show[index] * torch.tanh(torch.tensor(read.new_cell[index], dtype=torch.float64)).item()
                self.assertAlmostEqual(read.new_hidden[index], expected_h, places=14)
            hidden = torch.tensor(read.new_hidden, dtype=torch.float64)
            weights = torch.tensor(NOTE_HEAD_WEIGHTS, dtype=torch.float64)
            self.assertEqual(read.readout.scores, tuple((weights @ hidden).tolist()))
            self.assertEqual(read.readout.probabilities, tuple((weights @ hidden).softmax(0).tolist()))
            self.assertNotEqual(read.readout, note_readout(torch.tensor(read.new_cell, dtype=torch.float64)))

    def test_a3_snapshot_is_the_existing_lessons_native_step(self):
        actual = record_reads()[1]
        existing = load_state_roles().baseline
        self.assertEqual(actual.old_hidden, tuple(existing.old_hidden.tolist()))
        self.assertEqual(actual.old_cell, tuple(existing.old_cell.tolist()))
        self.assertEqual(actual.new_cell, tuple(existing.carried_cell.tolist()))
        self.assertEqual(actual.new_hidden, tuple(existing.carried_hidden.tolist()))

    def test_old_c_supplies_content_but_does_not_choose_standard_gates(self):
        cell, inputs = load_cell_and_inputs()
        read = record_reads()[1]
        hidden = torch.tensor([read.old_hidden], dtype=torch.float64)
        memory = torch.tensor([read.old_cell], dtype=torch.float64)
        baseline = record_lstm_layer_step(cell, inputs[1:2], hidden, memory)
        delta = torch.tensor([[0.1, -0.2]], dtype=torch.float64)
        altered = record_lstm_layer_step(cell, inputs[1:2], hidden, memory + delta)
        for name in ("input_gate", "forget_gate", "candidate", "output_gate"):
            torch.testing.assert_close(getattr(baseline, name), getattr(altered, name), rtol=0, atol=0)
        torch.testing.assert_close(
            altered.new_cell - baseline.new_cell, baseline.forget_gate * delta[0], rtol=1e-12, atol=1e-12,
        )
        self.assertFalse(torch.equal(altered.new_hidden, baseline.new_hidden))
        changed_h = record_lstm_layer_step(cell, inputs[1:2], hidden + delta, memory)
        self.assertEqual(tuple(changed_h.old_cell.tolist()), read.old_cell)
        self.assertFalse(torch.equal(changed_h.input_mix + changed_h.state_mix, baseline.input_mix + baseline.state_mix))

    def test_closing_show_preserves_c_but_removes_its_exposure_to_the_head(self):
        for read in record_reads():
            self.assertEqual(read.closed_show_cell, read.new_cell)
            self.assertNotEqual(read.new_cell, (0, 0))
            self.assertEqual(read.closed_show_hidden, (0, 0))
            self.assertEqual(read.closed_show_readout.scores, (0, 0, 0))
            self.assertEqual(read.closed_show_readout.probabilities, (1 / 3,) * 3)

    def test_diagram_dependencies_require_new_h_before_prediction(self):
        boxes, wires, _ = scene(record_reads()[1])
        controls = next(box for box in boxes if box.name == "controls")
        self.assertEqual(controls.labels[0].text, "Compute controls")
        self.assertIn("Shared weights", controls.labels[1].text)
        incoming = {}
        for wire in wires:
            incoming.setdefault(wire.target, set()).add(wire.source)
        self.assertEqual(incoming["controls"], {"input", "old-hidden"})
        self.assertEqual(incoming["prediction"], {"new-hidden"})
        self.assertEqual(incoming["new-cell"], {"add"})
        self.assertEqual(incoming["add"], {"kept", "written"})
        self.assertEqual(incoming["new-hidden"], {"show"})
        self.assertEqual(incoming["show"], {"new-cell", "controls"})
        self.assertEqual(incoming["next-cell"], {"new-cell"})
        self.assertEqual(incoming["next-hidden"], {"new-hidden"})

    def test_recording_is_deterministic_without_global_rng_or_thread_changes(self):
        rng = torch.random.get_rng_state().clone()
        threads = torch.get_num_threads()
        self.assertEqual(record_reads(), record_reads())
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        self.assertEqual(threads, torch.get_num_threads())
        data = read_recording()
        self.assertEqual(data["version"], 1)
        self.assertEqual(data["state_width"], 2)
        self.assertEqual(data["head_weights"], [list(row) for row in NOTE_HEAD_WEIGHTS])
        self.assertEqual(data["steps"], json.loads(json.dumps([asdict(read) for read in record_reads()])))
        self.assertEqual(len(data["scenes"]), 3)
        self.assertEqual(len(PHASES), 7)
        for svg in data["scenes"]:
            root = ElementTree.fromstring(svg)
            self.assertEqual(root.attrib["viewBox"], "0 0 1110 740")
            self.assertEqual(len([node for node in root.iter("image")]), 2)
            self.assertEqual(len([node for node in root.iter("polyline") if node.attrib.get("data-to") == "prediction"]), 1)

    def test_exports_are_self_contained_and_reproducible(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / PLAYER.name
            export_player(path)
            self.assertEqual(path.read_bytes(), PLAYER.read_bytes())
            self.assertEqual(render_player(), PLAYER.read_text(encoding="utf-8"))
            png = path.with_suffix(".png")
            export_storyboard(png)
            with Image.open(STORYBOARD) as saved, Image.open(png) as rebuilt:
                self.assertEqual(saved.tobytes(), rebuilt.tobytes())
                self.assertEqual(saved.info["HeadInput"], "new_hidden")
                self.assertEqual(saved.info["StateWidth"], "2")
        for role, image in role_images().items():
            self.assertIn(role, ("cell", "hidden"))
            self.assertEqual(image.size, (240, 180))
        manifest = json.loads(ART_MANIFEST.read_text(encoding="utf-8"))[0]
        self.assertEqual(manifest["model"], "MAI-Image-2.6")
        self.assertEqual(manifest["filename"], ART.name)

    def test_native_markdown_embed_has_no_code_box_and_protects_manual_edits(self):
        result = saved_replay(LSTM_FORWARD_FILE).data
        self.assertIn('sandbox="allow-scripts"', result)
        self.assertNotIn("allow-same-origin", result)
        document = unescape(re.search(r'srcdoc="([^"]+)"', result)[1])
        self.assertIn('data-player="lstm-forward-walkthrough"', document)
        self.assertIn('data-from=\\"new-hidden\\"', document)
        self.assertNotIn('src="http', document)
        self.assertNotIn('href="http', document)
        display = {"cell_type": "markdown", "source": [], "metadata": {"saved_replay": {"file": LSTM_FORWARD_FILE}}}
        book = {"cells": [display]}
        refresh_embedded_replays(book)
        self.assertNotIn("outputs", display)
        self.assertNotIn("execution_count", display)
        self.assertNotIn("\n", display["source"][0])
        snapshot = deepcopy(book)
        refresh_embedded_replays(book)
        self.assertEqual(book, snapshot)
        display["source"][0] += "manual change"
        with self.assertRaisesRegex(ValueError, "Markdown player changed"):
            refresh_embedded_replays(book)

    def test_invalid_head_inputs_are_reported(self):
        for hidden in (torch.zeros(3), torch.tensor([float("nan"), 0])):
            with self.assertRaises(ValueError):
                note_readout(hidden)


if __name__ == "__main__":
    unittest.main()
