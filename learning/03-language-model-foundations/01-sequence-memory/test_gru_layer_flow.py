"""Verify that the token-layer diagrams show native GRU computations and output semantics."""

from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import unittest

import matplotlib.pyplot as plt
from PIL import Image
import torch
from torch import nn

from gru_layer_flow import LAYER_PHASES, draw_layer, record_layer_step
from memory_gate_flows import fingerprint, recorded_foundation_layer
from memory_visuals import MOTION_FRAMES


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
        notebook = json.loads((here / "01A-sequence-memory-theory.ipynb").read_text(encoding="utf-8"))
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


if __name__ == "__main__":
    unittest.main()
