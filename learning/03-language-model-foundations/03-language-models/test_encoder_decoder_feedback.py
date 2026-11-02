"""Measured recurrent-state and embedding-feedback checks for the 03A preview."""

import ast
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import unittest
from tempfile import TemporaryDirectory

import matplotlib.pyplot as plt
from PIL import Image
import torch
from torch import nn

from encoder_decoder_feedback import (
    compare_feedback_routes, draw_state_playback, focus_coordinate, playback_stages,
    record_gru_step, record_state_playback,
    draw_token_layers, token_layer_stages,
    draw_teacher_forcing, teacher_forcing_stages,
)
from language_model import TrainableLanguageModel


class EncoderFeedbackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        notebook = json.loads((Path(__file__).parent / "03A-training-skeleton-and-word-objectives.ipynb").read_text(encoding="utf-8"))
        source = "\n\n".join(
            "".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"
        )
        definition = next(
            node for node in ast.parse(source).body if isinstance(node, ast.ClassDef)
            and node.name == "EncoderDecoderLM"
        )
        namespace = {"nn": nn, "TrainableLanguageModel": TrainableLanguageModel}
        exec(compile(ast.Module(body=[definition], type_ignores=[]), "preview-model", "exec"), namespace)
        cls.model_type = namespace["EncoderDecoderLM"]

    def setUp(self):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(17)
            self.model = self.model_type(9, 4, 6)
        self.source = torch.tensor([[1, 2, 3]])
        self.targets = torch.tensor([[7, 4, 5, 8]])
        self.words = ["<unk>", "the", "sleepy", "cat", "sat", "on", "mat", "<bos>", "<eos>"]

    def compare(self):
        with redirect_stdout(StringIO()):
            return compare_feedback_routes(self.model, self.source, self.targets, self.words)

    def test_unrolled_states_match_the_actual_fused_model_and_keep_all_feedback_routes(self):
        result = self.compare()
        self.assertLess(result.fused_logit_error, 1e-5)
        self.assertLess(result.fused_gradient_error, 1e-5)
        self.assertEqual(len(result.connected.encoder_states), 3)
        self.assertEqual(len(result.connected.decoder_states), 3)
        for state in result.connected.encoder_states + result.connected.decoder_states:
            self.assertEqual(tuple(state.shape), (1, 1, 6))
        self.assertTrue(all(norm > 0 for norm in result.connected.encoder_gradient_norms))
        self.assertTrue(all(norm > 0 for norm in result.connected.decoder_gradient_norms))
        self.assertGreater(result.connected.embedding_gradient[3].norm().item(), 0)
        self.assertGreater(result.connected.row_changes[3].item(), 0)

    def test_cutting_the_handoff_keeps_values_but_blocks_reader_and_source_only_feedback(self):
        result = self.compare()
        torch.testing.assert_close(result.connected.logits, result.detached.logits, rtol=0, atol=0)
        self.assertEqual(result.detached.encoder_gradient_norms, (0, 0, 0))
        self.assertEqual(result.detached.embedding_gradient[3].count_nonzero().item(), 0)
        self.assertEqual(result.detached.row_changes[3].item(), 0)
        self.assertTrue(all(norm > 0 for norm in result.detached.decoder_gradient_norms))
        self.assertGreater(result.detached.embedding_gradient[7].norm().item(), 0)
        self.assertFalse(any(name.startswith("encoder.") for name in result.detached.parameter_gradients))

    def test_stop_is_checked_by_the_head_without_being_read_as_an_input_row(self):
        result = self.compare()
        for trace in (result.connected, result.detached):
            self.assertEqual(trace.embedding_gradient[8].count_nonzero().item(), 0)
            self.assertEqual(trace.row_changes[8].item(), 0)
            self.assertNotEqual(trace.parameter_gradients["output.bias"][8].item(), 0)

    def test_diagnostic_copies_preserve_main_parameters_gradients_modes_ids_and_rng(self):
        self.model.train()
        self.model.encoder.eval()
        for parameter in self.model.parameters():
            parameter.grad = torch.ones_like(parameter)
        parameters = {name: value.detach().clone() for name, value in self.model.named_parameters()}
        gradients = {name: value.grad.clone() for name, value in self.model.named_parameters()}
        modes = [module.training for module in self.model.modules()]
        rng_before = torch.random.get_rng_state().clone()
        source_before, targets_before = self.source.clone(), self.targets.clone()
        self.compare()
        for name, value in self.model.named_parameters():
            torch.testing.assert_close(value, parameters[name], rtol=0, atol=0)
            torch.testing.assert_close(value.grad, gradients[name], rtol=0, atol=0)
        self.assertEqual([module.training for module in self.model.modules()], modes)
        self.assertTrue(torch.equal(torch.random.get_rng_state(), rng_before))
        torch.testing.assert_close(self.source, source_before)
        torch.testing.assert_close(self.targets, targets_before)

    def test_trace_boundaries_are_explicit(self):
        with self.assertRaisesRegex(ValueError, "source sequence"):
            compare_feedback_routes(self.model, torch.empty((1, 0), dtype=torch.long), self.targets, self.words)
        with self.assertRaisesRegex(ValueError, "absent from the writer"):
            compare_feedback_routes(self.model, self.source, torch.tensor([[7, 3, 5, 8]]), self.words)
        self.model.encoder = nn.LSTM(4, 6, batch_first=True)
        with self.assertRaisesRegex(TypeError, "GRU"):
            self.compare()

    def test_playback_records_teacher_inputs_and_generation_guesses_as_distinct_chains(self):
        with torch.no_grad():
            self.model.output.weight.zero_()
            self.model.output.bias.zero_()
            self.model.output.bias[2] = 3
        with redirect_stdout(StringIO()):
            playback = record_state_playback(
                self.model, self.source, self.targets, self.words, epochs=6,
            )
        self.assertEqual(playback.epochs, 6)
        self.assertEqual(playback.feedback.target_ids[0, :-1].tolist(), [7, 4, 5])
        self.assertEqual([step.input_id for step in playback.generation], [7, 2, 2])
        self.assertEqual([step.predicted_id for step in playback.generation], [2, 2, 2])
        torch.testing.assert_close(
            playback.generation[0].state_before, playback.feedback.connected.encoder_states[-1],
        )
        for previous, following in zip(playback.generation, playback.generation[1:]):
            self.assertEqual(following.input_id, previous.predicted_id)
            torch.testing.assert_close(following.state_before, previous.state_after)
        self.assertEqual(playback.omitted_state_error, 0)
        trace = playback.feedback.connected
        torch.testing.assert_close(
            trace.table_after, trace.table_before - 0.1 * trace.embedding_gradient,
        )

    def test_generation_stops_when_its_own_prediction_is_eos(self):
        with torch.no_grad():
            self.model.output.weight.zero_()
            self.model.output.bias.zero_()
            self.model.output.bias[8] = 5
        with redirect_stdout(StringIO()):
            playback = record_state_playback(self.model, self.source, self.targets, self.words)
        self.assertEqual(len(playback.generation), 1)
        self.assertEqual(playback.generation[0].input_id, 7)
        self.assertEqual(playback.generation[0].predicted_id, 8)

    def test_state_playback_preserves_the_main_model_and_records_zero_start_scores(self):
        self.model.train()
        before = {name: value.detach().clone() for name, value in self.model.named_parameters()}
        rng = torch.random.get_rng_state().clone()
        with redirect_stdout(StringIO()):
            playback = record_state_playback(self.model, self.source, self.targets, self.words)
        with torch.inference_mode():
            embedded = self.model.embedding(self.targets[:, :1])
            notes, _ = self.model.decoder(embedded)
            expected = self.model.output(notes[:, -1])
        torch.testing.assert_close(playback.zero_initial_logits, expected)
        for name, value in self.model.named_parameters():
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)
            self.assertIsNone(value.grad)
        self.assertTrue(self.model.training)
        self.assertTrue(torch.equal(torch.random.get_rng_state(), rng))

    def test_rendered_stages_use_recorded_values_and_explicit_training_generation_labels(self):
        with redirect_stdout(StringIO()):
            playback = record_state_playback(self.model, self.source, self.targets, self.words, epochs=6)
        stages = playback_stages(playback)
        self.assertEqual(stages[:4], [("overview", 0), ("encoder", 0),
                                     ("encoder-proposal", 0), ("encoder-blend", 0)])
        self.assertLess(stages.index(("update", 0)), stages.index(("generation", 0)))
        self.assertLessEqual(len(stages) * 2, 90)
        figure, axis = plt.subplots(figsize=(16, 9))
        try:
            for phase, required in (
                ("encoder", "Current word profile x"), ("handoff", "E3 = D0"),
                ("teacher", "Next practice input"), ("writer-backward", "gradient at D"),
                ("update", "cat row after"), ("zero", "PyTorch supplies an all-zero"),
                ("generation", "Generation: construct a hidden state"),
            ):
                index = next(index for index, value in enumerate(stages) if value[0] == phase)
                draw_state_playback(axis, playback, self.words, index)
                text = "\n".join(artist.get_text() for artist in axis.texts)
                self.assertIn(required, text)
                self.assertTrue("Recorded after 6 epochs" in text or "actual 6-epoch cat checkpoint" in text)
        finally:
            plt.close(figure)

    def test_gate_records_reconstruct_every_encoder_and_decoder_hidden_state(self):
        with redirect_stdout(StringIO()):
            playback = record_state_playback(self.model, self.source, self.targets, self.words)
        trace = playback.feedback.connected
        for mechanics, state in zip(
            trace.encoder_mechanics + trace.decoder_mechanics,
            trace.encoder_states + trace.decoder_states,
        ):
            torch.testing.assert_close(mechanics.new_state, state.reshape(-1), rtol=1e-5, atol=1e-6)
            torch.testing.assert_close(mechanics.reset_memory, mechanics.reset_gate * mechanics.memory_proposal)
            torch.testing.assert_close(mechanics.candidate, (mechanics.input_proposal + mechanics.reset_memory).tanh())
            torch.testing.assert_close(mechanics.retained, mechanics.update_gate * mechanics.old_state)
            torch.testing.assert_close(mechanics.written, (1 - mechanics.update_gate) * mechanics.candidate)
            torch.testing.assert_close(mechanics.new_state, mechanics.retained + mechanics.written)
            for gate in (mechanics.reset_gate, mechanics.update_gate):
                self.assertTrue(bool(((gate >= 0) & (gate <= 1)).all()))
        for earlier, later in zip(trace.encoder_mechanics, trace.encoder_mechanics[1:]):
            torch.testing.assert_close(earlier.new_state, later.old_state, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(trace.encoder_mechanics[0].old_state, torch.zeros(6))
        torch.testing.assert_close(
            trace.decoder_mechanics[0].old_state, trace.encoder_states[-1].reshape(-1),
        )

    def test_update_gate_limits_retain_old_state_or_write_the_candidate(self):
        with torch.random.fork_rng(devices=[]):
            gru = nn.GRU(2, 3, batch_first=True)
        inputs = torch.tensor([[0.7, -0.2]])
        previous = torch.tensor([[[0.25, -0.4, 0.9]]])
        for gate_bias in (50.0, -50.0):
            with torch.no_grad():
                gru.weight_ih_l0[3:6].zero_()
                gru.weight_hh_l0[3:6].zero_()
                gru.bias_hh_l0[3:6].zero_()
                gru.bias_ih_l0[3:6].fill_(gate_bias)
            step = record_gru_step(gru, inputs, previous)
            _, actual = gru(inputs[:, None], previous)
            expected = previous.reshape(-1) if gate_bias > 0 else step.candidate
            torch.testing.assert_close(step.new_state, expected, rtol=1e-5, atol=1e-6)
            torch.testing.assert_close(step.new_state, actual.reshape(-1), rtol=1e-5, atol=1e-6)

    def test_pytorch_reset_gate_acts_after_old_state_projection(self):
        with torch.random.fork_rng(devices=[]):
            gru = nn.GRU(2, 3, batch_first=True)
        inputs = torch.tensor([[0.7, -0.2]])
        previous = torch.tensor([[[0.25, -0.4, 0.9]]])
        step = record_gru_step(gru, inputs, previous)
        projected = nn.functional.linear(previous[0], gru.weight_hh_l0[6:], gru.bias_hh_l0[6:])
        torch.testing.assert_close(step.reset_memory, (step.reset_gate * projected).reshape(-1))
        _, actual = gru(inputs[:, None], previous)
        torch.testing.assert_close(step.new_state, actual.reshape(-1), rtol=1e-5, atol=1e-6)

    def test_hidden_state_closeup_displays_full_vectors_and_real_keep_write_arithmetic(self):
        with redirect_stdout(StringIO()):
            playback = record_state_playback(self.model, self.source, self.targets, self.words)
        stage = playback_stages(playback).index(("encoder-blend", 1))
        step = playback.feedback.connected.encoder_mechanics[1]
        q = focus_coordinate(playback)
        figure, axis = plt.subplots(figsize=(16, 9))
        try:
            draw_state_playback(axis, playback, self.words, stage)
            self.assertEqual(len(axis.images), 6)
            torch.testing.assert_close(
                torch.as_tensor(axis.images[1].get_array().data), step.old_state.reshape(1, -1),
            )
            text = "\n".join(artist.get_text() for artist in axis.texts)
            for item in ("Reset gate r", "Update gate z", "Proposed content n", "KEPT:", "WRITTEN:", f"NEW h[{q}]"):
                self.assertIn(item, text)
            self.assertIn(f"{step.new_state[q]:+.4f}", text)
        finally:
            plt.close(figure)

    def test_applied_block_playback_uses_states_and_word_routes_without_a_second_gate_lesson(self):
        with redirect_stdout(StringIO()):
            playback = record_state_playback(self.model, self.source, self.targets, self.words, epochs=6)
        stages = playback_stages(playback, mechanics=False)
        self.assertEqual(stages[:5], [
            ("overview", 0), ("encoder", 0), ("encoder", 1), ("encoder", 2), ("handoff", 0),
        ])
        self.assertLessEqual(len(stages) * 2, 60)
        self.assertFalse(any("proposal" in phase or "blend" in phase for phase, _ in stages))
        figure, axis = plt.subplots(figsize=(16, 9))
        try:
            for phase, required in (
                ("encoder", "old state E"), ("teacher", "Decoder practice"),
                ("handoff", "E3 = D0"), ("generation", "feeds back its own chosen word"),
            ):
                stage = next(index for index, value in enumerate(stages) if value[0] == phase)
                draw_state_playback(axis, playback, self.words, stage, mechanics=False)
                text = "\n".join(artist.get_text() for artist in axis.texts)
                self.assertIn(required, text)
                self.assertNotIn("Reset gate r", text)
                self.assertEqual(len(axis.images), 0)
        finally:
            plt.close(figure)

    def test_token_layer_pair_preserves_native_output_state_handoff_and_separate_head(self):
        with redirect_stdout(StringIO()):
            playback = record_state_playback(self.model, self.source, self.targets, self.words)
        pair = playback.token_layers
        for step in (pair.encoder, pair.decoder):
            self.assertEqual(step.input_mix.shape, (18,))
            self.assertEqual(step.state_mix.shape, (18,))
            torch.testing.assert_close(step.output, step.carried, rtol=0, atol=0)
            torch.testing.assert_close(step.new_state, step.carried, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(pair.encoder.carried, pair.decoder.old_state, rtol=0, atol=0)
        torch.testing.assert_close(
            pair.logits, nn.functional.linear(pair.decoder.output[None], pair.head_weight, pair.head_bias),
            rtol=0, atol=0,
        )
        torch.testing.assert_close(pair.logits, playback.feedback.connected.logits[:, 0], rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(
            pair.logits, playback.generation[0].logits, rtol=1e-5, atol=1e-6,
        )

    def test_layer_animation_zooms_one_cat_read_and_one_bos_read_not_an_invented_stack(self):
        with redirect_stdout(StringIO()):
            playback = record_state_playback(self.model, self.source, self.targets, self.words, epochs=6)
        stages = token_layer_stages()
        self.assertEqual(sum(role == "encoder-layer" for role, _ in stages), 7)
        self.assertEqual(sum(role == "decoder-layer" for role, _ in stages), 7)
        self.assertLessEqual(len(stages), 24)
        figure = plt.figure(figsize=(18, 11))
        try:
            draw_token_layers(figure, playback, self.words, stages.index(("head", 0)))
            text = "\n".join(
                item.get_text() for item in figure.texts + [
                    text for axis in figure.axes for text in axis.texts
                ]
            )
            for term in ("cat + E2", "<bos> + D0", "Linear: 4 -> 18", "Linear: 6 -> 18",
                         "same h_t values", "Linear 6 -> 9", "ignored by encode()"):
                self.assertIn(term, text)
        finally:
            plt.close(figure)

    def test_teacher_forcing_view_uses_correct_inputs_even_when_the_head_is_forced_wrong(self):
        with torch.no_grad():
            self.model.output.weight.zero_()
            self.model.output.bias.zero_()
            self.model.output.bias[6] = 8
        with redirect_stdout(StringIO()):
            playback = record_state_playback(self.model, self.source, self.targets, self.words)
        self.assertEqual(playback.feedback.target_ids[0, :-1].tolist(), [7, 4, 5])
        self.assertEqual([step.input_id for step in playback.generation], [7, 6, 6])
        self.assertEqual(teacher_forcing_stages(playback), [
            ("practice", 0), ("practice", 1), ("practice", 2),
            ("generation", 0), ("generation", 1), ("generation", 2),
        ])
        figure, axis = plt.subplots(figsize=(14, 9))
        try:
            draw_teacher_forcing(axis, playback, self.words, 1)
            text = "\n".join(item.get_text() for item in axis.texts)
            for label in ("Shown now: 'sat'", "Shown now: 'mat'", "Check answer: 'sat'",
                          "No reference word is supplied", "hidden state; do not reset it"):
                self.assertIn(label, text)
        finally:
            plt.close(figure)

    def test_saved_teacher_forcing_view_keeps_inputs_guesses_and_answers_separate(self):
        from encoder_decoder_feedback import TEACHER_FORCING_IMAGE, source_fingerprint
        with Image.open(TEACHER_FORCING_IMAGE) as image:
            self.assertEqual(image.info["SourceFingerprint"], source_fingerprint())
            self.assertEqual(image.info["TrainingEpochs"], "6")
            self.assertEqual(json.loads(image.info["TeacherInputs"]), ["<bos>", "sat", "on"])
            self.assertEqual(json.loads(image.info["CheckingAnswers"]), ["sat", "on", "<eos>"])
            guesses = json.loads(image.info["GenerationGuesses"])
            generation_inputs = json.loads(image.info["GenerationInputs"])
            self.assertEqual(generation_inputs, ["<bos>"] + guesses[:-1])
            self.assertEqual(json.loads(image.info["TeacherGuesses"])[0], guesses[0])
            self.assertTrue(all(0 <= value <= 1 for value in json.loads(image.info["CorrectWordShares"])))
        with Image.open(TEACHER_FORCING_IMAGE.with_suffix(".gif")) as image:
            self.assertEqual(image.n_frames, 3 + len(guesses))
        html = TEACHER_FORCING_IMAGE.with_suffix(".html").read_text(encoding="utf-8")
        self.assertNotIn("https:", html)
        self.assertIn(">Next frame</button>", html)

    def test_saved_diagram_matches_the_current_executable_sources(self):
        from encoder_decoder_feedback import IMAGE, source_fingerprint
        from PIL import Image

        with Image.open(IMAGE) as image:
            self.assertEqual(image.format, "PNG")
            self.assertGreaterEqual(image.width, 1800)
            self.assertGreaterEqual(image.height, 1000)
            self.assertEqual(image.info["SourceFingerprint"], source_fingerprint())
            self.assertEqual(image.info["SourceOnlyWord"], "cat")
            self.assertGreater(float(image.info["ConnectedRowChange"]), 0)
            self.assertEqual(float(image.info["DetachedRowChange"]), 0)
            self.assertEqual(image.info["TrainingEpochs"], "6")

    def test_saved_animation_is_bounded_and_matches_the_actual_cat_probe(self):
        from encoder_decoder_feedback import ANIMATION, HTML_ANIMATION, source_fingerprint
        from PIL import Image

        with Image.open(ANIMATION) as image:
            metadata = json.loads(image.info["comment"])
            self.assertGreaterEqual(image.width, 1200)
            self.assertGreaterEqual(image.height, 900)
            self.assertGreaterEqual(image.n_frames, len(metadata["Stages"]))
            self.assertLessEqual(image.n_frames, 90)
            self.assertEqual(metadata["SourceFingerprint"], source_fingerprint())
            self.assertEqual(metadata["Source"], ["the", "sleepy", "cat"])
            self.assertEqual(metadata["CheckingAnswers"], ["sat", "on", "<eos>"])
            self.assertEqual(metadata["TrainingEpochs"], 6)
            self.assertEqual(metadata["OmittedStateError"], 0)
            self.assertTrue(1 <= len(metadata["Generated"]) <= 3)
            self.assertIn("reset projected memory", metadata["GateMechanics"])
            self.assertTrue(0 <= metadata["FocusCoordinate"] < 48)
            self.assertEqual(metadata["PlaybackView"], "one-token-through-each-layer")
            self.assertEqual(metadata["EncoderToken"], "cat")
            self.assertEqual(metadata["DecoderToken"], "<bos>")
            self.assertEqual(metadata["RecurrentLayers"], 1)
            self.assertTrue(metadata["GRUOutputEqualsCarriedState"])
            self.assertFalse(any("proposal" in phase or "blend" in phase for phase, _ in metadata["Stages"]))
        html = HTML_ANIMATION.read_text(encoding="utf-8")
        self.assertIn("new Animation", html)
        self.assertIn("frames[0]", html)
        self.assertNotIn("https:", html)
        self.assertIn(">Next frame</button>", html)

    def test_static_hidden_mechanics_preserves_the_measured_keep_write_sum(self):
        from encoder_decoder_feedback import MECHANICS_IMAGE, source_fingerprint
        from PIL import Image

        with Image.open(MECHANICS_IMAGE) as image:
            self.assertGreaterEqual(image.width, 1500)
            self.assertGreaterEqual(image.height, 800)
            self.assertEqual(image.info["SourceFingerprint"], source_fingerprint())
            self.assertLess(float(image.info["ReconstructionError"]), 1e-5)
            self.assertTrue(0 <= float(image.info["ResetGate"]) <= 1)
            self.assertTrue(0 <= float(image.info["UpdateGate"]) <= 1)
            self.assertAlmostEqual(
                float(image.info["Retained"]) + float(image.info["Written"]),
                float(image.info["NewState"]), places=6,
            )

    def test_focused_embedding_and_order_views_report_measured_boundaries(self):
        from encoder_decoder_feedback import (
            EMBEDDING_LEARNING_IMAGE, ORDER_PRELUDE_IMAGE, PROFILE_LEARNING_IMAGE, source_fingerprint,
        )
        with Image.open(EMBEDDING_LEARNING_IMAGE) as image:
            self.assertEqual(image.info["SourceFingerprint"], source_fingerprint())
            self.assertEqual(image.info["Checkpoint"], "fresh model before training")
            self.assertGreater(float(image.info["ConnectedCatRowChange"]), 0)
            self.assertEqual(float(image.info["DetachedCatRowChange"]), 0)
        with Image.open(PROFILE_LEARNING_IMAGE) as image:
            self.assertEqual(image.info["SourceFingerprint"], source_fingerprint())
            self.assertEqual(image.info["TrainingEpochs"], "6")
            self.assertEqual(float(image.info["EOSRowChange"]), 0)
            self.assertEqual(set(json.loads(image.info["PairSimilarities"])), {"cat/dog", "cat/mat"})
        with Image.open(ORDER_PRELUDE_IMAGE) as image:
            self.assertEqual(image.info["SourceFingerprint"], source_fingerprint())
            self.assertLess(float(image.info["MeanDifference"]), 1e-6)
            self.assertGreater(float(image.info["HiddenDifference"]), 1e-5)
            self.assertGreater(float(image.info["ScoreDifference"]), 1e-5)
            self.assertEqual(image.info["AttentionImplemented"], "False")

    def test_explicit_rebuild_refreshes_only_the_embedded_gif(self):
        import base64
        from encoder_decoder_feedback import ANIMATION, refresh_saved_replay_attachment
        notebook = {"cells": [
            {"cell_type": "markdown", "source": ["saved replay"], "metadata": {},
             "attachments": {ANIMATION.name: {"image/gif": "stale"}, "other.png": {"image/png": "kept"}}},
            {"cell_type": "code", "source": ["value = 1"], "metadata": {},
             "outputs": [], "execution_count": None},
        ]}
        with TemporaryDirectory() as directory:
            path = Path(directory) / "notebook.ipynb"
            path.write_bytes((json.dumps(notebook, ensure_ascii=False, indent=1) + "\n").encode("utf-8"))
            refresh_saved_replay_attachment(path, ANIMATION)
            updated = json.loads(path.read_bytes())
        self.assertEqual(updated["cells"][1], notebook["cells"][1])
        self.assertEqual(updated["cells"][0]["source"], notebook["cells"][0]["source"])
        self.assertEqual(updated["cells"][0]["metadata"], notebook["cells"][0]["metadata"])
        self.assertEqual(updated["cells"][0]["attachments"]["other.png"], {"image/png": "kept"})
        self.assertEqual(base64.b64decode(updated["cells"][0]["attachments"][ANIMATION.name]["image/gif"]),
                         ANIMATION.read_bytes())


if __name__ == "__main__":
    unittest.main()
